"""Decoder helpers for CTC-based inference.

This module keeps decoding logic separate from the acoustic model so we can
progressively add stronger decoders without rewriting inference call sites.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.utils import BLANK_IDX, MAX_FRAMES, char_indices_to_text, parse_alignment_text


ROOT_DIR = Path(__file__).resolve().parent.parent
GRID_NGRAM_ARTIFACT = ROOT_DIR / "temp_reports" / "decoder_artifacts" / "grid_word_trigram.json"
GRID_ALIGN_GLOB = "data/**/align/*.align"

DEFAULT_DECODER_MODE = "greedy_ctc"
DEFAULT_BEAM_WIDTH = 10
DEFAULT_DEBUG_TOP_K = 5
DEFAULT_NGRAM_ALPHA = 0.8
DEFAULT_NGRAM_TOP_CANDIDATES = 10

DECODER_SPECS = {
    "greedy_ctc": {
        "label": "Greedy CTC",
        "description": "Fast baseline decoder using argmax-style CTC greedy decoding.",
    },
    "beam_ctc": {
        "label": "CTC Beam Search",
        "description": "Keeps multiple candidate paths and returns top beam hypotheses.",
    },
    "beam_ngram_grid": {
        "label": "CTC Beam + GRID N-gram",
        "description": "Reranks CTC beam candidates with a GRID transcript trigram language model.",
    },
}


@dataclass
class DecodeResult:
    mode: str
    label: str
    final_text: str
    collapsed_indices: list[int]
    hypotheses: list[dict[str, Any]]
    beam_width: int
    debug_top_k: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WordNGramLM:
    vocabulary: list[str]
    unigram_counts: dict[str, int]
    bigram_counts: dict[str, int]
    trigram_counts: dict[str, int]
    total_tokens: int
    add_k: float = 0.5

    def score_text(self, text: str) -> float:
        words = [w for w in text.strip().split() if w]
        tokens = ["<s>", "<s>", *words, "</s>"]
        score = 0.0
        for idx in range(2, len(tokens)):
            score += self._score_next(tokens[idx - 2], tokens[idx - 1], tokens[idx])
        return score

    def _score_next(self, left2: str, left1: str, word: str) -> float:
        vocab_size = max(len(self.vocabulary), 1)

        trigram_key = _join_ngram((left2, left1, word))
        bigram_context = _join_ngram((left2, left1))
        trigram_count = self.trigram_counts.get(trigram_key, 0)
        bigram_context_count = self.bigram_counts.get(bigram_context, 0)
        if bigram_context_count > 0:
            prob = (trigram_count + self.add_k) / (
                bigram_context_count + self.add_k * vocab_size
            )
            return math.log(prob)

        bigram_key = _join_ngram((left1, word))
        unigram_context_count = self.unigram_counts.get(left1, 0)
        bigram_count = self.bigram_counts.get(bigram_key, 0)
        if unigram_context_count > 0:
            prob = (bigram_count + self.add_k) / (
                unigram_context_count + self.add_k * vocab_size
            )
            return math.log(prob)

        unigram_count = self.unigram_counts.get(word, 0)
        prob = (unigram_count + self.add_k) / (
            self.total_tokens + self.add_k * vocab_size
        )
        return math.log(prob)


_WORD_NGRAM_CACHE: WordNGramLM | None = None


def list_decoder_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for mode, spec in DECODER_SPECS.items():
        entry = {
            "mode": mode,
            "label": spec["label"],
            "description": spec["description"],
            "available": True,
        }
        if mode == "beam_ngram_grid":
            entry["artifact_path"] = str(GRID_NGRAM_ARTIFACT)
            entry["auto_build"] = True
        specs.append(entry)
    return specs


def decode_logits(
    logits: tf.Tensor | np.ndarray,
    mode: str = DEFAULT_DECODER_MODE,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    debug_top_k: int = DEFAULT_DEBUG_TOP_K,
    ngram_alpha: float = DEFAULT_NGRAM_ALPHA,
) -> DecodeResult:
    if mode not in DECODER_SPECS:
        raise ValueError(f"Unsupported decoder mode: {mode}")

    logits_np = logits.numpy() if isinstance(logits, tf.Tensor) else np.asarray(logits)
    if logits_np.ndim != 3 or logits_np.shape[0] != 1:
        raise ValueError("decode_logits expects logits with shape (1, T, C).")

    if mode == "greedy_ctc":
        return _decode_greedy(logits_np, debug_top_k=debug_top_k)
    if mode == "beam_ctc":
        return _decode_beam(logits_np, beam_width=beam_width, debug_top_k=debug_top_k)
    return _decode_beam_ngram_grid(
        logits_np,
        beam_width=beam_width,
        debug_top_k=debug_top_k,
        ngram_alpha=ngram_alpha,
    )


def _decode_greedy(logits_np: np.ndarray, debug_top_k: int) -> DecodeResult:
    logits_t = tf.transpose(tf.convert_to_tensor(logits_np), [1, 0, 2])
    input_length = tf.fill([logits_np.shape[0]], MAX_FRAMES)
    decoded, _ = tf.nn.ctc_greedy_decoder(logits_t, input_length, blank_index=BLANK_IDX)
    dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()

    indices = dense[0]
    indices = indices[indices >= 0].tolist()
    text = char_indices_to_text(indices).lower()

    return DecodeResult(
        mode="greedy_ctc",
        label=DECODER_SPECS["greedy_ctc"]["label"],
        final_text=text,
        collapsed_indices=indices,
        hypotheses=[
            {
                "rank": 1,
                "text": text,
                "collapsed_indices": indices,
                "acoustic_score": None,
                "lm_score": None,
                "combined_score": None,
            }
        ],
        beam_width=1,
        debug_top_k=max(1, debug_top_k),
    )


def _decode_beam(logits_np: np.ndarray, beam_width: int, debug_top_k: int) -> DecodeResult:
    beam_width = max(2, int(beam_width))
    top_paths = max(1, min(int(debug_top_k), beam_width))
    hypotheses = _beam_hypotheses(logits_np, beam_width=beam_width, top_paths=top_paths)

    final = hypotheses[0] if hypotheses else {"text": "", "collapsed_indices": []}
    return DecodeResult(
        mode="beam_ctc",
        label=DECODER_SPECS["beam_ctc"]["label"],
        final_text=final["text"],
        collapsed_indices=final["collapsed_indices"],
        hypotheses=hypotheses,
        beam_width=beam_width,
        debug_top_k=top_paths,
    )


def _decode_beam_ngram_grid(
    logits_np: np.ndarray,
    beam_width: int,
    debug_top_k: int,
    ngram_alpha: float,
) -> DecodeResult:
    beam_width = max(2, int(beam_width))
    top_paths = max(DEFAULT_NGRAM_TOP_CANDIDATES, min(beam_width, max(debug_top_k, 3)))
    base_hypotheses = _beam_hypotheses(logits_np, beam_width=beam_width, top_paths=top_paths)

    lm = _get_or_build_word_ngram_lm()
    rescored: list[dict[str, Any]] = []
    for item in base_hypotheses:
        lm_score = lm.score_text(item["text"])
        acoustic = item["acoustic_score"] if item["acoustic_score"] is not None else 0.0
        combined = acoustic + ngram_alpha * lm_score
        rescored.append(
            {
                **item,
                "lm_score": lm_score,
                "combined_score": combined,
            }
        )

    rescored.sort(key=lambda hyp: hyp["combined_score"], reverse=True)
    for rank, item in enumerate(rescored, start=1):
        item["rank"] = rank

    final = rescored[0] if rescored else {"text": "", "collapsed_indices": []}
    return DecodeResult(
        mode="beam_ngram_grid",
        label=DECODER_SPECS["beam_ngram_grid"]["label"],
        final_text=final["text"],
        collapsed_indices=final["collapsed_indices"],
        hypotheses=rescored[: max(1, debug_top_k)],
        beam_width=beam_width,
        debug_top_k=max(1, debug_top_k),
        metadata={
            "lm_type": "grid_word_trigram",
            "lm_artifact": str(GRID_NGRAM_ARTIFACT),
            "ngram_alpha": ngram_alpha,
        },
    )


def _beam_hypotheses(
    logits_np: np.ndarray,
    beam_width: int,
    top_paths: int,
) -> list[dict[str, Any]]:
    logits_t = tf.transpose(tf.convert_to_tensor(logits_np), [1, 0, 2])
    input_length = tf.fill([logits_np.shape[0]], MAX_FRAMES)
    decoded_paths, log_probs = tf.nn.ctc_beam_search_decoder(
        logits_t,
        input_length,
        beam_width=beam_width,
        top_paths=top_paths,
    )

    hypotheses: list[dict[str, Any]] = []
    for rank, sparse_path in enumerate(decoded_paths, start=1):
        dense = tf.sparse.to_dense(sparse_path, default_value=-1).numpy()
        indices = dense[0]
        indices = indices[indices >= 0].tolist()
        text = char_indices_to_text(indices).lower()
        score = float(log_probs.numpy()[0, rank - 1])
        hypotheses.append(
            {
                "rank": rank,
                "text": text,
                "collapsed_indices": indices,
                "acoustic_score": score,
                "lm_score": None,
                "combined_score": score,
            }
        )
    return hypotheses


def _get_or_build_word_ngram_lm() -> WordNGramLM:
    global _WORD_NGRAM_CACHE
    if _WORD_NGRAM_CACHE is not None:
        return _WORD_NGRAM_CACHE

    artifact = GRID_NGRAM_ARTIFACT
    if not artifact.exists():
        artifact.parent.mkdir(parents=True, exist_ok=True)
        _build_grid_ngram_artifact(artifact)

    with artifact.open("r", encoding="utf-8") as f:
        data = json.load(f)

    _WORD_NGRAM_CACHE = WordNGramLM(
        vocabulary=data["vocabulary"],
        unigram_counts=data["unigram_counts"],
        bigram_counts=data["bigram_counts"],
        trigram_counts=data["trigram_counts"],
        total_tokens=data["total_tokens"],
        add_k=float(data.get("add_k", 0.5)),
    )
    return _WORD_NGRAM_CACHE


def _build_grid_ngram_artifact(output_path: Path) -> None:
    align_paths = sorted(ROOT_DIR.glob(GRID_ALIGN_GLOB))
    if not align_paths:
        raise FileNotFoundError(f"No alignment files found with glob {GRID_ALIGN_GLOB}")

    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    trigram_counts: Counter[str] = Counter()
    vocabulary: set[str] = {"</s>", "<s>"}
    total_tokens = 0

    for align_path in align_paths:
        text = parse_alignment_text(str(align_path)).lower().strip()
        if not text:
            continue
        words = [word for word in text.split() if word]
        if not words:
            continue

        vocabulary.update(words)
        tokens = ["<s>", "<s>", *words, "</s>"]
        total_tokens += len(words) + 1

        for idx, token in enumerate(tokens):
            unigram_counts[token] += 1
            if idx >= 1:
                bigram_counts[_join_ngram((tokens[idx - 1], token))] += 1
            if idx >= 2:
                trigram_counts[_join_ngram((tokens[idx - 2], tokens[idx - 1], token))] += 1

    payload = {
        "vocabulary": sorted(vocabulary),
        "unigram_counts": dict(unigram_counts),
        "bigram_counts": dict(bigram_counts),
        "trigram_counts": dict(trigram_counts),
        "total_tokens": total_tokens,
        "add_k": 0.5,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _join_ngram(parts: tuple[str, ...]) -> str:
    return "\t".join(parts)
