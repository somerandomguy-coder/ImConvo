"""Live per-word game: vocab/slot maps, pass-rule, preprocessing, scoring."""
from __future__ import annotations

import base64
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from experiments.isolated_word_level.common import IDX_TO_WORD, NUM_WORD_CLASSES, SLOT_VOCABS, WORD_TO_IDX
from experiments.isolated_word_level.model import build_lipreading_isolated_word_classifier
from experiments.isolated_word_level.temporal_slicer import slice_and_pad_word_clip
from src.utils import FRAME_HEIGHT, FRAME_WIDTH, _mediapipe_lip_bbox

SLOT_NAMES: tuple[str, ...] = (
    "command",
    "color",
    "preposition",
    "letter",
    "digit",
    "adverb",
)

# Tunables — single source of truth.
BUFFER_FRAMES = 25
TARGET_FRAMES = 25
SCORE_EVERY_N_FRAMES = 3
PASS_THRESHOLD = 0.6
LETTER_SLOT_INDEX = 3
LETTER_TOPK_PASS = 3


def slot_candidate_indices(slot_index: int) -> list[int]:
    """Return the FLAT_VOCAB class indices for a given GRID slot."""
    return [WORD_TO_IDX[w] for w in SLOT_VOCABS[slot_index]]


def evaluate_pass(
    slot_index: int,
    target: str,
    ranked_words: list[str],
    confidence: float,
) -> bool:
    """Decide whether a scored window passes the round.

    ranked_words: the slot's candidate words sorted by probability desc.
    confidence:   probability mass of ranked_words[0] (the top word).
    """
    if slot_index == LETTER_SLOT_INDEX:
        return target in ranked_words[:LETTER_TOPK_PASS]
    return bool(ranked_words and ranked_words[0] == target and confidence >= PASS_THRESHOLD)


def decode_jpeg(b64: str) -> np.ndarray:
    """Decode a base64 JPEG string into a BGR uint8 frame."""
    try:
        raw = base64.b64decode(b64, validate=True)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not decode JPEG: {exc}") from exc
    if img is None:
        raise ValueError("Could not decode JPEG: empty image")
    return img


def preprocess_frames(bgr_frames: list[np.ndarray]) -> np.ndarray | None:
    """Lip-crop, grayscale, resize and pad frames to the model input tensor."""
    crops: list[np.ndarray] = []
    for frame in bgr_frames:
        bbox = _mediapipe_lip_bbox(frame)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        lip = frame[y0:y1, x0:x1]
        if lip.size == 0:
            continue
        gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT))
        crops.append(resized.astype(np.float32) / 255.0)

    if not crops:
        return None

    stacked = np.stack(crops, axis=0)  # (t, H, W)
    clip = slice_and_pad_word_clip(
        stacked, 0, stacked.shape[0], target_frames=TARGET_FRAMES, pad_mode="repeat_last"
    )  # (TARGET_FRAMES, H, W, 1)
    return clip[np.newaxis, ...].astype(np.float32)


_ROOT = Path(__file__).resolve().parent.parent
_ISOLATED_DIR = _ROOT / "checkpoints" / "isolated_word_level"

_classifier = None
_classifier_lock = threading.Lock()


class IsolatedWordClassifier:
    """Lazy-loaded singleton wrapper around the isolated-word model."""

    def __init__(self) -> None:
        self.model = build_lipreading_isolated_word_classifier(
            model_variant="conformer_lite",
            frontend_model="flatten",
            num_word_classes=NUM_WORD_CLASSES,
        )
        dummy = np.zeros((1, TARGET_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1), dtype=np.float32)
        _ = self.model(dummy, training=False)
        self.model.load_weights(self._resolve_weights())

    @staticmethod
    def _resolve_weights() -> str:
        env = os.environ.get("IMCONVO_ISOLATED_WEIGHTS")
        if env:
            return env
        files = sorted(_ISOLATED_DIR.glob("*.weights.h5"))
        if not files:
            raise FileNotFoundError(f"No isolated-word weights in {_ISOLATED_DIR}")
        return str(files[-1])

    def predict_logits(self, frames: np.ndarray) -> np.ndarray:
        return np.asarray(self.model(frames, training=False))


def get_classifier() -> IsolatedWordClassifier:
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = IsolatedWordClassifier()
    return _classifier


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def score_window(frames: np.ndarray, slot_index: int, *, classifier=None, top_k: int = 3) -> dict:
    clf = classifier if classifier is not None else get_classifier()
    logits = clf.predict_logits(frames)[0]  # (51,)
    cand = slot_candidate_indices(slot_index)
    cand_logits = np.array([logits[i] for i in cand], dtype=np.float32)
    probs = _softmax(cand_logits)
    order = np.argsort(-probs)
    ranked_words = [IDX_TO_WORD[cand[i]] for i in order]
    ranked_probs = [float(probs[i]) for i in order]
    topk = [{"word": w, "confidence": p} for w, p in zip(ranked_words[:top_k], ranked_probs[:top_k])]
    return {
        "top_word": ranked_words[0],
        "confidence": ranked_probs[0],
        "topk": topk,
        "ranked_words": ranked_words,
    }
