"""Shared constants and helpers for the GRID word-level prediction model."""

from __future__ import annotations

import os
import re
from datetime import datetime
import sys

import numpy as np
from src.utils import SILENCE_TOKENS

SLOT_NAMES: tuple[str, ...] = (
    "command",
    "color",
    "preposition",
    "letter",
    "digit",
    "adverb",
)

SLOT_VOCABS: tuple[tuple[str, ...], ...] = (
    ("bin", "lay", "place", "set"),
    ("blue", "green", "red", "white"),
    ("at", "by", "in", "with"),
    (
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "x",
        "y",
        "z",
    ),
    (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ),
    ("again", "now", "please", "soon"),
)

NUM_SLOTS = len(SLOT_NAMES)
SLOT_VOCAB_SIZES: tuple[int, ...] = tuple(len(vocab) for vocab in SLOT_VOCABS)

SLOT_WORD_TO_IDX: tuple[dict[str, int], ...] = tuple(
    {word: idx for idx, word in enumerate(vocab)} for vocab in SLOT_VOCABS
)
SLOT_IDX_TO_WORD: tuple[tuple[str, ...], ...] = SLOT_VOCABS


def parse_alignment_words(align_path: str) -> list[str]:
    """Read a GRID .align file and return the 6 spoken words."""
    words: list[str] = []
    with open(align_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2] not in SILENCE_TOKENS:
                words.append(parts[2].lower())

    if len(words) != NUM_SLOTS:
        raise ValueError(
            f"Expected {NUM_SLOTS} words in {align_path}, found {len(words)}: {words}"
        )
    return words


def encode_alignment_words(align_path: str) -> np.ndarray:
    """Map alignment words to per-position categorical indices."""
    words = parse_alignment_words(align_path)
    encoded = np.zeros((NUM_SLOTS,), dtype=np.int32)

    for slot_idx, word in enumerate(words):
        word_to_idx = SLOT_WORD_TO_IDX[slot_idx]
        if word not in word_to_idx:
            slot_name = SLOT_NAMES[slot_idx]
            allowed = ", ".join(SLOT_VOCABS[slot_idx])
            raise ValueError(
                f"Unknown word '{word}' for slot '{slot_name}' in {align_path}. "
                f"Allowed: {allowed}"
            )
        encoded[slot_idx] = word_to_idx[word]

    return encoded


def decode_word_indices(indices: list[int] | np.ndarray) -> list[str]:
    """Convert per-slot categorical indices back to words."""
    arr = np.asarray(indices).astype(np.int32).reshape(-1)
    if arr.size < NUM_SLOTS:
        raise ValueError(f"Expected at least {NUM_SLOTS} indices, got {arr.size}")

    words: list[str] = []
    for slot_idx in range(NUM_SLOTS):
        vocab = SLOT_IDX_TO_WORD[slot_idx]
        idx = int(arr[slot_idx])
        if idx < 0 or idx >= len(vocab):
            words.append("<unk>")
        else:
            words.append(vocab[idx])
    return words


def indices_to_sentence(indices: list[int] | np.ndarray) -> str:
    """Convert 6 categorical indices to a full GRID sentence string."""
    return " ".join(decode_word_indices(indices))


def infer_frontend_from_checkpoint_path(checkpoint_path: str) -> str:
    """Infer frontend variant from checkpoint name suffix."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0].lower()
    if stem.endswith("_resnet18"):
        return "resnet18"
    if stem.endswith("_gap_proj"):
        return "gap_proj"
    return "flatten"


def infer_variant_from_checkpoint_path(checkpoint_path: str) -> str | None:
    """Infer model variant from checkpoint filename when possible."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0].lower()
    variants = (
        "transformer_medium",
        "conformer_lite",
        "transformer",
        "bilstm",
        "bigru",
        "tcn",
        "gru",
    )
    for variant in variants:
        variant_slug = variant.replace("_", "-")
        if f"mv-{variant_slug}" in stem:
            return variant
        if re.search(rf"(^|_){re.escape(variant)}(_|$)", stem):
            return variant
    return None
