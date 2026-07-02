"""Shared constants and utilities for isolated-word GRID classification."""

from __future__ import annotations

from dataclasses import dataclass

SILENCE_TOKENS = {"sil", "sp"}

# GRID sentence slots (kept explicit for deterministic flattened vocabulary order)
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
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    ("again", "now", "please", "soon"),
)


def _flatten_vocab_in_order() -> list[str]:
    flat: list[str] = []
    seen: set[str] = set()
    for group in SLOT_VOCABS:
        for word in group:
            if word in seen:
                continue
            seen.add(word)
            flat.append(word)
    return flat


FLAT_VOCAB: tuple[str, ...] = tuple(_flatten_vocab_in_order())
NUM_WORD_CLASSES = len(FLAT_VOCAB)  # expected 51 for GRID
WORD_TO_IDX = {word: idx for idx, word in enumerate(FLAT_VOCAB)}
IDX_TO_WORD = FLAT_VOCAB


def word_idx_to_text(index: int) -> str:
    if 0 <= int(index) < NUM_WORD_CLASSES:
        return IDX_TO_WORD[int(index)]
    return "<unk>"


def parse_align_entries(align_path: str) -> list[tuple[int, int, str]]:
    """Parse raw entries from a GRID .align file: (start, end, token)."""
    entries: list[tuple[int, int, str]] = []

    with open(align_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                continue
            token = parts[2].lower()
            if end <= start:
                continue
            entries.append((start, end, token))
    return entries
