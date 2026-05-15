"""Shared constants and utilities for isolated-word GRID experiments."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


@dataclass(frozen=True)
class ExperimentTag:
    run_id: str
    model_variant: str
    frontend_model: str
    backend_type: str


def word_idx_to_text(index: int) -> str:
    if 0 <= int(index) < NUM_WORD_CLASSES:
        return IDX_TO_WORD[int(index)]
    return "<unk>"


def parse_align_entries(align_path: str) -> list[tuple[int, int, str]]:
    """Parse raw entries from a GRID .align file: (start, end, token)."""
    entries: list[tuple[int, int, str]] = []

    with open(align_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
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

    if not entries:
        raise ValueError(f"No valid alignment entries parsed from {align_path}")

    return entries


def make_run_id(model_variant: str, frontend_model: str, backend_type: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    mv = re.sub(r"[^a-zA-Z0-9]+", "-", model_variant.lower()).strip("-")
    fe = re.sub(r"[^a-zA-Z0-9]+", "-", frontend_model.lower()).strip("-")
    be = re.sub(r"[^a-zA-Z0-9]+", "-", backend_type.lower()).strip("-")
    return f"{timestamp}_{mv}_{fe}_{be}_isolated_word_v1"


def make_checkpoint_filename(
    run_id: str,
    model_variant: str,
    frontend_model: str,
    backend_type: str,
    batch_size: int,
    target_frames: int,
) -> str:
    mv = re.sub(r"[^a-zA-Z0-9]+", "-", model_variant.lower()).strip("-")
    fe = re.sub(r"[^a-zA-Z0-9]+", "-", frontend_model.lower()).strip("-")
    be = re.sub(r"[^a-zA-Z0-9]+", "-", backend_type.lower()).strip("-")
    return (
        f"best_mv-{mv}_fe-{fe}_be-{be}_bs-{int(batch_size)}_t-{int(target_frames)}_"
        f"{run_id}.weights.h5"
    )


def infer_frontend_from_checkpoint_path(checkpoint_path: str) -> str:
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0].lower()
    if "_resnet18" in stem or "fe-resnet18" in stem:
        return "resnet18"
    if "_gap_proj" in stem or "fe-gap-proj" in stem or "fe-gap_proj" in stem:
        return "gap_proj"
    return "flatten"


def infer_variant_from_checkpoint_path(checkpoint_path: str) -> str | None:
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
        slug = variant.replace("_", "-")
        if f"mv-{slug}" in stem:
            return variant
        if re.search(rf"(^|_){re.escape(variant)}(_|$)", stem):
            return variant
    return None


def detect_preprocessed_roi_backend(project_root: str | Path = ".") -> tuple[str, bool]:
    """
    Heuristic check for mouth ROI preprocessing backend.

    Returns:
      backend_name: one of {'mediapipe', 'haar', 'unknown'}
      roi_centered: whether we can trust mouth-centered ROI in saved .npy files.
    """
    root = Path(project_root)
    preprocess_script = root / "scripts" / "preprocess.py"
    utils_script = root / "src" / "utils.py"

    preprocess_text = preprocess_script.read_text(encoding="utf-8", errors="ignore") if preprocess_script.exists() else ""
    utils_text = utils_script.read_text(encoding="utf-8", errors="ignore") if utils_script.exists() else ""

    has_mediapipe = "mediapipe" in utils_text.lower() or "face_landmarker" in utils_text.lower()
    has_haar = "haarcascade" in preprocess_text.lower() or "haarcascade" in utils_text.lower()
    uses_extract_lip_frames = "extract_lip_frames" in preprocess_text

    if has_mediapipe and uses_extract_lip_frames:
        return "mediapipe", True
    if has_haar:
        return "haar", False
    return "unknown", False
