#!/usr/bin/env python3
"""Verify GRID word-level TCN dataset artifacts.

Checks:
- label file has expected cardinality and excludes sil/sp
- metadata rows are well-formed and file paths exist
- label/index consistency across directories and metadata
- optional spot checks compare metadata frame bounds against align timestamps
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np

SILENCE_TOKENS = {"sil", "sp"}
CLIP_RE = re.compile(r"^(?P<sample>.+)_w(?P<idx>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify GRID word-level dataset outputs")
    parser.add_argument(
        "--dataset-dir",
        default="/home/nam/ImConvo/Lipreading_using_Temporal_Convolutional_Networks/datasets/visual_data_grid51",
        help="Root output directory from convert_grid_preprocessed_to_tcn_words.py",
    )
    parser.add_argument(
        "--labels-path",
        default="",
        help="Optional labels file path (defaults to <dataset-dir>/grid51_words.txt)",
    )
    parser.add_argument(
        "--metadata-path",
        default="",
        help="Optional metadata path (defaults to <dataset-dir>/metadata.csv)",
    )
    parser.add_argument(
        "--source-preprocessed-dir",
        default="/home/nam/ImConvo/data/preprocessed",
        help="Original preprocessed GRID dir with align/ for frame-bound spot checks",
    )
    parser.add_argument(
        "--expected-classes",
        type=int,
        default=51,
        help="Expected number of classes in label file (set <=0 to skip this check)",
    )
    parser.add_argument(
        "--spot-checks",
        type=int,
        default=25,
        help="How many random metadata rows to verify against align timestamps",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_align(align_path: Path) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    with align_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start, end, tok = parts
            rows.append((int(start), int(end), tok.lower()))
    return rows


def timestamp_to_frame_bounds(start: int, end: int, clip_start: int, clip_end: int, num_frames: int) -> tuple[int, int]:
    duration = max(1, clip_end - clip_start)
    start_ratio = (start - clip_start) / duration
    end_ratio = (end - clip_start) / duration

    start_frame = int(np.floor(start_ratio * num_frames))
    end_exclusive = int(np.ceil(end_ratio * num_frames))

    start_frame = max(0, min(start_frame, num_frames - 1))
    end_exclusive = max(start_frame + 1, min(end_exclusive, num_frames))

    return start_frame, end_exclusive - 1


def extract_word_segments(rows: list[tuple[int, int, str]], num_frames: int) -> list[tuple[str, int, int]]:
    if not rows:
        return []
    clip_start = min(s for s, _, _ in rows)
    clip_end = max(e for _, e, _ in rows)
    out: list[tuple[str, int, int]] = []
    for start, end, tok in rows:
        if tok in SILENCE_TOKENS:
            continue
        fs, fe = timestamp_to_frame_bounds(start, end, clip_start, clip_end, num_frames)
        out.append((tok, fs, fe))
    return out


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    labels_path = Path(args.labels_path).resolve() if args.labels_path else dataset_dir / "grid51_words.txt"
    metadata_path = Path(args.metadata_path).resolve() if args.metadata_path else dataset_dir / "metadata.csv"
    source_dir = Path(args.source_preprocessed_dir).resolve()

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_set = set(labels)

    print(f"Labels found: {len(labels)}")
    if args.expected_classes > 0 and len(labels) != args.expected_classes:
        raise AssertionError(f"Expected {args.expected_classes} classes, found {len(labels)}")
    if label_set & SILENCE_TOKENS:
        raise AssertionError("Labels include silence tokens sil/sp")

    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise AssertionError("Metadata CSV is empty")

    for col in ["clip_id", "speaker", "word", "split", "frame_start", "frame_end", "path"]:
        if col not in rows[0]:
            raise AssertionError(f"Metadata missing column: {col}")

    bad_paths = 0
    bad_words = 0
    bad_splits = 0

    for row in rows:
        word = row["word"]
        split = row["split"]
        rel_path = Path(row["path"])
        full_path = dataset_dir / rel_path

        if word not in label_set:
            bad_words += 1
        if split not in {"train", "val", "test"}:
            bad_splits += 1
        if not full_path.is_file():
            bad_paths += 1
            continue

        # Directory shape sanity: <word>/<split>/<clip_id>.npz
        parts = rel_path.parts
        if len(parts) < 3 or parts[0] != word or parts[1] != split:
            bad_paths += 1

    if bad_words:
        raise AssertionError(f"Rows with unknown words: {bad_words}")
    if bad_splits:
        raise AssertionError(f"Rows with invalid split: {bad_splits}")
    if bad_paths:
        raise AssertionError(f"Rows with missing/mismatched file paths: {bad_paths}")

    rng = random.Random(args.seed)
    n_checks = min(args.spot_checks, len(rows))
    sample_rows = rng.sample(rows, n_checks)

    for row in sample_rows:
        clip_id = row["clip_id"]
        m = CLIP_RE.match(clip_id)
        if not m:
            raise AssertionError(f"clip_id does not match expected pattern: {clip_id}")

        sample_id = m.group("sample")
        word_idx = int(m.group("idx"))

        npy_path = source_dir / f"{sample_id}.npy"
        align_path = source_dir / "align" / f"{sample_id}.align"
        if not npy_path.is_file() or not align_path.is_file():
            raise AssertionError(f"Missing source pair for spot check: {sample_id}")

        frames = np.load(npy_path)
        rows_align = parse_align(align_path)
        segments = extract_word_segments(rows_align, frames.shape[0])

        if word_idx >= len(segments):
            raise AssertionError(f"Word index {word_idx} out of range for {sample_id}")

        expected_word, expected_fs, expected_fe = segments[word_idx]
        if expected_word != row["word"]:
            raise AssertionError(
                f"Word mismatch for {clip_id}: metadata={row['word']} expected={expected_word}"
            )

        got_fs = int(row["frame_start"])
        got_fe = int(row["frame_end"])
        if (got_fs, got_fe) != (expected_fs, expected_fe):
            raise AssertionError(
                f"Frame bounds mismatch for {clip_id}: metadata=({got_fs},{got_fe}) expected=({expected_fs},{expected_fe})"
            )

    print(f"Metadata rows: {len(rows)}")
    print(f"Spot checks passed: {n_checks}/{n_checks}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
