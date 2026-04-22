#!/usr/bin/env python3
"""Convert GRID preprocessed sentence clips into word-level clips for TCN training.

Input layout (existing):
  <input_dir>/<sample_id>.npy
  <input_dir>/align/<sample_id>.align

Output layout (TCN-compatible):
  <output_dir>/<word>/<train|val|test>/<clip_id>.npz

Each output npz stores key "data" with shape (T, 96, 96), dtype uint8.
Metadata CSV schema:
  clip_id,speaker,word,split,frame_start,frame_end,path
where frame_start/frame_end are inclusive indices in the source 75-frame clip.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SILENCE_TOKENS = {"sil", "sp"}
SPEAKER_RE = re.compile(r"^(s\d+)_processed_")


@dataclass
class WordSegment:
    word: str
    start: int
    end: int
    start_frame: int
    end_frame: int  # inclusive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GRID word-level dataset for TCN")
    parser.add_argument(
        "--input-dir",
        default="/home/nam/ImConvo/data/preprocessed",
        help="Directory containing preprocessed .npy files and align/ folder",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/nam/ImConvo/Lipreading_using_Temporal_Convolutional_Networks/datasets/visual_data_grid51",
        help="Output directory for TCN-style word clips",
    )
    parser.add_argument(
        "--labels-path",
        default="",
        help="Optional explicit output path for grid51_words.txt",
    )
    parser.add_argument(
        "--metadata-path",
        default="",
        help="Optional explicit output path for metadata CSV",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=96,
        help="Output height/width per frame (default: 96)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on number of source sentence clips to process (0 = all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output clips",
    )
    return parser.parse_args()


def speaker_to_split(speaker: str) -> str:
    speaker_id = int(speaker[1:])
    if 1 <= speaker_id <= 28:
        return "train"
    if 29 <= speaker_id <= 32:
        return "val"
    if 33 <= speaker_id <= 34:
        return "test"
    raise ValueError(f"Speaker {speaker} does not map to train/val/test split")


def parse_align(align_path: Path) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    with align_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start, end, token = parts
            rows.append((int(start), int(end), token.lower()))
    return rows


def timestamp_to_frame_bounds(
    start: int,
    end: int,
    clip_start: int,
    clip_end: int,
    num_frames: int,
) -> tuple[int, int]:
    """Return inclusive [start_frame, end_frame] clamped to [0, num_frames-1]."""
    duration = max(1, clip_end - clip_start)

    start_ratio = (start - clip_start) / duration
    end_ratio = (end - clip_start) / duration

    start_frame = int(np.floor(start_ratio * num_frames))
    end_exclusive = int(np.ceil(end_ratio * num_frames))

    start_frame = max(0, min(start_frame, num_frames - 1))
    end_exclusive = max(start_frame + 1, min(end_exclusive, num_frames))

    return start_frame, end_exclusive - 1


def normalize_frame_dtype(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    frames = frames.astype(np.float32, copy=False)
    max_val = float(np.max(frames)) if frames.size else 0.0
    if max_val <= 1.5:
        frames = np.clip(frames * 255.0, 0.0, 255.0)
    else:
        frames = np.clip(frames, 0.0, 255.0)
    return frames.astype(np.uint8)


def resize_clip(frames: np.ndarray, target_size: int) -> np.ndarray:
    resized = [
        cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    return np.stack(resized, axis=0).astype(np.uint8)


def extract_word_segments(align_rows: list[tuple[int, int, str]], num_frames: int) -> list[WordSegment]:
    if not align_rows:
        return []

    clip_start = min(s for s, _, _ in align_rows)
    clip_end = max(e for _, e, _ in align_rows)

    segments: list[WordSegment] = []
    for start, end, token in align_rows:
        if token in SILENCE_TOKENS:
            continue
        start_frame, end_frame = timestamp_to_frame_bounds(start, end, clip_start, clip_end, num_frames)
        segments.append(
            WordSegment(
                word=token,
                start=start,
                end=end,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    return segments


def discover_sample_ids(input_dir: Path) -> list[str]:
    align_dir = input_dir / "align"
    ids = [p.stem for p in sorted(align_dir.glob("*.align"))]
    return ids


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    align_dir = input_dir / "align"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not align_dir.is_dir():
        raise FileNotFoundError(f"Align directory not found: {align_dir}")

    labels_path = Path(args.labels_path).resolve() if args.labels_path else output_dir / "grid51_words.txt"
    metadata_path = Path(args.metadata_path).resolve() if args.metadata_path else output_dir / "metadata.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = discover_sample_ids(input_dir)
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]

    metadata_rows: list[dict[str, str | int]] = []
    vocab: set[str] = set()

    missing_npy = 0
    skipped_invalid = 0
    written_clips = 0

    for sample_id in tqdm(sample_ids, desc="Converting GRID sentences"):
        match = SPEAKER_RE.match(sample_id)
        if not match:
            skipped_invalid += 1
            continue

        speaker = match.group(1)
        try:
            split = speaker_to_split(speaker)
        except ValueError:
            skipped_invalid += 1
            continue

        npy_path = input_dir / f"{sample_id}.npy"
        align_path = align_dir / f"{sample_id}.align"

        if not npy_path.is_file() or not align_path.is_file():
            missing_npy += 1
            continue

        clip = np.load(npy_path)
        if clip.ndim != 3:
            skipped_invalid += 1
            continue

        clip = normalize_frame_dtype(clip)
        num_frames = clip.shape[0]

        align_rows = parse_align(align_path)
        word_segments = extract_word_segments(align_rows, num_frames)

        for idx, seg in enumerate(word_segments):
            start = seg.start_frame
            end = seg.end_frame + 1
            word_clip = clip[start:end]
            if word_clip.size == 0:
                continue

            word_clip = resize_clip(word_clip, args.target_size)
            clip_id = f"{sample_id}_w{idx:02d}"
            rel_path = Path(seg.word) / split / f"{clip_id}.npz"
            dst = output_dir / rel_path

            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or args.overwrite:
                np.savez_compressed(dst, data=word_clip)

            vocab.add(seg.word)
            metadata_rows.append(
                {
                    "clip_id": clip_id,
                    "speaker": speaker,
                    "word": seg.word,
                    "split": split,
                    "frame_start": seg.start_frame,
                    "frame_end": seg.end_frame,
                    "path": rel_path.as_posix(),
                }
            )
            written_clips += 1

    labels = sorted(vocab)
    with labels_path.open("w", encoding="utf-8") as f:
        for token in labels:
            f.write(f"{token}\n")

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_id", "speaker", "word", "split", "frame_start", "frame_end", "path"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\nDone.")
    print(f"Input samples scanned: {len(sample_ids)}")
    print(f"Word clips written (including existing reused clips): {written_clips}")
    print(f"Unique words found: {len(labels)}")
    print(f"Missing sample pairs: {missing_npy}")
    print(f"Skipped invalid samples: {skipped_invalid}")
    print(f"Labels file: {labels_path}")
    print(f"Metadata file: {metadata_path}")

    if len(labels) != 51:
        print("WARNING: Expected 51 GRID words but found a different vocabulary size.")


if __name__ == "__main__":
    main()
