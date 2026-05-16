"""Quick validation for isolated-word slicing dataset readiness."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.isolated_word_level.common import FLAT_VOCAB
from experiments.isolated_word_level.dataset import build_word_segments_from_ids
from src.dataset import load_split_ids


def summarize_split(preprocessed_dir: str, split_dir: str, split_name: str):
    sample_ids = load_split_ids(split_dir=split_dir, split_name=split_name)
    segments, stats = build_word_segments_from_ids(preprocessed_dir=preprocessed_dir, sample_ids=sample_ids)

    class_counts = Counter(seg.class_idx for seg in segments)
    lengths = np.array([seg.num_frames for seg in segments], dtype=np.int32)

    print(f"\n[{split_name}]")
    print(f"  samples: {len(sample_ids)}")
    print(f"  segments: {len(segments)}")
    print(f"  stats: {stats}")
    if len(lengths) > 0:
        print(
            "  segment_len min/mean/max: "
            f"{int(lengths.min())}/{float(lengths.mean()):.2f}/{int(lengths.max())}"
        )
    print(f"  classes observed: {len(class_counts)}/{len(FLAT_VOCAB)}")

    top_classes = class_counts.most_common(10)
    if top_classes:
        print("  top10 classes:")
        for class_idx, count in top_classes:
            print(f"    - {FLAT_VOCAB[class_idx]}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate isolated-word slicing dataset")
    parser.add_argument("--preprocessed-dir", default="./data/preprocessed/")
    parser.add_argument("--split-dir", default="./splits/grid_v1")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val_oos", "val_is", "test_oos", "test_is"],
    )
    args = parser.parse_args()

    print(f"Expected vocabulary size: {len(FLAT_VOCAB)}")
    print(f"Vocabulary: {FLAT_VOCAB}")

    for split in args.splits:
        summarize_split(
            preprocessed_dir=args.preprocessed_dir,
            split_dir=args.split_dir,
            split_name=split,
        )
