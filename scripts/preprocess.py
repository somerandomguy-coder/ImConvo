"""
Preprocessing script for the GRID lip reading corpus.

Converts raw .mpg video files → preprocessed .npy arrays (run once).

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --data_dir ./data/ --output_dir ./data/preprocessed/
"""

import argparse
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import extract_lip_frames


def preprocess_dataset(data_dir: str, output_dir: str):
    """Preprocess all videos in data_dir and save as .npy files."""
    align_dir = os.path.join(data_dir, "align")
    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mpg"))
    print(f"Found {len(video_files)} video files in {data_dir}")

    valid_names = []
    start_time = time.time()

    for i, vf in enumerate(video_files):
        name = os.path.splitext(vf)[0]
        align_path = os.path.join(align_dir, f"{name}.align")

        if not os.path.exists(align_path):
            print(f"  [SKIP] {name} — no alignment file")
            continue

        output_path = os.path.join(output_dir, f"{name}.npy")

        # Skip if already preprocessed
        if os.path.exists(output_path):
            valid_names.append(name)
            continue

        video_path = os.path.join(data_dir, vf)
        frames = extract_lip_frames(video_path)
        np.save(output_path, frames)
        valid_names.append(name)

        if (i + 1) % 100 == 0 or (i + 1) == len(video_files):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{i + 1}/{len(video_files)}] {rate:.1f} videos/sec — {name}.npy saved")

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        for name in sorted(valid_names):
            f.write(name + "\n")

    elapsed = time.time() - start_time
    print(f"\n✓ Preprocessing complete!")
    print(f"  {len(valid_names)} samples preprocessed in {elapsed:.1f}s")
    print(f"  Output directory: {output_dir}")
    print(f"  Manifest: {manifest_path}")

    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith(".npy")
    )
    print(f"  Total size: {total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GRID corpus videos")
    parser.add_argument("--data_dir", default="./data/", help="Raw data directory")
    parser.add_argument("--output_dir", default="./data/preprocessed/", help="Output directory")
    args = parser.parse_args()

    preprocess_dataset(args.data_dir, args.output_dir)
