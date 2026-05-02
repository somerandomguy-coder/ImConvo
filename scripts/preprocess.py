"""
Preprocessing script for the GRID lip reading corpus.

Expected layout:  data/s{N}_processed/*.mpg + data/s{N}_processed/align/*.align

Converts raw .mpg video files → preprocessed .npy arrays (run once).
Output is a flat directory of .npy files + a manifest.txt.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --data_dir ./data/ --output_dir ./data/preprocessed/
"""

import argparse
import os
import re
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import extract_lip_frames


def discover_video_samples(data_dir: str):
    """
    Discover all (video_path, align_path) pairs from per-speaker directories.

    Expected layout: data/s{N}_processed/*.mpg + data/s{N}_processed/align/*.align
    """
    samples = []  # list of (video_path, align_path, unique_name)

    speaker_dirs = sorted(
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and re.match(r"s\d+(_processed)?$", d)
    )

    if not speaker_dirs:
        print(f"[ERROR] No speaker directories (s*_processed) found in {data_dir}")
        return samples

    for speaker in speaker_dirs:
        speaker_path = os.path.join(data_dir, speaker)
        align_dir = os.path.join(speaker_path, "align")

        if not os.path.isdir(align_dir):
            print(f"  [SKIP] {speaker} — no align/ subdirectory")
            continue

        video_files = sorted(
            f for f in os.listdir(speaker_path) if f.endswith(".mpg")
        )

        for vf in video_files:
            name = os.path.splitext(vf)[0]
            align_path = os.path.join(align_dir, f"{name}.align")
            if os.path.exists(align_path):
                unique_name = f"{speaker}_{name}"
                samples.append(
                    (os.path.join(speaker_path, vf), align_path, unique_name)
                )

    print(f"Discovered {len(samples)} videos across {len(speaker_dirs)} speakers")
    return samples


def preprocess_dataset(data_dir: str, output_dir: str, force: bool = False):
    """Preprocess all videos in data_dir and save as .npy files.
    
    Args:
        data_dir: Path to raw data directory.
        output_dir: Path to output directory.
        force: If True, re-process all videos even if .npy files already exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    samples = discover_video_samples(data_dir)
    if not samples:
        print("[ERROR] No video samples found. Check your data directory layout.")
        return

    # Copy alignment files into output_dir/align so training can resolve
    # preprocessed_dir/align/<sample_id>.align consistently.
    align_output_dir = os.path.join(output_dir, "align")
    os.makedirs(align_output_dir, exist_ok=True)

    valid_names = []
    skipped = 0
    start_time = time.time()

    for i, (video_path, align_path, unique_name) in enumerate(samples):
        output_path = os.path.join(output_dir, f"{unique_name}.npy")

        # Skip if already preprocessed (unless --force)
        if os.path.exists(output_path) and not force:
            valid_names.append(unique_name)
            # Still copy alignment if missing
            align_dst = os.path.join(align_output_dir, f"{unique_name}.align")
            if not os.path.exists(align_dst):
                _copy_file(align_path, align_dst)
            skipped += 1
            continue

        try:
            frames = extract_lip_frames(video_path)
            if frames is None:
                # Face detection failed — skip this sample
                continue
            np.save(output_path, frames)
            valid_names.append(unique_name)

            # Copy alignment file to flat output
            align_dst = os.path.join(align_output_dir, f"{unique_name}.align")
            _copy_file(align_path, align_dst)

        except Exception as e:
            print(f"  [ERROR] {unique_name}: {e}")
            continue

        if (i + 1) % 200 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i + 1}/{len(samples)}] {rate:.1f} videos/sec — {unique_name}.npy saved"
            )

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        for name in sorted(valid_names):
            f.write(name + "\n")

    elapsed = time.time() - start_time
    print(f"\n✓ Preprocessing complete!")
    print(f"  {len(valid_names)} samples ({skipped} cached, {len(valid_names) - skipped} new)")
    print(f"  Output directory: {output_dir}")
    print(f"  Alignment directory: {align_output_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Time: {elapsed:.1f}s")

    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".npy")
    )
    print(f"  Total size: {total_bytes / 1e6:.1f} MB")


def _copy_file(src: str, dst: str):
    """Copy a file from src to dst."""
    import shutil
    shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GRID corpus videos")
    parser.add_argument("--data_dir", default="./data/", help="Raw data directory")
    parser.add_argument(
        "--output_dir", default="./data/preprocessed/", help="Output directory"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process all videos even if .npy files already exist"
    )
    args = parser.parse_args()

    if args.force:
        print("⚠️  Force mode: re-processing all videos (overwriting existing .npy files)")

    preprocess_dataset(args.data_dir, args.output_dir, force=args.force)
