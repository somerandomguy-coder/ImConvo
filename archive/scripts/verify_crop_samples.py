"""Verify preprocessed lip-crop quality after running preprocess.py.

Loads N random samples from data/preprocessed/, renders a 6-frame strip per
sample, and writes a single composite PNG to reports/crop_samples/.

Usage:
    python scripts/verify_crop_samples.py
    python scripts/verify_crop_samples.py --n 40 --seed 99
    python scripts/verify_crop_samples.py --preprocessed_dir ./data/preprocessed/ --output_dir ./reports/crop_samples/
    python scripts/verify_crop_samples.py --samples bbab8n bbac1a   # specific clips (partial match)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FRAMES_PER_STRIP = 6
THUMB_W = 120
THUMB_H = 80
PAD = 4
LABEL_H = 18
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.38
FONT_THICKNESS = 1
DARK_THRESHOLD = 20   # mean brightness below this → likely failed crop


def _read_align(align_path: str) -> str:
    """Return the spoken sentence from a .align file, or '' if missing."""
    if not os.path.exists(align_path):
        return ""
    words = []
    with open(align_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2] not in ("sil", "sp"):
                words.append(parts[2])
    return " ".join(words)


def _make_strip(frames: np.ndarray, label: str, flag_dark: bool) -> np.ndarray:
    """Build an RGB strip: 6 thumbnails side by side + a label bar on top."""
    T = frames.shape[0]
    indices = [int(i * (T - 1) / (FRAMES_PER_STRIP - 1)) for i in range(FRAMES_PER_STRIP)]

    strip_w = FRAMES_PER_STRIP * THUMB_W + (FRAMES_PER_STRIP + 1) * PAD
    strip_h = LABEL_H + PAD + THUMB_H + PAD

    strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)

    # Label bar
    bar_color = (60, 20, 20) if flag_dark else (30, 30, 30)
    strip[:LABEL_H, :] = bar_color
    text_color = (80, 80, 220) if flag_dark else (200, 200, 200)
    cv2.putText(strip, label, (PAD, LABEL_H - 4), FONT, FONT_SCALE, text_color, FONT_THICKNESS, cv2.LINE_AA)

    # Thumbnails
    for col, idx in enumerate(indices):
        frame = frames[idx]  # (H, W) uint8 grayscale
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        x0 = PAD + col * (THUMB_W + PAD)
        y0 = LABEL_H + PAD
        strip[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = rgb

        # Frame index annotation bottom-right of thumbnail
        fi_text = f"f{idx}"
        cv2.putText(strip, fi_text, (x0 + 2, y0 + THUMB_H - 3), FONT, 0.3, (80, 200, 80), 1, cv2.LINE_AA)

    return strip


def _load_sample(npy_path: str) -> np.ndarray | None:
    try:
        arr = np.load(npy_path)
        if arr.ndim == 4:
            # (T, H, W, 1) → (T, H, W)
            arr = arr[..., 0]
        if arr.ndim != 3:
            return None
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr
    except Exception:
        return None


def _find_npy_files(preprocessed_dir: str, name_filters: list[str] | None) -> list[str]:
    all_npy = sorted(Path(preprocessed_dir).glob("*.npy"))
    if not name_filters:
        return [str(p) for p in all_npy]
    filtered = []
    for p in all_npy:
        stem = p.stem
        if any(f.lower() in stem.lower() for f in name_filters):
            filtered.append(str(p))
    return filtered


def run(
    preprocessed_dir: str,
    output_dir: str,
    n: int,
    seed: int,
    name_filters: list[str] | None,
):
    all_files = _find_npy_files(preprocessed_dir, name_filters)
    if not all_files:
        print(f"[ERROR] No .npy files found in {preprocessed_dir}")
        return

    rng = random.Random(seed)
    chosen = all_files if name_filters else rng.sample(all_files, min(n, len(all_files)))

    print(f"Verifying {len(chosen)} samples from {preprocessed_dir}")

    align_dir = Path(preprocessed_dir) / "align"
    strips = []
    stats = {"total": 0, "dark": 0, "missing_align": 0}

    for npy_path in chosen:
        stem = Path(npy_path).stem
        frames = _load_sample(npy_path)
        stats["total"] += 1

        if frames is None:
            print(f"  [SKIP] {stem} — could not load")
            continue

        mean_brightness = float(frames.mean())
        flag_dark = mean_brightness < DARK_THRESHOLD
        if flag_dark:
            stats["dark"] += 1

        align_path = align_dir / f"{stem}.align"
        sentence = _read_align(str(align_path))
        if not sentence:
            stats["missing_align"] += 1

        label = f"{stem}  |  {frames.shape}  |  μ={mean_brightness:.0f}"
        if sentence:
            label += f"  |  \"{sentence}\""
        if flag_dark:
            label += "  [DARK!]"

        strip = _make_strip(frames, label, flag_dark)
        strips.append(strip)

    if not strips:
        print("[ERROR] No valid strips to render.")
        return

    # Stack all strips vertically with a separator line
    sep = np.full((2, strips[0].shape[1], 3), 50, dtype=np.uint8)
    parts = []
    for s in strips:
        parts.append(s)
        parts.append(sep)
    composite = np.vstack(parts[:-1])

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"crop_verify_n{len(strips)}_seed{seed}.png"
    cv2.imwrite(str(out_path), composite)

    print(f"\nSummary:")
    print(f"  Samples checked : {stats['total']}")
    print(f"  Dark crops      : {stats['dark']}  (mean brightness < {DARK_THRESHOLD})")
    print(f"  Missing aligns  : {stats['missing_align']}")
    print(f"\nOutput → {out_path}")
    if stats["dark"] > 0:
        print(f"\nWARNING: {stats['dark']} dark frame(s) detected — MediaPipe may have missed the face.")
        print("  Re-run preprocess.py with --force to regenerate those clips.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify preprocessed lip-crop quality")
    parser.add_argument("--preprocessed_dir", default="./data/preprocessed/")
    parser.add_argument("--output_dir", default="./reports/crop_samples/")
    parser.add_argument("--n", type=int, default=30, help="Number of random samples to check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples", nargs="*", metavar="NAME",
        help="Filter by clip name substring (e.g. bbab8n). Overrides --n."
    )
    args = parser.parse_args()
    run(args.preprocessed_dir, args.output_dir, args.n, args.seed, args.samples)
