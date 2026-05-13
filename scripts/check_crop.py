"""
Visualise lip crops — two modes:

  --mode npy    (default) Load saved .npy files and render a contact sheet.
                This shows EXACTLY what the model sees after preprocessing.

  --mode video  Run MediaPipe detection live on raw .mpg files and show
                the detected bounding box + crop side-by-side.

Output goes to reports/crop_check/.

Usage (from project root, venv active):
    python scripts/check_crop.py                            # 5 random npy from s1
    python scripts/check_crop.py --speaker s5 --n 6        # 6 npy from s5
    python scripts/check_crop.py --mode video --speaker s1 # live detection on videos
    python scripts/check_crop.py --npy data/preprocessed/s1_processed_bbaf2n.npy
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAMES,
    LIP_Y_START, LIP_Y_END, LIP_X_START, LIP_X_END,
    _check_mediapipe, _detect_mouth_bbox,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "crop_check")
PREPROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed")


# ---------------------------------------------------------------------------
# Mode 1: inspect saved .npy files (what the model actually sees)
# ---------------------------------------------------------------------------

def check_npy(npy_path: str):
    name = os.path.splitext(os.path.basename(npy_path))[0]
    arr = np.load(npy_path)

    dtype = arr.dtype
    if dtype == np.uint8:
        frames_f32 = arr.astype(np.float32) / 255.0
        tag = "uint8 ✓ (new)"
    else:
        frames_f32 = arr.astype(np.float32)
        tag = "float32 (old)"

    print(f"  {name}: dtype={dtype} [{tag}]  shape={arr.shape}  "
          f"range=[{frames_f32.min():.2f}, {frames_f32.max():.2f}]")

    # Pick 6 evenly-spaced frames for the contact sheet
    indices = np.linspace(0, MAX_FRAMES - 1, 6, dtype=int)
    thumbs = []
    for i in indices:
        frame = (frames_f32[i] * 255).astype(np.uint8)
        thumb = cv2.resize(frame, (FRAME_WIDTH * 2, FRAME_HEIGHT * 2),
                           interpolation=cv2.INTER_LINEAR)
        # Add frame index label
        cv2.putText(thumb, f"f{i}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 200, 1)
        thumbs.append(thumb)

    # Arrange 6 thumbs in a 2-row × 3-col grid
    row1 = np.hstack(thumbs[:3])
    row2 = np.hstack(thumbs[3:])
    sheet = np.vstack([row1, row2])

    # Header bar with metadata
    header = np.full((28, sheet.shape[1]), 40, dtype=np.uint8)
    cv2.putText(header, f"{name}  |  {tag}  |  shape {arr.shape}",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 220, 1)
    out_img = np.vstack([header, sheet])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_npy.png")
    cv2.imwrite(out_path, out_img)
    print(f"  saved → {out_path}")


def run_npy_mode(speaker: str, n: int, npy_path: str | None):
    if npy_path:
        files = [npy_path]
    else:
        pattern = os.path.join(PREPROCESSED_DIR, f"{speaker}_processed_*.npy")
        all_files = sorted(glob.glob(pattern))
        if not all_files:
            # fallback: any npy with speaker prefix
            pattern = os.path.join(PREPROCESSED_DIR, f"{speaker}_*.npy")
            all_files = sorted(glob.glob(pattern))
        if not all_files:
            print(f"No .npy files found for speaker '{speaker}' in {PREPROCESSED_DIR}")
            sys.exit(1)
        files = random.sample(all_files, min(n, len(all_files)))

    # Also report overall dtype split
    all_npy = glob.glob(os.path.join(PREPROCESSED_DIR, "*.npy"))
    uint8_n = sum(1 for f in all_npy if np.load(f, mmap_mode='r').dtype == np.uint8)
    total_n = len(all_npy)
    print(f"\nDataset status: {uint8_n:,}/{total_n:,} files are uint8 "
          f"({uint8_n/total_n*100:.1f}% re-processed)\n")

    for f in files:
        check_npy(f)


# ---------------------------------------------------------------------------
# Mode 2: live detection on raw .mpg videos (bbox + crop side-by-side)
# ---------------------------------------------------------------------------

def _mid_frame(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames[len(frames) // 2] if frames else None


def check_video(video_path: str):
    name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n--- {name} ---")

    frame = _mid_frame(video_path)
    if frame is None:
        print("  [SKIP] could not read video")
        return

    h, w = frame.shape[:2]
    vis = frame.copy()
    bbox = _detect_mouth_bbox(frame, h, w)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "MediaPipe" if _check_mediapipe() else "Haar",
                    (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  bbox: x={x1}-{x2}  y={y1}-{y2}  size={x2-x1}×{y2-y1}px")
        lip = frame[y1:y2, x1:x2]
        lip_gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(lip_gray, (FRAME_WIDTH, FRAME_HEIGHT))
    else:
        cv2.rectangle(vis, (LIP_X_START, LIP_Y_START), (LIP_X_END, LIP_Y_END),
                      (0, 165, 255), 2)
        cv2.putText(vis, "FALLBACK", (LIP_X_START, max(0, LIP_Y_START - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        print("  [WARN] no face detected — fallback region shown")
        roi = frame[LIP_Y_START:LIP_Y_END, LIP_X_START:LIP_X_END]
        crop = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                          (FRAME_WIDTH, FRAME_HEIGHT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_bbox.png"), vis)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_crop.png"),
                cv2.resize(crop, (FRAME_WIDTH * 3, FRAME_HEIGHT * 3),
                           interpolation=cv2.INTER_NEAREST))
    print(f"  saved → {OUTPUT_DIR}/{name}_bbox.png  +  _crop.png")


def run_video_mode(speaker: str, n: int, video_path: str | None):
    print(f"MediaPipe available: {_check_mediapipe()}")
    if video_path:
        videos = [video_path]
    else:
        pattern = os.path.join("data", f"{speaker}_processed", "*.mpg")
        all_videos = sorted(glob.glob(pattern))
        if not all_videos:
            print(f"No videos found: {pattern}")
            sys.exit(1)
        videos = random.sample(all_videos, min(n, len(all_videos)))
    for v in videos:
        check_video(v)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise lip crops")
    parser.add_argument("--mode", choices=["npy", "video"], default="npy",
                        help="npy = inspect saved files (default); video = live detection")
    parser.add_argument("--speaker", default="s1", help="Speaker prefix, e.g. s1, s5")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to check")
    parser.add_argument("--npy", default=None, help="Path to a single .npy file (npy mode)")
    parser.add_argument("--video", default=None, help="Path to a single .mpg file (video mode)")
    args = parser.parse_args()

    if args.mode == "npy":
        run_npy_mode(speaker=args.speaker, n=args.n, npy_path=args.npy)
    else:
        run_video_mode(speaker=args.speaker, n=args.n, video_path=args.video)

    print(f"\nDone. Images saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
