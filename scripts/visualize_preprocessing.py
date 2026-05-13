"""
Side-by-side comparison of raw video frames vs preprocessed .npy crops.

For each sample, produces one PNG with two rows:
  Row 1 (RAW)    — 6 evenly-spaced frames from the .mpg with the
                   MediaPipe/Haar bounding box drawn in green.
  Row 2 (CROP)   — The same 6 frame indices loaded from the saved .npy,
                   showing exactly what the model receives.

Usage (from project root, venv active):
    python scripts/visualize_preprocessing.py                    # 4 random from s1
    python scripts/visualize_preprocessing.py --speaker s5 --n 6
    python scripts/visualize_preprocessing.py --speakers s1 s5 s15 --n 2
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

DATA_DIR          = os.path.join(os.path.dirname(__file__), "..", "data")
PREPROCESSED_DIR  = os.path.join(DATA_DIR, "preprocessed")
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), "..", "reports", "preprocess_viz")

N_FRAMES   = 6          # number of columns per sample
THUMB_W    = 160        # display width per thumb
THUMB_H    = 120        # display height per thumb
LABEL_H    = 22         # height of the row label strip
PAD        = 4          # pixels between thumbs


def _read_raw_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def _get_stable_bbox(raw_frames: list[np.ndarray]):
    """Compute median bbox across sampled frames (same logic as preprocessing)."""
    if not raw_frames:
        return None
    h, w = raw_frames[0].shape[:2]
    sample_interval = max(1, len(raw_frames) // 10)
    bboxes = []
    for i in range(0, len(raw_frames), sample_interval):
        bbox = _detect_mouth_bbox(raw_frames[i], h, w)
        if bbox is not None:
            bboxes.append(bbox)
    if not bboxes:
        return None
    return (
        int(np.median([b[0] for b in bboxes])),
        int(np.median([b[1] for b in bboxes])),
        int(np.median([b[2] for b in bboxes])),
        int(np.median([b[3] for b in bboxes])),
    )


def _make_label_strip(text: str, width: int, height: int = LABEL_H,
                      bg: int = 30, fg: int = 220) -> np.ndarray:
    strip = np.full((height, width), bg, dtype=np.uint8)
    cv2.putText(strip, text, (6, height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fg, 1, cv2.LINE_AA)
    return strip


def visualize_sample(video_path: str, npy_path: str) -> np.ndarray | None:
    """
    Build a comparison image for one sample.
    Returns an H×W uint8 numpy array, or None on failure.
    """
    name = os.path.splitext(os.path.basename(video_path))[0]
    speaker = os.path.basename(os.path.dirname(video_path)).replace("_processed", "")

    # ---- Load raw video ----
    raw_frames = _read_raw_frames(video_path)
    if not raw_frames:
        print(f"  [SKIP] {name}: could not read video")
        return None

    # ---- Load preprocessed npy ----
    if not os.path.exists(npy_path):
        print(f"  [SKIP] {name}: .npy not found at {npy_path}")
        return None
    arr = np.load(npy_path)
    dtype_tag = "uint8 ✓" if arr.dtype == np.uint8 else "float32"
    if arr.dtype == np.uint8:
        npy_frames = arr.astype(np.float32) / 255.0
    else:
        npy_frames = arr.astype(np.float32)

    # ---- Compute stable bbox (same as preprocess pipeline) ----
    bbox = _get_stable_bbox(raw_frames)

    # ---- Pick N evenly-spaced frame indices ----
    indices = np.linspace(0, min(len(raw_frames), MAX_FRAMES) - 1,
                          N_FRAMES, dtype=int)

    # ---- Build RAW row ----
    raw_thumbs = []
    h_raw, w_raw = raw_frames[0].shape[:2]
    for idx in indices:
        frame = raw_frames[min(idx, len(raw_frames) - 1)].copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_raw, x2), min(h_raw, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        else:
            cv2.rectangle(frame,
                          (LIP_X_START, LIP_Y_START), (LIP_X_END, LIP_Y_END),
                          (0, 140, 255), 2)

        # Convert BGR→gray for uniform look, then resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"f{idx}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 1, cv2.LINE_AA)
        raw_thumbs.append(thumb)

    # ---- Build CROP row ----
    crop_thumbs = []
    for idx in indices:
        frame_f = npy_frames[min(idx, MAX_FRAMES - 1)]
        crop_uint8 = (frame_f * 255).astype(np.uint8)
        thumb = cv2.resize(crop_uint8, (THUMB_W, THUMB_H),
                           interpolation=cv2.INTER_LINEAR)
        crop_thumbs.append(thumb)

    # ---- Assemble contact sheet ----
    def hstack_with_pad(thumbs: list[np.ndarray]) -> np.ndarray:
        padded = []
        for t in thumbs:
            padded.append(t)
            padded.append(np.full((THUMB_H, PAD), 15, dtype=np.uint8))
        padded.pop()  # remove trailing pad
        return np.hstack(padded)

    row_raw  = hstack_with_pad(raw_thumbs)
    row_crop = hstack_with_pad(crop_thumbs)
    total_w  = row_raw.shape[1]

    detector = "MediaPipe" if (_check_mediapipe() and bbox is not None) else \
               ("Haar fallback" if bbox is not None else "NO DETECTION")

    label_raw  = _make_label_strip(
        f"RAW ({w_raw}×{h_raw})  |  {detector}  |  speaker: {speaker}",
        total_w)
    label_crop = _make_label_strip(
        f"PREPROCESSED  |  {dtype_tag}  |  shape {arr.shape}  |  {name}",
        total_w, bg=20)

    separator = np.full((3, total_w), 80, dtype=np.uint8)

    sheet = np.vstack([label_raw, row_raw, separator, label_crop, row_crop])
    return sheet


def main():
    parser = argparse.ArgumentParser(description="Visualise raw vs preprocessed")
    parser.add_argument("--speakers", nargs="+", default=["s1"],
                        help="One or more speaker IDs, e.g. --speakers s1 s5 s15")
    parser.add_argument("--speaker", default=None,
                        help="Single speaker shorthand (overrides --speakers)")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of samples per speaker")
    args = parser.parse_args()

    speakers = [args.speaker] if args.speaker else args.speakers
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"MediaPipe: {_check_mediapipe()}")
    print(f"Output → {os.path.abspath(OUTPUT_DIR)}\n")

    for speaker in speakers:
        pattern = os.path.join(DATA_DIR, f"{speaker}_processed", "*.mpg")
        all_videos = sorted(glob.glob(pattern))
        if not all_videos:
            print(f"[WARN] No videos found for speaker '{speaker}' ({pattern})")
            continue

        samples = random.sample(all_videos, min(args.n, len(all_videos)))
        print(f"Speaker {speaker}: checking {len(samples)} samples")

        for video_path in samples:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            npy_path = os.path.join(PREPROCESSED_DIR,
                                    f"{speaker}_processed_{stem}.npy")

            sheet = visualize_sample(video_path, npy_path)
            if sheet is None:
                continue

            out_path = os.path.join(OUTPUT_DIR, f"{speaker}_{stem}.png")
            cv2.imwrite(out_path, sheet)
            print(f"  saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
