"""
Haar Cascade mouth extraction demo — for report visualisation only.
Generates comparison images: raw frame | face bbox | mouth crop | resized output.

Usage:
    python scripts/haar_cascade_demo.py
    python scripts/haar_cascade_demo.py --video data/s1_processed/bbaf2n.mpg --n_frames 6
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import _mediapipe_lip_bbox, FRAME_HEIGHT, FRAME_WIDTH

OUTPUT_DIR = "reports/haar_cascade_demo"

# Same hardcoded fallback coordinates used before MediaPipe
HARDCODED_Y_START = 190
HARDCODED_Y_END = 270
HARDCODED_X_START = 120
HARDCODED_X_END = 240


def mediapipe_median_bbox(raw_frames):
    """Replicate extract_lip_frames strategy: sparse sample → median bbox."""
    h, w = raw_frames[0].shape[:2]
    sample_interval = max(1, len(raw_frames) // 10)
    bboxes = []
    for i in range(0, len(raw_frames), sample_interval):
        bbox = _mediapipe_lip_bbox(raw_frames[i])
        if bbox is not None:
            bboxes.append(bbox)

    num_sampled = len(range(0, len(raw_frames), sample_interval))
    if len(bboxes) < num_sampled * 0.5 and sample_interval > 1:
        bboxes = [_mediapipe_lip_bbox(f) for f in raw_frames]
        bboxes = [b for b in bboxes if b is not None]

    if not bboxes:
        return None

    x_start = max(0,  int(np.median([b[0] for b in bboxes])))
    y_start = max(0,  int(np.median([b[1] for b in bboxes])))
    x_end   = min(w,  int(np.median([b[2] for b in bboxes])))
    y_end   = min(h,  int(np.median([b[3] for b in bboxes])))
    return (x_start, y_start, x_end, y_end)


def load_cascades():
    data_path = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(os.path.join(data_path, "haarcascade_frontalface_default.xml"))
    mouth_cascade = cv2.CascadeClassifier(os.path.join(data_path, "haarcascade_smile.xml"))
    return face_cascade, mouth_cascade


def extract_mouth_haar(frame_bgr, face_cascade, mouth_cascade):
    """
    Returns:
        mouth_crop  — cropped mouth region (BGR), or None if not detected
        face_bbox   — (x, y, w, h) of detected face, or None
        mouth_bbox  — (x, y, w, h) relative to full frame, or None
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return None, None, None

    # Largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face_roi_gray = gray[y:y+h, x:x+w]
    face_roi_bgr  = frame_bgr[y:y+h, x:x+w]

    # Search for mouth in lower half of face
    lower_half = face_roi_gray[h//2:, :]
    mouths = mouth_cascade.detectMultiScale(lower_half, scaleFactor=1.7, minNeighbors=11, minSize=(25, 15))

    if len(mouths) == 0:
        # Fall back to lower-third crop of face
        m_y = h * 2 // 3
        mouth_crop = face_roi_bgr[m_y:, :]
        mouth_abs = (x, y + m_y, w, h - m_y)
        return mouth_crop, (x, y, w, h), mouth_abs

    mx, my, mw, mh = max(mouths, key=lambda r: r[2] * r[3])
    my_abs = y + h // 2 + my
    mouth_crop = frame_bgr[my_abs:my_abs+mh, x+mx:x+mx+mw]
    return mouth_crop, (x, y, w, h), (x + mx, my_abs, mw, mh)


def make_comparison_row(frame_bgr, face_cascade, mouth_cascade, mp_median_bbox):
    """Build a single side-by-side comparison image for one frame.

    mp_median_bbox: pre-computed median bbox from mediapipe_median_bbox() —
                    matches the actual preprocessing strategy exactly.
    """
    h_orig, w_orig = frame_bgr.shape[:2]

    # --- Panel 1: raw frame ---
    panel1 = frame_bgr.copy()
    label(panel1, "Raw frame")

    # --- Panel 2: hardcoded crop (earliest approach) ---
    panel2 = frame_bgr.copy()
    cv2.rectangle(panel2,
                  (HARDCODED_X_START, HARDCODED_Y_START),
                  (HARDCODED_X_END,   HARDCODED_Y_END),
                  (0, 0, 255), 2)
    label(panel2, "Hardcoded bbox")

    # --- Panel 3: Haar Cascade detection ---
    panel3 = frame_bgr.copy()
    mouth_crop, face_bbox, mouth_bbox = extract_mouth_haar(frame_bgr, face_cascade, mouth_cascade)

    if face_bbox is not None:
        fx, fy, fw, fh = face_bbox
        cv2.rectangle(panel3, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
    if mouth_bbox is not None:
        mx, my, mw, mh = mouth_bbox
        cv2.rectangle(panel3, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
    label(panel3, "Haar Cascade")

    # --- Panel 4: Haar resized output ---
    target_h = h_orig
    if mouth_crop is not None and mouth_crop.size > 0:
        mouth_gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
        mouth_resized = cv2.resize(mouth_gray, (FRAME_WIDTH, FRAME_HEIGHT))
        scale = target_h / FRAME_HEIGHT
        mouth_display = cv2.resize(mouth_resized, (int(FRAME_WIDTH * scale), target_h))
        haar_out = cv2.cvtColor(mouth_display, cv2.COLOR_GRAY2BGR)
    else:
        haar_out = np.zeros((target_h, int(FRAME_WIDTH * target_h / FRAME_HEIGHT), 3), dtype=np.uint8)
    label(haar_out, f"Haar output ({FRAME_WIDTH}x{FRAME_HEIGHT})")

    # --- Panel 5: MediaPipe bbox (median across all frames — same as preprocessing) ---
    panel5 = frame_bgr.copy()
    mp_out = np.zeros((target_h, int(FRAME_WIDTH * target_h / FRAME_HEIGHT), 3), dtype=np.uint8)
    if mp_median_bbox is not None:
        x1, y1, x2, y2 = mp_median_bbox
        cv2.rectangle(panel5, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lip = frame_bgr[y1:y2, x1:x2]
        if lip.size > 0:
            lip_gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
            lip_resized = cv2.resize(lip_gray, (FRAME_WIDTH, FRAME_HEIGHT))
            scale = target_h / FRAME_HEIGHT
            disp = cv2.resize(lip_resized, (int(FRAME_WIDTH * scale), target_h))
            mp_out = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    label(panel5, "MediaPipe (median bbox)")

    # --- Panel 6: MediaPipe resized output ---
    label(mp_out, f"MediaPipe output ({FRAME_WIDTH}x{FRAME_HEIGHT})")

    # Resize all panels to same height
    panels = [panel1, panel2, panel3, haar_out, panel5, mp_out]
    panels = [cv2.resize(p, (int(p.shape[1] * target_h / p.shape[0]), target_h)) for p in panels]

    row = np.hstack(panels)
    return row


def label(img, text, pos=(10, 25)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def process_video(video_path, face_cascade, mouth_cascade, n_frames=6):
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        return []

    # Compute MediaPipe median bbox once across all frames — identical to preprocessing
    mp_median_bbox = mediapipe_median_bbox(raw_frames)

    indices = set(np.linspace(0, len(raw_frames) - 1, n_frames).astype(int).tolist())
    rows = []
    for i, frame in enumerate(raw_frames):
        if i in indices:
            row = make_comparison_row(frame, face_cascade, mouth_cascade, mp_median_bbox)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/s1_processed/bbaf2n.mpg")
    parser.add_argument("--n_frames", type=int, default=6, help="Number of sample frames")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    face_cascade, mouth_cascade = load_cascades()

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    print(f"Processing {args.video} ...")

    rows = process_video(args.video, face_cascade, mouth_cascade, args.n_frames)
    if not rows:
        print("[ERROR] No frames extracted.")
        return

    # Save individual frame comparisons
    for idx, row in enumerate(rows):
        out_path = os.path.join(args.output_dir, f"{video_name}_frame{idx:02d}.png")
        cv2.imwrite(out_path, row)
        print(f"  Saved {out_path}")

    # Save a vertical strip of all frames
    strip = np.vstack(rows)
    strip_path = os.path.join(args.output_dir, f"{video_name}_strip.png")
    cv2.imwrite(strip_path, strip)
    print(f"\nFull strip saved: {strip_path}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
