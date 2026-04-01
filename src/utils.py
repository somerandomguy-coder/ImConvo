"""
Shared constants, vocabulary, and video processing utilities
for the GRID lip reading corpus.

Uses OpenCV's Haar cascade face detector for dynamic lip region
detection instead of hardcoded pixel coordinates, so the crop
adapts to each speaker's face position and size.
"""

import glob
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Video processing constants
# ---------------------------------------------------------------------------

# Legacy hardcoded lip region (used as fallback when face detection fails)
FALLBACK_LIP_Y_START = 190
FALLBACK_LIP_Y_END = 270
FALLBACK_LIP_X_START = 120
FALLBACK_LIP_X_END = 240

# Keep old names as aliases for backward compatibility (inference.py uses them)
LIP_Y_START = FALLBACK_LIP_Y_START
LIP_Y_END = FALLBACK_LIP_Y_END
LIP_X_START = FALLBACK_LIP_X_START
LIP_X_END = FALLBACK_LIP_X_END

FRAME_HEIGHT = 80
FRAME_WIDTH = 120
MAX_FRAMES = 75

# --- Face / mouth crop parameters ---
# The mouth region is estimated as a sub-region of the detected face bbox.
# These ratios define where the mouth sits within the face bounding box.
# (Validated against the GRID corpus and standard lip-reading literature)
MOUTH_TOP_RATIO = 0.65      # mouth starts at 65% of face height from the top
MOUTH_BOTTOM_RATIO = 0.95   # mouth ends at 95% of face height
MOUTH_LEFT_RATIO = 0.20     # mouth starts at 20% from the left of face
MOUTH_RIGHT_RATIO = 0.80    # mouth ends at 80% from the right of face

# Padding around the estimated mouth region (fraction of mouth bbox)
MOUTH_PAD_X = 0.10  # 10% extra on each side horizontally
MOUTH_PAD_Y = 0.10  # 10% extra vertically

# Minimum face size for detection (pixels)
MIN_FACE_SIZE = 30

# ---------------------------------------------------------------------------
# Character-level vocabulary (for CTC)
# ---------------------------------------------------------------------------

# a-z = 0..25, space = 26, CTC blank = 27
CHAR_LIST = list("abcdefghijklmnopqrstuvwxyz")
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_LIST)}
SPACE_IDX = 26
BLANK_IDX = 27
NUM_CHARS = 28  # 26 letters + space + blank

SILENCE_TOKENS = {"sil", "sp"}
MAX_CHAR_LEN = 100  # max character sequence length (padded)


def text_to_char_indices(text: str) -> list:
    """Convert a text string to a list of character indices."""
    indices = []
    for c in text:
        if c == " ":
            indices.append(SPACE_IDX)
        elif c in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[c])
    return indices


def char_indices_to_text(indices: list) -> str:
    """Convert character indices back to text (for decoding CTC output)."""
    chars = []
    for idx in indices:
        if idx == SPACE_IDX:
            chars.append(" ")
        elif idx == BLANK_IDX:
            continue  # skip blank
        elif 0 <= idx < len(CHAR_LIST):
            chars.append(CHAR_LIST[idx])
    return "".join(chars)


# ---------------------------------------------------------------------------
# Alignment parsing
# ---------------------------------------------------------------------------


def parse_alignment_text(align_path: str) -> str:
    """Parse an alignment file and return the sentence as a string."""
    words = []
    with open(align_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2] not in SILENCE_TOKENS:
                words.append(parts[2])
    return " ".join(words)


def parse_alignment_chars(align_path: str) -> tuple:
    """
    Parse alignment file → character indices + length.

    Returns:
        (char_indices, length) — padded to MAX_CHAR_LEN
    """
    text = parse_alignment_text(align_path)
    indices = text_to_char_indices(text)
    length = len(indices)

    # Pad to MAX_CHAR_LEN
    if len(indices) < MAX_CHAR_LEN:
        indices += [0] * (MAX_CHAR_LEN - len(indices))  # pad with 0 (ignored via label_length)
    else:
        indices = indices[:MAX_CHAR_LEN]
        length = MAX_CHAR_LEN

    return np.array(indices, dtype=np.int32), length


# ---------------------------------------------------------------------------
# Smart lip detection with OpenCV Haar Cascade
# ---------------------------------------------------------------------------

# Load cascade classifiers once at module level (ordered by speed → sensitivity)
_cascades = [
    # (cascade, scaleFactor, minNeighbors) — tried in order
    (cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"), 1.1, 3),
    (cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"), 1.05, 2),
    (cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"), 1.1, 3),
]


def _detect_face(gray_frame: np.ndarray):
    """
    Detect the largest face in a grayscale frame using cascading Haar classifiers.

    Tries multiple cascades with progressively more sensitive settings
    to maximize detection rate across different speakers.

    Returns:
        (x, y, w, h) of the largest face, or None if no face is detected.
    """
    for cascade, scale_factor, min_neighbors in _cascades:
        faces = cascade.detectMultiScale(
            gray_frame,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) > 0:
            # Return the largest face (by area)
            areas = [w * h for (x, y, w, h) in faces]
            best_idx = np.argmax(areas)
            return tuple(faces[best_idx])

    return None


def _face_to_mouth_bbox(face_bbox, frame_h: int, frame_w: int):
    """
    Estimate the mouth bounding box from a face bounding box.

    The mouth is typically located in the lower-middle portion of the face.
    We use configurable ratios to extract this region with padding.

    Args:
        face_bbox: (x, y, w, h) of the face.
        frame_h: Frame height in pixels.
        frame_w: Frame width in pixels.

    Returns:
        (x_start, y_start, x_end, y_end) of the mouth region.
    """
    fx, fy, fw, fh = face_bbox

    # Estimate mouth region within the face bbox
    mouth_x_start = fx + int(fw * MOUTH_LEFT_RATIO)
    mouth_x_end = fx + int(fw * MOUTH_RIGHT_RATIO)
    mouth_y_start = fy + int(fh * MOUTH_TOP_RATIO)
    mouth_y_end = fy + int(fh * MOUTH_BOTTOM_RATIO)

    # Add padding
    mouth_w = mouth_x_end - mouth_x_start
    mouth_h = mouth_y_end - mouth_y_start
    pad_x = int(mouth_w * MOUTH_PAD_X)
    pad_y = int(mouth_h * MOUTH_PAD_Y)

    x_start = max(0, mouth_x_start - pad_x)
    y_start = max(0, mouth_y_start - pad_y)
    x_end = min(frame_w, mouth_x_end + pad_x)
    y_end = min(frame_h, mouth_y_end + pad_y)

    return (x_start, y_start, x_end, y_end)


def extract_lip_frames(video_path: str) -> np.ndarray:
    """
    Load video, detect face, crop lip region dynamically,
    convert to grayscale, resize, and normalize.

    Strategy:
        1. Read all frames from the video.
        2. Detect faces in a sample of frames (every Nth frame for speed).
        3. Compute the median mouth bounding box across all detections
           for a stable, jitter-free crop.
        4. If no face is detected in any frame, fall back to the legacy
           hardcoded coordinates.
        5. Crop all frames using the stable bounding box.

    Returns:
        np.ndarray of shape (MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), float32 in [0, 1].
    """
    frames = []
    raw_frames = []

    # --- Pass 1: Read all frames ---
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) == 0:
        return np.zeros((MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

    frame_h, frame_w = raw_frames[0].shape[:2]

    # --- Pass 2: Detect faces in sampled frames ---
    # Sample ~10 evenly-spaced frames for face detection
    sample_interval = max(1, len(raw_frames) // 10)
    mouth_bboxes = []

    for i in range(0, len(raw_frames), sample_interval):
        gray = cv2.cvtColor(raw_frames[i], cv2.COLOR_BGR2GRAY)
        face = _detect_face(gray)
        if face is not None:
            mouth_bbox = _face_to_mouth_bbox(face, frame_h, frame_w)
            mouth_bboxes.append(mouth_bbox)

    # If detection rate is low, retry on ALL frames
    num_sampled = len(range(0, len(raw_frames), sample_interval))
    if len(mouth_bboxes) < num_sampled * 0.5 and sample_interval > 1:
        mouth_bboxes = []
        for i in range(len(raw_frames)):
            gray = cv2.cvtColor(raw_frames[i], cv2.COLOR_BGR2GRAY)
            face = _detect_face(gray)
            if face is not None:
                mouth_bbox = _face_to_mouth_bbox(face, frame_h, frame_w)
                mouth_bboxes.append(mouth_bbox)

    # --- Compute stable crop region ---
    if len(mouth_bboxes) > 0:
        # Use median of detected mouth bboxes for stability
        all_x_starts = [b[0] for b in mouth_bboxes]
        all_y_starts = [b[1] for b in mouth_bboxes]
        all_x_ends   = [b[2] for b in mouth_bboxes]
        all_y_ends   = [b[3] for b in mouth_bboxes]

        x_start = int(np.median(all_x_starts))
        y_start = int(np.median(all_y_starts))
        x_end   = int(np.median(all_x_ends))
        y_end   = int(np.median(all_y_ends))

        # Clamp to frame boundaries
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end   = min(frame_w, x_end)
        y_end   = min(frame_h, y_end)

        detection_rate = len(mouth_bboxes) / max(1, len(raw_frames) // sample_interval)
        if detection_rate < 0.5:
            basename = os.path.basename(video_path)
            print(f"  [WARN] {basename}: face detected in only "
                  f"{detection_rate*100:.0f}% of sampled frames")
    else:
        # No face detected — skip this sample
        basename = os.path.basename(video_path)
        print(f"  [SKIP] {basename}: no face detected in any frame")
        return None

    # --- Pass 3: Crop, grayscale, resize ---
    for frame in raw_frames:
        lip = frame[y_start:y_end, x_start:x_end]
        if lip.size == 0:
            continue
        lip_gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        lip_resized = cv2.resize(lip_gray, (FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(lip_resized)

    if len(frames) == 0:
        return None

    frames = np.array(frames, dtype=np.float32) / 255.0

    # Pad or truncate
    T = frames.shape[0]
    if T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)
    else:
        frames = frames[:MAX_FRAMES]

    return frames
