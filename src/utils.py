"""
Shared constants, vocabulary, and video processing utilities
for the GRID lip reading corpus.

Uses MediaPipe Face Mesh for precise lip landmark detection —
yields a tight, speaker-adaptive lip crop on every frame.
"""

import glob
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Video processing constants
# ---------------------------------------------------------------------------

# Fallback region (used only when MediaPipe detects no face at all)
FALLBACK_LIP_Y_START = 190
FALLBACK_LIP_Y_END = 270
FALLBACK_LIP_X_START = 120
FALLBACK_LIP_X_END = 240

# Legacy aliases kept for inference.py webcam overlay
LIP_Y_START = FALLBACK_LIP_Y_START
LIP_Y_END = FALLBACK_LIP_Y_END
LIP_X_START = FALLBACK_LIP_X_START
LIP_X_END = FALLBACK_LIP_X_END

FRAME_HEIGHT = 80
FRAME_WIDTH = 120
MAX_FRAMES = 75
TARGET_FPS = 25.0  # GRID corpus native fps; all videos resampled to this

# Padding around the MediaPipe lip bounding box (fraction of lip size)
LIP_PAD_X = 0.25
LIP_PAD_Y = 0.35

# ---------------------------------------------------------------------------
# MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10)
# ---------------------------------------------------------------------------

# Lip landmark indices from the 478-point FaceLandmarker model
_LIP_LANDMARK_INDICES: list[int] = [
    # outer upper lip
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    # outer lower lip
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    # inner upper lip
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    # inner lower lip
    95, 88, 178, 87, 14, 317, 402, 318, 324,
]

_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_FACE_LANDMARKER_MODEL_PATH = Path(__file__).resolve().parent.parent / "reports" / "face_landmarker.task"

_FACE_LANDMARKER: "mp.tasks.vision.FaceLandmarker | None" = None


def _get_face_landmarker() -> "mp.tasks.vision.FaceLandmarker":
    global _FACE_LANDMARKER
    if _FACE_LANDMARKER is not None:
        return _FACE_LANDMARKER

    model_path = _FACE_LANDMARKER_MODEL_PATH
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[MediaPipe] Downloading face_landmarker model to {model_path} ...")
        urllib.request.urlretrieve(_FACE_LANDMARKER_MODEL_URL, model_path)
        print("[MediaPipe] Download complete.")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
    )
    _FACE_LANDMARKER = FaceLandmarker.create_from_options(options)
    return _FACE_LANDMARKER


def _mediapipe_lip_bbox(
    bgr_frame: np.ndarray,
) -> "tuple[int, int, int, int] | None":
    """Return (x_start, y_start, x_end, y_end) for the lip region, or None."""
    h, w = bgr_frame.shape[:2]
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_face_landmarker().detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    xs = [landmarks[i].x * w for i in _LIP_LANDMARK_INDICES]
    ys = [landmarks[i].y * h for i in _LIP_LANDMARK_INDICES]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    lip_w = x_max - x_min
    lip_h = y_max - y_min

    x_start = max(0, int(x_min - lip_w * LIP_PAD_X))
    x_end   = min(w, int(x_max + lip_w * LIP_PAD_X))
    y_start = max(0, int(y_min - lip_h * LIP_PAD_Y))
    y_end   = min(h, int(y_max + lip_h * LIP_PAD_Y))

    return (x_start, y_start, x_end, y_end)

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
MAX_CHAR_LEN = 40 # max character sequence length (padded)
# hotfix: sentence usually only around 40 character


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
        indices += [BLANK_IDX] * (MAX_CHAR_LEN - len(indices))  # pad with 0 (ignored via label_length)
        # hotfix: pad with blank_idx instead
    else:
        indices = indices[:MAX_CHAR_LEN]
        length = MAX_CHAR_LEN

    return np.array(indices, dtype=np.int32), length


# ---------------------------------------------------------------------------
# Video lip extraction using MediaPipe 
# ---------------------------------------------------------------------------

def extract_lip_frames(video_path: str) -> "np.ndarray | None":
    """Load video, detect lip landmarks with MediaPipe, crop, resize, normalise.

    Strategy:
        1. Read all frames; resample to TARGET_FPS for temporal consistency.
        2. Run MediaPipe on a sparse sample (~10 frames) for speed.
        3. If detection rate < 50 %, retry on every frame.
        4. Take the median bounding box across detections for a stable crop.
        5. Crop + grayscale + resize every frame to (FRAME_HEIGHT, FRAME_WIDTH).

    Returns:
        float32 array of shape (MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH) in [0, 1],
        or None if no face is detected.
    """
    raw_frames: list[np.ndarray] = []

    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        return None

    # Resample to TARGET_FPS so inference matches training temporal rate
    if abs(source_fps - TARGET_FPS) > 0.5:
        target_count = max(1, int(round(len(raw_frames) * TARGET_FPS / source_fps)))
        indices = np.linspace(0, len(raw_frames) - 1, target_count).round().astype(int)
        raw_frames = [raw_frames[i] for i in indices]

    frame_h, frame_w = raw_frames[0].shape[:2]

    # --- Pass 2: detect lip bbox on sampled frames ---
    sample_interval = max(1, len(raw_frames) // 10)
    lip_bboxes: list[tuple[int, int, int, int]] = []

    for i in range(0, len(raw_frames), sample_interval):
        bbox = _mediapipe_lip_bbox(raw_frames[i])
        if bbox is not None:
            lip_bboxes.append(bbox)

    # Retry on all frames if detection rate is low
    num_sampled = len(range(0, len(raw_frames), sample_interval))
    if len(lip_bboxes) < num_sampled * 0.5 and sample_interval > 1:
        lip_bboxes = []
        for frame in raw_frames:
            bbox = _mediapipe_lip_bbox(frame)
            if bbox is not None:
                lip_bboxes.append(bbox)

    if not lip_bboxes:
        print(f"  [SKIP] {os.path.basename(video_path)}: no face detected")
        return None

    # Median bbox for stability across frames
    x_start = max(0,       int(np.median([b[0] for b in lip_bboxes])))
    y_start = max(0,       int(np.median([b[1] for b in lip_bboxes])))
    x_end   = min(frame_w, int(np.median([b[2] for b in lip_bboxes])))
    y_end   = min(frame_h, int(np.median([b[3] for b in lip_bboxes])))

    detection_rate = len(lip_bboxes) / max(1, num_sampled)
    if detection_rate < 0.5:
        print(f"  [WARN] {os.path.basename(video_path)}: face detected in "
              f"{detection_rate*100:.0f}% of sampled frames")

    # --- Pass 3: crop, grayscale, resize ---
    frames: list[np.ndarray] = []
    for frame in raw_frames:
        y1 = max(0, y_start)
        y2 = min(frame.shape[0], y_end)
        x1 = max(0, x_start)
        x2 = min(frame.shape[1], x_end)

        lip = frame[y1:y2, x1:x2]
        if lip.size == 0:
            lip_resized = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        else:
            lip_gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
            lip_resized = cv2.resize(lip_gray, (FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(lip_resized)

    if not frames:
        return None

    out = np.array(frames, dtype=np.float32) / 255.0

    T = out.shape[0]
    if T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        out = np.concatenate([out, pad], axis=0)
    elif T > MAX_FRAMES:
        indices = np.linspace(0, T - 1, MAX_FRAMES).round().astype(int)
        out = out[indices]

    return out
