"""
Shared constants, vocabulary, and video processing utilities
for the GRID lip reading corpus.

Uses OpenCV's Haar cascade face detector for dynamic lip region
detection instead of hardcoded pixel coordinates, so the crop
adapts to each speaker's face position and size.
"""

import glob
import os
import urllib.request
from pathlib import Path

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
MAX_CHAR_LEN = 40 # max character sequence length (padded)
# hotfix: sentence usually only around 40 character

# ---------------------------------------------------------------------------
# MediaPipe lip landmark indices (Face Mesh 468-point model)
# ---------------------------------------------------------------------------

# Outer lip contour — used to compute a tight, landmark-anchored crop bbox
LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
           291, 308, 415, 310, 312, 13, 82, 81, 80, 191]

# MediaPipe gives a tight contour around the red lip surface only (~10px tall
# for a closed mouth). These larger multipliers + pixel floor expand the crop to
# include surrounding skin context, matching roughly what Haar-based cropping
# produces via face-ratio estimates.
_MP_PAD_X_RATIO = 0.5   # 50% of lip width added to each horizontal side
_MP_PAD_Y_RATIO = 2.0   # 200% of lip height added to each vertical side
_MP_PAD_MIN_PX  = 20    # minimum pixels of padding on every side

_mediapipe_available: bool | None = None
_face_mesh_offline = None  # per-process singleton (safe in multiprocessing)

# MediaPipe Tasks API — face landmarker model (auto-downloaded on first use, ~1.8 MB)
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_FACE_LANDMARKER_CACHE = (
    Path(__file__).resolve().parent.parent / "temp_reports" / "face_landmarker.task"
)


def _check_mediapipe() -> bool:
    global _mediapipe_available
    if _mediapipe_available is None:
        try:
            import mediapipe  # noqa: F401
            _mediapipe_available = True
        except ImportError:
            _mediapipe_available = False
    return _mediapipe_available


def _ensure_face_landmarker_model() -> str:
    """Download the MediaPipe face landmarker .task file once and cache it."""
    if not _FACE_LANDMARKER_CACHE.exists():
        _FACE_LANDMARKER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("[MediaPipe] Downloading face landmarker model (~1.8 MB)...")
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, str(_FACE_LANDMARKER_CACHE))
        print(f"[MediaPipe] Saved to {_FACE_LANDMARKER_CACHE}")
    return str(_FACE_LANDMARKER_CACHE)


def _get_face_mesh_offline():
    """Lazily create a per-process FaceLandmarker for offline/batch use (IMAGE mode)."""
    global _face_mesh_offline
    if _face_mesh_offline is None:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        model_path = _ensure_face_landmarker_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _face_mesh_offline = mp_vision.FaceLandmarker.create_from_options(options)
    return _face_mesh_offline


def _mediapipe_mouth_bbox(frame_bgr: np.ndarray, frame_h: int, frame_w: int):
    """
    Detect lip bounding box via MediaPipe Face Landmarker landmark min/max.
    Returns (x_start, y_start, x_end, y_end) or None if detection fails.
    """
    import mediapipe as mp
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = _get_face_mesh_offline().detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    xs = [lm[i].x * frame_w for i in LIP_IDX]
    ys = [lm[i].y * frame_h for i in LIP_IDX]
    lip_w = max(xs) - min(xs)
    lip_h = max(ys) - min(ys)
    pad_x = max(lip_w * _MP_PAD_X_RATIO, _MP_PAD_MIN_PX)
    pad_y = max(lip_h * _MP_PAD_Y_RATIO, _MP_PAD_MIN_PX)
    x_start = max(0, int(min(xs) - pad_x))
    y_start = max(0, int(min(ys) - pad_y))
    x_end = min(frame_w, int(max(xs) + pad_x))
    y_end = min(frame_h, int(max(ys) + pad_y))
    return (x_start, y_start, x_end, y_end)


def create_live_face_landmarker():
    """
    Create a MediaPipe FaceLandmarker for per-frame live inference (IMAGE mode).
    Returns the landmarker object, or None if MediaPipe is unavailable.
    """
    if not _check_mediapipe():
        return None
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        model_path = _ensure_face_landmarker_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"[MediaPipe] Failed to create live landmarker: {e}")
        return None


def detect_lip_bbox_frame(landmarker, frame_bgr: np.ndarray):
    """
    Run MediaPipe face landmark detection on a single BGR frame.
    Returns (x1, y1, x2, y2) or None.
    """
    import mediapipe as mp
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    xs = [lm[i].x * w for i in LIP_IDX]
    ys = [lm[i].y * h for i in LIP_IDX]
    lip_w = max(xs) - min(xs)
    lip_h = max(ys) - min(ys)
    pad_x = max(lip_w * _MP_PAD_X_RATIO, _MP_PAD_MIN_PX)
    pad_y = max(lip_h * _MP_PAD_Y_RATIO, _MP_PAD_MIN_PX)
    x1 = max(0, int(min(xs) - pad_x))
    y1 = max(0, int(min(ys) - pad_y))
    x2 = min(w, int(max(xs) + pad_x))
    y2 = min(h, int(max(ys) + pad_y))
    return (x1, y1, x2, y2)


def _detect_mouth_bbox(frame_bgr: np.ndarray, frame_h: int, frame_w: int):
    """Try MediaPipe first; fall back to Haar cascade."""
    if _check_mediapipe():
        bbox = _mediapipe_mouth_bbox(frame_bgr, frame_h, frame_w)
        if bbox is not None:
            return bbox
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face = _detect_face(gray)
    if face is not None:
        return _face_to_mouth_bbox(face, frame_h, frame_w)
    return None


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
    Load video, detect lip region (MediaPipe primary, Haar fallback),
    crop, convert to grayscale, resize, and normalize.

    Uses a stable median bbox computed across sampled frames to avoid
    jitter, then applies it uniformly to all frames.

    Returns:
        np.ndarray of shape (MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), float32 in [0, 1],
        or None if no face/lips were detected in any frame.
    """
    raw_frames = []

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

    # Detect on ~10 evenly-spaced frames for speed
    sample_interval = max(1, len(raw_frames) // 10)
    mouth_bboxes = []
    for i in range(0, len(raw_frames), sample_interval):
        bbox = _detect_mouth_bbox(raw_frames[i], frame_h, frame_w)
        if bbox is not None:
            mouth_bboxes.append(bbox)

    # Retry on all frames if detection rate is low
    num_sampled = len(range(0, len(raw_frames), sample_interval))
    if len(mouth_bboxes) < num_sampled * 0.5 and sample_interval > 1:
        mouth_bboxes = []
        for frame in raw_frames:
            bbox = _detect_mouth_bbox(frame, frame_h, frame_w)
            if bbox is not None:
                mouth_bboxes.append(bbox)

    if len(mouth_bboxes) == 0:
        print(f"  [SKIP] {os.path.basename(video_path)}: no lips detected in any frame")
        return None

    # Stable crop: median bbox across all detections
    x_start = max(0, int(np.median([b[0] for b in mouth_bboxes])))
    y_start = max(0, int(np.median([b[1] for b in mouth_bboxes])))
    x_end   = min(frame_w, int(np.median([b[2] for b in mouth_bboxes])))
    y_end   = min(frame_h, int(np.median([b[3] for b in mouth_bboxes])))

    detection_rate = len(mouth_bboxes) / max(1, num_sampled)
    if detection_rate < 0.5:
        print(f"  [WARN] {os.path.basename(video_path)}: "
              f"detected in only {detection_rate*100:.0f}% of sampled frames")

    frames = []
    for frame in raw_frames:
        y1, y2 = max(0, y_start), min(frame.shape[0], y_end)
        x1, x2 = max(0, x_start), min(frame.shape[1], x_end)
        lip = frame[y1:y2, x1:x2]
        if lip.size == 0:
            lip_resized = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        else:
            lip_gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
            lip_resized = cv2.resize(lip_gray, (FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(lip_resized)

    if len(frames) == 0:
        return None

    frames = np.array(frames, dtype=np.float32) / 255.0

    T = frames.shape[0]
    if T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)
    else:
        frames = frames[:MAX_FRAMES]

    return frames
