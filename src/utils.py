"""
Shared constants, vocabulary, and video processing utilities
for the GRID lip reading corpus.
"""

import glob
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Video processing constants
# ---------------------------------------------------------------------------

LIP_Y_START = 190
LIP_Y_END = 270
LIP_X_START = 120
LIP_X_END = 240

FRAME_HEIGHT = 80
FRAME_WIDTH = 120
MAX_FRAMES = 75

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

MAX_LABEL_LEN = 6   # GRID sentences have exactly 6 content words
PAD_IDX = -1
SILENCE_TOKENS = {"sil", "sp"}
SLOT_NAMES = ["command", "color", "preposition", "letter", "digit", "adverb"]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def build_vocab(align_dir: str) -> dict:
    """Build a word-to-index vocabulary from alignment files."""
    words = set()
    for path in glob.glob(os.path.join(align_dir, "*.align")):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3 and parts[2] not in SILENCE_TOKENS:
                    words.add(parts[2])
    word2idx = {w: i for i, w in enumerate(sorted(words))}
    return word2idx


def parse_alignment(align_path: str, word2idx: dict) -> list:
    """Parse an alignment file and return a list of word indices."""
    indices = []
    with open(align_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2] not in SILENCE_TOKENS:
                if parts[2] in word2idx:
                    indices.append(word2idx[parts[2]])
    return indices


def pad_label(label_indices: list) -> np.ndarray:
    """Pad or truncate label to MAX_LABEL_LEN."""
    if len(label_indices) < MAX_LABEL_LEN:
        label_indices += [PAD_IDX] * (MAX_LABEL_LEN - len(label_indices))
    else:
        label_indices = label_indices[:MAX_LABEL_LEN]
    return np.array(label_indices, dtype=np.int32)


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------


def extract_lip_frames(video_path: str) -> np.ndarray:
    """
    Load video, crop lip region, convert to grayscale, resize, normalize.

    Returns:
        np.ndarray of shape (MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), float32 in [0, 1].
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lip = frame[LIP_Y_START:LIP_Y_END, LIP_X_START:LIP_X_END]
        lip_gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        lip_resized = cv2.resize(lip_gray, (FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(lip_resized)
    cap.release()

    if len(frames) == 0:
        return np.zeros((MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

    frames = np.array(frames, dtype=np.float32) / 255.0

    # Pad or truncate
    T = frames.shape[0]
    if T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)
    else:
        frames = frames[:MAX_FRAMES]

    return frames
