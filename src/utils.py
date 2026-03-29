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
