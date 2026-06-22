"""Live per-word game: vocab/slot maps, pass-rule, preprocessing, scoring."""
from __future__ import annotations

import base64

import cv2
import numpy as np

from experiments.isolated_word_level.common import SLOT_VOCABS, WORD_TO_IDX
from experiments.isolated_word_level.temporal_slicer import slice_and_pad_word_clip
from src.utils import FRAME_HEIGHT, FRAME_WIDTH, _mediapipe_lip_bbox

SLOT_NAMES: tuple[str, ...] = (
    "command",
    "color",
    "preposition",
    "letter",
    "digit",
    "adverb",
)

# Tunables — single source of truth.
BUFFER_FRAMES = 25
TARGET_FRAMES = 25
SCORE_EVERY_N_FRAMES = 3
PASS_THRESHOLD = 0.6
LETTER_SLOT_INDEX = 3
LETTER_TOPK_PASS = 3


def slot_candidate_indices(slot_index: int) -> list[int]:
    """Return the FLAT_VOCAB class indices for a given GRID slot."""
    return [WORD_TO_IDX[w] for w in SLOT_VOCABS[slot_index]]


def evaluate_pass(
    slot_index: int,
    target: str,
    ranked_words: list[str],
    confidence: float,
) -> bool:
    """Decide whether a scored window passes the round.

    ranked_words: the slot's candidate words sorted by probability desc.
    confidence:   probability mass of ranked_words[0] (the top word).
    """
    if slot_index == LETTER_SLOT_INDEX:
        return target in ranked_words[:LETTER_TOPK_PASS]
    return bool(ranked_words and ranked_words[0] == target and confidence >= PASS_THRESHOLD)


def decode_jpeg(b64: str) -> np.ndarray:
    """Decode a base64 JPEG string into a BGR uint8 frame."""
    try:
        raw = base64.b64decode(b64, validate=False)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not decode JPEG: {exc}") from exc
    if img is None:
        raise ValueError("Could not decode JPEG: empty image")
    return img


def preprocess_frames(bgr_frames: list[np.ndarray]) -> np.ndarray | None:
    """Lip-crop, grayscale, resize and pad frames to the model input tensor."""
    crops: list[np.ndarray] = []
    for frame in bgr_frames:
        bbox = _mediapipe_lip_bbox(frame)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        lip = frame[y0:y1, x0:x1]
        if lip.size == 0:
            continue
        gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT))
        crops.append(resized.astype(np.float32) / 255.0)

    if not crops:
        return None

    stacked = np.stack(crops, axis=0)  # (t, H, W)
    clip = slice_and_pad_word_clip(
        stacked, 0, stacked.shape[0], target_frames=TARGET_FRAMES, pad_mode="repeat_last"
    )  # (TARGET_FRAMES, H, W, 1)
    return clip[np.newaxis, ...].astype(np.float32)
