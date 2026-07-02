"""Temporal slicing utilities: .align timestamps -> word frame segments."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from src.isolated_word.common import (
    SILENCE_TOKENS,
    WORD_TO_IDX,
    parse_align_entries,
)


@dataclass(frozen=True)
class WordSegment:
    sample_id: str
    npy_path: str
    align_path: str
    word: str
    class_idx: int
    start_frame: int
    end_frame: int
    num_frames: int
    raw_start: int
    raw_end: int
    was_clipped: bool


@dataclass
class SegmentStats:
    missing_align: int = 0
    missing_npy: int = 0
    bad_align: int = 0
    unknown_word: int = 0
    out_of_bounds_fixed: int = 0
    empty_segment_fixed: int = 0


def _timestamp_to_frame(timestamp: int, t0: int, duration: int, total_frames: int) -> float:
    rel = float(timestamp - t0) / float(max(duration, 1))
    return rel * float(total_frames)


def alignment_to_segments(
    sample_id: str,
    npy_path: str,
    align_path: str,
    total_frames: int,
    stats: SegmentStats | None = None,
) -> list[WordSegment]:
    """Convert one .align file into word-level frame segments for the given video."""
    if total_frames <= 0:
        if stats is not None:
            stats.bad_align += 1
        return []

    try:
        entries = parse_align_entries(align_path)
    except Exception:
        if stats is not None:
            stats.bad_align += 1
        return []

    t0 = min(e[0] for e in entries)
    t1 = max(e[1] for e in entries)
    duration = max(1, t1 - t0)

    segments: list[WordSegment] = []

    for raw_start, raw_end, token in entries:
        if token in SILENCE_TOKENS:
            continue
        if token not in WORD_TO_IDX:
            if stats is not None:
                stats.unknown_word += 1
            continue

        start_float = _timestamp_to_frame(raw_start, t0=t0, duration=duration, total_frames=total_frames)
        end_float = _timestamp_to_frame(raw_end, t0=t0, duration=duration, total_frames=total_frames)

        start_idx = int(math.floor(start_float))
        end_idx = int(math.ceil(end_float))
        clipped = False

        if start_idx < 0:
            start_idx = 0
            clipped = True
        if end_idx > total_frames:
            end_idx = total_frames
            clipped = True
        if start_idx >= total_frames:
            start_idx = total_frames - 1
            clipped = True

        if end_idx <= start_idx:
            end_idx = min(total_frames, start_idx + 1)
            clipped = True
            if stats is not None:
                stats.empty_segment_fixed += 1

        if clipped and stats is not None:
            stats.out_of_bounds_fixed += 1

        segments.append(
            WordSegment(
                sample_id=sample_id,
                npy_path=npy_path,
                align_path=align_path,
                word=token,
                class_idx=int(WORD_TO_IDX[token]),
                start_frame=int(start_idx),
                end_frame=int(end_idx),
                num_frames=int(end_idx - start_idx),
                raw_start=int(raw_start),
                raw_end=int(raw_end),
                was_clipped=bool(clipped),
            )
        )

    return segments


def slice_and_pad_word_clip(
    frames: np.ndarray,
    start_frame: int,
    end_frame: int,
    target_frames: int,
    pad_mode: str = "repeat_last",
) -> np.ndarray:
    """Slice word clip and pad/truncate to fixed target_frames.

    pad_mode:
      - 'repeat_last': repeat final frame
      - 'zeros': zero pad
    """
    if frames.ndim == 3:
        frames = frames[..., np.newaxis]

    t = int(frames.shape[0])
    h = int(frames.shape[1])
    w = int(frames.shape[2])
    c = int(frames.shape[3])

    start = max(0, min(int(start_frame), t - 1))
    end = max(start + 1, min(int(end_frame), t))

    clip = frames[start:end]
    if clip.shape[0] == 0:
        clip = frames[max(0, min(start, t - 1)) : max(1, min(start + 1, t))]

    if clip.shape[0] > target_frames:
        # center-crop in time to keep core articulation region
        overflow = clip.shape[0] - target_frames
        left = overflow // 2
        right = left + target_frames
        clip = clip[left:right]

    if clip.shape[0] < target_frames:
        pad_len = target_frames - clip.shape[0]
        if pad_mode == "zeros":
            pad = np.zeros((pad_len, h, w, c), dtype=clip.dtype)
        else:
            last = clip[-1:]
            pad = np.repeat(last, repeats=pad_len, axis=0)
        clip = np.concatenate([clip, pad], axis=0)

    return clip.astype(np.float32)
