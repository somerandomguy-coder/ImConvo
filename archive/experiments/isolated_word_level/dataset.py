"""Dataset pipeline for isolated-word GRID classification."""

from __future__ import annotations

import os
from dataclasses import asdict

import numpy as np
import tensorflow as tf
import sys, os
sys.path.append(os.getcwd())
from src.dataset import (
    AUGMENTATION_PROFILES,
    _apply_spatial_augment,
    _apply_temporal_augment,
    load_split_ids,
)
from src.utils import FRAME_HEIGHT, FRAME_WIDTH

from experiments.isolated_word_level.temporal_slicer import (
    SegmentStats,
    WordSegment,
    alignment_to_segments,
    slice_and_pad_word_clip,
)


DEFAULT_TARGET_FRAMES = 25


def _resolve_sample_paths(preprocessed_dir: str, sample_id: str) -> tuple[str, str]:
    npy_path = os.path.join(preprocessed_dir, f"{sample_id}.npy")
    align_path = os.path.join(preprocessed_dir, "align", f"{sample_id}.align")
    return npy_path, align_path


def build_word_segments_from_ids(
    preprocessed_dir: str,
    sample_ids: list[str],
) -> tuple[list[WordSegment], SegmentStats]:
    """Build word-level segment metadata from split sample IDs."""
    stats = SegmentStats()
    segments: list[WordSegment] = []

    for sample_id in sample_ids:
        npy_path, align_path = _resolve_sample_paths(preprocessed_dir, sample_id)

        if not os.path.exists(npy_path):
            stats.missing_npy += 1
            continue
        if not os.path.exists(align_path):
            stats.missing_align += 1
            continue

        try:
            # mmap_mode avoids loading entire array into RAM during metadata pass.
            frames = np.load(npy_path, mmap_mode="r")
            total_frames = int(frames.shape[0])
        except Exception:
            stats.bad_align += 1
            continue

        sample_segments = alignment_to_segments(
            sample_id=sample_id,
            npy_path=npy_path,
            align_path=align_path,
            total_frames=total_frames,
            stats=stats,
        )
        segments.extend(sample_segments)

    return segments, stats


def segments_to_arrays(segments: list[WordSegment]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    npy_paths = [seg.npy_path for seg in segments]
    starts = np.array([seg.start_frame for seg in segments], dtype=np.int32)
    ends = np.array([seg.end_frame for seg in segments], dtype=np.int32)
    labels = np.array([seg.class_idx for seg in segments], dtype=np.int32)
    return npy_paths, starts, ends, labels


def _apply_center_focus_crop(frames: tf.Tensor, crop_ratio: float = 0.8) -> tf.Tensor:
    """Optional fallback crop if upstream preprocessing is not mouth-centered."""
    crop_h = max(8, int(round(FRAME_HEIGHT * float(crop_ratio))))
    crop_w = max(8, int(round(FRAME_WIDTH * float(crop_ratio))))
    start_y = max(0, (FRAME_HEIGHT - crop_h) // 2)
    start_x = max(0, (FRAME_WIDTH - crop_w) // 2)

    cropped = frames[:, start_y : start_y + crop_h, start_x : start_x + crop_w, :]
    resized = tf.image.resize(cropped, [FRAME_HEIGHT, FRAME_WIDTH], method="bilinear")
    return tf.cast(resized, tf.float32)


def create_isolated_word_dataset(
    npy_paths: list[str],
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    pad_mode: str = "repeat_last",
    shuffle: bool = True,
    seed: int | None = None,
    training: bool = False,
    augmentation_profile: str = "off",
    enforce_center_crop: bool = False,
) -> tf.data.Dataset:
    """Create tf.data dataset yielding (word_clip, {'label': class_idx})."""
    augmentation_profile = augmentation_profile.lower().strip()
    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unsupported augmentation_profile='{augmentation_profile}'. "
            f"Supported: {AUGMENTATION_PROFILES}"
        )

    if pad_mode not in {"repeat_last", "zeros"}:
        raise ValueError("pad_mode must be one of {'repeat_last', 'zeros'}")

    active_profile = augmentation_profile if training else "off"

    dataset = tf.data.Dataset.from_tensor_slices((npy_paths, start_frames, end_frames, labels))
    if shuffle:
        dataset = dataset.shuffle(len(npy_paths), seed=seed, reshuffle_each_iteration=True)

    def load_word_clip(path: tf.Tensor, start: tf.Tensor, end: tf.Tensor, label: tf.Tensor):
        def _read_and_slice(p: tf.Tensor, s: tf.Tensor, e: tf.Tensor) -> np.ndarray:
            frames = np.load(p.numpy().decode())
            if frames.ndim == 3:
                frames = frames[..., np.newaxis]
            if frames.dtype == np.uint8:
                frames = frames.astype(np.float32) / 255.0
            else:
                frames = frames.astype(np.float32)

            clip = slice_and_pad_word_clip(
                frames=frames,
                start_frame=int(s.numpy()),
                end_frame=int(e.numpy()),
                target_frames=int(target_frames),
                pad_mode=pad_mode,
            )
            return clip

        clip = tf.py_function(_read_and_slice, [path, start, end], Tout=tf.float32)
        clip.set_shape((target_frames, FRAME_HEIGHT, FRAME_WIDTH, 1))

        if enforce_center_crop:
            clip = _apply_center_focus_crop(clip)
            clip.set_shape((target_frames, FRAME_HEIGHT, FRAME_WIDTH, 1))

        if active_profile in {"spatial", "spatiotemporal", "strong"}:
            clip = _apply_spatial_augment(clip)
        if active_profile in {"spatiotemporal", "strong"}:
            clip = _apply_temporal_augment(clip)

        clip.set_shape((target_frames, FRAME_HEIGHT, FRAME_WIDTH, 1))
        label = tf.cast(label, tf.int32)
        label.set_shape(())
        return clip, {"label": label}

    dataset = dataset.map(load_word_clip, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(5)
    return dataset


def create_isolated_word_dataset_pipeline(
    preprocessed_dir: str,
    split_dir: str,
    batch_size: int,
    seed: int | None = None,
    train_split: str = "train",
    val_split: str = "val_oos",
    target_frames: int = DEFAULT_TARGET_FRAMES,
    pad_mode: str = "repeat_last",
    train_augmentation_profile: str = "off",
    enforce_center_crop: bool = False,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    list[WordSegment],
    SegmentStats,
    list[WordSegment],
    SegmentStats,
]:
    train_ids = load_split_ids(split_dir=split_dir, split_name=train_split)
    val_ids = load_split_ids(split_dir=split_dir, split_name=val_split)

    train_segments, train_stats = build_word_segments_from_ids(
        preprocessed_dir=preprocessed_dir,
        sample_ids=train_ids,
    )
    val_segments, val_stats = build_word_segments_from_ids(
        preprocessed_dir=preprocessed_dir,
        sample_ids=val_ids,
    )

    if not train_segments:
        raise ValueError("No train word segments were built. Check preprocessed_dir/splits/align files.")
    if not val_segments:
        raise ValueError("No val word segments were built. Check preprocessed_dir/splits/align files.")

    train_paths, train_starts, train_ends, train_labels = segments_to_arrays(train_segments)
    val_paths, val_starts, val_ends, val_labels = segments_to_arrays(val_segments)

    print(
        "[Dataset:isolated-word] "
        f"train_segments={len(train_segments)} val_segments={len(val_segments)}"
    )
    print(f"[Dataset:isolated-word] train_stats={asdict(train_stats)}")
    print(f"[Dataset:isolated-word] val_stats={asdict(val_stats)}")

    train_ds = create_isolated_word_dataset(
        train_paths,
        train_starts,
        train_ends,
        train_labels,
        batch_size=batch_size,
        target_frames=target_frames,
        pad_mode=pad_mode,
        shuffle=True,
        seed=seed,
        training=True,
        augmentation_profile=train_augmentation_profile,
        enforce_center_crop=enforce_center_crop,
    )
    val_ds = create_isolated_word_dataset(
        val_paths,
        val_starts,
        val_ends,
        val_labels,
        batch_size=batch_size,
        target_frames=target_frames,
        pad_mode=pad_mode,
        shuffle=False,
        seed=None,
        training=False,
        augmentation_profile="off",
        enforce_center_crop=enforce_center_crop,
    )

    return train_ds, val_ds, train_segments, train_stats, val_segments, val_stats
