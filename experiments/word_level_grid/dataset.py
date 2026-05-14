"""Dataset pipeline for GRID word-level classification experiments."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.dataset import (
    AUGMENTATION_PROFILES,
    _apply_spatial_augment,
    _apply_temporal_augment,
    load_split_ids,
    resolve_sample_ids,
)
from src.utils import FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAMES

from experiments.word_level_grid.common import NUM_SLOTS, encode_alignment_words


def build_split_word_arrays(
    preprocessed_dir: str,
    sample_ids: list[str],
) -> tuple[list[str], np.ndarray]:
    """Resolve sample IDs into frame paths + per-slot word labels."""
    samples = resolve_sample_ids(preprocessed_dir=preprocessed_dir, sample_ids=sample_ids)

    npy_paths: list[str] = []
    labels: list[np.ndarray] = []

    for npy_path, align_path in samples:
        npy_paths.append(npy_path)
        labels.append(encode_alignment_words(align_path))

    return npy_paths, np.array(labels, dtype=np.int32)


def create_word_dataset(
    npy_paths: list[str],
    word_labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
    training: bool = False,
    augmentation_profile: str = "off",
) -> tf.data.Dataset:
    """Create tf.data pipeline yielding (frames, {'word_labels': [B, 6]})."""
    augmentation_profile = augmentation_profile.lower().strip()
    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unsupported augmentation_profile='{augmentation_profile}'. "
            f"Supported: {AUGMENTATION_PROFILES}"
        )

    active_profile = augmentation_profile if training else "off"

    dataset = tf.data.Dataset.from_tensor_slices((npy_paths, word_labels))

    if shuffle:
        dataset = dataset.shuffle(
            len(npy_paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_npy_file(path: tf.Tensor, label: tf.Tensor):
        def _read_file(p: tf.Tensor) -> np.ndarray:
            frames = np.load(p.numpy().decode())
            if frames.dtype == np.uint8:
                frames = frames[..., np.newaxis].astype(np.float32) / 255.0
            else:
                frames = frames[..., np.newaxis].astype(np.float32)
            return frames

        frames = tf.py_function(_read_file, [path], tf.float32)
        frames.set_shape((MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1))

        if active_profile in {"spatial", "spatiotemporal", "strong"}:
            frames = _apply_spatial_augment(frames)
        if active_profile in {"spatiotemporal", "strong"}:
            frames = _apply_temporal_augment(frames)

        frames.set_shape((MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1))
        label = tf.cast(label, tf.int32)
        label.set_shape((NUM_SLOTS,))
        return frames, {"word_labels": label}

    dataset = dataset.map(load_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(5)
    return dataset


def create_word_dataset_pipeline(
    preprocessed_dir: str,
    split_dir: str,
    batch_size: int,
    seed: int | None = None,
    train_split: str = "train",
    val_split: str = "val_oos",
    train_augmentation_profile: str = "off",
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    list[str],
    np.ndarray,
    list[str],
    np.ndarray,
]:
    """Build train/val pipelines for word-level classification."""
    train_ids = load_split_ids(split_dir=split_dir, split_name=train_split)
    val_ids = load_split_ids(split_dir=split_dir, split_name=val_split)

    train_paths, train_labels = build_split_word_arrays(
        preprocessed_dir=preprocessed_dir,
        sample_ids=train_ids,
    )
    val_paths, val_labels = build_split_word_arrays(
        preprocessed_dir=preprocessed_dir,
        sample_ids=val_ids,
    )

    print(f"[Dataset:word] Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_ds = create_word_dataset(
        train_paths,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        training=True,
        augmentation_profile=train_augmentation_profile,
    )
    val_ds = create_word_dataset(
        val_paths,
        val_labels,
        batch_size=batch_size,
        shuffle=False,
        training=False,
        augmentation_profile="off",
    )

    return train_ds, val_ds, train_paths, train_labels, val_paths, val_labels
