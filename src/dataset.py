"""
Dataset module for CTC-based lip reading on the GRID corpus.

Loads preprocessed .npy video files and character-level labels for CTC training.
Supports deterministic split manifests to prevent data leakage.
"""

import os

import numpy as np
import tensorflow as tf
from src.utils import FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAMES, parse_alignment_chars

AUGMENTATION_PROFILES = ("off", "spatial", "spatiotemporal", "strong")


def _apply_gaussian_blur(frames: tf.Tensor) -> tf.Tensor:
    """Apply a tiny 3x3 Gaussian blur independently to each frame."""
    kernel = tf.constant(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=tf.float32,
    )
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    return tf.nn.depthwise_conv2d(
        frames,
        filter=kernel,
        strides=[1, 1, 1, 1],
        padding="SAME",
    )


def _apply_spatial_augment(frames: tf.Tensor) -> tf.Tensor:
    """Conservative spatial + photometric perturbations."""
    scale = tf.random.uniform([], minval=0.95, maxval=1.05, dtype=tf.float32)
    scaled_h = tf.cast(tf.round(scale * FRAME_HEIGHT), tf.int32)
    scaled_w = tf.cast(tf.round(scale * FRAME_WIDTH), tf.int32)
    frames = tf.image.resize(frames, [scaled_h, scaled_w], method="bilinear")
    frames = tf.image.resize_with_crop_or_pad(frames, FRAME_HEIGHT, FRAME_WIDTH)

    max_shift = 5
    padded = tf.pad(
        frames,
        [[0, 0], [max_shift, max_shift], [max_shift, max_shift], [0, 0]],
        mode="REFLECT",
    )
    offset_y = tf.random.uniform([], minval=0, maxval=(2 * max_shift) + 1, dtype=tf.int32)
    offset_x = tf.random.uniform([], minval=0, maxval=(2 * max_shift) + 1, dtype=tf.int32)
    frames = padded[:, offset_y:offset_y + FRAME_HEIGHT, offset_x:offset_x + FRAME_WIDTH, :]

    frames = tf.image.random_brightness(frames, max_delta=0.06)
    frames = tf.image.random_contrast(frames, lower=0.9, upper=1.1)
    gamma = tf.random.uniform([], minval=0.95, maxval=1.05, dtype=tf.float32)
    frames = tf.image.adjust_gamma(tf.maximum(frames, 1e-6), gamma=gamma)

    noise = tf.random.normal(tf.shape(frames), mean=0.0, stddev=0.01, dtype=tf.float32)
    frames = frames + noise

    frames = tf.cond(
        tf.random.uniform([], 0.0, 1.0) < 0.1,
        lambda: _apply_gaussian_blur(frames),
        lambda: frames,
    )
    return tf.clip_by_value(frames, 0.0, 1.0)


def _apply_temporal_shift(frames: tf.Tensor) -> tf.Tensor:
    """Length-preserving temporal shift with edge padding."""
    shift = tf.random.uniform([], minval=-3, maxval=4, dtype=tf.int32)
    shifted = tf.roll(frames, shift=shift, axis=0)

    t = tf.range(MAX_FRAMES)
    front_mask = tf.logical_and(shift > 0, t < shift)
    back_mask = tf.logical_and(shift < 0, t >= (MAX_FRAMES + shift))
    first_frame = tf.repeat(frames[:1], repeats=MAX_FRAMES, axis=0)
    last_frame = tf.repeat(frames[-1:], repeats=MAX_FRAMES, axis=0)
    shifted = tf.where(front_mask[:, tf.newaxis, tf.newaxis, tf.newaxis], first_frame, shifted)
    shifted = tf.where(back_mask[:, tf.newaxis, tf.newaxis, tf.newaxis], last_frame, shifted)
    return shifted


def _apply_temporal_dropout_repeat(frames: tf.Tensor) -> tf.Tensor:
    """Drop frames by replacing with previous frame, preserving length."""
    drop_prob = tf.random.uniform([], minval=0.02, maxval=0.05, dtype=tf.float32)
    drop_mask = tf.random.uniform([MAX_FRAMES], 0.0, 1.0) < drop_prob
    previous = tf.concat([frames[:1], frames[:-1]], axis=0)
    return tf.where(drop_mask[:, tf.newaxis, tf.newaxis, tf.newaxis], previous, frames)


def _apply_temporal_time_mask(frames: tf.Tensor) -> tf.Tensor:
    """Mask 2-6 consecutive frames to improve temporal robustness."""
    mask_len = tf.random.uniform([], minval=2, maxval=7, dtype=tf.int32)
    max_start = MAX_FRAMES - mask_len + 1
    start = tf.random.uniform([], minval=0, maxval=max_start, dtype=tf.int32)
    t = tf.range(MAX_FRAMES)
    mask = tf.logical_and(t >= start, t < (start + mask_len))
    return tf.where(
        mask[:, tf.newaxis, tf.newaxis, tf.newaxis],
        tf.zeros_like(frames),
        frames,
    )


def _apply_temporal_augment(frames: tf.Tensor) -> tf.Tensor:
    """Apply conservative temporal augmentations while preserving T=MAX_FRAMES."""
    frames = _apply_temporal_shift(frames)
    frames = _apply_temporal_dropout_repeat(frames)
    frames = _apply_temporal_time_mask(frames)
    return frames

# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------


def discover_samples(preprocessed_dir: str) -> list[tuple[str, str]]:
    """
    Discover valid samples from manifest or directory scan.

    Alignment files are expected in preprocessed_dir/align/ with matching names
    to the .npy files in preprocessed_dir (may include speaker prefix).

    Returns:
        list of (npy_path, align_path) tuples
    """
    align_dir = os.path.join(preprocessed_dir, "align")
    manifest_path = os.path.join(preprocessed_dir, "manifest.txt")

    with open(manifest_path) as f:
        names = [line.strip() for line in f if line.strip()]

    samples = []
    for name in names:
        npy_path = os.path.join(preprocessed_dir, f"{name}.npy")
        align_path = os.path.join(align_dir, f"{name}.align")
        if os.path.exists(npy_path) and os.path.exists(align_path):
            samples.append((npy_path, align_path))

    return samples


def load_split_ids(split_dir: str, split_name: str) -> list[str]:
    """
    Load sample IDs from a split manifest file.

    Args:
        split_dir: Directory containing split text files.
        split_name: Split key, e.g. "train", "val_oos".

    Returns:
        Ordered list of sample IDs.
    """
    split_path = os.path.join(split_dir, f"{split_name}.txt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, encoding="utf-8") as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    if not sample_ids:
        raise ValueError(f"Split file is empty: {split_path}")
    return sample_ids


def resolve_sample_ids(
    preprocessed_dir: str,
    sample_ids: list[str],
) -> list[tuple[str, str]]:
    """
    Resolve sample IDs into strict (npy_path, align_path) pairs.
    Raises if any referenced sample is missing.
    """
    align_dir = os.path.join(preprocessed_dir, "align")
    resolved: list[tuple[str, str]] = []
    missing: list[str] = []

    for sample_id in sample_ids:
        npy_path = os.path.join(preprocessed_dir, f"{sample_id}.npy")
        align_path = os.path.join(align_dir, f"{sample_id}.align")
        if not (os.path.exists(npy_path) and os.path.exists(align_path)):
            missing.append(sample_id)
            continue
        resolved.append((npy_path, align_path))

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} sample IDs from split are missing .npy/.align. "
            f"First missing: {preview}"
        )
    return resolved


def build_split_arrays(
    preprocessed_dir: str,
    sample_ids: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Build paths + label arrays for a specific split.
    """
    samples = resolve_sample_ids(preprocessed_dir=preprocessed_dir, sample_ids=sample_ids)

    npy_paths: list[str] = []
    labels: list[np.ndarray] = []
    lengths: list[int] = []
    for npy_path, align_path in samples:
        char_indices, length = parse_alignment_chars(align_path)
        npy_paths.append(npy_path)
        labels.append(char_indices)
        lengths.append(length)

    return npy_paths, np.array(labels, dtype=np.int32), np.array(lengths, dtype=np.int32)


# ---------------------------------------------------------------------------
# tf.data pipeline for CTC
# ---------------------------------------------------------------------------
def create_ctc_dataset(
    npy_paths: list[str],
    char_labels: np.ndarray,
    label_lengths: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
    training: bool = False,
    augmentation_profile: str = "off",
) -> tf.data.Dataset:
    """
    Creates a high-performance tf.data.Dataset for CTC training.

    This pipeline handles parallel loading of .npy files, shape restoration,
    batching, and prefetching to ensure maximum GPU utilization.

    Args:
        npy_paths: List of absolute paths to preprocessed .npy video files.
        char_labels: 2D array of shape (N, MAX_CHAR_LEN) containing padded label indices.
        label_lengths: 1D array of shape (N,) containing the true length of each sentence.
        batch_size: Number of samples per training batch.
        shuffle: Whether to shuffle the data at the start of each epoch.
        seed: Shuffle seed for deterministic reproducibility.
        training: Whether this dataset is used for training.
        augmentation_profile: Augmentation level: off|spatial|spatiotemporal|strong.

    Returns:
        A tf.data.Dataset yielding (frames, label_dict) batches.
    """
    augmentation_profile = augmentation_profile.lower().strip()
    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unsupported augmentation_profile='{augmentation_profile}'. "
            f"Supported: {AUGMENTATION_PROFILES}"
        )
    active_profile = augmentation_profile if training else "off"

    # 1. Create a dataset of indices or paths
    dataset = tf.data.Dataset.from_tensor_slices(
        (npy_paths, char_labels, label_lengths)
    )

    if shuffle:
        dataset = dataset.shuffle(
            len(npy_paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    # 2. Define a loading function
    def load_npy_file(
        path: tf.Tensor, label: tf.Tensor, length: tf.Tensor
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """
        TensorFlow mapper function to load video frames and format labels.

        Args:
            path: A scalar string Tensor containing the path to the .npy file.
            label: An integer Tensor containing the padded character indices.
            length: A scalar integer Tensor representing the actual (unpadded) label length.

        Returns:
            A tuple (frames, label_dict) where:
                - frames: A float32 Tensor of shape (MAX_FRAMES, H, W, 1).
                - label_dict: A dictionary containing 'labels' and 'label_length'
                  required for Keras CTC loss.
        """

        # We use tf.py_function because np.load isn't native TensorFlow
        def _read_file(p: tf.Tensor) -> np.ndarray:
            frames = np.load(p.numpy().decode())
            frames = frames[..., np.newaxis].astype(np.float32)
            return frames

        frames = tf.py_function(_read_file, [path], tf.float32)
        # Re-set shapes (tf.py_function loses them)
        frames.set_shape((MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1))

        if active_profile in {"spatial", "spatiotemporal", "strong"}:
            frames = _apply_spatial_augment(frames)
        if active_profile in {"spatiotemporal", "strong"}:
            frames = _apply_temporal_augment(frames)
        frames.set_shape((MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1))

        return frames, {"labels": label, "label_length": length}

    # 3. USE PARALLEL CALLS HERE!
    # This replaces the single-threaded generator loop
    dataset = dataset.map(load_npy_file, num_parallel_calls=tf.data.AUTOTUNE)

    # 4. Batch, Repeat, and Prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(5)
    return dataset


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


def create_dataset_pipeline(
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
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    """
    Build train/validation tf.data pipelines for CTC training using hard split manifests.

    Args:
        preprocessed_dir: Root directory containing .npy samples and align/ labels.
        split_dir: Directory with split manifest txt files.
        batch_size: Number of samples per batch.
        seed: Shuffle seed for training dataset.
        train_split: Split key for training IDs.
        val_split: Split key for validation IDs.
        train_augmentation_profile: Train-time augmentation profile.

    Returns:
        train_ds, val_ds,
        train_paths, train_labels, train_label_lengths,
        val_paths, val_labels, val_label_lengths
    """
    train_ids = load_split_ids(split_dir=split_dir, split_name=train_split)
    val_ids = load_split_ids(split_dir=split_dir, split_name=val_split)

    train_paths, train_labels, train_lengths = build_split_arrays(
        preprocessed_dir=preprocessed_dir,
        sample_ids=train_ids,
    )
    val_paths, val_labels, val_lengths = build_split_arrays(
        preprocessed_dir=preprocessed_dir,
        sample_ids=val_ids,
    )

    print(f"[Dataset] Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_ds = create_ctc_dataset(
        train_paths,
        train_labels,
        train_lengths,
        batch_size,
        shuffle=True,
        seed=seed,
        training=True,
        augmentation_profile=train_augmentation_profile,
    )
    val_ds = create_ctc_dataset(
        val_paths,
        val_labels,
        val_lengths,
        batch_size,
        shuffle=False,
        training=False,
        augmentation_profile="off",
    )

    return (
        train_ds,
        val_ds,
        train_paths,
        train_labels,
        train_lengths,
        val_paths,
        val_labels,
        val_lengths,
    )
