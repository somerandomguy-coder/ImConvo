"""
Dataset module for CTC-based lip reading on the GRID corpus.

Loads preprocessed .npy video files and character-level labels for CTC training.
Supports deterministic split manifests to prevent data leakage.
"""

import os

import numpy as np
import tensorflow as tf
from src.utils import FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAMES, parse_alignment_chars

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

    Returns:
        A tf.data.Dataset yielding (frames, label_dict) batches.
    """
    # 1. Create a dataset of indices or paths
    dataset = tf.data.Dataset.from_tensor_slices(
        (npy_paths, char_labels, label_lengths)
    )

    if shuffle:
        dataset = dataset.shuffle(len(npy_paths))

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
    train_split: str = "train",
    val_split: str = "val_oos",
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
        train_paths, train_labels, train_lengths, batch_size, shuffle=True
    )
    val_ds = create_ctc_dataset(
        val_paths, val_labels, val_lengths, batch_size, shuffle=False
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
