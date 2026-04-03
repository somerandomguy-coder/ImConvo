"""
Dataset module for CTC-based lip reading on the GRID corpus.

Lazily loads preprocessed .npy video files and provides
character-level labels for CTC training.
"""

import os
from collections.abc import Iterator

import numpy as np
import tensorflow as tf

from src.utils import (BLANK_IDX, FRAME_HEIGHT, FRAME_WIDTH, MAX_CHAR_LEN,
                       MAX_FRAMES, SILENCE_TOKENS, parse_alignment_chars)

# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------


def discover_samples(preprocessed_dir: str, manifest_path: str = "") -> list[tuple[str, str]]:
    """
    Discover valid samples from manifest or directory scan.

    Alignment files are expected in preprocessed_dir/align/ with matching names
    to the .npy files in preprocessed_dir (may include speaker prefix).

    Returns:
        list of (npy_path, align_path) tuples
    """
    align_dir = os.path.join(preprocessed_dir, "align")
    # Fallback file if can't find manifest.txt
    manifest_path = manifest_path or os.path.join(preprocessed_dir, "manifest.txt")

    with open(manifest_path) as f:
        names = [line.strip() for line in f if line.strip()]

    samples = []
    for name in names:
        npy_path = os.path.join(preprocessed_dir, f"{name}.npy")
        align_path = os.path.join(align_dir, f"{name}.align")
        if os.path.exists(npy_path) and os.path.exists(align_path):
            samples.append((npy_path, align_path))

    return samples


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
            try:
                frames = np.load(p.numpy().decode())
                frames = frames[..., np.newaxis].astype(np.float32)
                return frames
            except Exception as e:
                # If a file is corrupted, return a "Zero" tensor 
                # so the training doesn't crash.
                print(f"\n[ERROR] Failed to load {p.numpy().decode()}: {e}")
                return np.zeros((MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1), dtype=np.float32)

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
        preprocessed_dir: str, batch_size: int, val_split: float = 0.2, seed: int = 42, manifest_path: str = "")-> tuple[tf.data.Dataset, tf.data.Dataset, list[str], np.ndarray, np.ndarray]:
    """
    Build train/val tf.data pipelines for CTC training.

    Returns:
        train_ds, val_ds, val_paths, val_labels, val_label_lengths
    """
    samples = discover_samples(preprocessed_dir=preprocessed_dir, manifest_path=manifest_path)
    print(f"[Dataset] Discovered {len(samples)} preprocessed samples.")

    all_npy_paths = []
    all_labels = []
    all_lengths = []

    for npy_path, align_path in samples:
        char_indices, length = parse_alignment_chars(align_path)
        all_npy_paths.append(npy_path)
        all_labels.append(char_indices)
        all_lengths.append(length)

    all_labels = np.array(all_labels, dtype=np.int32)
    all_lengths = np.array(all_lengths, dtype=np.int32)

    # Train / val split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_npy_paths))
    rng.shuffle(indices)

    split = int(len(indices) * (1 - val_split))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_paths = [all_npy_paths[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    train_lengths = all_lengths[train_idx]

    val_paths = [all_npy_paths[i] for i in val_idx]
    val_labels = all_labels[val_idx]
    val_lengths = all_lengths[val_idx]

    print(f"[Dataset] Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_ds = create_ctc_dataset(
        train_paths, train_labels, train_lengths, batch_size, shuffle=True
    )
    val_ds = create_ctc_dataset(
        val_paths, val_labels, val_lengths, batch_size, shuffle=False
    )

    return train_ds, val_ds, val_paths, val_labels, val_lengths
