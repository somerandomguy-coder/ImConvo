"""
Dataset module for CTC-based lip reading on the GRID corpus.

Lazily loads preprocessed .npy video files and provides
character-level labels for CTC training.
"""

import os

import numpy as np
import tensorflow as tf

from src.utils import (
    MAX_FRAMES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_CHAR_LEN,
    BLANK_IDX,
    SILENCE_TOKENS,
    parse_alignment_chars,
)

# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------


def discover_samples(data_dir: str, preprocessed_dir: str):
    """
    Discover valid samples from manifest or directory scan.

    Alignment files are expected in data_dir/align/ with matching names
    to the .npy files in preprocessed_dir (may include speaker prefix).

    Returns:
        list of (npy_path, align_path) tuples
    """
    align_dir = os.path.join(preprocessed_dir, "align")
    manifest_path = os.path.join(preprocessed_dir, "manifest.txt")

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(preprocessed_dir)
            if f.endswith(".npy")
        )

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
def create_ctc_dataset(npy_paths: list,
    char_labels: np.ndarray,
    label_lengths: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
):
    """
    Create a tf.data.Dataset for CTC training.

    Yields:
        (video_frames, {"labels": char_indices, "label_length": length})
    """

    # 1. Create a dataset of indices or paths
    dataset = tf.data.Dataset.from_tensor_slices((npy_paths, char_labels, label_lengths))

    if shuffle:
        dataset = dataset.shuffle(len(npy_paths))

    # 2. Define a loading function
    def load_npy_file(path, label, length):
        # We use tf.py_function because np.load isn't native TensorFlow
        def _read_file(p):
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
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


def create_dataset_pipeline(
    data_dir: str,
    preprocessed_dir: str,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Build train/val tf.data pipelines for CTC training.

    Returns:
        train_ds, val_ds, val_paths, val_labels, val_label_lengths
    """
    samples = discover_samples(data_dir, preprocessed_dir)
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
