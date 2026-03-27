"""
Dataset module for the GRID lip reading corpus.

Provides lazy tf.data pipelines that load preprocessed .npy files
one at a time for memory-efficient training.
"""

import os

import numpy as np
import tensorflow as tf

from src.utils import (
    MAX_FRAMES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_LABEL_LEN,
    PAD_IDX,
    parse_alignment,
    pad_label,
)

# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------


def discover_samples(data_dir: str, preprocessed_dir: str):
    """
    Discover valid samples by reading the manifest or scanning directories.

    Returns:
        list of (npy_path, align_path) tuples
    """
    align_dir = os.path.join(data_dir, "align")
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
# tf.data pipeline
# ---------------------------------------------------------------------------


def create_dataset_from_samples(
    npy_paths: list,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
):
    """
    Create a tf.data.Dataset that lazily loads preprocessed .npy files.
    """

    def generator():
        indices = np.arange(len(npy_paths))
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            frames = np.load(npy_paths[idx])         # (T, H, W)
            frames = frames[..., np.newaxis]          # (T, H, W, 1)
            frames = frames.astype(np.float32)
            label = labels[idx]
            label_dict = {f"slot_{i}": label[i] for i in range(MAX_LABEL_LEN)}
            yield frames, label_dict

    output_sig = (
        tf.TensorSpec(shape=(MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1), dtype=tf.float32),
        {
            f"slot_{i}": tf.TensorSpec(shape=(), dtype=tf.int32)
            for i in range(MAX_LABEL_LEN)
        },
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


def create_dataset_pipeline(
    data_dir: str,
    preprocessed_dir: str,
    word2idx: dict,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Build train/val tf.data pipelines from preprocessed data.

    Returns:
        train_ds, val_ds, val_labels, val_paths
    """
    samples = discover_samples(data_dir, preprocessed_dir)
    print(f"[Dataset] Discovered {len(samples)} preprocessed samples.")

    all_npy_paths = []
    all_labels = []
    for npy_path, align_path in samples:
        label_indices = parse_alignment(align_path, word2idx)
        label = pad_label(label_indices)
        all_npy_paths.append(npy_path)
        all_labels.append(label)

    all_labels = np.array(all_labels, dtype=np.int32)

    # Train / val split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_npy_paths))
    rng.shuffle(indices)

    split = int(len(indices) * (1 - val_split))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_paths = [all_npy_paths[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_paths = [all_npy_paths[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    print(f"[Dataset] Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_ds = create_dataset_from_samples(
        train_paths, train_labels, batch_size, shuffle=True
    )
    val_ds = create_dataset_from_samples(
        val_paths, val_labels, batch_size, shuffle=False
    )

    return train_ds, val_ds, val_labels, val_paths
