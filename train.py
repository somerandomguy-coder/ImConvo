"""
Training script for CTC-based lip reading model.

Prerequisites:
    python scripts/preprocess.py      # run once to convert .mpg → .npy
    python train.py                   # train the model
"""
import os
import sys

if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("\n[!] WARNING: You are not running in a virtual environment.")
    print("Please run 'source .venv/bin/activate' or use './setup_and_run.sh'\n")
    sys.exit(1)

import json
import math

import numpy as np
import tensorflow as tf
from clearml import Task
from src import NUM_CHARS, LipReadingCTC, char_indices_to_text, count_parameters
from src.dataset import (
    build_split_arrays,
    create_ctc_dataset,
    create_dataset_pipeline,
    load_split_ids,
)

# ---------------------------------------------------------------------------
# ClearML (minimal lifecycle only)
# ---------------------------------------------------------------------------
task = Task.init(
    project_name="ImConvo",
    task_name="LipReadingCTC_Training",
    task_type=Task.TaskTypes.training,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "data_dir": "./data/",
    "preprocessed_dir": "./data/preprocessed/",
    "split_dir": "./splits/grid_v1",
    "batch_size": 48,
    "num_epochs": 1,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "patience": 7,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between two strings."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Levenshtein distance at word level
    r = len(ref_words)
    h = len(hyp_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,     # deletion
                    d[i][j - 1] + 1,     # insertion
                    d[i - 1][j - 1] + 1  # substitution
                )

    return d[r][h] / max(r, 1)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between two strings."""
    r = len(reference)
    h = len(hypothesis)
    d = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1
                )

    return d[r][h] / max(r, 1)


def evaluate_split(
    model: LipReadingCTC,
    dataset: tf.data.Dataset,
    steps: int,
    split_name: str,
    preview_limit: int = 10,
) -> tuple[float, float, int]:
    """Decode and evaluate a dataset split, returning (wer, cer, num_samples)."""
    print(f"\n{'='*60}")
    print(f"CTC Decoding — {split_name}")
    print(f"{'='*60}\n")

    total_wer = 0.0
    total_cer = 0.0
    num_samples = 0

    for batch in dataset.take(steps):
        x, y = batch
        logits = model(x, training=False)
        decoded_batch = model.decode_greedy(logits)

        labels = y["labels"].numpy()
        lengths = y["label_length"].numpy()

        for i in range(len(labels)):
            gt_indices = labels[i][:lengths[i]]
            gt_text = char_indices_to_text(gt_indices.tolist())

            pred_indices = decoded_batch[i]
            pred_indices = pred_indices[pred_indices >= 0]
            pred_text = char_indices_to_text(pred_indices.tolist())

            wer = compute_wer(gt_text, pred_text)
            cer = compute_cer(gt_text, pred_text)
            total_wer += wer
            total_cer += cer
            num_samples += 1

            if num_samples <= preview_limit:
                print(f"  GT:   '{gt_text}'")
                print(f"  Pred: '{pred_text}'")
                print(f"  WER: {wer:.2f} | CER: {cer:.2f}")
                print()

    avg_wer = total_wer / max(num_samples, 1)
    avg_cer = total_cer / max(num_samples, 1)
    print(f"{'='*60}")
    print(f"{split_name} Average WER: {avg_wer:.4f} ({avg_wer*100:.1f}%)")
    print(f"{split_name} Average CER: {avg_cer:.4f} ({avg_cer*100:.1f}%)")
    print(f"{'='*60}")

    return avg_wer, avg_cer, num_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    tf.random.set_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

    # ---- Check preprocessing ----
    if not os.path.isdir(CONFIG["preprocessed_dir"]):
        print(
            f"\n[ERROR] Preprocessed data not found at: {CONFIG['preprocessed_dir']}\n"
            f"Run this first:  python scripts/preprocess.py\n"
        )
        task.close()
        return
    if not os.path.isdir(CONFIG["split_dir"]):
        print(
            f"\n[ERROR] Split manifests not found at: {CONFIG['split_dir']}\n"
            f"Run this first: python scripts/build_split_manifests.py\n"
        )
        task.close()
        return

    # ---- Create tf.data pipelines ----
    (
        train_ds,
        val_ds,
        train_paths,
        train_labels,
        train_label_lengths,
        val_paths,
        val_labels,
        val_label_lengths,
    ) = create_dataset_pipeline(
        preprocessed_dir=CONFIG["preprocessed_dir"],
        split_dir=CONFIG["split_dir"],
        batch_size=CONFIG["batch_size"],
        train_split="train",
        val_split="val_oos",
    )

    # Compute steps from known hard-split sizes
    num_train_samples = len(train_paths)
    num_val_samples = len(val_paths)
    steps_per_epoch = math.ceil(num_train_samples / CONFIG["batch_size"])
    validation_steps = math.ceil(num_val_samples / CONFIG["batch_size"])

    # ---- Build model ----
    model = LipReadingCTC(num_chars=NUM_CHARS)

    # Build model by running a forward pass
    for batch in train_ds.take(1):
        x, _ = batch
        _ = model(x)
        break

    print(f"\nModel parameters: {count_parameters(model):,}")

    # ---- Compile (loss handled in train_step) ----
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    model.compile(optimizer=optimizer)

    # ---- Callbacks ----
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=CONFIG["patience"],
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_ctc_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
    ]
    # ---- Train ----
    print(f"\n{'='*60}")
    print("Starting CTC Training")
    print(f"{'='*60}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["num_epochs"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Decode & Evaluate ----
    val_oos_wer, val_oos_cer, _ = evaluate_split(
        model=model,
        dataset=val_ds,
        steps=validation_steps,
        split_name="val_oos",
    )

    val_is_ids = load_split_ids(split_dir=CONFIG["split_dir"], split_name="val_is")
    val_is_paths, val_is_labels, val_is_lengths = build_split_arrays(
        preprocessed_dir=CONFIG["preprocessed_dir"],
        sample_ids=val_is_ids,
    )
    val_is_ds = create_ctc_dataset(
        val_is_paths,
        val_is_labels,
        val_is_lengths,
        CONFIG["batch_size"],
        shuffle=False,
    )
    val_is_steps = math.ceil(len(val_is_paths) / CONFIG["batch_size"])
    val_is_wer, val_is_cer, _ = evaluate_split(
        model=model,
        dataset=val_is_ds,
        steps=val_is_steps,
        split_name="val_is",
    )

    # ---- Save history ----
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_data["eval"] = {
        "val_oos": {"wer": float(val_oos_wer), "cer": float(val_oos_cer)},
        "val_is": {"wer": float(val_is_wer), "cer": float(val_is_cer)},
    }
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    task.close()

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
