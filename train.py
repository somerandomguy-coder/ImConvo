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

from src import (BLANK_IDX, NUM_CHARS, LipReadingCTC, char_indices_to_text,
                 count_parameters, create_dataset_pipeline)

# ---------------------------------------------------------------------------
# ClearML (optional)
# ---------------------------------------------------------------------------

os.environ["CLEARML_CONFIG_FILE"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "clearml.conf"
)

USE_CLEARML = False
task = None
try:
    from clearml import Task

    task = Task.init(
        project_name="ImConvo",
        task_name="LipReadingCTC_Training_full_image_no_crop",
        task_type=Task.TaskTypes.training,
    )
    USE_CLEARML = True
    print("[ClearML] Connected successfully.")
except BaseException as e:
    print(
        f"[ClearML] Not available ({type(e).__name__}: {e}). "
        "Training locally without tracking."
    )
    task = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "data_dir": "./data/",
    "preprocessed_dir": "./data/preprocessed/",
    "batch_size": 48,
    "num_epochs": 100,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "patience": 7,
    "seed": 42,
}

if USE_CLEARML and task:
    task.connect(CONFIG)
    print(f"[ClearML] Training Node active. Using data from: {CONFIG['preprocessed_dir']}")

# ---------------------------------------------------------------------------
# ClearML callback
# ---------------------------------------------------------------------------


class ClearMLCallback(tf.keras.callbacks.Callback):
    def __init__(self, clearml_task):
        super().__init__()
        self.task = clearml_task

    def on_epoch_end(self, epoch, logs=None):
        if not self.task or not logs:
            return
        logger = self.task.get_logger()
        for key, value in logs.items():
            series = "val" if key.startswith("val_") else "train"
            title = key[4:] if key.startswith("val_") else key
            logger.report_scalar(title, series, float(value), iteration=epoch + 1)


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    tf.random.set_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    
    if USE_CLEARML and task:
        # Get the pipeline details (if running as part of one)
        parent_tasks = task.get_parent_tasks()
        
        if parent_tasks:
            # We assume the first parent is the Preprocessing task
            preprocess_task = parent_tasks[0]
            print(f"[ClearML] Pipeline detected! Parent Task: {preprocess_task.id}")
            
            # Fetch the manifest artifact from the Preprocessing task
            manifest_artifact = preprocess_task.artifacts.get('manifest')
            
            if manifest_artifact:
                # This doesn't download the 100GB; it just gets the small .txt file
                manifest_local_path = manifest_artifact.get_local_copy()
                print(f"✓ Manifest verified: {manifest_local_path}")
                
                # OPTIONAL: You can parse the manifest here to get the exact count
                with open(manifest_local_path, 'r') as f:
                    num_samples = len(f.readlines())
                print(f"✓ Found {num_samples} valid preprocessed samples.")
            else:
                print("[WARN] No manifest artifact found in parent task. Using local fallback.")

    # ---- Check preprocessing ----
    if not os.path.isdir(CONFIG["preprocessed_dir"]):
        print(
            f"\n[ERROR] Preprocessed data not found at: {CONFIG['preprocessed_dir']}\n"
            f"Run this first:  python scripts/preprocess.py\n"
        )
        return

    # ---- Create tf.data pipelines ----
    train_ds, val_ds, val_paths, val_labels, val_label_lengths = create_dataset_pipeline(
        preprocessed_dir=CONFIG["preprocessed_dir"],
        batch_size=CONFIG["batch_size"],
        val_split=CONFIG["val_split"],
        seed=CONFIG["seed"],
    )

    # Compute steps from known dataset sizes
    num_train = len(val_paths)  # val_paths length == num_val
    # Recompute from split ratio
    total_samples = int(len(val_paths) / CONFIG["val_split"])  # approximate
    num_train_samples = total_samples - len(val_paths)
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
    if USE_CLEARML and task:
        callbacks.append(ClearMLCallback(task))

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
    print(f"\n{'='*60}")
    print("CTC Decoding — Validation Samples")
    print(f"{'='*60}\n")

    total_wer = 0.0
    total_cer = 0.0
    num_samples = 0

    for batch in val_ds.take(validation_steps):
        x, y = batch
        logits = model(x, training=False)
        decoded_batch = model.decode_greedy(logits)

        labels = y["labels"].numpy()
        lengths = y["label_length"].numpy()

        for i in range(len(labels)):
            # Ground truth
            gt_indices = labels[i][:lengths[i]]
            gt_text = char_indices_to_text(gt_indices.tolist())

            # Prediction
            pred_indices = decoded_batch[i]
            pred_indices = pred_indices[pred_indices >= 0]  # remove padding
            pred_text = char_indices_to_text(pred_indices.tolist())

            wer = compute_wer(gt_text, pred_text)
            cer = compute_cer(gt_text, pred_text)

            total_wer += wer
            total_cer += cer
            num_samples += 1

            # Print first 10 examples
            if num_samples <= 10:
                print(f"  GT:   '{gt_text}'")
                print(f"  Pred: '{pred_text}'")
                print(f"  WER: {wer:.2f} | CER: {cer:.2f}")
                print()

    avg_wer = total_wer / max(num_samples, 1)
    avg_cer = total_cer / max(num_samples, 1)

    print(f"{'='*60}")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.1f}%)")
    print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.1f}%)")
    print(f"{'='*60}")

    if USE_CLEARML and task:
        logger = task.get_logger()
        logger.report_scalar("WER", "val", avg_wer, iteration=CONFIG["num_epochs"])
        logger.report_scalar("CER", "val", avg_cer, iteration=CONFIG["num_epochs"])

    # ---- Save history ----
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_data["avg_wer"] = avg_wer
    history_data["avg_cer"] = avg_cer
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    if USE_CLEARML and task:
        try:
            task.close()
        except BaseException:
            pass

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
