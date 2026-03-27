"""
Training script for the lip reading CNN model.

Prerequisites:
    python scripts/preprocess.py      # run once to convert .mpg → .npy
    python train.py                   # train the model
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from src import (
    build_vocab,
    create_dataset_pipeline,
    LipReadingCNN,
    count_parameters,
    MAX_LABEL_LEN,
    PAD_IDX,
    SLOT_NAMES,
)

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
        task_name="LipReadingCNN_TF_Training",
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
    "batch_size": 8,
    "num_epochs": 1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "patience": 7,
    "seed": 42,
}

if USE_CLEARML and task:
    task.connect(CONFIG)

# ---------------------------------------------------------------------------
# Custom masked loss & metric
# ---------------------------------------------------------------------------


class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """Cross entropy that ignores samples where the label == PAD_IDX."""

    def __init__(self, pad_idx=-1, **kwargs):
        super().__init__(**kwargs)
        self.pad_idx = pad_idx

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, self.pad_idx)
        mask_f = tf.cast(mask, tf.float32)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        y_true_safe = tf.where(mask, y_true, tf.zeros_like(y_true))

        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_safe, y_pred)
        loss = loss * mask_f

        return tf.reduce_sum(loss) / (tf.reduce_sum(mask_f) + 1e-8)


class MaskedAccuracy(tf.keras.metrics.Metric):
    """Accuracy metric that ignores PAD_IDX labels."""

    def __init__(self, pad_idx=-1, name="masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_idx = pad_idx
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, self.pad_idx)
        y_true = tf.cast(y_true, tf.int64)
        y_pred_classes = tf.cast(tf.argmax(y_pred, axis=-1), tf.int64)

        correct = tf.equal(y_true, y_pred_classes)
        correct = tf.logical_and(correct, mask)

        self.correct.assign_add(tf.reduce_sum(tf.cast(correct, tf.float32)))
        self.total.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))

    def result(self):
        return self.correct / (self.total + 1e-8)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


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
        return

    # ---- Build vocabulary ----
    align_dir = os.path.join(CONFIG["data_dir"], "align")
    word2idx = build_vocab(align_dir)
    idx2word = {v: k for k, v in word2idx.items()}
    num_classes = len(word2idx)
    print(f"Vocabulary size: {num_classes} words")

    # ---- Create tf.data pipelines ----
    train_ds, val_ds, val_labels, _ = create_dataset_pipeline(
        data_dir=CONFIG["data_dir"],
        preprocessed_dir=CONFIG["preprocessed_dir"],
        word2idx=word2idx,
        batch_size=CONFIG["batch_size"],
        val_split=CONFIG["val_split"],
        seed=CONFIG["seed"],
    )

    # ---- Build model ----
    model = LipReadingCNN(num_classes=num_classes, num_slots=MAX_LABEL_LEN)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # ---- Compile ----
    losses = {f"slot_{i}": MaskedSparseCategoricalCrossentropy() for i in range(MAX_LABEL_LEN)}
    metrics = {f"slot_{i}": [MaskedAccuracy()] for i in range(MAX_LABEL_LEN)}

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

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
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
    ]
    if USE_CLEARML and task:
        callbacks.append(ClearMLCallback(task))

    # ---- Train ----
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["num_epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Final Evaluation ----
    print(f"\n{'='*60}")
    print("Final Evaluation on Validation Set")
    print(f"{'='*60}")

    results = model.evaluate(val_ds, verbose=1, return_dict=True)
    print("\nMetrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # ---- Per-slot classification report ----
    predictions = model.predict(val_ds, verbose=1)

    for slot, name in enumerate(SLOT_NAMES):
        slot_key = f"slot_{slot}"
        if slot_key in predictions:
            slot_preds = np.argmax(predictions[slot_key], axis=-1)
            slot_labels = val_labels[:, slot]
            mask = slot_labels != PAD_IDX

            if mask.any():
                unique_labels = sorted(set(slot_labels[mask].tolist()))
                target_names = [idx2word.get(i, f"unk_{i}") for i in unique_labels]
                print(f"\n--- {name.upper()} slot ---")
                print(classification_report(
                    slot_labels[mask], slot_preds[mask],
                    labels=unique_labels, target_names=target_names,
                    zero_division=0,
                ))

    # ---- Save history ----
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
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
