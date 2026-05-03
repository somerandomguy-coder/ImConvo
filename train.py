"""
Training script for CTC-based lip reading model.

Prerequisites:
    python scripts/preprocess.py      # run once to convert .mpg → .npy
    python train.py                   # train the model
"""
import os
import sys
import argparse
import random

if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("\n[!] WARNING: You are not running in a virtual environment.")
    print("Please run 'source .venv/bin/activate' or use './setup_and_run.sh'\n")
    sys.exit(1)

import json
import math
import traceback
from datetime import datetime

import numpy as np
import tensorflow as tf
from clearml import Task
from src import (
    MODEL_VARIANTS,
    NUM_CHARS,
    LipReadingCTC,
    build_lipreading_ctc,
    char_indices_to_text,
    count_parameters,
)
from src.dataset import (
    AUGMENTATION_PROFILES,
    build_split_arrays,
    create_ctc_dataset,
    create_dataset_pipeline,
    load_split_ids,
)

task = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "data_dir": "./data/",
    "preprocessed_dir": "./data/preprocessed/",
    "split_dir": "./splits/grid_v1",
    "batch_size": 48,
    "num_epochs": 100,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "patience": 9,
    "seed": 42,
    "model_config": {
        "backbone_dropout": 0.3,
        "head_dropout": 0.3,
    },
    "variant_model_config": {
        "conformer_lite": {
            "backbone_dropout": 0.2,
            "head_dropout": 0.2,
        },
        "transformer_medium": {
            "backbone_dropout": 0.2,
            "head_dropout": 0.2,
        },
    },
    "optimizer_config": {
        "warmup_epochs": 0,
        "scheduler": "reduce_on_plateau",  # none|reduce_on_plateau|cosine
        "plateau_factor": 0.5,
        "plateau_patience": 3,
        "min_lr": 1e-6,
        "cosine_alpha": 0.1,  # min lr ratio for cosine phase
    },
    "variant_optimizer_config": {
        "conformer_lite": {
            "learning_rate": 1e-4,
            "warmup_epochs": 3,
            "scheduler": "cosine",
            "cosine_alpha": 0.1,
        },
        "transformer_medium": {
            "learning_rate": 1e-4,
            "warmup_epochs": 3,
            "scheduler": "cosine",
            "cosine_alpha": 0.1,
        },
    },
    "resume_from_best_checkpoint": False,
    "model_variant": "bigru",
    "augmentation_profile": "off",  # off|spatial|spatiotemporal|strong
    "freeze_config": {
        "enabled": False,
        "warmup_epochs": 0,
        "warmup_freeze": "frontend",  # none|frontend|backbone|frontend_backbone
        "post_warmup": "full_unfreeze",
    },
}

VARIANT_CHECKPOINT_MAP = {
    "bigru": "best_ctc_model_bigru.keras",
    "gru": "best_ctc_model_gru.keras",
    "bilstm": "best_ctc_model_bilstm.keras",
    "transformer": "best_ctc_model_transformer.keras",
    "tcn": "best_ctc_model_tcn.keras",
    "conformer_lite": "best_ctc_model_conformer_lite.keras",
    "transformer_medium": "best_ctc_model_transformer_medium.keras",
}

LEGACY_BASELINE_CHECKPOINT = "best_ctc_model.keras"

FREEZE_TARGETS = {"none", "frontend", "backbone", "frontend_backbone"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train lipreading CTC model")
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default=None,
        help="Temporal backbone variant override",
    )
    parser.add_argument(
        "--augmentation-profile",
        choices=AUGMENTATION_PROFILES,
        default=None,
        help="Train-time augmentation profile",
    )
    return parser.parse_args()


def set_global_determinism(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _resolve_variant_model_config(model_variant: str) -> dict[str, float]:
    base_cfg = dict(CONFIG.get("model_config", {}))
    variant_cfg = dict(CONFIG.get("variant_model_config", {}).get(model_variant, {}))
    base_cfg.update(variant_cfg)
    return {
        "backbone_dropout": float(base_cfg.get("backbone_dropout", 0.3)),
        "head_dropout": float(base_cfg.get("head_dropout", 0.3)),
    }


def _resolve_variant_optimizer_config(model_variant: str) -> dict[str, float | int | str]:
    cfg = {
        "learning_rate": float(CONFIG["learning_rate"]),
        "weight_decay": float(CONFIG["weight_decay"]),
        "warmup_epochs": 0,
        "scheduler": "reduce_on_plateau",
        "plateau_factor": 0.5,
        "plateau_patience": 3,
        "min_lr": 1e-6,
        "cosine_alpha": 0.1,
    }
    cfg.update(dict(CONFIG.get("optimizer_config", {})))
    cfg.update(dict(CONFIG.get("variant_optimizer_config", {}).get(model_variant, {})))
    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["warmup_epochs"] = int(cfg.get("warmup_epochs", 0))
    cfg["scheduler"] = str(cfg.get("scheduler", "reduce_on_plateau")).lower()
    cfg["plateau_factor"] = float(cfg.get("plateau_factor", 0.5))
    cfg["plateau_patience"] = int(cfg.get("plateau_patience", 3))
    cfg["min_lr"] = float(cfg.get("min_lr", 1e-6))
    cfg["cosine_alpha"] = float(cfg.get("cosine_alpha", 0.1))
    return cfg


def _set_optimizer_lr(optimizer: tf.keras.optimizers.Optimizer, lr: float):
    if hasattr(optimizer.learning_rate, "assign"):
        optimizer.learning_rate.assign(lr)
    else:
        tf.keras.backend.set_value(optimizer.learning_rate, lr)


def build_optimizer(opt_cfg: dict[str, float | int | str]) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.AdamW(
        learning_rate=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )


def set_layers_trainable(layers_list, trainable: bool):
    for layer in layers_list:
        layer.trainable = trainable


def apply_freeze_state(model: LipReadingCTC, freeze_target: str):
    freeze_target = freeze_target.lower()
    if freeze_target not in FREEZE_TARGETS:
        raise ValueError(
            f"Unsupported freeze target '{freeze_target}'. Supported: {sorted(FREEZE_TARGETS)}"
        )

    # Defaults
    set_layers_trainable(model.get_frontend_layers(), True)
    set_layers_trainable(model.get_backbone_layers(), True)
    set_layers_trainable(model.get_head_layers(), True)  # head always trainable

    if freeze_target == "frontend":
        set_layers_trainable(model.get_frontend_layers(), False)
    elif freeze_target == "backbone":
        set_layers_trainable(model.get_backbone_layers(), False)
    elif freeze_target == "frontend_backbone":
        set_layers_trainable(model.get_frontend_layers(), False)
        set_layers_trainable(model.get_backbone_layers(), False)


def summarize_freeze_state(model: LipReadingCTC) -> str:
    front_trainable = sum(int(layer.trainable) for layer in model.get_frontend_layers())
    back_trainable = sum(int(layer.trainable) for layer in model.get_backbone_layers())
    head_trainable = sum(int(layer.trainable) for layer in model.get_head_layers())
    return (
        f"frontend_trainable={front_trainable}/{len(model.get_frontend_layers())}, "
        f"backbone_trainable={back_trainable}/{len(model.get_backbone_layers())}, "
        f"head_trainable={head_trainable}/{len(model.get_head_layers())}"
    )


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, [])
            merged[key].extend([float(v) for v in values])
    return merged


class EpochHistoryCollector(tf.keras.callbacks.Callback):
    """Collect epoch-end logs so partial progress is preserved on crashes."""

    def __init__(self):
        super().__init__()
        self.history: dict[str, list[float]] = {}

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        for key, value in logs.items():
            if value is None:
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            self.history.setdefault(key, [])
            self.history[key].append(val)


class WarmupThenCosineCallback(tf.keras.callbacks.Callback):
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        target_lr: float,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float,
    ):
        super().__init__()
        self.target_lr = float(target_lr)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs))
        self.min_lr = float(min_lr)

    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            lr = self.target_lr * float(epoch + 1) / float(self.warmup_epochs)
            _set_optimizer_lr(self.model.optimizer, lr)
            return

        decay_epochs = max(self.total_epochs - self.warmup_epochs, 1)
        progress = min(max(epoch - self.warmup_epochs, 0), decay_epochs - 1) / max(decay_epochs - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.min_lr + (self.target_lr - self.min_lr) * cosine
        _set_optimizer_lr(self.model.optimizer, lr)


class WarmupOnlyCallback(tf.keras.callbacks.Callback):
    """Linear warmup to target lr, then keep lr unchanged."""

    def __init__(self, target_lr: float, warmup_epochs: int):
        super().__init__()
        self.target_lr = float(target_lr)
        self.warmup_epochs = max(0, int(warmup_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs <= 0:
            return
        if epoch < self.warmup_epochs:
            lr = self.target_lr * float(epoch + 1) / float(self.warmup_epochs)
            _set_optimizer_lr(self.model.optimizer, lr)
        elif epoch == self.warmup_epochs:
            _set_optimizer_lr(self.model.optimizer, self.target_lr)


class DelayedReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    """ReduceLROnPlateau that starts only after warmup."""

    def __init__(self, warmup_epochs: int, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = max(0, int(warmup_epochs))

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) <= self.warmup_epochs:
            return
        super().on_epoch_end(epoch, logs)


def load_history_container(path: str) -> dict:
    if not os.path.exists(path):
        return {"runs": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except BaseException:
        return {"runs": []}

    if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], list):
        return data

    # Backward compatibility with old flat history files.
    legacy_history = {k: v for k, v in data.items() if isinstance(v, list)} if isinstance(data, dict) else {}
    legacy = {
        "run_id": "legacy_migrated",
        "status": "unknown",
        "started_at": None,
        "ended_at": None,
        "model_variant": "unknown",
        "checkpoint_path": None,
        "history": legacy_history,
        "freeze": data.get("freeze") if isinstance(data, dict) else None,
        "eval": data.get("eval") if isinstance(data, dict) else None,
        "error": None,
    }
    return {"runs": [legacy]}

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
    global task
    args = parse_args()
    model_variant = (args.model_variant or CONFIG["model_variant"]).lower()
    augmentation_profile = (
        args.augmentation_profile or str(CONFIG.get("augmentation_profile", "off"))
    ).lower()
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported model variant '{model_variant}'. Supported: {MODEL_VARIANTS}"
        )
    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unsupported augmentation profile '{augmentation_profile}'. "
            f"Supported: {AUGMENTATION_PROFILES}"
        )
    feature_time_masking = augmentation_profile == "strong"
    model_cfg = _resolve_variant_model_config(model_variant)
    opt_cfg = _resolve_variant_optimizer_config(model_variant)
    set_global_determinism(int(CONFIG["seed"]))

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    print(f"Model variant: {model_variant}")
    print(f"Augmentation profile: {augmentation_profile}")
    print(f"Feature-time masking: {feature_time_masking}")
    print(
        "Model config: "
        f"backbone_dropout={model_cfg['backbone_dropout']}, "
        f"head_dropout={model_cfg['head_dropout']}"
    )
    print(
        "Optimizer config: "
        f"lr={opt_cfg['learning_rate']}, wd={opt_cfg['weight_decay']}, "
        f"warmup_epochs={opt_cfg['warmup_epochs']}, scheduler={opt_cfg['scheduler']}"
    )

    task = Task.init(
        project_name="ImConvo",
        task_name=f"LipReadingCTC_Training_{model_variant}_{augmentation_profile}",
        task_type=Task.TaskTypes.training,
    )
    run_started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_{model_variant}"

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
        seed=int(CONFIG["seed"]),
        train_split="train",
        val_split="val_oos",
        train_augmentation_profile=augmentation_profile,
    )

    # Compute steps from known hard-split sizes
    num_train_samples = len(train_paths)
    num_val_samples = len(val_paths)
    steps_per_epoch = math.ceil(num_train_samples / CONFIG["batch_size"])
    validation_steps = math.ceil(num_val_samples / CONFIG["batch_size"])

    # ---- Build model ----
    model = build_lipreading_ctc(
        model_variant=model_variant,
        num_chars=NUM_CHARS,
        feature_time_masking=feature_time_masking,
        backbone_dropout=float(model_cfg["backbone_dropout"]),
        head_dropout=float(model_cfg["head_dropout"]),
    )

    # Build model by running a forward pass
    for batch in train_ds.take(1):
        x, _ = batch
        _ = model(x)
        break

    print(f"\nModel parameters: {count_parameters(model):,}")

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        VARIANT_CHECKPOINT_MAP[model_variant],
    )
    bigru_checkpoint_path = os.path.join(
        checkpoint_dir,
        VARIANT_CHECKPOINT_MAP["bigru"],
    )
    legacy_checkpoint_path = os.path.join(checkpoint_dir, LEGACY_BASELINE_CHECKPOINT)

    # Keep backward compatibility with prior baseline checkpoint naming.
    if (
        model_variant == "bigru"
        and not os.path.exists(bigru_checkpoint_path)
        and os.path.exists(legacy_checkpoint_path)
    ):
        best_checkpoint_path = legacy_checkpoint_path

    # ---- Optional resume from previous best checkpoint ----
    if CONFIG["resume_from_best_checkpoint"] and os.path.exists(best_checkpoint_path):
        try:
            model.load_weights(best_checkpoint_path)
            print(f"[Checkpoint] Resumed weights from: {best_checkpoint_path}")
        except BaseException as e:
            print(
                f"[Checkpoint] Failed to resume from {best_checkpoint_path} "
                f"({type(e).__name__}: {e}). Training from current initialization."
            )
    elif CONFIG["resume_from_best_checkpoint"]:
        if model_variant != "bigru":
            # Partial transfer from BiGRU baseline into shared front-end/head layers.
            # Mismatched backbone layers are skipped by name/shape.
            transfer_source = None
            if os.path.exists(bigru_checkpoint_path):
                transfer_source = bigru_checkpoint_path
            elif os.path.exists(legacy_checkpoint_path):
                transfer_source = legacy_checkpoint_path

            if transfer_source is not None:
                try:
                    model.load_weights(transfer_source, skip_mismatch=True)
                    print(
                        "[Checkpoint] Variant checkpoint missing. "
                        f"Partially initialized from baseline: {transfer_source}"
                    )
                except BaseException as e:
                    print(
                        "[Checkpoint] Partial transfer failed "
                        f"({type(e).__name__}: {e}). Training from scratch."
                    )
            else:
                print(
                    f"[Checkpoint] No variant checkpoint at {best_checkpoint_path} "
                    "and no baseline BiGRU checkpoint found. Training from scratch."
                )
        else:
            print(
                f"[Checkpoint] No previous checkpoint found at {best_checkpoint_path}. "
                "Training from scratch."
            )

    freeze_cfg = dict(CONFIG.get("freeze_config", {}))
    freeze_enabled = bool(freeze_cfg.get("enabled", False))
    warmup_target = str(freeze_cfg.get("warmup_freeze", "none")).lower()
    if warmup_target not in FREEZE_TARGETS:
        raise ValueError(
            f"Invalid freeze_config.warmup_freeze='{warmup_target}'. "
            f"Supported: {sorted(FREEZE_TARGETS)}"
        )

    if freeze_enabled:
        apply_freeze_state(model, warmup_target)
        print(
            f"[Freeze] Warmup enabled for {freeze_cfg.get('warmup_epochs', 0)} epochs "
            f"with target='{warmup_target}' ({summarize_freeze_state(model)})"
        )
    else:
        apply_freeze_state(model, "none")
        print(f"[Freeze] Disabled ({summarize_freeze_state(model)})")

    # ---- Compile (loss handled in train_step) ----
    optimizer = build_optimizer(opt_cfg)
    model.compile(optimizer=optimizer)

    # ---- Callbacks ----
    callbacks: list[tf.keras.callbacks.Callback] = [EpochHistoryCollector()]
    scheduler_mode = str(opt_cfg.get("scheduler", "reduce_on_plateau")).lower()
    warmup_epochs_lr = max(0, int(opt_cfg.get("warmup_epochs", 0)))
    if scheduler_mode == "cosine":
        min_lr_ratio = float(opt_cfg.get("cosine_alpha", 0.1))
        min_lr = max(float(opt_cfg.get("min_lr", 1e-6)), float(opt_cfg["learning_rate"]) * min_lr_ratio)
        callbacks.append(
            WarmupThenCosineCallback(
                target_lr=float(opt_cfg["learning_rate"]),
                warmup_epochs=warmup_epochs_lr,
                total_epochs=int(CONFIG["num_epochs"]),
                min_lr=min_lr,
            )
        )
    elif scheduler_mode == "reduce_on_plateau":
        if warmup_epochs_lr > 0:
            callbacks.append(
                WarmupOnlyCallback(
                    target_lr=float(opt_cfg["learning_rate"]),
                    warmup_epochs=warmup_epochs_lr,
                )
            )
        callbacks.append(
            DelayedReduceLROnPlateau(
                warmup_epochs=warmup_epochs_lr,
                monitor="val_loss",
                factor=float(opt_cfg.get("plateau_factor", 0.5)),
                patience=int(opt_cfg.get("plateau_patience", 3)),
                min_lr=float(opt_cfg.get("min_lr", 1e-6)),
                verbose=1,
            )
        )
    elif scheduler_mode == "none":
        if warmup_epochs_lr > 0:
            callbacks.append(
                WarmupOnlyCallback(
                    target_lr=float(opt_cfg["learning_rate"]),
                    warmup_epochs=warmup_epochs_lr,
                )
            )
    else:
        raise ValueError(
            f"Unsupported optimizer scheduler '{scheduler_mode}'. "
            "Supported: none|reduce_on_plateau|cosine"
        )

    callbacks.extend(
        [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=CONFIG["patience"],
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_checkpoint_path,
                monitor="val_loss", save_best_only=True, verbose=1,
            ),
        ]
    )
    # ---- Train ----
    print(f"\n{'='*60}")
    print("Starting CTC Training")
    print(f"{'='*60}\n")
    warmup_epochs = int(freeze_cfg.get("warmup_epochs", 0)) if freeze_enabled else 0
    total_epochs = int(CONFIG["num_epochs"])
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    post_mode = str(freeze_cfg.get("post_warmup", "full_unfreeze")).lower()

    history_path = os.path.join(checkpoint_dir, "training_history.json")
    history_collector = callbacks[0]
    merged_history: dict[str, list[float]] = {}
    val_oos_wer = None
    val_oos_cer = None
    val_is_wer = None
    val_is_cer = None
    run_status = "completed"
    error_info = None

    try:
        warmup_history = None
        main_history = None
        if freeze_enabled and warmup_epochs > 0:
            print(f"[Freeze] Phase 1 warmup training for {warmup_epochs} epochs.")
            warmup_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=warmup_epochs,
                initial_epoch=0,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
            )

            if post_mode == "full_unfreeze" and warmup_epochs < total_epochs:
                apply_freeze_state(model, "none")
                model.compile(optimizer=build_optimizer(opt_cfg))
                print(
                    f"[Freeze] Phase 2 full unfreeze from epoch {warmup_epochs + 1} "
                    f"({summarize_freeze_state(model)})"
                )

        if warmup_epochs < total_epochs:
            main_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=total_epochs,
                initial_epoch=warmup_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
            )

        merged_history = merge_histories(warmup_history, main_history)

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
            training=False,
            augmentation_profile="off",
        )
        val_is_steps = math.ceil(len(val_is_paths) / CONFIG["batch_size"])
        val_is_wer, val_is_cer, _ = evaluate_split(
            model=model,
            dataset=val_is_ds,
            steps=val_is_steps,
            split_name="val_is",
        )
    except Exception as e:
        run_status = "crashed"
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"[ERROR] Training crashed: {type(e).__name__}: {e}")
    finally:
        if not merged_history:
            merged_history = dict(history_collector.history)

        freeze_meta = {
            "enabled": freeze_enabled,
            "warmup_epochs": int(freeze_cfg.get("warmup_epochs", 0)),
            "warmup_freeze": warmup_target,
            "post_warmup": str(freeze_cfg.get("post_warmup", "full_unfreeze")).lower(),
            "transition_epoch": int(freeze_cfg.get("warmup_epochs", 0)) if freeze_enabled else None,
        }

        eval_meta = {
            "val_oos": {
                "wer": float(val_oos_wer) if val_oos_wer is not None else None,
                "cer": float(val_oos_cer) if val_oos_cer is not None else None,
            },
            "val_is": {
                "wer": float(val_is_wer) if val_is_wer is not None else None,
                "cer": float(val_is_cer) if val_is_cer is not None else None,
            },
        }

        run_ended_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        run_record = {
            "run_id": run_id,
            "status": run_status,
            "started_at": run_started_at,
            "ended_at": run_ended_at,
            "model_variant": model_variant,
            "augmentation_profile": augmentation_profile,
            "feature_time_masking": feature_time_masking,
            "model_config": model_cfg,
            "optimizer": opt_cfg,
            "checkpoint_path": best_checkpoint_path,
            "history": merged_history,
            "freeze": freeze_meta,
            "eval": eval_meta,
            "error": error_info,
        }

        container = load_history_container(history_path)
        container.setdefault("runs", [])
        container["runs"].append(run_record)

        # Keep top-level latest run history keys for compatibility with plotting utils.
        for key, values in merged_history.items():
            if isinstance(values, list):
                container[key] = values
        container["augmentation"] = {
            "profile": augmentation_profile,
            "feature_time_masking": feature_time_masking,
        }
        container["model_config"] = model_cfg
        container["optimizer"] = opt_cfg
        container["freeze"] = freeze_meta
        container["eval"] = eval_meta
        container["last_run_id"] = run_id
        container["last_status"] = run_status

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(container, f, indent=2)
        print(f"\nTraining history saved to {history_path} (run_id={run_id}, status={run_status})")

        task.close()

        if run_status == "crashed":
            raise RuntimeError(
                f"Training crashed for run_id={run_id}. "
                f"Partial history was saved to {history_path}."
            )

    history_data = merged_history
    history_data["augmentation"] = {
        "profile": augmentation_profile,
        "feature_time_masking": feature_time_masking,
    }
    history_data["model_config"] = model_cfg
    history_data["optimizer"] = opt_cfg
    history_data["freeze"] = {
        "enabled": freeze_enabled,
        "warmup_epochs": int(freeze_cfg.get("warmup_epochs", 0)),
        "warmup_freeze": warmup_target,
        "post_warmup": str(freeze_cfg.get("post_warmup", "full_unfreeze")).lower(),
        "transition_epoch": int(freeze_cfg.get("warmup_epochs", 0)) if freeze_enabled else None,
    }
    history_data["eval"] = {
        "val_oos": {
            "wer": float(val_oos_wer) if val_oos_wer is not None else None,
            "cer": float(val_oos_cer) if val_oos_cer is not None else None,
        },
        "val_is": {
            "wer": float(val_is_wer) if val_is_wer is not None else None,
            "cer": float(val_is_cer) if val_is_cer is not None else None,
        },
    }
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
