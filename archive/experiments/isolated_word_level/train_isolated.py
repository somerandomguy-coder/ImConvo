"""Train isolated-word GRID model with temporal slicing + cross-entropy."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from dataclasses import asdict
from datetime import datetime

if not hasattr(sys, "real_prefix") and not (
    hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
):
    print("\n[!] WARNING: You are not running in a virtual environment.")
    print("Please run 'source .venv/bin/activate' or use './setup_and_run.sh'\n")
    sys.exit(1)

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.isolated_word_level.common import (
    NUM_WORD_CLASSES,
    detect_preprocessed_roi_backend,
    make_checkpoint_filename,
    make_run_id,
    word_idx_to_text,
)
from experiments.isolated_word_level.dataset import (
    build_word_segments_from_ids,
    create_isolated_word_dataset,
    create_isolated_word_dataset_pipeline,
)
from experiments.isolated_word_level.model import (
    FRONTEND_MODELS,
    MODEL_VARIANTS,
    build_lipreading_isolated_word_classifier,
)
from src.dataset import load_split_ids

CONFIG = {
    "preprocessed_dir": "./data/preprocessed/",
    "split_dir": "./splits/grid_v1",
    "checkpoint_dir": "./checkpoints/isolated_word_level",
    "batch_size": 48,
    "num_epochs": 60,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "patience": 8,
    "seed": 42,
    "resume_from_best_checkpoint": False,
    "model_variant": "bigru",
    "frontend_model": "flatten",
    "augmentation_profile": "off",  # off|spatial|spatiotemporal|strong
    "target_frames": 25,
    "pad_mode": "repeat_last",  # repeat_last|zeros
    "force_center_crop": False,
    "model_config": {
        "backbone_dropout": 0.3,
        "head_dropout": 0.3,
        "classifier_hidden_dim": 256,
    },
    "variant_model_config": {
        "tcn": {
            "backbone_dropout": 0.35,
            "head_dropout": 0.3,
            "classifier_hidden_dim": 320,
        },
        "conformer_lite": {
            "backbone_dropout": 0.2,
            "head_dropout": 0.2,
            "classifier_hidden_dim": 256,
        },
        "transformer_medium": {
            "backbone_dropout": 0.2,
            "head_dropout": 0.2,
            "classifier_hidden_dim": 320,
        },
    },
}

AUGMENTATION_PROFILES = ("off", "spatial", "spatiotemporal", "strong")
PAD_MODES = ("repeat_last", "zeros")


def parse_args():
    parser = argparse.ArgumentParser(description="Train isolated-word GRID classifier")
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default=None)
    parser.add_argument("--frontend-model", choices=FRONTEND_MODELS, default=None)
    parser.add_argument("--augmentation-profile", choices=AUGMENTATION_PROFILES, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-frames", type=int, default=None)
    parser.add_argument("--pad-mode", choices=PAD_MODES, default=None)
    parser.add_argument("--force-center-crop", action="store_true", default=False)
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


def _set_optimizer_lr(optimizer: tf.keras.optimizers.Optimizer, lr: float):
    if hasattr(optimizer.learning_rate, "assign"):
        optimizer.learning_rate.assign(lr)
    else:
        tf.keras.backend.set_value(optimizer.learning_rate, lr)


class EpochHistoryCollector(tf.keras.callbacks.Callback):
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
                v = float(value)
            except (TypeError, ValueError):
                continue
            self.history.setdefault(key, [])
            self.history[key].append(v)


class WarmupOnlyCallback(tf.keras.callbacks.Callback):
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


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, [])
            merged[key].extend([float(v) for v in values])
    return merged


def load_history_container(path: str) -> dict:
    if not os.path.exists(path):
        return {"runs": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except BaseException:
        return {"runs": []}

    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        return data
    return {"runs": []}


def _resolve_variant_model_config(model_variant: str) -> dict[str, float]:
    base_cfg = dict(CONFIG.get("model_config", {}))
    variant_cfg = dict(CONFIG.get("variant_model_config", {}).get(model_variant, {}))
    base_cfg.update(variant_cfg)
    return {
        "backbone_dropout": float(base_cfg.get("backbone_dropout", 0.3)),
        "head_dropout": float(base_cfg.get("head_dropout", 0.3)),
        "classifier_hidden_dim": float(base_cfg.get("classifier_hidden_dim", 256)),
    }


def evaluate_split_topk(
    model,
    dataset: tf.data.Dataset,
    steps: int,
    split_name: str,
    preview_limit: int = 10,
) -> dict[str, float | int]:
    total_top1 = 0.0
    total_top5 = 0.0
    num_samples = 0

    print(f"\n{'='*60}")
    print(f"Isolated-word evaluation — {split_name}")
    print(f"{'='*60}")

    for batch in dataset.take(steps):
        x, y = batch
        labels = y["label"].numpy().astype(np.int32)
        logits = model(x, training=False).numpy()

        top1_pred = np.argmax(logits, axis=1)
        top5_pred = np.argsort(logits, axis=1)[:, ::-1][:, :5]

        for i in range(len(labels)):
            gt = int(labels[i])
            pred1 = int(top1_pred[i])
            pred5 = top5_pred[i].tolist()

            is_top1 = float(pred1 == gt)
            is_top5 = float(gt in pred5)

            total_top1 += is_top1
            total_top5 += is_top5
            num_samples += 1

            if num_samples <= preview_limit:
                gt_word = word_idx_to_text(gt)
                p1_word = word_idx_to_text(pred1)
                p5_words = [word_idx_to_text(idx) for idx in pred5]
                print(
                    f"  GT='{gt_word}' | top1='{p1_word}' | "
                    f"top5={p5_words} | top1={is_top1:.0f} top5={is_top5:.0f}"
                )

    avg_top1 = total_top1 / max(num_samples, 1)
    avg_top5 = total_top5 / max(num_samples, 1)

    print(f"{split_name} top1_acc: {avg_top1:.4f}")
    print(f"{split_name} top5_acc: {avg_top5:.4f}")

    return {
        "top1_acc": float(avg_top1),
        "top5_acc": float(avg_top5),
        "count": int(num_samples),
    }


def main():
    args = parse_args()

    model_variant = (args.model_variant or CONFIG["model_variant"]).lower()
    frontend_model = (args.frontend_model or CONFIG["frontend_model"]).lower()
    augmentation_profile = (args.augmentation_profile or CONFIG["augmentation_profile"]).lower()
    batch_size = int(args.batch_size or CONFIG["batch_size"])
    num_epochs = int(args.num_epochs or CONFIG["num_epochs"])
    learning_rate = float(args.learning_rate or CONFIG["learning_rate"])
    target_frames = int(args.target_frames or CONFIG["target_frames"])
    pad_mode = (args.pad_mode or CONFIG["pad_mode"]).lower()

    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model_variant={model_variant}")
    if frontend_model not in FRONTEND_MODELS:
        raise ValueError(f"Unsupported frontend_model={frontend_model}")
    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(f"Unsupported augmentation_profile={augmentation_profile}")
    if pad_mode not in PAD_MODES:
        raise ValueError(f"Unsupported pad_mode={pad_mode}")
    if target_frames <= 0:
        raise ValueError("target_frames must be > 0")

    feature_time_masking = augmentation_profile == "strong"
    model_cfg = _resolve_variant_model_config(model_variant)

    set_global_determinism(int(CONFIG["seed"]))

    backend_name, roi_centered = detect_preprocessed_roi_backend(project_root=".")
    enforce_center_crop = bool(args.force_center_crop)
    if not roi_centered:
        enforce_center_crop = True

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    print(f"Model variant: {model_variant}")
    print(f"Frontend model: {frontend_model}")
    print(f"Augmentation profile: {augmentation_profile}")
    print(f"Target frames: {target_frames} | pad_mode={pad_mode}")
    print(f"Detected preprocessing backend: {backend_name} | roi_centered={roi_centered}")
    print(f"Center-focus crop in loader: {enforce_center_crop}")

    if not os.path.isdir(CONFIG["preprocessed_dir"]):
        raise FileNotFoundError(f"Preprocessed dir not found: {CONFIG['preprocessed_dir']}")
    if not os.path.isdir(CONFIG["split_dir"]):
        raise FileNotFoundError(f"Split dir not found: {CONFIG['split_dir']}")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    run_id = make_run_id(
        model_variant=model_variant,
        frontend_model=frontend_model,
        backend_type=model_variant,
    )
    run_started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    checkpoint_filename = make_checkpoint_filename(
        run_id=run_id,
        model_variant=model_variant,
        frontend_model=frontend_model,
        backend_type=model_variant,
        batch_size=batch_size,
        target_frames=target_frames,
    )
    best_checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], checkpoint_filename)

    (
        train_ds,
        val_ds,
        train_segments,
        train_stats,
        val_segments,
        val_stats,
    ) = create_isolated_word_dataset_pipeline(
        preprocessed_dir=CONFIG["preprocessed_dir"],
        split_dir=CONFIG["split_dir"],
        batch_size=batch_size,
        seed=int(CONFIG["seed"]),
        train_split="train",
        val_split="val_oos",
        target_frames=target_frames,
        pad_mode=pad_mode,
        train_augmentation_profile=augmentation_profile,
        enforce_center_crop=enforce_center_crop,
    )

    steps_per_epoch = math.ceil(len(train_segments) / batch_size)
    validation_steps = math.ceil(len(val_segments) / batch_size)

    model = build_lipreading_isolated_word_classifier(
        model_variant=model_variant,
        frontend_model=frontend_model,
        num_word_classes=NUM_WORD_CLASSES,
        feature_time_masking=feature_time_masking,
        backbone_dropout=float(model_cfg["backbone_dropout"]),
        head_dropout=float(model_cfg["head_dropout"]),
        classifier_hidden_dim=int(model_cfg["classifier_hidden_dim"]),
    )

    for batch in train_ds.take(1):
        x, _ = batch
        _ = model(x)
        break

    if CONFIG.get("resume_from_best_checkpoint", False) and os.path.exists(best_checkpoint_path):
        model.load_weights(best_checkpoint_path)
        print(f"[Checkpoint] resumed from {best_checkpoint_path}")

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=float(CONFIG["weight_decay"]),
    )
    model.compile(optimizer=optimizer)

    callbacks: list[tf.keras.callbacks.Callback] = [
        EpochHistoryCollector(),
        WarmupOnlyCallback(target_lr=learning_rate, warmup_epochs=2),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(CONFIG["patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    history_path = os.path.join(CONFIG["checkpoint_dir"], "training_history_isolated_word_v1.json")
    history_collector = callbacks[0]
    run_status = "completed"
    error_info = None
    merged_history: dict[str, list[float]] = {}

    eval_meta = {
        "val_oos": None,
        "val_is": None,
    }

    try:
        print(f"\n{'='*60}")
        print("Starting isolated-word training")
        print(f"{'='*60}\n")

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )
        merged_history = merge_histories(history)

        eval_meta["val_oos"] = evaluate_split_topk(
            model=model,
            dataset=val_ds,
            steps=validation_steps,
            split_name="val_oos",
        )

        val_is_ids = load_split_ids(split_dir=CONFIG["split_dir"], split_name="val_is")
        val_is_segments, val_is_stats = build_word_segments_from_ids(
            preprocessed_dir=CONFIG["preprocessed_dir"],
            sample_ids=val_is_ids,
        )
        if not val_is_segments:
            raise ValueError("No val_is segments found.")

        val_is_paths = [seg.npy_path for seg in val_is_segments]
        val_is_starts = np.array([seg.start_frame for seg in val_is_segments], dtype=np.int32)
        val_is_ends = np.array([seg.end_frame for seg in val_is_segments], dtype=np.int32)
        val_is_labels = np.array([seg.class_idx for seg in val_is_segments], dtype=np.int32)

        val_is_ds = create_isolated_word_dataset(
            val_is_paths,
            val_is_starts,
            val_is_ends,
            val_is_labels,
            batch_size=batch_size,
            target_frames=target_frames,
            pad_mode=pad_mode,
            shuffle=False,
            seed=None,
            training=False,
            augmentation_profile="off",
            enforce_center_crop=enforce_center_crop,
        )
        val_is_steps = math.ceil(len(val_is_segments) / batch_size)
        eval_meta["val_is"] = evaluate_split_topk(
            model=model,
            dataset=val_is_ds,
            steps=val_is_steps,
            split_name="val_is",
        )
        eval_meta["val_is_stats"] = asdict(val_is_stats)

    except Exception as e:
        run_status = "crashed"
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"[ERROR] training crashed: {type(e).__name__}: {e}")
    finally:
        if not merged_history:
            merged_history = dict(history_collector.history)

        run_ended_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        run_record = {
            "run_id": run_id,
            "status": run_status,
            "started_at": run_started_at,
            "ended_at": run_ended_at,
            "experiment_family": "isolated_word_level",
            "label_unit": "word",
            "loss_function": "cross_entropy",
            "target_frames": target_frames,
            "pad_mode": pad_mode,
            "model_variant": model_variant,
            "frontend_model": frontend_model,
            "augmentation_profile": augmentation_profile,
            "feature_time_masking": feature_time_masking,
            "preprocess_backend": backend_name,
            "roi_centered_detected": roi_centered,
            "enforce_center_crop": enforce_center_crop,
            "model_config": model_cfg,
            "optimizer": {
                "learning_rate": learning_rate,
                "weight_decay": float(CONFIG["weight_decay"]),
            },
            "checkpoint_path": best_checkpoint_path,
            "history": merged_history,
            "dataset_stats": {
                "train": asdict(train_stats),
                "val_oos": asdict(val_stats),
            },
            "eval": eval_meta,
            "error": error_info,
        }

        container = load_history_container(history_path)
        container.setdefault("runs", [])
        container["runs"].append(run_record)

        for key, values in merged_history.items():
            if isinstance(values, list):
                container[key] = values
        container["eval"] = eval_meta
        container["last_run_id"] = run_id
        container["last_status"] = run_status

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(container, f, indent=2)

        print(f"\nTraining history saved to {history_path} (run_id={run_id}, status={run_status})")

        if run_status == "crashed":
            raise RuntimeError(
                f"Training crashed for run_id={run_id}. "
                f"Partial history saved to {history_path}."
            )

    print("\n✓ Isolated-word training complete.")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
