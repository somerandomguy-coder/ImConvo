"""Train GRID word-level lipreading model with slot-wise cross-entropy loss."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from datetime import datetime
sys.path.append(os.getcwd())

if not hasattr(sys, "real_prefix") and not (
    hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
):
    print("\n[!] WARNING: You are not running in a virtual environment.")
    print("Please run 'source .venv/bin/activate' or use './setup_and_run.sh'\n")
    sys.exit(1)

import numpy as np
import tensorflow as tf

from experiments.word_level_grid.common import (
    SLOT_VOCAB_SIZES,
    indices_to_sentence,
    make_checkpoint_filename,
    make_run_id,
)
from experiments.word_level_grid.dataset import (
    build_split_word_arrays,
    create_word_dataset,
    create_word_dataset_pipeline,
)
from experiments.word_level_grid.metrics import compute_cer, compute_wer
from experiments.word_level_grid.model import (
    FRONTEND_MODELS,
    MODEL_VARIANTS,
    build_lipreading_word_classifier,
)

CONFIG = {
    "preprocessed_dir": "./data/preprocessed/",
    "split_dir": "./splits/grid_v1",
    "checkpoint_dir": "./checkpoints/word_level_grid",
    "batch_size": 48,
    "num_epochs": 80,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 9,
    "seed": 42,
    "resume_from_best_checkpoint": False,
    "model_variant": "bigru",
    "frontend_model": "flatten",
    "augmentation_profile": "off",  # off|spatial|spatiotemporal|strong
    "model_config": {
        "backbone_dropout": 0.3,
        "head_dropout": 0.3,
        "slot_projection_dim": 256,
    },
    "variant_model_config": {
        "tcn": {
            "backbone_dropout": 0.35,
            "head_dropout": 0.3,
            "slot_projection_dim": 320,
        },
        "conformer_lite": {
            "backbone_dropout": 0.2,
            "head_dropout": 0.2,
            "slot_projection_dim": 256,
        },
        "transformer_medium": {
            "backbone_dropout": 0.2,
            "head_dropout": 0.2,
            "slot_projection_dim": 320,
        },
    },
}

AUGMENTATION_PROFILES = ("off", "spatial", "spatiotemporal", "strong")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRID word-level lipreading model")
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default=None)
    parser.add_argument("--frontend-model", choices=FRONTEND_MODELS, default=None)
    parser.add_argument(
        "--augmentation-profile",
        choices=AUGMENTATION_PROFILES,
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
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

    if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], list):
        return data
    return {"runs": []}


def _resolve_variant_model_config(model_variant: str) -> dict[str, float]:
    base_cfg = dict(CONFIG.get("model_config", {}))
    variant_cfg = dict(CONFIG.get("variant_model_config", {}).get(model_variant, {}))
    base_cfg.update(variant_cfg)
    return {
        "backbone_dropout": float(base_cfg.get("backbone_dropout", 0.3)),
        "head_dropout": float(base_cfg.get("head_dropout", 0.3)),
        "slot_projection_dim": float(base_cfg.get("slot_projection_dim", 256)),
    }


def evaluate_split_word(
    model,
    dataset: tf.data.Dataset,
    steps: int,
    split_name: str,
    preview_limit: int = 8,
) -> dict[str, float | int]:
    """Evaluate a split and return WER/CER/sentence_acc/slot_acc stats."""
    print(f"\n{'='*60}")
    print(f"Word-level evaluation — {split_name}")
    print(f"{'='*60}\n")

    total_wer = 0.0
    total_cer = 0.0
    total_sentence_acc = 0.0
    total_slot_acc = 0.0
    num_samples = 0

    for batch in dataset.take(steps):
        x, y = batch
        logits_list = model(x, training=False)
        pred = model.decode_greedy_slots(logits_list).numpy()
        labels = y["word_labels"].numpy()

        for i in range(len(labels)):
            gt = labels[i]
            pd = pred[i]
            gt_sentence = indices_to_sentence(gt)
            pd_sentence = indices_to_sentence(pd)

            wer = compute_wer(gt_sentence, pd_sentence)
            cer = compute_cer(gt_sentence, pd_sentence)
            sent_acc = float(np.array_equal(gt, pd))
            slot_acc = float(np.mean(gt == pd))

            total_wer += wer
            total_cer += cer
            total_sentence_acc += sent_acc
            total_slot_acc += slot_acc
            num_samples += 1

            if num_samples <= preview_limit:
                print(f"  GT:   '{gt_sentence}'")
                print(f"  Pred: '{pd_sentence}'")
                print(
                    f"  WER: {wer:.2f} | CER: {cer:.2f} | "
                    f"SentAcc: {sent_acc:.1f} | SlotAcc: {slot_acc:.2f}"
                )
                print()

    avg_wer = total_wer / max(num_samples, 1)
    avg_cer = total_cer / max(num_samples, 1)
    avg_sent_acc = total_sentence_acc / max(num_samples, 1)
    avg_slot_acc = total_slot_acc / max(num_samples, 1)

    print(f"{split_name} WER: {avg_wer:.4f}")
    print(f"{split_name} CER: {avg_cer:.4f}")
    print(f"{split_name} Sentence Acc: {avg_sent_acc:.4f}")
    print(f"{split_name} Slot Acc: {avg_slot_acc:.4f}")

    return {
        "wer": float(avg_wer),
        "cer": float(avg_cer),
        "sentence_acc": float(avg_sent_acc),
        "slot_acc": float(avg_slot_acc),
        "count": int(num_samples),
    }


def main():
    args = parse_args()

    model_variant = (args.model_variant or CONFIG["model_variant"]).lower()
    frontend_model = (args.frontend_model or CONFIG["frontend_model"]).lower()
    augmentation_profile = (
        args.augmentation_profile or CONFIG.get("augmentation_profile", "off")
    ).lower()

    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model_variant='{model_variant}'")
    if frontend_model not in FRONTEND_MODELS:
        raise ValueError(f"Unsupported frontend_model='{frontend_model}'")
    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(f"Unsupported augmentation_profile='{augmentation_profile}'")

    batch_size = int(args.batch_size or CONFIG["batch_size"])
    num_epochs = int(args.num_epochs or CONFIG["num_epochs"])
    learning_rate = float(args.learning_rate or CONFIG["learning_rate"])
    feature_time_masking = augmentation_profile == "strong"
    model_cfg = _resolve_variant_model_config(model_variant)

    set_global_determinism(int(CONFIG["seed"]))

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    print(f"Model variant: {model_variant}")
    print(f"Frontend model: {frontend_model}")
    print(f"Augmentation profile: {augmentation_profile}")

    if not os.path.isdir(CONFIG["preprocessed_dir"]):
        raise FileNotFoundError(
            f"Preprocessed data not found at: {CONFIG['preprocessed_dir']} "
            "(run scripts/preprocess.py first)"
        )
    if not os.path.isdir(CONFIG["split_dir"]):
        raise FileNotFoundError(
            f"Split manifests not found at: {CONFIG['split_dir']} "
            "(run scripts/build_split_manifests.py first)"
        )

    run_id = make_run_id(model_variant, frontend_model, augmentation_profile)
    run_started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    checkpoint_dir = CONFIG["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = make_checkpoint_filename(
        run_id=run_id,
        model_variant=model_variant,
        frontend_model=frontend_model,
        augmentation_profile=augmentation_profile,
        batch_size=batch_size,
        seed=int(CONFIG["seed"]),
    )
    best_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    (
        train_ds,
        val_ds,
        train_paths,
        train_labels,
        val_paths,
        val_labels,
    ) = create_word_dataset_pipeline(
        preprocessed_dir=CONFIG["preprocessed_dir"],
        split_dir=CONFIG["split_dir"],
        batch_size=batch_size,
        seed=int(CONFIG["seed"]),
        train_split="train",
        val_split="val_oos",
        train_augmentation_profile=augmentation_profile,
    )

    steps_per_epoch = math.ceil(len(train_paths) / batch_size)
    validation_steps = math.ceil(len(val_paths) / batch_size)

    model = build_lipreading_word_classifier(
        model_variant=model_variant,
        frontend_model=frontend_model,
        slot_vocab_sizes=SLOT_VOCAB_SIZES,
        feature_time_masking=feature_time_masking,
        backbone_dropout=float(model_cfg["backbone_dropout"]),
        head_dropout=float(model_cfg["head_dropout"]),
        slot_projection_dim=int(model_cfg["slot_projection_dim"]),
    )

    for batch in train_ds.take(1):
        x, _ = batch
        _ = model(x)
        break

    if CONFIG.get("resume_from_best_checkpoint", False) and os.path.exists(best_checkpoint_path):
        model.load_weights(best_checkpoint_path)
        print(f"[Checkpoint] Resumed weights from {best_checkpoint_path}")

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

    history_path = os.path.join(checkpoint_dir, "training_history_word.json")
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
        print("Starting word-level GRID training")
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

        eval_meta["val_oos"] = evaluate_split_word(
            model=model,
            dataset=val_ds,
            steps=validation_steps,
            split_name="val_oos",
        )

        val_is_ids = [line.strip() for line in open(os.path.join(CONFIG["split_dir"], "val_is.txt"), encoding="utf-8") if line.strip()]
        val_is_paths, val_is_labels = build_split_word_arrays(
            preprocessed_dir=CONFIG["preprocessed_dir"],
            sample_ids=val_is_ids,
        )
        val_is_ds = create_word_dataset(
            val_is_paths,
            val_is_labels,
            batch_size=batch_size,
            shuffle=False,
            training=False,
            augmentation_profile="off",
        )
        val_is_steps = math.ceil(len(val_is_paths) / batch_size)
        eval_meta["val_is"] = evaluate_split_word(
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

        run_ended_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        run_record = {
            "run_id": run_id,
            "status": run_status,
            "started_at": run_started_at,
            "ended_at": run_ended_at,
            "experiment_family": "word_level_grid",
            "label_unit": "word",
            "loss_function": "cross_entropy",
            "model_variant": model_variant,
            "frontend_model": frontend_model,
            "augmentation_profile": augmentation_profile,
            "feature_time_masking": feature_time_masking,
            "model_config": model_cfg,
            "optimizer": {
                "learning_rate": learning_rate,
                "weight_decay": float(CONFIG["weight_decay"]),
            },
            "checkpoint_path": best_checkpoint_path,
            "history": merged_history,
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

    print("\n✓ Word-level training complete.")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
