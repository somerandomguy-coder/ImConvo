"""Append word-level experiment records to a JSONL file in experiments.jsonl-style format."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
import sys, os
sys.path.append(os.getcwd())

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "reports" / "eval_result_word"
JSONL_PATH = ROOT / "experiments_word_level.jsonl"
TRAIN_PY = ROOT / "experiments" / "word_level_grid" / "train_word.py"
TRAINING_HISTORY_PATH = ROOT / "checkpoints" / "word_level_grid" / "training_history_word.json"


def detect_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def find_latest_eval_report() -> Path | None:
    if not EVAL_DIR.exists():
        return None
    candidates = sorted(
        [p for p in EVAL_DIR.glob("eval_report_word_*.txt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_eval_metrics(report_path: Path | None) -> dict[str, float | None]:
    metrics = {
        "val_oos_wer": None,
        "val_oos_cer": None,
        "val_oos_sentence_acc": None,
        "val_oos_slot_acc": None,
        "val_is_wer": None,
        "val_is_cer": None,
        "val_is_sentence_acc": None,
        "val_is_slot_acc": None,
        "test_oos_wer": None,
        "test_oos_cer": None,
        "test_oos_sentence_acc": None,
        "test_oos_slot_acc": None,
        "test_is_wer": None,
        "test_is_cer": None,
        "test_is_sentence_acc": None,
        "test_is_slot_acc": None,
    }
    if report_path is None or not report_path.exists():
        return metrics

    text = report_path.read_text(encoding="utf-8", errors="ignore")
    row_pattern = re.compile(
        r"^(val_oos|val_is|test_oos|test_is):\s+count=\d+,\s+"
        r"WER=([0-9.]+),\s+CER=([0-9.]+),\s+"
        r"SENT_ACC=([0-9.]+),\s+SLOT_ACC=([0-9.]+)\s*$",
        re.MULTILINE,
    )
    for split, wer, cer, sentence_acc, slot_acc in row_pattern.findall(text):
        metrics[f"{split}_wer"] = float(wer)
        metrics[f"{split}_cer"] = float(cer)
        metrics[f"{split}_sentence_acc"] = float(sentence_acc)
        metrics[f"{split}_slot_acc"] = float(slot_acc)
    return metrics


def extract_config_from_train() -> dict:
    defaults = {
        "batch_size": None,
        "learning_rate": None,
        "weight_decay": None,
        "patience": None,
        "num_epochs": None,
        "resume_from_best_checkpoint": None,
        "split_dir": "./splits/grid_v1",
        "model_variant": "bigru",
        "frontend_model": "flatten",
        "augmentation_profile": "off",
    }
    try:
        source = TRAIN_PY.read_text(encoding="utf-8")
        module = ast.parse(source)
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "CONFIG":
                        value = ast.literal_eval(node.value)
                        if isinstance(value, dict):
                            defaults.update(value)
                        return defaults
    except Exception:
        pass
    return defaults


def extract_latest_training_run() -> dict:
    if not TRAINING_HISTORY_PATH.exists():
        return {}
    try:
        payload = json.loads(TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        return {}
    latest = runs[-1]
    return latest if isinstance(latest, dict) else {}


def resolve_setting(
    cli_value: str | None,
    latest_run_value: str | None,
    default_value: str,
) -> str:
    if cli_value:
        return str(cli_value).lower()
    if latest_run_value:
        return str(latest_run_value).lower()
    return str(default_value).lower()


def make_run_id(branch: str, model_variant: str, frontend_model: str, split_version: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    compact_model = re.sub(r"[^a-zA-Z0-9]+", "-", model_variant.lower()).strip("-")
    compact_frontend = re.sub(r"[^a-zA-Z0-9]+", "-", frontend_model.lower()).strip("-")
    compact_branch = re.sub(r"[^a-zA-Z0-9]+", "-", branch.lower()).strip("-")
    return f"{date_str}_wordgrid_{compact_model}_{compact_frontend}_{compact_branch}_{split_version}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a word-level experiment record to JSONL")
    parser.add_argument("--run-id", default=None, help="Optional explicit run_id")
    parser.add_argument("--model-variant", choices=[
        "bigru", "gru", "bilstm", "transformer", "tcn", "conformer_lite", "transformer_medium",
    ], default=None)
    parser.add_argument("--frontend-model", choices=["flatten", "gap_proj", "resnet18"], default=None)
    parser.add_argument("--augmentation-profile", choices=["off", "spatial", "spatiotemporal", "strong"], default=None)
    parser.add_argument("--notes", default="Auto-logged from word-level local artifacts.")
    parser.add_argument("--jsonl-path", default=str(JSONL_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    branch = detect_branch()
    latest_report = find_latest_eval_report()
    metrics = parse_eval_metrics(latest_report)
    cfg = extract_config_from_train()
    latest_run = extract_latest_training_run()

    split_dir = str(cfg.get("split_dir") or "./splits/grid_v1")
    split_version = Path(split_dir).name if split_dir else "unknown_split"

    latest_model_variant = latest_run.get("model_variant")
    latest_frontend_model = latest_run.get("frontend_model")
    latest_augmentation_profile = latest_run.get("augmentation_profile")
    latest_checkpoint_path = latest_run.get("checkpoint_path")

    default_model_variant = str(cfg.get("model_variant") or "bigru")
    default_frontend_model = str(cfg.get("frontend_model") or "flatten")
    default_augmentation_profile = str(cfg.get("augmentation_profile") or "off")

    model_variant = resolve_setting(
        cli_value=args.model_variant,
        latest_run_value=latest_model_variant,
        default_value=default_model_variant,
    )
    frontend_model = resolve_setting(
        cli_value=args.frontend_model,
        latest_run_value=latest_frontend_model,
        default_value=default_frontend_model,
    )
    augmentation_profile = resolve_setting(
        cli_value=args.augmentation_profile,
        latest_run_value=latest_augmentation_profile,
        default_value=default_augmentation_profile,
    )

    run_id = args.run_id or make_run_id(
        branch=branch,
        model_variant=model_variant,
        frontend_model=frontend_model,
        split_version=split_version,
    )

    if latest_checkpoint_path:
        latest_checkpoint = Path(str(latest_checkpoint_path))
        if not latest_checkpoint.is_absolute():
            latest_checkpoint = (ROOT / latest_checkpoint).resolve()
        try:
            base_checkpoint = str(latest_checkpoint.relative_to(ROOT))
        except ValueError:
            base_checkpoint = str(latest_checkpoint)
    else:
        base_checkpoint = "checkpoints/word_level_grid/unknown.weights.h5"

    record = {
        "run_id": run_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "branch": branch,
        "experiment_family": "word_level_grid",
        "label_unit": "word",
        "loss_function": "cross_entropy",
        "base_checkpoint": base_checkpoint,
        "resume_from_checkpoint": bool(cfg.get("resume_from_best_checkpoint")),
        "split_version": split_version,
        "train_split": "train",
        "val_oos_split": "val_oos",
        "model_variant": model_variant,
        "frontend_model": frontend_model,
        "augmentation_profile": augmentation_profile,
        "feature_time_masking": augmentation_profile == "strong",
        "freeze_enabled": False,
        "decoder": "slot_argmax",
        "batch_size": int(cfg.get("batch_size") or 0),
        "learning_rate": float(cfg.get("learning_rate") or 0.0),
        "weight_decay": float(cfg.get("weight_decay") or 0.0),
        "patience": int(cfg.get("patience") or 0),
        "num_epochs": int(cfg.get("num_epochs") or 0),
        "freeze_warmup_epochs": 0,
        "freeze_warmup_target": "none",
        "freeze_post_warmup": "none",
        "val_oos_wer": metrics["val_oos_wer"],
        "val_oos_cer": metrics["val_oos_cer"],
        "val_oos_sentence_acc": metrics["val_oos_sentence_acc"],
        "val_oos_slot_acc": metrics["val_oos_slot_acc"],
        "val_is_wer": metrics["val_is_wer"],
        "val_is_cer": metrics["val_is_cer"],
        "val_is_sentence_acc": metrics["val_is_sentence_acc"],
        "val_is_slot_acc": metrics["val_is_slot_acc"],
        "test_oos_wer": metrics["test_oos_wer"],
        "test_oos_cer": metrics["test_oos_cer"],
        "test_oos_sentence_acc": metrics["test_oos_sentence_acc"],
        "test_oos_slot_acc": metrics["test_oos_slot_acc"],
        "test_is_wer": metrics["test_is_wer"],
        "test_is_cer": metrics["test_is_cer"],
        "test_is_sentence_acc": metrics["test_is_sentence_acc"],
        "test_is_slot_acc": metrics["test_is_slot_acc"],
        "eval_report_path": (
            str(latest_report.relative_to(ROOT))
            if latest_report and latest_report.exists()
            else None
        ),
        "notes": args.notes,
    }

    out_path = Path(args.jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps(record, indent=2))
        return

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Appended word-level experiment record to: {out_path}")
    print(f"run_id: {record['run_id']}")


if __name__ == "__main__":
    main()
