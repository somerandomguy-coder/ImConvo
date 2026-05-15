"""Append isolated-word experiment metadata to log_isolated.jsonl."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "reports" / "eval_result_isolated_word"
JSONL_PATH = ROOT / "log_isolated.jsonl"
TRAIN_PY = ROOT / "experiments" / "isolated_word_level" / "train_isolated.py"
TRAINING_HISTORY_PATH = ROOT / "checkpoints" / "isolated_word_level" / "training_history_isolated_word_v1.json"


def detect_branch() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True)
            .strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def find_latest_eval_report() -> Path | None:
    if not EVAL_DIR.exists():
        return None
    reports = sorted(
        [p for p in EVAL_DIR.glob("eval_report_isolated_word_*.txt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def parse_eval_metrics(report_path: Path | None) -> dict[str, float | int | None]:
    metrics = {
        "val_oos_top1": None,
        "val_oos_top5": None,
        "val_is_top1": None,
        "val_is_top5": None,
        "test_oos_top1": None,
        "test_oos_top5": None,
        "test_is_top1": None,
        "test_is_top5": None,
    }
    if report_path is None or not report_path.exists():
        return metrics

    text = report_path.read_text(encoding="utf-8", errors="ignore")
    row_pattern = re.compile(
        r"^(val_oos|val_is|test_oos|test_is):\s+count=\d+,\s+TOP1=([0-9.]+),\s+TOP5=([0-9.]+)\s*$",
        re.MULTILINE,
    )
    for split, top1, top5 in row_pattern.findall(text):
        metrics[f"{split}_top1"] = float(top1)
        metrics[f"{split}_top5"] = float(top5)

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
        "target_frames": 25,
        "pad_mode": "repeat_last",
    }
    try:
        source = TRAIN_PY.read_text(encoding="utf-8")
        module = ast.parse(source)
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "CONFIG":
                        cfg = ast.literal_eval(node.value)
                        if isinstance(cfg, dict):
                            defaults.update(cfg)
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

    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or not runs:
        return {}
    latest = runs[-1]
    return latest if isinstance(latest, dict) else {}


def resolve_setting(cli_value: str | None, latest_run_value: str | None, default_value: str) -> str:
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
    return f"{date_str}_{compact_model}_{compact_frontend}_{compact_branch}_{split_version}_isolated_word_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Append isolated-word experiment log row")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--frontend-model", default=None)
    parser.add_argument("--augmentation-profile", default=None)
    parser.add_argument("--notes", default="Auto-logged from isolated-word artifacts.")
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

    model_variant = resolve_setting(
        cli_value=args.model_variant,
        latest_run_value=latest_run.get("model_variant"),
        default_value=str(cfg.get("model_variant") or "bigru"),
    )
    frontend_model = resolve_setting(
        cli_value=args.frontend_model,
        latest_run_value=latest_run.get("frontend_model"),
        default_value=str(cfg.get("frontend_model") or "flatten"),
    )
    augmentation_profile = resolve_setting(
        cli_value=args.augmentation_profile,
        latest_run_value=latest_run.get("augmentation_profile"),
        default_value=str(cfg.get("augmentation_profile") or "off"),
    )

    run_id = args.run_id or make_run_id(
        branch=branch,
        model_variant=model_variant,
        frontend_model=frontend_model,
        split_version=split_version,
    )

    latest_checkpoint_path = latest_run.get("checkpoint_path")
    if latest_checkpoint_path:
        ckpt = Path(str(latest_checkpoint_path))
        if not ckpt.is_absolute():
            ckpt = (ROOT / ckpt).resolve()
        try:
            base_checkpoint = str(ckpt.relative_to(ROOT))
        except ValueError:
            base_checkpoint = str(ckpt)
    else:
        base_checkpoint = "checkpoints/isolated_word_level/unknown_isolated_word_v1.weights.h5"

    record = {
        "run_id": run_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "branch": branch,
        "experiment_family": "isolated_word_level",
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
        "decoder": "single_head_argmax",
        "batch_size": int(cfg.get("batch_size") or 0),
        "learning_rate": float(cfg.get("learning_rate") or 0.0),
        "weight_decay": float(cfg.get("weight_decay") or 0.0),
        "patience": int(cfg.get("patience") or 0),
        "num_epochs": int(cfg.get("num_epochs") or 0),
        "target_frames": int(cfg.get("target_frames") or 25),
        "pad_mode": str(cfg.get("pad_mode") or "repeat_last"),
        # Compatibility placeholders with experiments.jsonl schema
        "val_oos_wer": None,
        "val_oos_cer": None,
        "val_is_wer": None,
        "val_is_cer": None,
        "test_oos_wer": None,
        "test_oos_cer": None,
        "test_is_wer": None,
        "test_is_cer": None,
        "val_oos_top1": metrics["val_oos_top1"],
        "val_oos_top5": metrics["val_oos_top5"],
        "val_is_top1": metrics["val_is_top1"],
        "val_is_top5": metrics["val_is_top5"],
        "test_oos_top1": metrics["test_oos_top1"],
        "test_oos_top5": metrics["test_oos_top5"],
        "test_is_top1": metrics["test_is_top1"],
        "test_is_top5": metrics["test_is_top5"],
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

    print(f"Appended isolated-word experiment record to: {out_path}")
    print(f"run_id: {record['run_id']}")


if __name__ == "__main__":
    main()
