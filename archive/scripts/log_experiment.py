"""
Append an experiment record to experiments.jsonl from local artifacts.

What it auto-detects:
- Current git branch
- Latest eval report in reports/eval_result
- Split metrics from report final summary (val_oos, val_is, test_oos, test_is)
- Key training config values from train.py CONFIG dict

Usage:
    ./.venv/bin/python scripts/log_experiment.py
    ./.venv/bin/python scripts/log_experiment.py --run-id my_custom_run_id
    ./.venv/bin/python scripts/log_experiment.py --dry-run
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "reports" / "eval_result"
JSONL_PATH = ROOT / "experiments.jsonl"
TRAIN_PY = ROOT / "train.py"
TRAINING_HISTORY_PATH = ROOT / "checkpoints" / "training_history.json"
FRONTEND_MODELS = ("flatten", "gap_proj", "resnet18")
VARIANT_CHECKPOINT_MAP = {
    "bigru": "best_ctc_model_bigru.keras",
    "gru": "best_ctc_model_gru.keras",
    "bilstm": "best_ctc_model_bilstm.keras",
    "transformer": "best_ctc_model_transformer.keras",
    "tcn": "best_ctc_model_tcn.keras",
    "conformer_lite": "best_ctc_model_conformer_lite.keras",
    "transformer_medium": "best_ctc_model_transformer_medium.keras",
}


def resolve_checkpoint_filename(model_variant: str, frontend_model: str) -> str:
    base_name = VARIANT_CHECKPOINT_MAP.get(model_variant, "best_ctc_model.keras")
    if frontend_model == "flatten":
        return base_name
    stem, ext = Path(base_name).stem, Path(base_name).suffix
    return f"{stem}_{frontend_model}{ext}"


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
        [p for p in EVAL_DIR.glob("eval_report_*.txt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_eval_metrics(report_path: Path | None) -> dict[str, float | None]:
    metrics = {
        "val_oos_wer": None,
        "val_oos_cer": None,
        "val_is_wer": None,
        "val_is_cer": None,
        "test_oos_wer": None,
        "test_oos_cer": None,
        "test_is_wer": None,
        "test_is_cer": None,
    }
    if report_path is None or not report_path.exists():
        return metrics

    text = report_path.read_text(encoding="utf-8", errors="ignore")
    # Expected line format from current test.py final summary:
    # val_oos: count=1000, WER=0.1234, CER=0.5678
    row_pattern = re.compile(
        r"^(val_oos|val_is|test_oos|test_is):\s+count=\d+,\s+WER=([0-9.]+),\s+CER=([0-9.]+)\s*$",
        re.MULTILINE,
    )
    for split, wer, cer in row_pattern.findall(text):
        metrics[f"{split}_wer"] = float(wer)
        metrics[f"{split}_cer"] = float(cer)
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
        "freeze_config": {},
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
    """Read the latest run metadata captured by train.py if available."""
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
    """Resolve setting priority: CLI override > latest run metadata > defaults."""
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
    return f"{date_str}_{compact_model}_{compact_frontend}_{compact_branch}_{split_version}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Append an experiment record to experiments.jsonl")
    parser.add_argument("--run-id", default=None, help="Optional explicit run_id")
    parser.add_argument(
        "--model-variant",
        choices=[
            "bigru",
            "gru",
            "bilstm",
            "transformer",
            "tcn",
            "conformer_lite",
            "transformer_medium",
        ],
        default=None,
        help="Model variant label override for tracking",
    )
    parser.add_argument(
        "--augmentation-profile",
        choices=["off", "spatial", "spatiotemporal", "strong"],
        default=None,
        help="Augmentation profile override for tracking",
    )
    parser.add_argument(
        "--frontend-model",
        choices=FRONTEND_MODELS,
        default=None,
        help="Visual frontend override for tracking",
    )
    parser.add_argument(
        "--notes",
        default="Auto-logged from local artifacts.",
        help="Free-text notes",
    )
    parser.add_argument(
        "--jsonl-path",
        default=str(JSONL_PATH),
        help="Destination jsonl file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON only, do not append",
    )
    args = parser.parse_args()

    branch = detect_branch()
    latest_report = find_latest_eval_report()
    metrics = parse_eval_metrics(latest_report)
    cfg = extract_config_from_train()
    latest_run = extract_latest_training_run()

    split_dir = str(cfg.get("split_dir") or "./splits/grid_v1")
    split_version = Path(split_dir).name if split_dir else "unknown_split"
    freeze_cfg = cfg.get("freeze_config") if isinstance(cfg.get("freeze_config"), dict) else {}

    model_variant = (args.model_variant or str(cfg.get("model_variant") or "bigru")).lower()
    frontend_model = (args.frontend_model or str(cfg.get("frontend_model") or "flatten")).lower()
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

    if latest_model_variant and latest_model_variant != default_model_variant:
        print(
            "[WARN] model_variant mismatch: "
            f"train.py default='{default_model_variant}' vs "
            f"latest_run='{latest_model_variant}'. "
            "Using resolved priority (CLI > latest run > default)."
        )
    if (
        latest_augmentation_profile
        and latest_augmentation_profile != default_augmentation_profile
    ):
        print(
            "[WARN] augmentation_profile mismatch: "
            f"train.py default='{default_augmentation_profile}' vs "
            f"latest_run='{latest_augmentation_profile}'. "
            "Using resolved priority (CLI > latest run > default)."
        )
    if latest_frontend_model and latest_frontend_model != default_frontend_model:
        print(
            "[WARN] frontend_model mismatch: "
            f"train.py default='{default_frontend_model}' vs "
            f"latest_run='{latest_frontend_model}'. "
            "Using resolved priority (CLI > latest run > default)."
        )

    run_id = args.run_id or make_run_id(
        branch=branch,
        model_variant=model_variant,
        frontend_model=frontend_model,
        split_version=split_version,
    )

    latest_frontend_for_match = (
        str(latest_frontend_model).lower() if latest_frontend_model else "flatten"
    )
    use_latest_checkpoint = (
        bool(latest_checkpoint_path)
        and str(latest_model_variant or "").lower() == model_variant
        and latest_frontend_for_match == frontend_model
    )

    if use_latest_checkpoint:
        latest_checkpoint = Path(str(latest_checkpoint_path))
        if not latest_checkpoint.is_absolute():
            latest_checkpoint = (ROOT / latest_checkpoint).resolve()
        try:
            base_checkpoint = str(latest_checkpoint.relative_to(ROOT))
        except ValueError:
            base_checkpoint = str(latest_checkpoint)
    else:
        base_checkpoint = str(
            (ROOT / "checkpoints" / resolve_checkpoint_filename(model_variant, frontend_model)).relative_to(ROOT)
        )

    record = {
        "run_id": run_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "branch": branch,
        "base_checkpoint": base_checkpoint,
        "resume_from_checkpoint": bool(cfg.get("resume_from_best_checkpoint")),
        "split_version": split_version,
        "train_split": "train",
        "val_oos_split": "val_oos",
        "model_variant": model_variant,
        "frontend_model": frontend_model,
        "augmentation_profile": augmentation_profile,
        "feature_time_masking": augmentation_profile == "strong",
        "freeze_enabled": bool(freeze_cfg.get("enabled", False)),
        "decoder": "ctc_greedy",
        "batch_size": cfg.get("batch_size"),
        "learning_rate": cfg.get("learning_rate"),
        "weight_decay": cfg.get("weight_decay"),
        "patience": cfg.get("patience"),
        "num_epochs": cfg.get("num_epochs"),
        "freeze_warmup_epochs": int(freeze_cfg.get("warmup_epochs", 0)),
        "freeze_warmup_target": str(freeze_cfg.get("warmup_freeze", "none")).lower(),
        "freeze_post_warmup": str(freeze_cfg.get("post_warmup", "full_unfreeze")).lower(),
        "val_oos_wer": metrics["val_oos_wer"],
        "val_oos_cer": metrics["val_oos_cer"],
        "val_is_wer": metrics["val_is_wer"],
        "val_is_cer": metrics["val_is_cer"],
        "test_oos_wer": metrics["test_oos_wer"],
        "test_oos_cer": metrics["test_oos_cer"],
        "test_is_wer": metrics["test_is_wer"],
        "test_is_cer": metrics["test_is_cer"],
        "eval_report_path": (
            str(latest_report.relative_to(ROOT)) if latest_report else None
        ),
        "notes": args.notes,
    }

    line = json.dumps(record, ensure_ascii=True)
    if args.dry_run:
        print(line)
        return

    jsonl_path = Path(args.jsonl_path)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[OK] Appended experiment record to: {jsonl_path}")
    print(f"[Info] branch={branch}")
    print(f"[Info] report={record['eval_report_path']}")
    print(f"[Info] run_id={run_id}")


if __name__ == "__main__":
    main()
