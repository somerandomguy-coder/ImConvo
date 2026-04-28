#!/usr/bin/env python3
"""
Plot model comparison bar charts from evaluation JSON files.

Expected JSON format: output from scripts/evaluate_model_all_splits.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3 model-comparison bar plots (accuracy, WER, CER) from "
            "evaluation JSON files."
        )
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Paths to model evaluation JSON files (one per model).",
    )
    parser.add_argument(
        "--split",
        default="overall",
        choices=["train", "valid", "test", "overall"],
        help="Which split to visualize (default: overall).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "valid", "test", "overall"],
        default=None,
        help=(
            "Optional list of splits to visualize in one run. "
            "Example: --splits train valid test (creates 9 plots for 3 metrics). "
            "If omitted, --split is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="temp_reports/plots",
        help="Directory to save PNG plots.",
    )
    parser.add_argument(
        "--prefix",
        default="model_compare",
        help="Filename prefix for plot outputs.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Model Comparison",
        help="Title prefix for all plots.",
    )
    parser.add_argument(
        "--combined-json",
        default="",
        help="Optional output path for merged summary JSON.",
    )
    return parser.parse_args()


def load_reports(report_paths: List[str]) -> List[Dict]:
    reports = []
    for p in report_paths:
        path = Path(p).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        reports.append(payload)
    return reports


def bar_plot(
    model_labels: List[str],
    values: List[float],
    metric_name: str,
    split_name: str,
    output_path: Path,
    title_prefix: str,
) -> None:
    x = np.arange(len(model_labels))
    fig, ax = plt.subplots(figsize=(9, 5))

    if metric_name == "accuracy":
        color = "#2E8B57"
        y_label = "Accuracy"
    elif metric_name == "wer":
        color = "#1F77B4"
        y_label = "Word Error Rate (WER)"
    else:
        color = "#D2691E"
        y_label = "Character Error Rate (CER)"

    bars = ax.bar(x, values, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix} - {metric_name.upper()} ({split_name})")
    ax.grid(True, axis="y", alpha=0.25)

    if metric_name == "accuracy":
        ax.set_ylim(0, 1.0)
    else:
        ymax = max(values) if values else 0.0
        ax.set_ylim(0, max(1.0, ymax * 1.15))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    reports = load_reports(args.reports)
    if len(reports) < 2:
        raise ValueError("Provide at least 2 report files for comparison.")

    target_splits = args.splits if args.splits else [args.split]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files_by_split: Dict[str, Dict[str, str]] = {}
    metrics_by_split: Dict[str, Dict[str, List[float]]] = {}
    model_labels: List[str] = []

    # Model labels are consistent across splits
    for report in reports:
        model_name = report.get("model_name") or Path(report["_path"]).stem
        model_labels.append(model_name)

    for split_name in target_splits:
        accuracy_vals = []
        wer_vals = []
        cer_vals = []

        for report in reports:
            metrics = report.get("metrics", {})
            split_metrics = metrics.get(split_name)
            if not split_metrics:
                raise ValueError(
                    f"Report {report['_path']} has no metrics for split '{split_name}'."
                )

            accuracy_vals.append(float(split_metrics["accuracy"]))
            wer_vals.append(float(split_metrics["wer"]))
            cer_vals.append(float(split_metrics["cer"]))

        metric_to_values = {
            "accuracy": accuracy_vals,
            "wer": wer_vals,
            "cer": cer_vals,
        }
        metrics_by_split[split_name] = metric_to_values

        created_files: Dict[str, str] = {}
        for metric_name, values in metric_to_values.items():
            output_path = output_dir / f"{args.prefix}_{metric_name}_{split_name}.png"
            bar_plot(
                model_labels=model_labels,
                values=values,
                metric_name=metric_name,
                split_name=split_name,
                output_path=output_path,
                title_prefix=args.title_prefix,
            )
            created_files[metric_name] = str(output_path)
            print(f"[Plot] {metric_name.upper()} ({split_name}) saved: {output_path}")
        created_files_by_split[split_name] = created_files

    if args.combined_json:
        combined_payload = {
            "created_at": datetime.now().isoformat(),
            "splits": target_splits,
            "models": model_labels,
            "metrics": metrics_by_split,
            "plots": created_files_by_split,
            "source_reports": [str(Path(r["_path"]).resolve()) for r in reports],
        }
        combined_path = Path(args.combined_json).resolve()
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")
        print(f"[Output] Combined summary JSON saved: {combined_path}")


if __name__ == "__main__":
    main()
