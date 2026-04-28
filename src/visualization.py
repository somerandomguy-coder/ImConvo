import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def save_loss_plot(history_path, output_dir="reports/plots"):
    """Read training history JSON and save a per-model summary plot."""
    if not os.path.exists(history_path):
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    if "avg_wer" in history:
        plt.subplot(1, 2, 2)
        plt.bar(
            ["WER", "CER"],
            [history["avg_wer"], history.get("avg_cer", np.nan)],
            color=["blue", "green"],
        )
        plt.title("Final Error Rates")
        plt.ylim(0, 1)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_summary.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


_MODEL_JSON_PATTERN = re.compile(r"^model_(.+)\.json$")
_WER_PATTERN = re.compile(r"Average Word Error Rate:\s*([0-9]*\.?[0-9]+)")
_CER_PATTERN = re.compile(r"Average Char Error Rate:\s*([0-9]*\.?[0-9]+)")
_SAMPLES_PATTERN = re.compile(r"Total Samples Evaluated:\s*(\d+)")
_DATE_PATTERN = re.compile(r"Date:\s*(.+)")


@dataclass
class ModelReport:
    key: str
    label: str
    history_path: Path
    loss: List[float]
    val_loss: List[float]
    learning_rate: List[float]
    avg_wer_json: Optional[float]
    avg_cer_json: Optional[float]
    eval_wer: Optional[float]
    eval_cer: Optional[float]
    eval_samples: Optional[int]
    eval_date: Optional[str]

    def metric_source(self) -> str:
        if self.eval_wer is not None and self.eval_cer is not None:
            return "eval_report"
        if self.avg_wer_json is not None and self.avg_cer_json is not None:
            return "json_history"
        return "missing"

    def chosen_wer(self) -> Optional[float]:
        if self.eval_wer is not None:
            return self.eval_wer
        return self.avg_wer_json

    def chosen_cer(self) -> Optional[float]:
        if self.eval_cer is not None:
            return self.eval_cer
        return self.avg_cer_json


def _sort_key_for_model_key(model_key: str) -> Tuple[int, object]:
    try:
        return (0, int(model_key))
    except ValueError:
        return (1, model_key)


def _to_float_list(values) -> List[float]:
    if not isinstance(values, list):
        return []
    return [float(v) for v in values]


def _to_float_or_none(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_matching_eval_report(report_dir: Path, model_key: str) -> Optional[Path]:
    exact = report_dir / f"eval_report_model_{model_key}.txt"
    if exact.exists():
        return exact

    matches = sorted(
        report_dir.glob(f"eval_report_model_{model_key}*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    return None


def _parse_eval_report(report_path: Path) -> Dict[str, Optional[object]]:
    text = report_path.read_text(encoding="utf-8")

    wer_match = _WER_PATTERN.search(text)
    cer_match = _CER_PATTERN.search(text)
    samples_match = _SAMPLES_PATTERN.search(text)
    date_match = _DATE_PATTERN.search(text)

    return {
        "wer": float(wer_match.group(1)) if wer_match else None,
        "cer": float(cer_match.group(1)) if cer_match else None,
        "samples": int(samples_match.group(1)) if samples_match else None,
        "date": date_match.group(1).strip() if date_match else None,
    }


def load_model_reports(report_dir: str = "temp_reports") -> List[ModelReport]:
    """Load model histories and matching eval reports from a report directory."""
    base = Path(report_dir)
    if not base.exists():
        return []

    model_reports: List[ModelReport] = []
    for history_path in base.glob("model_*.json"):
        match = _MODEL_JSON_PATTERN.match(history_path.name)
        if not match:
            continue

        model_key = match.group(1)
        history = json.loads(history_path.read_text(encoding="utf-8"))

        eval_path = _find_matching_eval_report(base, model_key)
        eval_metrics: Dict[str, Optional[object]] = {}
        if eval_path:
            eval_metrics = _parse_eval_report(eval_path)

        model_reports.append(
            ModelReport(
                key=model_key,
                label=f"Model {model_key}",
                history_path=history_path,
                loss=_to_float_list(history.get("loss", [])),
                val_loss=_to_float_list(history.get("val_loss", [])),
                learning_rate=_to_float_list(history.get("learning_rate", [])),
                avg_wer_json=_to_float_or_none(history.get("avg_wer")),
                avg_cer_json=_to_float_or_none(history.get("avg_cer")),
                eval_wer=_to_float_or_none(eval_metrics.get("wer")),
                eval_cer=_to_float_or_none(eval_metrics.get("cer")),
                eval_samples=eval_metrics.get("samples"),
                eval_date=eval_metrics.get("date"),
            )
        )

    model_reports.sort(key=lambda report: _sort_key_for_model_key(report.key))
    return model_reports


def _plot_series(ax, reports: List[ModelReport], attr_name: str, title: str, y_label: str):
    has_data = False
    for report in reports:
        series = getattr(report, attr_name)
        if not series:
            continue
        epochs = range(1, len(series) + 1)
        ax.plot(epochs, series, linewidth=2, label=report.label)
        has_data = True

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    if has_data:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def _annotate_bar_values(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_model_comparison_plot(
    report_dir: str = "temp_reports",
    output_path: Optional[str] = None,
    title: str = "Model Comparison",
) -> str:
    """
    Generate a comparison plot for all `model_*.json` files in `report_dir`.

    The output includes:
    - Train loss curves
    - Validation loss curves
    - Learning rate curves
    - WER/CER grouped bars (prefers eval report values, falls back to JSON)
    """
    reports = load_model_reports(report_dir)
    if not reports:
        raise ValueError(f"No model report JSON files found in: {report_dir}")

    if output_path is None:
        output_path = str(Path(report_dir) / "model_comparison.png")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_train, ax_val = axes[0]
    ax_lr, ax_metrics = axes[1]

    _plot_series(ax_train, reports, "loss", "Train Loss by Epoch", "Loss")
    _plot_series(ax_val, reports, "val_loss", "Validation Loss by Epoch", "Loss")
    _plot_series(ax_lr, reports, "learning_rate", "Learning Rate by Epoch", "Learning Rate")
    if any(report.learning_rate for report in reports):
        ax_lr.set_yscale("log")

    wer_values: List[float] = []
    cer_values: List[float] = []
    x_labels: List[str] = []

    for report in reports:
        wer = report.chosen_wer()
        cer = report.chosen_cer()

        wer_values.append(np.nan if wer is None else wer)
        cer_values.append(np.nan if cer is None else cer)

        source = report.metric_source()
        source_label = "eval" if source == "eval_report" else ("json" if source == "json_history" else "n/a")
        x_labels.append(f"{report.label}\n({source_label})")

    x = np.arange(len(reports))
    width = 0.35

    wer_bars = ax_metrics.bar(x - width / 2, wer_values, width, label="WER")
    cer_bars = ax_metrics.bar(x + width / 2, cer_values, width, label="CER")
    _annotate_bar_values(ax_metrics, wer_bars)
    _annotate_bar_values(ax_metrics, cer_bars)

    ax_metrics.set_title("Final WER/CER per Model")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(x_labels)
    ax_metrics.set_ylabel("Error Rate")
    ax_metrics.grid(True, axis="y", alpha=0.25)
    ax_metrics.legend()

    if np.all(np.isnan(wer_values)) and np.all(np.isnan(cer_values)):
        ax_metrics.text(0.5, 0.5, "No WER/CER metrics", ha="center", va="center", transform=ax_metrics.transAxes)
    else:
        metric_max = np.nanmax(np.array(wer_values + cer_values, dtype=float))
        ax_metrics.set_ylim(0, max(1.0, float(metric_max) * 1.15))

    summary_lines = ["Model summary:"]
    for report in reports:
        best_val = min(report.val_loss) if report.val_loss else None
        final_val = report.val_loss[-1] if report.val_loss else None
        samples = report.eval_samples if report.eval_samples is not None else "n/a"
        summary_lines.append(
            f"{report.label}: best_val={best_val:.4f}" if best_val is not None else f"{report.label}: best_val=n/a"
        )
        summary_lines[-1] += (
            f", final_val={final_val:.4f}" if final_val is not None else ", final_val=n/a"
        )
        summary_lines[-1] += f", samples={samples}, metric_source={report.metric_source()}"

    fig.suptitle(title, fontsize=16, y=0.99)
    fig.text(
        0.01,
        0.01,
        "\n".join(summary_lines),
        fontsize=8,
        family="monospace",
        va="bottom",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return str(output)
