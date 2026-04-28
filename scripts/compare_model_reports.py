#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization import save_model_comparison_plot


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a comparison plot from model report files "
            "(model_*.json and optional eval_report_model_*.txt)."
        )
    )
    parser.add_argument(
        "--report-dir",
        default="temp_reports",
        help="Directory containing model report artifacts (default: temp_reports).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: <report-dir>/model_comparison.png).",
    )
    parser.add_argument(
        "--title",
        default="Model Comparison",
        help="Figure title.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = save_model_comparison_plot(
        report_dir=args.report_dir,
        output_path=args.output,
        title=args.title,
    )
    print(f"Comparison plot saved to: {output_path}")


if __name__ == "__main__":
    main()
