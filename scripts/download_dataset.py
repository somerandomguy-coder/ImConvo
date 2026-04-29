"""
Download a ClearML dataset to the local data directory.

Usage:
    python scripts/download_dataset.py <dataset_id>
"""

import argparse
import os

from clearml import Dataset, Task

PROJECT_NAME = "ImConvo"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_dataset(dataset_id: str):
    """Download a ClearML dataset to the local data directory."""
    print("=" * 60)
    print(f"Downloading dataset: {dataset_id}")
    print("=" * 60)

    task = Task.init(
        project_name=PROJECT_NAME,
        task_name="download_dataset",
    )

    ds = Dataset.get(dataset_id=dataset_id)
    local_copy = ds.get_mutable_local_copy(DATA_DIR)

    task.close()
    print(f"\n✓ Dataset downloaded to: {local_copy}")
    return str(local_copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a ClearML dataset"
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        help="ClearML dataset ID to download",
    )
    args = parser.parse_args()

    download_dataset(args.dataset_id)
