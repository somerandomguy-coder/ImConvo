"""
Preprocess raw GRID videos and upload the results to ClearML.

Usage:
    python scripts/sync_dataset.py
"""

import argparse
import os
import sys

# Add project root so we can import src.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clearml import Dataset, Task

PROJECT_NAME = "ImConvo"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
ALIGN_DIR = os.path.join(DATA_DIR, "align")


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------


def run_preprocessing():
    """Run the full preprocessing pipeline (mpg → npy)."""
    from scripts.preprocess import preprocess_dataset

    print("=" * 60)
    print("Step 1/2 — Preprocessing videos")
    print("=" * 60)
    preprocess_dataset(data_dir=DATA_DIR, output_dir=PREPROCESSED_DIR)
    print()


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_dataset(dataset_name: str = "GRID_Preprocessed"):
    """
    Upload preprocessed .npy files + alignment .align files to ClearML.

    The uploaded dataset contains two top-level folders:
        preprocessed/   ← .npy frames + manifest.txt
        align/          ← .align label files
    """
    print("=" * 60)
    print("Step 2/2 — Uploading dataset to ClearML")
    print("=" * 60)

    # Validate that data exists
    if not os.path.isdir(PREPROCESSED_DIR):
        print(f"[ERROR] Preprocessed directory not found: {PREPROCESSED_DIR}")
        print("Run preprocessing first.")
        return
    if not os.path.isdir(ALIGN_DIR):
        print(f"[ERROR] Alignment directory not found: {ALIGN_DIR}")
        return

    npy_count = len([f for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npy")])
    align_count = len([f for f in os.listdir(ALIGN_DIR) if f.endswith(".align")])
    print(f"  Preprocessed files (.npy): {npy_count}")
    print(f"  Alignment files (.align): {align_count}")

    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=f"upload_dataset_{dataset_name}",
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=True,
    )

    ds = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=PROJECT_NAME,
        use_current_task=True,
    )

    # Add preprocessed .npy files (stored under "preprocessed/" in the dataset)
    print(f"\n  Adding preprocessed files from: {PREPROCESSED_DIR}")
    ds.add_files(
        path=PREPROCESSED_DIR,
        dataset_path="preprocessed",
    )

    # Add alignment files (stored under "align/" in the dataset)
    print(f"  Adding alignment files from:    {ALIGN_DIR}")
    ds.add_files(
        path=ALIGN_DIR,
        dataset_path="align",
    )

    print("\n  Finalizing and uploading (this may take a while)...")
    ds.finalize(auto_upload=True)

    dataset_id = ds.id
    task.close()

    print(f"\n{'=' * 60}")
    print(f"✓ Dataset uploaded successfully!")
    print(f"  Dataset name : {dataset_name}")
    print(f"  Dataset ID   : {dataset_id}")
    print(f"  Project      : {PROJECT_NAME}")
    print(f"{'=' * 60}")
    print(f"\nTo download this dataset later, run:")
    print(f"  python scripts/download_dataset.py {dataset_id}")

    return dataset_id


if __name__ == "__main__":
    run_preprocessing()
    upload_dataset()
