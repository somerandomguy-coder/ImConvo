import argparse
import hashlib
import os

from clearml import Dataset, Task

PROJECT_NAME = "ImConvo"
# We'll use a relative path that works inside the repo structure
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data"
)


def main(dataset_id=None):
    # 1. Initialize Task
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name="dataset_ingestion",
        task_type=Task.TaskTypes.data_processing,
    )

    # 2. UI-Editable Configuration
    # This allows you to change the ID or path directly in the ClearML Web UI
    args = {
        "dataset_id": dataset_id,
        "destination_dir": DEFAULT_DATA_DIR,
    }
    task.connect(args)

    print(f"--- Synchronizing Dataset: {args['dataset_id']} ---")

    # 3. ClearML Managed Sync
    # This automatically handles basic 'presence' checks and downloads missing files
    ds = Dataset.get(dataset_id=args["dataset_id"])

    # We use get_mutable_local_copy so you can preprocess into the same folder later
    local_path = ds.get_mutable_local_copy(target_path=args["destination_dir"])

    # 5. Output the path for the Pipeline
    # This allows the 'Preprocess' node to know where the data landed
    task.upload_artifact("dataset_path", artifact_object={"path": local_path})

    print(f"\n✓ Dataset ready at: {local_path}")
    task.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id", type=str, nargs="?", help="ClearML dataset ID")
    cli_args = parser.parse_args()
    main(cli_args.dataset_id)
