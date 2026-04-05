import argparse
import hashlib
import os

from clearml import Dataset, Task

PROJECT_NAME = "ImConvo"




def main(args):
    # 1. Initialize Task
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name="dataset_ingestion",
        task_type=Task.TaskTypes.data_processing,
    )
    taskID = task.task_id

    # 2. UI-Editable Configuration
    # This allows you to change the ID or path directly in the ClearML Web UI
    
    print('Arguments: {}'.format(args))

    data_dir = args.persistent_data_dir
    remote = args.remote
    if remote:
        task.execute_remotely()
        data_dir = "/home/nam/.remote/" # Persistent directory because after finish a task the data got wipe out

        

    print(f"--- Synchronizing Dataset: {args.dataset_id} ---")

    # 3. ClearML Managed Sync
    # This automatically handles basic 'presence' checks and downloads missing files
    ds = Dataset.get(dataset_id=args.dataset_id)

    from pathlib import Path

    # This gets the absolute path of the directory where your script is currently located
    data_root = Path(data_dir).absolute() 
    DEFAULT_DATA_DIR = data_root / "data" 

    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    # We use get_mutable_local_copy so you can preprocess into the same folder later
    local_path = ds.get_mutable_local_copy(target_folder=DEFAULT_DATA_DIR)

    # 5. Output the path for the Pipeline
    # This allows the 'Preprocess' node to know where the data landed
    task.upload_artifact("dataset_path", artifact_object={"path": local_path})
    task.upload_artifact("taskID", artifact_object={"id": taskID})

    print(f"\n✓ Dataset ready at: {local_path}")
    task.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, nargs="?", help="ClearML dataset ID")
    parser.add_argument("--persistent_data_dir", default="./../..", type=str, help="Data directory")
    parser.add_argument("--remote", action="store_true", help="Overwrite existing files")
    cli_args = parser.parse_args()
    main(cli_args)
