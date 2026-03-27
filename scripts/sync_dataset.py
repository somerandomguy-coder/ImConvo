import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Dataset, Task

PROJECT_NAME = "ImConvo"
DATA_FOLDER = "./data/"


def download_dataset(dataset_id) -> str:
    task = Task.init(project_name=PROJECT_NAME, task_name="download_dataset")

    # get metadata nua
    ds = Dataset.get(dataset_id=dataset_id)

    local_copy = ds.get_mutable_local_copy(DATA_FOLDER)

    task.close()
    print(f"local_copy feature path: {local_copy}")
    return str(local_copy)


def upload_dataset(dataset_path, name):
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=f"upload_dataset {name}",
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=True,
    )
    # set metadata nua
    ds = Dataset.create(
        dataset_name=name, dataset_project=PROJECT_NAME, use_current_task=True
    )
    ds.add_files(path=dataset_path)
    ds.finalize(auto_upload=True)
    task.close()
    print("Succesfully upload dataset!")



if __name__ == "__main__":
    download_dataset("22622e6dcfa948f1ba1d3de1666fd068")
