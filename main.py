import os

# Point ClearML to the local config file before importing it
os.environ["CLEARML_CONFIG_FILE"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clearml.conf")

from sync_dataset import upload_dataset, download_dataset

path = "./data/"

# upload_dataset(path, "og_dataset")
download_dataset("22622e6dcfa948f1ba1d3de1666fd068")
