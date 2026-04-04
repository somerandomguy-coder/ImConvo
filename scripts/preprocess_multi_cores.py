import argparse
import os
import re
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from clearml import Dataset, Task

# Add project root to path so we can find src.utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import extract_lip_frames


def _copy_file(src, dst):
    """Helper to copy alignment files to the flat directory."""
    import shutil
    try:
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    except Exception as e:
        print(f"  [WARN] Failed to copy {src}: {e}")

def process_single_sample(sample_info, output_dir, align_output_dir, force):
    """
    Worker function: This runs in a separate process for each video.
    """
    video_path, align_path, unique_name = sample_info
    output_path = os.path.join(output_dir, f"{unique_name}.npy")
    align_dst = os.path.join(align_output_dir, f"{unique_name}.align")

    # 1. Skip if already exists (unless --force)
    if os.path.exists(output_path) and not force:
        # Ensure the alignment file is also there
        _copy_file(align_path, align_dst)
        return unique_name

    # 2. Process Video
    try:
        frames = extract_lip_frames(video_path)
        if frames is not None:
            np.save(output_path, frames)
            _copy_file(align_path, align_dst)
            return unique_name
    except Exception as e:
        # We print to stderr so it shows up even during multiprocessing
        sys.stderr.write(f"\n[ERROR] {unique_name}: {str(e)}\n")
    
    return None

def discover_video_samples(data_dir):
    """Finds all s*_processed/*.mpg and matching align/*.align files."""
    samples = []
    if not os.path.exists(data_dir):
        return []

    speaker_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and re.match(r"s\d+(_processed)?$", d)
    ])

    for speaker in speaker_dirs:
        speaker_path = os.path.join(data_dir, speaker)
        align_dir = os.path.join(speaker_path, "align")
        if not os.path.isdir(align_dir):
            continue

        video_files = [f for f in os.listdir(speaker_path) if f.endswith(".mpg")]
        for vf in video_files:
            name = os.path.splitext(vf)[0]
            align_path = os.path.join(align_dir, f"{name}.align")
            if os.path.exists(align_path):
                samples.append((
                    os.path.join(speaker_path, vf), 
                    align_path, 
                    f"{speaker}_{name}"
                ))
    return samples
def main():
    

    parser = argparse.ArgumentParser(description="Multi-core GRID Preprocessing")
    # We define defaults, but ClearML will override these if changed in the UI
    parser.add_argument("--parent", default="ad79af81ff2e44368fa8384a5d96577e", help="Input download task id")
    parser.add_argument("--output_dir", default="./data/preprocessed/", help="NPY output folder")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="CPU cores")
    args = parser.parse_args()

    # 1. Initialize ClearML Task
    task = Task.init(
        project_name="ImConvo",
        task_name="preprocess_dataset",
        task_type=Task.TaskTypes.data_processing,
    )

    # 2. Link Arguments to UI
    task.connect(args)

    # 3. Get Parent task and its artifact (path)
    parent_tasks = Task.get_tasks(task_ids=[args.parent]) # This node have only 1 parent 
    
    if parent_tasks:
        # We assume the first parent is the Download task
        download_task = parent_tasks[0]
        print(f"[ClearML] Pipeline detected! Parent Task: {download_task.id}")
        
        # Fetch the path artifact from the Download task
        dataset_path_artifact = download_task.artifacts.get('dataset_path')
        if dataset_path_artifact:
            # Artifacts are often stored as dictionaries or objects
            # Based on our previous script, it's {'path': '...'}
            artifact_data = dataset_path_artifact.get()
            raw_data_dir = artifact_data.get('path')
            print(f"✓ Found path from parent artifact: {raw_data_dir}")

    # 4. Fallback/Manual Logic
    if not raw_data_dir:
        # If no parent or artifact found, we fallback to the Dataset ID 
        # (Make sure to add this to your argparse if you want to use it manually)
        print(f"--- Fetching Raw Data via Dataset ID ---")
        dataset_id = "98ddafeae8e64c359f89d435b52bea0f" # Or get from args
        ds = Dataset.get(dataset_id=dataset_id)
        raw_data_dir = ds.get_local_copy()

    # Final Check
    if not raw_data_dir or not os.path.exists(raw_data_dir):
        print(f"[ERROR] Could not locate raw data directory!")
        return
        



    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    align_output_dir = os.path.join(args.output_dir, "align")
    os.makedirs(align_output_dir, exist_ok=True)

    # ... [keep your discover_video_samples and Pool logic] ...
    # Replace args.data_dir with raw_data_dir in your logic
    samples = discover_video_samples(raw_data_dir)
    
    samples = discover_video_samples(raw_data_dir)
    if not samples:
        print(f"No samples found in {raw_data_dir}!")
        return

    print(f"--- GRID Multi-Core Preprocessing ---")
    print(f"Found {len(samples)} samples.")
    print(f"Utilizing {args.cores} cores.")

    # Prepare worker arguments
    worker_task = partial(
        process_single_sample, 
        output_dir=args.output_dir, 
        align_output_dir=align_output_dir, 
        force=args.force
    )

    start_time = time.time()
    valid_names = []

    # ... [Pool execution logic] ...
    # Execution Pool
    try:
        with Pool(processes=args.cores) as pool:
            # imap_unordered is faster as it doesn't wait for sequence order
            for i, result in enumerate(pool.imap_unordered(worker_task, samples), 1):
                if result:
                    valid_names.append(result)
                
                if i % 50 == 0 or i == len(samples):
                    elapsed = time.time() - start_time
                    vps = i / elapsed
                    print(f"\rProcessed {i}/{len(samples)} ({vps:.1f} videos/sec)", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping... Saving current manifest.")


    # 4. Finalizing and Reporting
    manifest_path = os.path.join(args.output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(sorted(valid_names)))

    # Upload manifest as artifact so the Train node knows what's ready
    task.upload_artifact("manifest", artifact_object=manifest_path)
    # We do NOT upload the 100GB .npy files to ClearML (saving your free tier!)
    
    print(f"\n✓ Preprocessing Complete. Manifest saved.")
    task.close()

if __name__ == "__main__":
    main()
