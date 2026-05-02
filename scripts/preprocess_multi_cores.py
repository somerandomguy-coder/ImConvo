import argparse
import os
import re
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

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
    parser.add_argument("--data_dir", default="./data/", help="Raw data root")
    parser.add_argument("--output_dir", default="./data/preprocessed/", help="NPY output folder")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="Number of CPU cores to use")
    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    align_output_dir = os.path.join(args.output_dir.rstrip('/'), "align")
    os.makedirs(align_output_dir, exist_ok=True)

    samples = discover_video_samples(args.data_dir)
    if not samples:
        print(f"No samples found in {args.data_dir}!")
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

    # Save manifest.txt
    manifest_path = os.path.join(args.output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(sorted(valid_names)))

    print(f"\n\n✓ Finished in {time.time() - start_time:.1f} seconds.")
    print(f"Total valid samples saved: {len(valid_names)}")

if __name__ == "__main__":
    main()
