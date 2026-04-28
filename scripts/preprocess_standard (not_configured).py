import argparse
import os
import re
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAMES


def _copy_file(src, dst):
    """Helper to copy alignment files to the flat directory."""
    import shutil

    try:
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    except Exception as e:
        print(f"  [WARN] Failed to copy {src}: {e}")


def process_video_to_numpy(video_path):
    """
    Reads video, converts to grayscale, and normalizes.
    Returns a numpy array of shape (T, H, W, 1) or (T, H, W).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize to the target dimensions (80x120) 
        # Even though we aren't cropping, we must resize the full frame 
        # so the model gets the expected input shape.
        resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # 3. Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        frames.append(normalized)
    
    cap.release()
    
    if len(frames) == 0:
        return None
        
    frames = np.array(frames, dtype=np.float32)

    # --- Padding or Truncating (The "Reshape" logic) ---
    # This ensures every .npy file is exactly (75, 80, 120)
    T = frames.shape[0]
    if T < MAX_FRAMES:
        # Create zero padding for the remaining frames
        pad = np.zeros((MAX_FRAMES - T, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)
    else:
        # Truncate if the video is longer than 75 frames
        frames = frames[:MAX_FRAMES]
        
    return frames

def process_single_sample(sample_info, output_dir, align_output_dir, force):
    video_path, align_path, unique_name = sample_info
    output_path = os.path.join(output_dir, f"{unique_name}.npy")
    align_dst = os.path.join(align_output_dir, f"{unique_name}.align")

    if os.path.exists(output_path) and not force:
        _copy_file(align_path, align_dst)
        return unique_name

    try:
        # Perform the manual preprocessing instead of calling extract_lip_frames
        frames = process_video_to_numpy(video_path)

        if frames is not None:
            np.save(output_path, frames)
            _copy_file(align_path, align_dst)
            return unique_name
    except Exception as e:
        sys.stderr.write(f"\n[ERROR] {unique_name}: {str(e)}\n")

    return None


def discover_video_samples(data_dir):
    """Finds all s*_processed/*.mpg and matching align/*.align files."""
    samples = []
    if not os.path.exists(data_dir):
        return []

    # Matches directories like 's1' or 's1_processed'
    speaker_dirs = sorted(
        [
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and re.match(r"s\d+(_processed)?$", d)
        ]
    )

    for speaker in speaker_dirs:
        speaker_path = os.path.join(data_dir, speaker)
        # Check for 'align' folder inside the speaker folder
        align_dir = os.path.join(speaker_path, "align")
        if not os.path.isdir(align_dir):
            continue

        video_files = [f for f in os.listdir(speaker_path) if f.endswith(".mpg")]
        for vf in video_files:
            name = os.path.splitext(vf)[0]
            align_path = os.path.join(align_dir, f"{name}.align")
            if os.path.exists(align_path):
                samples.append(
                    (os.path.join(speaker_path, vf), align_path, f"{speaker}_{name}")
                )
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Multi-core Video Preprocessing (No Cropping)"
    )
    parser.add_argument("--data_dir", default="./data/", help="Raw data root")
    parser.add_argument(
        "--output_dir", default="./data/preprocessed/", help="NPY output folder"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--cores", type=int, default=11, help="Number of CPU cores"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    align_output_dir = os.path.join(
        os.path.dirname(args.output_dir.rstrip("/")), "align"
    )
    os.makedirs(align_output_dir, exist_ok=True)

    samples = discover_video_samples(args.data_dir)
    if not samples:
        print(f"No samples found in {args.data_dir}!")
        return

    print(f"--- Video Processing Pipeline ---")
    print(f"Found {len(samples)} samples. Utilizing {args.cores} cores.")

    worker_task = partial(
        process_single_sample,
        output_dir=args.output_dir,
        align_output_dir=align_output_dir,
        force=args.force,
    )

    start_time = time.time()
    valid_names = []

    try:
        with Pool(processes=args.cores) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_task, samples), 1):
                if result:
                    valid_names.append(result)
                if i % 50 == 0 or i == len(samples):
                    elapsed = time.time() - start_time
                    vps = i / elapsed
                    print(
                        f"\rProcessed {i}/{len(samples)} ({vps:.1f} videos/sec)",
                        end="",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\nStopping...")

    manifest_path = os.path.join(args.output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(sorted(valid_names)))

    print(f"\n\n✓ Finished in {time.time() - start_time:.1f} seconds.")


if __name__ == "__main__":
    main()
