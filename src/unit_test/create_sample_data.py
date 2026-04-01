import os
import sys

# Get the absolute path of the 'unit_test' directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path of the project root (one level up)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Append the project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np

from src.utils import extract_lip_frames


def create_mirror_sample(src_root, dest_root, samples_per_speaker=2, max_speakers=2):
    """
    Creates a tiny version of the GRID dataset structure for testing.
    Structure:
    dest_root/
      s1_processed/
        align/
          video1.align
        video1.mpg
      preprocessed/
        align/
          s1_processed_video1.align
        s1_processed_video1.npy
        manifest.txt
    """
    src_root = os.path.abspath(src_root)
    dest_root = os.path.abspath(dest_root)

    if not os.path.exists(src_root):
        print(f"[ERROR] Source data path {src_root} does not exist.")
        return

    print(f"--- Creating Sample Dataset Mirror at: {dest_root} ---")

    # 1. Identify speaker directories
    speakers = sorted(
        [
            d
            for d in os.listdir(src_root)
            if os.path.isdir(os.path.join(src_root, d)) and d.startswith("s")
        ]
    )[:max_speakers]

    # Create the preprocessed flat dir structure used by dataset.py
    preprocessed_dir = os.path.join(dest_root, "preprocessed")
    preprocessed_align_dir = os.path.join(preprocessed_dir, "align")
    os.makedirs(preprocessed_align_dir, exist_ok=True)

    manifest_entries = []

    for speaker in speakers:
        src_speaker_path = os.path.join(src_root, speaker)
        dest_speaker_path = os.path.join(dest_root, speaker)
        os.makedirs(os.path.join(dest_speaker_path, "align"), exist_ok=True)

        # Find .mpg videos
        videos = sorted([f for f in os.listdir(src_speaker_path) if f.endswith(".mpg")])
        sample_videos = videos[:samples_per_speaker]

        valid_names = []
        for video in sample_videos:
            name = os.path.splitext(video)[0]
            unique_name = f"{speaker}_{name}"

            # Paths for raw sample
            v_src = os.path.join(src_speaker_path, video)
            v_dest = os.path.join(dest_speaker_path, video)
            a_src = os.path.join(src_speaker_path, "align", f"{name}.align")
            a_dest = os.path.join(dest_speaker_path, "align", f"{name}.align")

            # Link Raw Data
            for src, dst in [(v_src, v_dest), (a_src, a_dest)]:
                if os.path.exists(src) and not os.path.exists(dst):
                    os.symlink(src, dst)

            # -- REAL DATA ---
            npy_dest = os.path.join(preprocessed_dir, f"{unique_name}.npy")
            try:
                frames = extract_lip_frames(v_src)
                if frames is None:
                    # Face detection failed — skip this sample
                    continue
                np.save(npy_dest, frames)
                valid_names.append(unique_name)


            except Exception as e:
                print(f"  [ERROR] {unique_name}: {e}")
                continue

            prep_align_dest = os.path.join(
                preprocessed_align_dir, f"{unique_name}.align"
            )
            # Link alignment to the flat preprocessed/align folder
            if os.path.exists(a_src) and not os.path.exists(prep_align_dest):
                os.symlink(a_src, prep_align_dest)

            manifest_entries.append(unique_name)
            print(f"  [OK] Processed {unique_name}")

    # Write manifest.txt
    with open(os.path.join(preprocessed_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(sorted(manifest_entries)))

    print(f"\n✓ Sample data ready in {dest_root}")


if __name__ == "__main__":
    # Adjust these to your local paths
    REAL_DATA = "../../data/"
    SAMPLE_DATA = "./data_sample"
    create_mirror_sample(REAL_DATA, SAMPLE_DATA)
