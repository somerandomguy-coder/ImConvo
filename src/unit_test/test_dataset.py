# This is for fedora only
import os

# 1. Force X11 to stop the Wayland/Gnome warning
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 2. Silence the Font/Qt Logging (This stops the "Cannot find font directory" spam)
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.conf=false"

# 3. Suppress TensorFlow's own info/warning messages (optional but keeps things clean)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys

# Get the absolute path of the 'unit_test' directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path of the project root (one level up)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Append the project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import numpy as np
import tensorflow as tf

from src.dataset import create_ctc_dataset, discover_samples
from src.utils import (FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAMES,
                       parse_alignment_chars)


def test_dataset_pipeline(data_dir, preprocessed_dir):
    print("--- 1. Testing Sample Discovery ---")
    samples = discover_samples(data_dir, preprocessed_dir)
    print(f"Total samples found: {len(samples)}")

    if len(samples) == 0:
        print("ERROR: No samples found. Check your paths!")
        return

    # 2. Pick a random sample to check alignment parsing
    test_npy, test_align = samples[np.random.randint(len(samples))]
    print(f"Testing sample: {os.path.basename(test_npy)}")

    char_indices, length = parse_alignment_chars(test_align)
    print(f"Parsed Labels: {char_indices}")
    print(f"Label Length: {length}")
    
    # ... inside test_dataset_pipeline ...
    char_indices, length = parse_alignment_chars(test_align)
    print(f"Parsed Labels: {char_indices}")
    print(f"Label Length: {length}")
    
    # ADD THIS:
    from src.utils import char_indices_to_text

    # We only decode up to 'length' to ignore the padding (27s)
    decoded_text = char_indices_to_text(char_indices[:length])
    print(f"Decoded Text: '{decoded_text}'")

    # --- 3. Testing tf.data Pipeline ---
    print("\n--- 2. Testing tf.data Dataset (Batching & Shapes) ---")
    batch_size = 2
    # We only take a subset for the test to avoid 100GB overhead
    subset_paths = [s[0] for s in samples[:4]]
    subset_labels = []
    subset_lengths = []

    for s in samples[:4]:
        lbl, ln = parse_alignment_chars(s[1])
        subset_labels.append(lbl)
        subset_lengths.append(ln)

    ds = create_ctc_dataset(
        subset_paths,
        np.array(subset_labels),
        np.array(subset_lengths),
        batch_size=batch_size,
        shuffle=False,
    )

    # Grab one batch
    for frames, labels_dict in ds.take(1):
        labels = labels_dict["labels"]
        lengths = labels_dict["label_length"]

        print(f"Batch Frames Shape: {frames.shape}")  # Expect (2, MAX_FRAMES, H, W, 1)
        print(f"Batch Labels Shape: {labels.shape}")
        print(f"Batch Lengths Shape: {lengths.shape}")

        # Verification logic
        assert frames.shape == (batch_size, MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1)
        print("SUCCESS: Batch shapes are correct.")

        # --- 4. Visual Inspection (The cv2 part) ---
        print("\n--- 3. Visualizing Frames (cv2.imshow) ---")
        print("Press any key to see next frame, 'q' to stop visualization.")

        # Take the first video in the batch
        video = frames[0].numpy()

        for i, frame in enumerate(video):
            # Frames are likely 0-1 or -1 to 1 float; convert to 0-255 for display
            display_frame = (frame * 255).astype(np.uint8)

            # Remove channel dim if it's grayscale (H, W, 1) -> (H, W)
            if display_frame.shape[-1] == 1:
                display_frame = np.squeeze(display_frame, axis=-1)

            cv2.imshow("Dataset Test - Press Key", display_frame)
            cv2.putText(
                display_frame,
                f"Frame: {i}",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255),
                1,
            )

            key = cv2.waitKey(50)  # 50ms delay per frame
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        print("Visualization complete.")


if __name__ == "__main__":
    # SET YOUR PATHS HERE
    # Get the directory where THIS script (test_dataset.py) lives
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build absolute paths based on that location
    DATA_DIR = os.path.join(script_dir, "data_sample")
    PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

    print(f"Checking data in: {DATA_DIR}")
    print(f"Checking preprocessed in: {PREPROCESSED_DIR}")

    if not os.path.exists(PREPROCESSED_DIR):
        print(f"ERROR: Path not found: {PREPROCESSED_DIR}")
    else:
        test_dataset_pipeline(DATA_DIR, PREPROCESSED_DIR)
