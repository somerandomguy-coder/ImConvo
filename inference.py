import os
import sys

if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("\n[!] WARNING: You are not running in a virtual environment.")
    print("Please run 'source .venv/bin/activate' or use './setup_and_run.sh'\n")
    sys.exit(1)

import cv2
import numpy as np
import tensorflow as tf
import argparse

# Your local imports
from src import LipReadingCTC, char_indices_to_text
from src.utils import *



def get_args():
    parser = argparse.ArgumentParser(description="LipNet Real-time Inference")
    parser.add_argument("--ip", type=str, required=False, default=None,
                        help="IP Webcam URL (e.g., http://192.168.0.69:8080). "
                             "If not provided, uses laptop webcam.")
    return parser.parse_args()

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (120, 80)) 
    frame = frame / 255.0 
    return frame

def main():
    args = get_args()
    if args.ip:
        stream_url = args.ip if args.ip.endswith("/video") else f"{args.ip}/video"
        video_source = stream_url
        print("📱 Using IP Webcam stream:", video_source)
    else:
        video_source = 0
        print("💻 Using laptop webcam")

    # 1. Load Model
    checkpoint_path = "./checkpoints/best_ctc_model.keras"
    model = LipReadingCTC(num_chars=28)
    # Warm up
    _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("✓ Model weights loaded.")

    # 2. Setup Video
    cap = cv2.VideoCapture(video_source)
    frame_buffer = []
    BUFFER_SIZE = 75
    prediction_text = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- Step A: Process Mouth ---
        mouth_roi = frame[LIP_Y_START:LIP_Y_END, LIP_X_START:LIP_X_END]
        
        if mouth_roi.size > 0:
            processed_f = preprocess_frame(mouth_roi)
            frame_buffer.append(processed_f)
        
        if len(frame_buffer) > BUFFER_SIZE:
            frame_buffer.pop(0)

        # --- Step B: Inference (Only if buffer is full) ---
        if len(frame_buffer) == BUFFER_SIZE:
            input_tensor = np.expand_dims(frame_buffer, axis=(0, -1))
            input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
            
            logits = model(input_tensor, training=False)
            decoded = model.decode_greedy(logits)
            
            pred_indices = decoded[0]
            pred_indices = pred_indices[pred_indices >= 0]
            prediction_text = char_indices_to_text(pred_indices.tolist())

        # --- Step C: Visual Overlays ---
        # 1. Mirror for the user
        display_frame = cv2.flip(frame, 1)
        h, w, _ = display_frame.shape

        # 2. Draw Mouth Bounding Box (Note: flipped X coordinates)
        # Formula: w - start_x to w - end_x because of the flip
        cv2.rectangle(display_frame, (w-LIP_X_END, LIP_Y_START), (w-LIP_X_START, LIP_Y_END), (0, 255, 255), 2)
        cv2.putText(display_frame, "PLACE LIPS HERE", (w-LIP_X_END, LIP_Y_START-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 3. Buffer Progress Bar (Visual timing)
        progress = len(frame_buffer) / BUFFER_SIZE
        bar_width = int(w * progress)
        color = (0, 255, 0) if len(frame_buffer) == BUFFER_SIZE else (0, 165, 255)
        cv2.rectangle(display_frame, (0, h-10), (bar_width, h), color, -1)

        # 4. Status and Prediction
        status = "READY" if len(frame_buffer) == BUFFER_SIZE else f"BUFFERING: {len(frame_buffer)}/{BUFFER_SIZE}"
        cv2.rectangle(display_frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(display_frame, f"TEXT: {prediction_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('LipNet Live Monitor', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()