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
from src import MODEL_VARIANTS, NUM_CHARS, build_lipreading_ctc, char_indices_to_text
from src.utils import *

DEFAULT_CHECKPOINT_PATH = "./checkpoints/best_ctc_model.keras"


def get_args():
    parser = argparse.ArgumentParser(description="LipNet Real-time Inference")
    parser.add_argument(
        "--ip",
        type=str,
        required=False,
        default=None,
        help=(
            "IP Webcam URL (e.g., http://192.168.0.69:8080). "
            "If not provided, uses laptop webcam."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint path (.keras/.weights).",
    )
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default=None,
        help="Optional temporal backbone override. If omitted, infer from checkpoint name.",
    )
    return parser.parse_args()


def infer_variant_from_checkpoint_path(checkpoint_path: str) -> str:
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0].lower()
    if stem.endswith("_transformer_medium"):
        return "transformer_medium"
    if stem.endswith("_conformer_lite"):
        return "conformer_lite"
    if stem.endswith("_tcn"):
        return "tcn"
    if stem.endswith("_bilstm"):
        return "bilstm"
    if stem.endswith("_bigru"):
        return "bigru"
    if stem.endswith("_transformer"):
        return "transformer"
    if stem.endswith("_gru"):
        return "gru"
    return "bigru"


def load_model_for_inference(
    checkpoint_path: str,
    model_variant_override: str | None,
):
    inferred_variant = infer_variant_from_checkpoint_path(checkpoint_path)
    if model_variant_override:
        candidates = [model_variant_override.lower()]
    else:
        candidates = [inferred_variant]
        if "bigru" not in candidates:
            candidates.append("bigru")
        candidates.extend([v for v in MODEL_VARIANTS if v not in candidates])

    if not os.path.exists(checkpoint_path):
        fallback_variant = candidates[0]
        model = build_lipreading_ctc(model_variant=fallback_variant, num_chars=NUM_CHARS)
        _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
        print(f"[WARN] Checkpoint not found at {checkpoint_path}. Using random weights.")
        print(f"[Info] Initialized variant='{fallback_variant}'.")
        return model, fallback_variant

    load_errors = []
    for variant in candidates:
        try:
            model = build_lipreading_ctc(model_variant=variant, num_chars=NUM_CHARS)
            _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
            model.load_weights(checkpoint_path)
            return model, variant
        except Exception as exc:
            load_errors.append(f"{variant}: {type(exc).__name__}: {exc}")
            if model_variant_override:
                break

    joined = " | ".join(load_errors[:5]) if load_errors else "unknown"
    raise RuntimeError(
        "Failed to load checkpoint with available variants. "
        f"Tried {candidates}. Errors: {joined}"
    )


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (120, 80)) 
    frame = frame / 255.0 
    return frame

def main():
    args = get_args()
    if args.ip:
        if args.ip.isdigit():
            video_source = int(args.ip)
            print("💻 Using camera index:", video_source)
        else:
            stream_url = args.ip if args.ip.endswith("/video") else f"{args.ip}/video"
            video_source = stream_url
            print("📱 Using IP Webcam stream:", video_source)
    else:
        video_source = 0
        print("💻 Using laptop webcam")

    # 1. Load Model
    checkpoint_path = args.checkpoint_path
    model, resolved_variant = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        model_variant_override=args.model_variant,
    )
    print(f"[Info] Model variant: {resolved_variant}")
    if os.path.exists(checkpoint_path):
        print(f"✓ Model weights loaded from: {checkpoint_path}")

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
