#!/usr/bin/env python3
"""Live latency benchmark for TCN lipreading models.

Primary use:
  - rolling 29-frame inference windows
  - inference every N frames (default: 5)
  - reports model latency + end-to-end latency + display FPS

This is latency-first tooling. Predictions are meaningful only when the
model and dataset domain match.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from lipreading.dataloaders import get_preprocessing_pipelines
from lipreading.model import Lipreading
from lipreading.utils import load_model, read_txt_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live latency benchmark for TCN lipreading")
    parser.add_argument("--source", default="0", help="Camera index or stream URL")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Execution device")
    parser.add_argument("--model-path", default="./models/tiny_lrw_snv05x_tcn1x.pth", help="Checkpoint path")
    parser.add_argument("--config-path", default="./configs/lrw_snv05x_tcn1x.json", help="JSON config path")
    parser.add_argument("--label-path", default="./labels/500WordsSortedList.txt", help="Class label file path")
    parser.add_argument("--num-classes", type=int, default=500, help="Model output classes")
    parser.add_argument("--buffer-size", type=int, default=29, help="Rolling frame window size")
    parser.add_argument("--infer-every", type=int, default=5, help="Run model every N incoming frames")
    parser.add_argument("--max-windows", type=int, default=200, help="Stop after this many inference windows")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to print")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV UI display")
    parser.add_argument("--synthetic", action="store_true", help="Benchmark with random synthetic inputs")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested capture width")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested capture height")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_path: Path) -> tuple[dict, dict, dict]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    tcn_options = {}
    if cfg.get("tcn_num_layers", ""):
        tcn_options = {
            "num_layers": cfg["tcn_num_layers"],
            "kernel_size": cfg["tcn_kernel_size"],
            "dropout": cfg["tcn_dropout"],
            "dwpw": cfg["tcn_dwpw"],
            "width_mult": cfg["tcn_width_mult"],
        }

    densetcn_options = {}
    if cfg.get("densetcn_block_config", ""):
        densetcn_options = {
            "block_config": cfg["densetcn_block_config"],
            "growth_rate_set": cfg["densetcn_growth_rate_set"],
            "reduced_size": cfg["densetcn_reduced_size"],
            "kernel_size_set": cfg["densetcn_kernel_size_set"],
            "dilation_size_set": cfg["densetcn_dilation_size_set"],
            "squeeze_excitation": cfg["densetcn_se"],
            "dropout": cfg["densetcn_dropout"],
        }

    return cfg, tcn_options, densetcn_options


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[Lipreading, list[str]]:
    config_path = Path(args.config_path)
    model_path = Path(args.model_path)
    label_path = Path(args.label_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    cfg, tcn_options, densetcn_options = load_config(config_path)
    labels = read_txt_lines(str(label_path))

    model = Lipreading(
        modality="video",
        num_classes=args.num_classes,
        tcn_options=tcn_options,
        densetcn_options=densetcn_options,
        backbone_type=cfg["backbone_type"],
        relu_type=cfg["relu_type"],
        width_mult=cfg["width_mult"],
        use_boundary=cfg.get("use_boundary", False),
        extract_feats=False,
    ).to(device)

    model = load_model(str(model_path), model, map_location=device)
    model.eval()
    return model, labels


def parse_source(source: str) -> int | str:
    if source.isdigit():
        return int(source)
    if source.startswith("http") and not source.endswith("/video"):
        return f"{source}/video"
    return source


def detect_mouth_bbox(frame: np.ndarray, face_cascade: cv2.CascadeClassifier) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        x0 = x + int(0.20 * fw)
        x1 = x + int(0.80 * fw)
        y0 = y + int(0.65 * fh)
        y1 = y + int(0.95 * fh)
    else:
        x0 = int(0.30 * w)
        x1 = int(0.70 * w)
        y0 = int(0.55 * h)
        y1 = int(0.90 * h)

    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1


def preprocess_clip(buffer: deque[np.ndarray], preprocess_func) -> np.ndarray:
    clip = np.stack(buffer, axis=0)  # (T, 96, 96), uint8
    return preprocess_func(clip.copy())


def run_inference(
    model: Lipreading,
    preprocess_func,
    buffer: deque[np.ndarray],
    device: torch.device,
    labels: list[str],
    topk: int,
) -> tuple[str, list[str], float, float]:
    start_total = time.perf_counter()
    clip = preprocess_clip(buffer, preprocess_func)
    input_tensor = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_model = time.perf_counter()
    with torch.no_grad():
        logits = model(input_tensor, lengths=[clip.shape[0]])
        probs = torch.softmax(logits, dim=1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_ms = (time.perf_counter() - start_model) * 1000.0

    e2e_ms = (time.perf_counter() - start_total) * 1000.0

    k = min(topk, probs.numel())
    top_probs, top_idx = torch.topk(probs, k=k)

    pred_idx = int(top_idx[0].item())
    pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)

    top_lines = []
    for p, i in zip(top_probs.tolist(), top_idx.tolist()):
        label = labels[i] if i < len(labels) else str(i)
        top_lines.append(f"{label}:{p:.3f}")

    return pred_label, top_lines, model_ms, e2e_ms


def summarize(latencies: list[float], name: str) -> None:
    if not latencies:
        print(f"{name}: no samples")
        return
    arr = np.array(latencies, dtype=np.float64)
    print(f"{name}: mean={arr.mean():.2f} ms, p95={np.percentile(arr, 95):.2f} ms, n={len(arr)}")


def run_synthetic(args: argparse.Namespace, model: Lipreading, preprocess_func, device: torch.device, labels: list[str]) -> None:
    model_lat_ms: list[float] = []
    e2e_lat_ms: list[float] = []

    buffer = deque(maxlen=args.buffer_size)
    for _ in range(args.buffer_size):
        buffer.append(np.random.randint(0, 256, (96, 96), dtype=np.uint8))

    for window_idx in range(1, args.max_windows + 1):
        buffer.append(np.random.randint(0, 256, (96, 96), dtype=np.uint8))
        _, _, model_ms, e2e_ms = run_inference(model, preprocess_func, buffer, device, labels, args.topk)
        model_lat_ms.append(model_ms)
        e2e_lat_ms.append(e2e_ms)

        if window_idx % 20 == 0 or window_idx == args.max_windows:
            print(f"Synthetic windows: {window_idx}/{args.max_windows}")

    print("\nSynthetic benchmark complete")
    summarize(model_lat_ms, "Model latency")
    summarize(e2e_lat_ms, "End-to-end latency")


def run_live(args: argparse.Namespace, model: Lipreading, preprocess_func, device: torch.device, labels: list[str]) -> None:
    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frame_buffer: deque[np.ndarray] = deque(maxlen=args.buffer_size)

    model_lat_ms: list[float] = []
    e2e_lat_ms: list[float] = []

    prediction = ""
    topk_line = ""
    model_ms_last = 0.0
    e2e_ms_last = 0.0

    last_frame_time = time.perf_counter()
    fps = 0.0
    frame_idx = 0
    windows_done = 0

    last_bbox = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.perf_counter()
        dt = now - last_frame_time
        last_frame_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        frame_idx += 1

        if last_bbox is None or frame_idx % 3 == 0:
            last_bbox = detect_mouth_bbox(frame, face_cascade)
        x0, y0, x1, y1 = last_bbox

        mouth = frame[y0:y1, x0:x1]
        if mouth.size > 0:
            gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
            frame_buffer.append(gray.astype(np.uint8))

        if len(frame_buffer) == args.buffer_size and frame_idx % args.infer_every == 0:
            prediction, top_lines, model_ms_last, e2e_ms_last = run_inference(
                model, preprocess_func, frame_buffer, device, labels, args.topk
            )
            topk_line = " | ".join(top_lines)
            model_lat_ms.append(model_ms_last)
            e2e_lat_ms.append(e2e_ms_last)
            windows_done += 1

            if windows_done % 10 == 0 or windows_done == args.max_windows:
                print(
                    f"Windows {windows_done}/{args.max_windows} "
                    f"model={model_ms_last:.2f}ms e2e={e2e_ms_last:.2f}ms pred={prediction}"
                )

            if windows_done >= args.max_windows:
                break

        if not args.no_display:
            display = frame.copy()
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)

            overlay_lines = [
                f"device={device}",
                f"buffer={len(frame_buffer)}/{args.buffer_size}",
                f"windows={windows_done}/{args.max_windows}",
                f"fps={fps:.1f}",
                f"model={model_ms_last:.2f} ms",
                f"e2e={e2e_ms_last:.2f} ms",
                f"pred={prediction}",
                topk_line[:120],
            ]

            y = 20
            for line in overlay_lines:
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 255, 30), 1, cv2.LINE_AA)
                y += 20

            cv2.imshow("TCN Live Latency Benchmark", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("\nLive benchmark complete")
    summarize(model_lat_ms, "Model latency")
    summarize(e2e_lat_ms, "End-to-end latency")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model, labels = build_model(args, device)
    preprocess_func = get_preprocessing_pipelines("video")["test"]

    # warmup for steadier latency numbers
    dummy = np.random.randint(0, 256, (args.buffer_size, 96, 96), dtype=np.uint8)
    dummy = preprocess_func(dummy.copy())
    with torch.no_grad():
        dummy_t = torch.from_numpy(dummy).float().unsqueeze(0).unsqueeze(0).to(device)
        _ = model(dummy_t, lengths=[dummy.shape[0]])

    if args.synthetic:
        run_synthetic(args, model, preprocess_func, device, labels)
    else:
        run_live(args, model, preprocess_func, device, labels)


if __name__ == "__main__":
    main()
