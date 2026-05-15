"""Pure helper utilities — no model state, no FastAPI deps."""
from __future__ import annotations

import base64
import os
import platform
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import tensorflow as tf

from src import BLANK_IDX, CHAR_LIST, SPACE_IDX

ROOT_DIR = Path(__file__).resolve().parent.parent
PREVIEW_DIR = ROOT_DIR / "api" / "preview_cache"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _levenshtein(a: list, b: list) -> int:
    r, h = len(a), len(b)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): d[i][0] = i
    for j in range(h + 1): d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[r][h]


def compute_wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return _levenshtein(r, h) / max(len(r), 1)


def compute_cer(ref: str, hyp: str) -> float:
    return _levenshtein(list(ref), list(hyp)) / max(len(ref), 1)


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

def get_device_specs() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    gpu_names: list[str] = []
    gpu_mem_mb: int | None = None

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            gpu_names.append(tf.config.experimental.get_device_details(gpu).get("device_name") or gpu.name)
        except Exception:
            gpu_names.append(gpu.name)

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        if out.returncode == 0:
            mems = []
            for line in out.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    if not gpu_names: gpu_names.append(parts[0])
                    if parts[1].isdigit(): mems.append(int(parts[1]))
            if mems: gpu_mem_mb = sum(mems)
    except Exception:
        pass

    cpu = None
    try:
        if os.path.exists("/proc/cpuinfo"):
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    cpu = cpu or platform.processor() or None

    return {
        "cpu_model": cpu,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1024**3, 2),
        "ram_available_gb": round(vm.available / 1024**3, 2),
        "gpu_names": gpu_names,
        "gpu_memory_total_mb": gpu_mem_mb,
        "tf_version": tf.__version__,
        "device_used": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def get_video_metadata(path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return {"width": None, "height": None, "fps": None, "frame_count": None, "duration_sec": None}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_raw) if fps_raw and fps_raw > 0 else None
    fc_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fc = int(fc_raw) if fc_raw and fc_raw > 0 else None
    cap.release()
    return {"width": w, "height": h, "fps": fps, "frame_count": fc,
            "duration_sec": (fc / fps) if fps and fc else None}


def build_preview(temp_path: str, original_name: str | None) -> str | None:
    suffix = Path(original_name or "").suffix.lower()
    out = PREVIEW_DIR / f"{uuid.uuid4().hex}.mp4"
    if suffix == ".mp4":
        try: shutil.copyfile(temp_path, out); return f"/preview/{out.name}"
        except Exception: return None
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", temp_path,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-c:a", "aac", "-b:a", "128k", str(out)],
            capture_output=True, text=True, check=False, timeout=60,
        )
        if r.returncode == 0 and out.exists(): return f"/preview/{out.name}"
    except Exception:
        pass
    return None


def encode_sample_frames(frames: np.ndarray, n: int = 6) -> list[str]:
    T = frames.shape[0]
    indices = [int(i * (T - 1) / (n - 1)) for i in range(n)]
    result = []
    for idx in indices:
        gray = (frames[idx] * 255).astype(np.uint8)
        display = cv2.resize(gray, (480, 320), interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode(".png", display)
        result.append(f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode()}")
    return result


def token_from_index(idx: int) -> str:
    if idx == BLANK_IDX: return "<blank>"
    if idx == SPACE_IDX: return "<space>"
    if 0 <= idx < len(CHAR_LIST): return CHAR_LIST[idx]
    return f"<unk:{idx}>"
