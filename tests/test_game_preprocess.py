import base64

import cv2
import numpy as np

import api._game as game


def _jpeg_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_decode_jpeg_roundtrip():
    bgr = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    out = game.decode_jpeg(_jpeg_b64(bgr))
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8


def test_decode_jpeg_bad_input_raises():
    import pytest

    with pytest.raises(ValueError):
        game.decode_jpeg("not-base64-jpeg")


def test_preprocess_frames_shapes_when_face_found(monkeypatch):
    # Force a detected lip bbox so we exercise the crop/resize/pad path.
    monkeypatch.setattr(game, "_mediapipe_lip_bbox", lambda frame: (0, 0, frame.shape[1], frame.shape[0]))
    frames = [np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8) for _ in range(10)]
    arr = game.preprocess_frames(frames)
    assert arr is not None
    assert arr.shape == (1, game.TARGET_FRAMES, game.FRAME_HEIGHT, game.FRAME_WIDTH, 1)
    assert arr.dtype == np.float32
    assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0


def test_preprocess_frames_returns_none_when_no_face(monkeypatch):
    monkeypatch.setattr(game, "_mediapipe_lip_bbox", lambda frame: None)
    frames = [np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8) for _ in range(5)]
    assert game.preprocess_frames(frames) is None
