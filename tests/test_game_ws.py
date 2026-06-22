import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

import api._game as game
from api.main import app


def _jpeg_b64() -> str:
    bgr = np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _patch_pipeline(monkeypatch, *, face=True, top_word="blue", confidence=0.95):
    monkeypatch.setattr(game, "preprocess_frames", lambda frames: (None if not face else np.zeros((1, 1), dtype=np.float32)))
    monkeypatch.setattr(
        game,
        "score_window",
        lambda frames, slot_index, **kw: {
            "top_word": top_word,
            "confidence": confidence,
            "topk": [{"word": top_word, "confidence": confidence}],
            "ranked_words": [top_word, "green", "red", "white"],
        },
    )


def test_ws_pass_on_correct_word(monkeypatch):
    _patch_pipeline(monkeypatch, top_word="blue", confidence=0.95)
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "blue"})
        result = None
        for _ in range(game.SCORE_EVERY_N_FRAMES + 1):
            ws.send_json({"type": "frame", "data": _jpeg_b64()})
            msg = ws.receive_json()
            if msg["type"] == "result":
                result = msg
                break
        assert result is not None
        assert result["pass"] is True
        assert result["word"] == "blue"


def test_ws_reports_no_face(monkeypatch):
    _patch_pipeline(monkeypatch, face=False)
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "blue"})
        for _ in range(game.SCORE_EVERY_N_FRAMES):
            ws.send_json({"type": "frame", "data": _jpeg_b64()})
        msg = ws.receive_json()
        assert msg["type"] == "progress"
        assert msg["face"] is False


def test_ws_rejects_bad_target(monkeypatch):
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "purple"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
