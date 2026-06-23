import base64, time, cv2, numpy as np
from fastapi.testclient import TestClient
import api._game as game
from api.main import app

def _jpeg() -> str:
    bgr = np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr); assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _patch_pass(monkeypatch, top_word="blue"):
    monkeypatch.setattr(game, "preprocess_frames", lambda f: np.zeros((1,1), dtype=np.float32))
    monkeypatch.setattr(game, "score_window", lambda f, s, **k: {
        "top_word": top_word, "confidence": 0.99,
        "topk": [{"word": top_word, "confidence": 0.99}],
        "ranked_words": [top_word, "green", "red", "white"],
    })

def test_pass_requires_streak_and_warmup(monkeypatch):
    _patch_pass(monkeypatch, top_word="blue")
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "blue"})
        # First scoring window during warmup: must emit progress, not result
        for _ in range(game.SCORE_EVERY_N_FRAMES):
            ws.send_json({"type": "frame", "data": _jpeg()})
        msg = ws.receive_json()
        assert msg["type"] == "progress", msg
        # Wait past warmup, then send enough to complete the streak
        time.sleep((game.PASS_WARMUP_MS / 1000.0) + 0.05)
        result = None
        # Send batches of SCORE_EVERY_N_FRAMES frames, receive after each batch
        batches_needed = game.PASS_STREAK_REQUIRED + 1  # need 2 more matches plus one extra
        for batch_idx in range(batches_needed):
            for _ in range(game.SCORE_EVERY_N_FRAMES):
                ws.send_json({"type": "frame", "data": _jpeg()})
            m = ws.receive_json()
            if m["type"] == "result":
                result = m
                break
        assert result is not None and result["pass"] is True and result["word"] == "blue"

def test_no_pass_when_wrong_word(monkeypatch):
    _patch_pass(monkeypatch, top_word="green")  # wrong target
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "blue"})
        time.sleep((game.PASS_WARMUP_MS / 1000.0) + 0.05)
        saw_pass = False
        # Send several batches and check for pass
        for batch_idx in range(7):
            for _ in range(game.SCORE_EVERY_N_FRAMES):
                ws.send_json({"type": "frame", "data": _jpeg()})
            m = ws.receive_json()
            if m["type"] == "result" and m["pass"]:
                saw_pass = True
                break
        assert not saw_pass

def test_streak_resets_on_no_match(monkeypatch):
    import api._game as game_mod
    import time as _time
    client = TestClient(app)
    monkeypatch.setattr(game_mod, "preprocess_frames", lambda f: np.zeros((1,1), dtype=np.float32))
    state = {"top": "blue"}

    def patched(f, s, **k):
        return {
            "top_word": state["top"],
            "confidence": 0.99,
            "topk": [{"word": state["top"], "confidence": 0.99}],
            "ranked_words": [state["top"], "green", "red", "white"],
        }
    monkeypatch.setattr(game_mod, "score_window", patched)

    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "blue"})
        _time.sleep((game_mod.PASS_WARMUP_MS / 1000.0) + 0.05)

        # one matching window after warmup -> streak should be 1, no pass yet
        for _ in range(game_mod.SCORE_EVERY_N_FRAMES):
            ws.send_json({"type": "frame", "data": _jpeg()})
        msg = ws.receive_json()
        assert msg["type"] == "progress", msg

        # now switch to wrong word -> streak resets to 0
        state["top"] = "green"
        for _ in range(game_mod.SCORE_EVERY_N_FRAMES):
            ws.send_json({"type": "frame", "data": _jpeg()})
        msg = ws.receive_json()
        assert msg["type"] == "progress", msg

        # back to right word -> need 2 fresh consecutive matches before pass
        state["top"] = "blue"
        for _ in range(game_mod.SCORE_EVERY_N_FRAMES):
            ws.send_json({"type": "frame", "data": _jpeg()})
        msg = ws.receive_json()
        # streak == 1 again, still progress
        assert msg["type"] == "progress", msg

        for _ in range(game_mod.SCORE_EVERY_N_FRAMES):
            ws.send_json({"type": "frame", "data": _jpeg()})
        msg = ws.receive_json()
        # streak hits 2 -> pass
        assert msg["type"] == "result" and msg["pass"] is True
