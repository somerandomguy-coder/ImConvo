import numpy as np

import api._game as game
from experiments.isolated_word_level.common import WORD_TO_IDX, NUM_WORD_CLASSES


class _FakeClassifier:
    """Returns fixed logits so scoring is deterministic."""

    def __init__(self, logits: np.ndarray):
        self._logits = logits

    def predict_logits(self, frames: np.ndarray) -> np.ndarray:
        return self._logits


def _logits_favoring(word: str) -> np.ndarray:
    logits = np.zeros((1, NUM_WORD_CLASSES), dtype=np.float32)
    logits[0, WORD_TO_IDX[word]] = 10.0
    return logits


def test_score_window_restricts_to_slot_and_picks_top():
    frames = np.zeros((1, game.TARGET_FRAMES, game.FRAME_HEIGHT, game.FRAME_WIDTH, 1), dtype=np.float32)
    fake = _FakeClassifier(_logits_favoring("blue"))
    out = game.score_window(frames, slot_index=1, classifier=fake)  # color slot
    assert out["top_word"] == "blue"
    assert out["confidence"] > 0.9
    assert set(out["ranked_words"]) == {"blue", "green", "red", "white"}
    assert out["ranked_words"][0] == "blue"


def test_score_window_confidence_is_softmax_over_slot_only():
    frames = np.zeros((1, game.TARGET_FRAMES, game.FRAME_HEIGHT, game.FRAME_WIDTH, 1), dtype=np.float32)
    # Equal logits across the 4 colors -> ~0.25 each.
    logits = np.full((1, NUM_WORD_CLASSES), 0.0, dtype=np.float32)
    for w in ("blue", "green", "red", "white"):
        logits[0, WORD_TO_IDX[w]] = 1.0
    out = game.score_window(frames, slot_index=1, classifier=_FakeClassifier(logits))
    assert abs(out["confidence"] - 0.25) < 0.05
