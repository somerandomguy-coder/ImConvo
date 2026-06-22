"""Live per-word game: vocab/slot maps, pass-rule, preprocessing, scoring."""
from __future__ import annotations

from experiments.isolated_word_level.common import SLOT_VOCABS, WORD_TO_IDX

SLOT_NAMES: tuple[str, ...] = (
    "command",
    "color",
    "preposition",
    "letter",
    "digit",
    "adverb",
)

# Tunables — single source of truth.
BUFFER_FRAMES = 25
TARGET_FRAMES = 25
SCORE_EVERY_N_FRAMES = 3
PASS_THRESHOLD = 0.6
LETTER_SLOT_INDEX = 3
LETTER_TOPK_PASS = 3


def slot_candidate_indices(slot_index: int) -> list[int]:
    """Return the FLAT_VOCAB class indices for a given GRID slot."""
    return [WORD_TO_IDX[w] for w in SLOT_VOCABS[slot_index]]


def evaluate_pass(
    slot_index: int,
    target: str,
    ranked_words: list[str],
    confidence: float,
) -> bool:
    """Decide whether a scored window passes the round.

    ranked_words: the slot's candidate words sorted by probability desc.
    confidence:   probability mass of ranked_words[0] (the top word).
    """
    if slot_index == LETTER_SLOT_INDEX:
        return target in ranked_words[:LETTER_TOPK_PASS]
    return bool(ranked_words and ranked_words[0] == target and confidence >= PASS_THRESHOLD)
