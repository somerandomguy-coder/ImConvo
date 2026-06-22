from api._game import (
    SLOT_NAMES,
    LETTER_SLOT_INDEX,
    slot_candidate_indices,
    evaluate_pass,
)
from experiments.isolated_word_level.common import SLOT_VOCABS, WORD_TO_IDX


def test_slot_names_match_grid_positions():
    assert SLOT_NAMES == ("command", "color", "preposition", "letter", "digit", "adverb")
    assert len(SLOT_NAMES) == len(SLOT_VOCABS)


def test_slot_candidate_indices_color():
    color_idx = SLOT_NAMES.index("color")
    expected = [WORD_TO_IDX[w] for w in SLOT_VOCABS[color_idx]]
    assert slot_candidate_indices(color_idx) == expected
    assert len(slot_candidate_indices(color_idx)) == 4


def test_pass_normal_slot_allows_top2():
    # easy mode: non-letter slots pass if the target is within the top-2 guesses
    assert evaluate_pass(1, "blue", ["blue", "green", "red", "white"], 0.9) is True   # top-1
    assert evaluate_pass(1, "blue", ["green", "blue", "red", "white"], 0.4) is True   # top-2
    assert evaluate_pass(1, "blue", ["green", "red", "blue", "white"], 0.9) is False  # rank 3


def test_pass_letter_slot_allows_top5():
    ranked = ["c", "e", "b", "a", "d", "f"]
    assert LETTER_SLOT_INDEX == 3
    assert evaluate_pass(3, "d", ranked) is True   # d is rank 5 (within top-5)
    assert evaluate_pass(3, "f", ranked) is False  # f is rank 6
