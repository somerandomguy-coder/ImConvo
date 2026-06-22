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


def test_pass_normal_slot_requires_top1_and_threshold():
    # color slot (index 1): top word must equal target AND clear threshold
    assert evaluate_pass(1, "blue", ["blue", "green", "red", "white"], 0.9) is True
    assert evaluate_pass(1, "blue", ["blue", "green", "red", "white"], 0.4) is False
    assert evaluate_pass(1, "blue", ["green", "blue", "red", "white"], 0.9) is False


def test_pass_letter_slot_allows_top3():
    ranked = ["c", "e", "b", "a", "d"]
    assert LETTER_SLOT_INDEX == 3
    assert evaluate_pass(3, "b", ranked, 0.2) is True   # b is in top-3
    assert evaluate_pass(3, "a", ranked, 0.2) is False  # a is 4th
