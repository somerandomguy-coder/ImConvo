from api._game import (
    SLOT_NAMES,
    LETTER_SLOT_INDEX,
    PASS_CONFIDENCE_THRESHOLD,
    RELAX_AFTER_ATTEMPTS,
    RELAX2_AFTER_ATTEMPTS,
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


def test_pass_strict_normal_slot_requires_top1_and_threshold():
    # default (attempts=0): top-1 == target AND confidence >= threshold
    high = PASS_CONFIDENCE_THRESHOLD + 0.1
    low = PASS_CONFIDENCE_THRESHOLD - 0.1
    assert evaluate_pass(1, "blue", ["blue", "green", "red", "white"], high) is True
    assert evaluate_pass(1, "blue", ["blue", "green", "red", "white"], low) is False   # confidence too low
    assert evaluate_pass(1, "blue", ["green", "blue", "red", "white"], high) is False  # not top-1


def test_pass_strict_letter_slot_allows_top3_no_confidence_gate():
    ranked = ["c", "e", "b", "a", "d", "f"]
    assert LETTER_SLOT_INDEX == 3
    # letters: in top-3, confidence ignored
    assert evaluate_pass(3, "b", ranked, 0.05) is True   # b is rank 3
    assert evaluate_pass(3, "a", ranked, 0.99) is False  # a is rank 4 (out of strict top-3)


def test_pass_relaxes_after_three_failed_attempts():
    high = PASS_CONFIDENCE_THRESHOLD + 0.1
    # before relaxation: rank-2 still fails
    assert evaluate_pass(1, "blue", ["green", "blue", "red", "white"], high, attempts=RELAX_AFTER_ATTEMPTS - 1) is False
    # at relaxation threshold (3 prior fails): rank-2 passes
    assert evaluate_pass(1, "blue", ["green", "blue", "red", "white"], high, attempts=RELAX_AFTER_ATTEMPTS) is True
    # letters: rank-5 passes only after relaxation
    ranked_letters = ["c", "e", "b", "a", "d", "f"]
    assert evaluate_pass(3, "d", ranked_letters, attempts=0) is False
    assert evaluate_pass(3, "d", ranked_letters, attempts=RELAX_AFTER_ATTEMPTS) is True


def test_pass_relaxes_again_after_five_failed_attempts():
    high = PASS_CONFIDENCE_THRESHOLD + 0.1
    # rank-3 fails under tier-1 (top-2) but passes under tier-2 (top-3)
    ranked = ["green", "red", "blue", "white"]
    assert evaluate_pass(1, "blue", ranked, high, attempts=RELAX_AFTER_ATTEMPTS) is False
    assert evaluate_pass(1, "blue", ranked, high, attempts=RELAX2_AFTER_ATTEMPTS) is True
    # letters: rank-7 passes only after tier-2 (top-8)
    ranked_letters = list("cebadfghi")  # 9 letters; "h" at index 6 = rank 7
    assert evaluate_pass(3, "h", ranked_letters, attempts=RELAX_AFTER_ATTEMPTS) is False
    assert evaluate_pass(3, "h", ranked_letters, attempts=RELAX2_AFTER_ATTEMPTS) is True
