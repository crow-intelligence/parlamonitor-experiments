import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.readability import (
    HUNGARIAN_LONG_WORD_THRESHOLD,
    MAX_PLAUSIBLE_WORDS_PER_SENTENCE,
    is_long_word,
    measure,
)

# --- the Hungarian calibration ----------------------------------------------


def test_the_threshold_is_the_hungarian_one_not_the_swedish_default():
    # saphes.recommended_threshold("hu") is 8; LIX's familiar 6 is Swedish and
    # makes every Hungarian text look harder than it is.
    assert HUNGARIAN_LONG_WORD_THRESHOLD == 8


@pytest.mark.parametrize(
    ("word", "long"),
    [
        ("egészségügy", True),  # 11 characters, 9 letters
        ("asszony", False),  # 7 characters, 5 letters -- gy/sz/ny are one each
        ("dzsungel", False),  # 8 characters, 6 letters
        ("törvényjavaslatot", True),
        ("kormány", False),  # 7 characters, 6 letters
        ("a", False),
    ],
)
def test_length_is_counted_in_letters_not_characters(word, long):
    assert is_long_word(word) is long


def test_a_digraph_word_can_fall_below_the_line_a_character_count_would_cross():
    # 9 characters, but ny and sz are single letters, so 7 -- not long.
    assert len("asszonyok") == 9
    assert is_long_word("asszonyok") is False


# --- measurement ------------------------------------------------------------


def test_lix_matches_its_definition():
    # LIX = words/sentences + long_words/words*100
    words = ["egészségügy"] * 2 + ["a"] * 8
    result = measure(words, ["x"] * 10, n_sentences=2)
    assert result.n_long_words == 2
    assert result.lix == pytest.approx(10 / 2 + 2 / 10 * 100)


def test_rix_is_long_words_per_sentence():
    words = ["egészségügy"] * 4 + ["a"] * 6
    assert measure(words, ["x"] * 10, n_sentences=2).rix == pytest.approx(2.0)


def test_the_threshold_and_window_travel_with_the_result():
    result = measure(["a"] * 10, ["x"] * 10, n_sentences=2)
    assert result.long_word_threshold == HUNGARIAN_LONG_WORD_THRESHOLD
    assert result.mattr_window == 100


def test_a_short_text_reports_ttr_and_says_so():
    result = measure(["a"] * 10, ["x", "y"] * 5, n_sentences=2)
    assert result.mattr_windowed is False
    assert any("not windowed" in note for note in result.notes)


def test_a_long_text_reports_a_windowed_mattr():
    lemmas = [f"w{i % 40}" for i in range(300)]
    result = measure(["a"] * 300, lemmas, n_sentences=20)
    assert result.mattr_windowed is True


def test_an_empty_text_raises_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="no words"):
        measure([], [], n_sentences=1)


def test_zero_sentences_raises():
    with pytest.raises(ValueError, match="must be positive"):
        measure(["a"], ["a"], n_sentences=0)


def test_a_text_with_no_lemmas_scores_zero_diversity_and_says_so():
    result = measure(["a", "b"], [], n_sentences=1)
    assert result.mattr == 0.0
    assert any("no lemmas" in note for note in result.notes)


# --- the roll-call artefact -------------------------------------------------


def test_a_text_with_no_sentence_punctuation_is_flagged_unreliable():
    # The notary's roll-call: 590 words of names, one "sentence", LIX 594.
    result = measure(["Kovács"] * 590, ["kovács"] * 590, n_sentences=1)
    assert result.readability_reliable is False
    assert result.lix > 500


def test_ordinary_prose_is_reliable():
    result = measure(["szó"] * 400, ["szó"] * 400, n_sentences=20)
    assert result.words_per_sentence == 20
    assert result.readability_reliable is True


def test_the_flag_turns_exactly_at_the_documented_boundary():
    n = MAX_PLAUSIBLE_WORDS_PER_SENTENCE
    assert measure(["a"] * n, ["a"] * n, n_sentences=1).readability_reliable is True
    assert (
        measure(["a"] * (n + 1), ["a"] * (n + 1), n_sentences=1).readability_reliable
        is False
    )


@given(
    st.integers(min_value=1, max_value=500),
    st.integers(min_value=1, max_value=50),
)
def test_counts_are_always_consistent(n_words, n_sentences):
    words = ["egészségügy" if i % 3 == 0 else "a" for i in range(n_words)]
    result = measure(words, words, n_sentences=n_sentences)
    assert result.n_words == n_words
    assert 0 <= result.n_long_words <= n_words
    assert result.lix >= 0
    assert result.rix >= 0
