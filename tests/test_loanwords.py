from pathlib import Path

import pytest

from parlamonitor.loanwords import Lexicon, load_lexicon, measure, proper_noun_tags


@pytest.fixture
def lexicon(tmp_path):
    path = tmp_path / "idegenszavak.txt"
    path.write_text("# comment\nabszurd\nkoalíció\n\nDEFICIT\n", encoding="utf-8")
    return load_lexicon(path)


def test_entries_are_case_folded_and_comments_skipped(lexicon):
    assert lexicon.entries == frozenset({"abszurd", "koalíció", "deficit"})
    assert lexicon.size == 3


def test_the_lexicon_identifies_itself_by_hash(lexicon):
    assert lexicon.lexicon_id.startswith("idegenszavak.txt@")
    assert len(lexicon.sha256) == 64


def test_a_missing_lexicon_raises_rather_than_scoring_everything_native(tmp_path):
    # An empty fallback would report every text as 0% foreign and look real.
    with pytest.raises(FileNotFoundError, match="saphes ships none"):
        load_lexicon(tmp_path / "nope.txt")


def test_an_empty_lexicon_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_lexicon(path)


# --- the proper-noun proxy --------------------------------------------------


def test_a_capitalised_lemma_is_treated_as_a_proper_noun():
    assert proper_noun_tags(["taps", "Fidesz", "kormány"]) == ["X", "PROPN", "X"]


def test_matching_is_case_insensitive(lexicon):
    assert measure(["deficit"], lexicon).matched == 1


def test_a_capitalised_match_is_excluded_not_counted(lexicon):
    # A foreign surname must not count as a loan word every time it is said.
    result = measure(["Abszurd", "javaslat"], lexicon)
    assert (result.matched, result.excluded) == (0, 1)


# --- the ratio --------------------------------------------------------------


def test_the_ratio_is_matches_over_counted_lemmas(lexicon):
    result = measure(["a", "javaslat", "abszurd"], lexicon)
    assert result.ratio == pytest.approx(1 / 3)
    assert sorted(result.matches) == ["abszurd"]


def test_a_text_with_no_foreign_lemmas_scores_zero(lexicon):
    assert measure(["a", "javaslat"], lexicon).ratio == 0.0


def test_the_lexicon_id_travels_with_the_result(lexicon):
    assert measure(["abszurd"], lexicon).lexicon_id == lexicon.lexicon_id


def test_an_empty_text_is_null_not_zero(lexicon):
    assert measure([], lexicon) is None


def test_a_text_of_only_proper_nouns_is_null_not_zero(lexicon):
    # Nothing was counted, so there is no ratio -- 0.0 would read as "native".
    assert measure(["Kovács", "Fidesz"], lexicon) is None


def test_a_raw_string_is_refused(lexicon):
    # Surface forms miss the lexicon and return a plausible, low ratio.
    with pytest.raises(TypeError, match="not a string"):
        measure("abszurd javaslat", lexicon)


def test_lexicon_can_be_built_without_a_file():
    result = measure(["abszurd"], Lexicon(frozenset({"abszurd"}), Path("x"), "ab", 1))
    assert result.ratio == 1.0
