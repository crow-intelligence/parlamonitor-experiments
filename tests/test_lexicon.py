from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.lexicon import (
    EKMAN,
    EKMAN_FILES,
    NEG_SUFFIX,
    Lexicon,
    count_hits,
    fold,
    load_lexicon,
    mark_negation,
    register_vocabulary,
    score_emotion,
    score_sentiment,
    without,
)

TAGS = {"n": "[/N][Nom]", "v": "[/V]", "a": "[/Adv]", "p": "[Punct]"}


def lex(name, single=(), multi=()):
    return Lexicon(name, frozenset(single), frozenset(multi), Path(name), "deadbeef")


def sentence(lemmas, tags=None):
    return (list(lemmas), list(tags or ["[/N][Nom]"] * len(lemmas)))


# --- folding ----------------------------------------------------------------


def test_folding_lowercases_but_keeps_accents():
    # The opposite of the loanword key, and deliberately: "őr" and "or" are
    # different words and the lists separate them.
    assert fold("Bátorság") == "bátorság"
    assert fold("ŐSZINTE") == "őszinte"
    assert fold("Őr") != fold("Or")


# --- negation ---------------------------------------------------------------


def test_forward_negation_marks_what_follows():
    assert mark_negation(["ez", "nem", "probléma", "."]) == [
        "ez",
        "nem",
        "probléma" + NEG_SUFFIX,
        ".",
    ]


def test_the_cue_itself_is_not_marked():
    assert "nem" + NEG_SUFFIX not in mark_negation(["nem", "jó"])


def test_scope_stops_at_a_clause_boundary():
    marked = mark_negation(["nem", "jó", ",", "hanem", "rossz"])
    assert marked[1].endswith(NEG_SUFFIX)
    assert not marked[4].endswith(NEG_SUFFIX)


@pytest.mark.parametrize("cue", ["nem", "sem", "se", "ne", "nincs", "sincs"])
def test_every_forward_cue_works(cue):
    assert mark_negation([cue, "jó"])[1].endswith(NEG_SUFFIX)


def test_nelkul_is_a_postposition_and_scopes_backwards():
    # 'pénz nélkül' -- the negated word comes BEFORE the cue. A forward-only
    # rule would mark the wrong half of the sentence.
    marked = mark_negation(["siker", "nélkül", "zárult"])
    assert marked[0].endswith(NEG_SUFFIX)
    assert not marked[2].endswith(NEG_SUFFIX)


def test_backward_scope_also_stops_at_punctuation():
    marked = mark_negation(["jó", ",", "siker", "nélkül"])
    assert not marked[0].endswith(NEG_SUFFIX)
    assert marked[2].endswith(NEG_SUFFIX)


@given(st.lists(st.sampled_from(["a", "nem", ".", "jó", "nélkül"]), max_size=12))
def test_marking_preserves_length_and_never_raises(lemmas):
    assert len(mark_negation(lemmas)) == len(lemmas)


# --- matching ---------------------------------------------------------------


def test_multiword_entries_are_matched_as_ngrams():
    count, matched = count_hits(["ez", "jó", "hír"], lex("joy", multi=[("jó", "hír")]))
    assert (count, matched) == (1, ["jó hír"])


def test_a_multiword_match_is_not_also_counted_as_single_words():
    joy = lex("joy", single=["jó", "hír"], multi=[("jó", "hír")])
    count, matched = count_hits(["jó", "hír"], joy)
    assert count == 1 and matched == ["jó hír"]


def test_longer_multiword_entries_win():
    nested = lex("x", multi=[("a", "b"), ("a", "b", "c")])
    assert count_hits(["a", "b", "c"], nested)[1] == ["a b c"]


# --- sentiment --------------------------------------------------------------


def test_a_plain_positive_word_counts_positive():
    result = score_sentiment(
        [sentence(["ez", "jó", "."], [TAGS["n"], TAGS["a"], TAGS["p"]])],
        lex("positive", ["jó"]),
        lex("negative", ["rossz"]),
    )
    assert (result.positive, result.negative) == (1, 0)


def test_a_negated_negative_counts_positive():
    # 'nem probléma' is not a complaint.
    result = score_sentiment(
        [sentence(["nem", "probléma", "."], [TAGS["a"], TAGS["n"], TAGS["p"]])],
        lex("positive", ["jó"]),
        lex("negative", ["probléma"]),
    )
    assert (result.positive, result.negative, result.flipped) == (1, 0, 1)


def test_a_negated_positive_counts_negative():
    result = score_sentiment(
        [sentence(["nem", "jó", "."], [TAGS["a"], TAGS["a"], TAGS["p"]])],
        lex("positive", ["jó"]),
        lex("negative", ["rossz"]),
    )
    assert (result.positive, result.negative, result.flipped) == (0, 1, 1)


def test_punctuation_is_excluded_from_the_denominator():
    result = score_sentiment(
        [sentence(["jó", ".", "!"], [TAGS["a"], TAGS["p"], TAGS["p"]])],
        lex("positive", ["jó"]),
        lex("negative", []),
    )
    assert result.n_tokens == 1


def test_polarity_ignores_how_much_other_text_surrounded_the_hits():
    long_text = ["szó"] * 50 + ["jó"]
    result = score_sentiment(
        [sentence(long_text, [TAGS["n"]] * 51)],
        lex("positive", ["jó"]),
        lex("negative", []),
    )
    assert result.polarity == 1.0  # balance of hits
    assert result.score < 0.05  # density among tokens


def test_polarity_of_a_text_with_no_hits_is_zero():
    result = score_sentiment(
        [sentence(["szó"])], lex("positive", ["jó"]), lex("negative", [])
    )
    assert result.polarity == 0.0


def test_balanced_weighting_offsets_an_asymmetric_lexicon():
    # A 1:4 list makes an unweighted count lean negative by construction.
    pos, neg = lex("positive", ["jó"]), lex("negative", ["a", "b", "c", "d"])
    sent = [sentence(["jó", "a"], [TAGS["a"], TAGS["n"]])]
    plain = score_sentiment(sent, pos, neg, balanced=False)
    weighted = score_sentiment(sent, pos, neg, balanced=True)
    assert weighted.score > plain.score
    assert weighted.balanced is True


def test_an_empty_text_raises():
    with pytest.raises(ValueError, match="no sentences"):
        score_sentiment([], lex("positive"), lex("negative"))


# --- emotion ----------------------------------------------------------------


def test_emotion_counts_and_dominant():
    lexicons = {"joy": lex("joy", ["öröm"]), "anger": lex("anger", ["düh"])}
    result = score_emotion(
        [sentence(["nagy", "öröm", "."], [TAGS["a"], TAGS["n"], TAGS["p"]])],
        lexicons,
        min_hits=1,
    )
    assert result.counts == {"joy": 1, "anger": 0}
    assert result.dominant == "joy"


def test_a_text_below_the_threshold_is_flagged_sparse():
    # 0.0 means both "no anger" and "no evidence"; the flag separates them.
    result = score_emotion([sentence(["szó"])], {"joy": lex("joy", ["öröm"])})
    assert result.sparse is True
    assert result.dominant is None


def test_negation_is_not_applied_to_emotion():
    # 'nem félek' is not joy, so there is nothing to flip; the fear word is
    # still counted as a fear word.
    result = score_emotion(
        [sentence(["nem", "félelem"], [TAGS["a"], TAGS["n"]])],
        {"fear": lex("fear", ["félelem"])},
        min_hits=1,
    )
    assert result.counts["fear"] == 1


def test_categories_may_overlap():
    shared = {
        "anger": lex("anger", ["gyűlölet"]),
        "disgust": lex("disgust", ["gyűlölet"]),
    }
    result = score_emotion([sentence(["gyűlölet"])], shared, min_hits=1)
    assert result.counts == {"anger": 1, "disgust": 1}


# --- the register filter ----------------------------------------------------


def test_register_filter_takes_the_top_share():
    freq = {"a": 100, "b": 50, "c": 10, "d": 1}
    assert sorted(register_vocabulary(freq, percentile=50)) == ["a", "b"]


def test_a_zero_percentile_disables_the_filter():
    assert register_vocabulary({"a": 1}, percentile=0) == frozenset()


def test_dropping_entries_reports_what_went():
    kept, dropped = without(lex("joy", ["jó", "öröm"]), frozenset({"jó"}))
    assert sorted(kept.single) == ["öröm"]
    assert dropped == ["jó"]


def test_multiword_entries_survive_the_register_filter():
    # A multiword expression is not register vocabulary however common its
    # parts are.
    kept, _ = without(lex("joy", ["jó"], [("jó", "hír")]), frozenset({"jó"}))
    assert ("jó", "hír") in kept.multi


def test_the_filter_keeps_the_original_provenance():
    kept, _ = without(lex("joy", ["jó", "öröm"]), frozenset({"jó"}))
    assert kept.sha256 == "deadbeef"  # still points at the file on disk


# --- the declared scheme ----------------------------------------------------


def test_every_ekman_category_has_a_file():
    assert set(EKMAN) == set(EKMAN_FILES)
    assert len(EKMAN) == 6


def test_the_non_ekman_categories_are_not_scored():
    assert not any("feszultseg" in f or "szeretet" in f for f in EKMAN_FILES.values())


def test_a_missing_lexicon_raises_rather_than_scoring_everything_neutral(tmp_path):
    with pytest.raises(FileNotFoundError, match="see data/lexicons"):
        load_lexicon(tmp_path / "nope.txt", "joy")
