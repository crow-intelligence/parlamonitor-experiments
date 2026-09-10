import pytest

from parlamonitor.emtsv import Token
from parlamonitor.syntax import (
    EMTSV_PARSER,
    MIN_SENTENCE_LENGTH,
    PARSER,
    from_spacy_repaired,
    measure,
    measure_doc,
    to_dep_tokens,
)


def emtsv_sentence(heads, pos=None):
    """Build an emtsv sentence from 1-based head indices."""
    pos = pos or ["NOUN"] * len(heads)
    return [
        Token("w", "w", "[/N]", upostag=p, dep_id=i + 1, head=h)
        for i, (h, p) in enumerate(zip(heads, pos, strict=True))
    ]


class Tok:
    """Minimal stand-in for a spaCy token; saphes and this module duck-type."""

    def __init__(self, i, pos="NOUN"):
        self.i, self.pos_, self.head = i, pos, None

    @property
    def is_punct(self):
        return self.pos_ == "PUNCT"


def spacy_doc(sentences):
    class Doc:
        sents = sentences

    return Doc()


# --- adapting emtsv ---------------------------------------------------------


def test_punctuation_is_flagged_from_the_universal_tag():
    sentence = emtsv_sentence([2, 0, 0], ["DET", "VERB", "PUNCT"])
    assert [t.is_punct for t in to_dep_tokens(sentence)] == [False, False, True]


def test_a_sentence_with_no_parse_raises_rather_than_scoring_zero():
    # head defaults to -1, which is not the same as being a root.
    sentence = [Token("a", "a", "[/N]"), Token("b", "b", "[/N]")]
    with pytest.raises(ValueError, match="no dependency parse"):
        to_dep_tokens(sentence)


# --- the metric -------------------------------------------------------------


def test_mdd_matches_a_hand_computed_example():
    # 1->2 (distance 1), 2->3 (distance 1), 3 is root, 4 is punctuation.
    sentence = emtsv_sentence([2, 3, 0, 0], ["DET", "NOUN", "VERB", "PUNCT"])
    result = measure([sentence])
    assert result.mdd == pytest.approx(1.0)
    assert result.n_tokens == 3


def test_longer_dependencies_score_higher():
    near = emtsv_sentence([2, 3, 0, 3], ["DET", "NOUN", "VERB", "NOUN"])
    far = emtsv_sentence([4, 4, 4, 0], ["DET", "NOUN", "NOUN", "VERB"])
    assert measure([far]).mdd > measure([near]).mdd


def test_short_sentences_are_discarded_and_counted():
    long_one = emtsv_sentence([2, 3, 0, 3], ["DET", "NOUN", "VERB", "NOUN"])
    short = emtsv_sentence([0, 0], ["ADV", "PUNCT"])
    result = measure([long_one, short])
    assert result.n_sentences == 1
    assert result.n_sentences_discarded == 1


def test_a_text_of_only_short_sentences_is_null_not_zero():
    # "Köszönöm." has no measurable syntax; 0.0 would put it at the easy end.
    result = measure([emtsv_sentence([0, 0], ["ADV", "PUNCT"])])
    assert result.mdd is None and result.mhd is None


def test_the_minimum_length_is_the_documented_one():
    assert MIN_SENTENCE_LENGTH == 3


def test_an_empty_text_raises():
    with pytest.raises(ValueError, match="no sentences"):
        measure([])


# --- provenance -------------------------------------------------------------


def test_the_parser_is_recorded_and_the_two_are_distinguished():
    # Head conventions decide every distance, so the two must never be mixed.
    assert PARSER != EMTSV_PARSER
    assert measure([emtsv_sentence([2, 3, 0], ["DET", "NOUN", "VERB"])]).parser == (
        EMTSV_PARSER
    )


# --- the spaCy path and its repair ------------------------------------------


def test_a_spacy_doc_is_measured_and_labelled_with_the_spacy_parser():
    a, b, c = Tok(0, "DET"), Tok(1, "NOUN"), Tok(2, "VERB")
    a.head, b.head, c.head = b, c, c
    result = measure_doc(spacy_doc([[a, b, c]]))
    assert result.mdd == pytest.approx(1.0)
    assert result.parser == PARSER


def test_a_head_outside_its_sentence_becomes_a_local_root_and_is_counted():
    a, b, c = Tok(0), Tok(1), Tok(2)
    a.head, b.head, c.head = a, c, c  # b's governor is in the next sentence
    parses, repaired = from_spacy_repaired(spacy_doc([[a, b], [c]]))
    assert [(t.index, t.head) for t in parses[0]] == [(1, 0), (2, 0)]
    assert repaired == 1


def test_a_consistent_parse_needs_no_repair():
    a, b, c = Tok(0), Tok(1), Tok(2)
    a.head, b.head, c.head = b, c, c
    assert from_spacy_repaired(spacy_doc([[a, b, c]]))[1] == 0


def test_indices_are_renumbered_within_each_sentence():
    a, b, c, d = Tok(0), Tok(1), Tok(2), Tok(3)
    a.head, b.head, c.head, d.head = b, b, d, d
    parses, _ = from_spacy_repaired(spacy_doc([[a, b], [c, d]]))
    assert [t.index for t in parses[1]] == [1, 2]


def test_the_repair_count_reaches_the_result():
    a, b, c = Tok(0), Tok(1), Tok(2)
    a.head, b.head, c.head = a, c, c
    assert measure_doc(spacy_doc([[a, b], [c]])).n_heads_repaired == 1


def test_something_without_sentences_is_refused():
    with pytest.raises(TypeError, match="sents"):
        from_spacy_repaired(object())
