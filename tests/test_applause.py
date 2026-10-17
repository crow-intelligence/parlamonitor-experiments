from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.applause import (
    line_kinds,
    placed_reactions,
    sentence_features,
    side_relation,
    tokens_by_sentence,
)


def record(text, sentences, prefix=None):
    return {"text": text, "sentences": sentences, "speaker_prefix": prefix}


def test_applause_lands_on_the_sentence_before_it():
    rec = record(
        "Első mondat. Ez a lényeg. (Taps a TISZA soraiból.) Harmadik.",
        ["Első mondat.", "Ez a lényeg.", "Harmadik."],
    )
    [p] = placed_reactions(rec)
    assert (p.sentence_index, p.interrupted, p.closing) == (1, False, False)
    assert p.relative_position == 0.5


def test_a_heckle_mid_sentence_is_an_interruption_with_its_cut():
    rec = record(
        "Ezt a törvényt (Közbeszólás a Fidesz soraiból: Hazugság!) most elfogadjuk.",
        ["Ezt a törvényt most elfogadjuk."],
    )
    [p] = placed_reactions(rec)
    assert p.interrupted and p.cut == "Ezt a törvényt"
    assert p.quote == "Hazugság!"


def test_the_closing_formula_is_ritual_even_before_the_last_sentence():
    rec = record(
        "Ennyi. Köszönöm szépen a megtisztelő figyelmüket. (Taps.) Elnök úr!",
        ["Ennyi.", "Köszönöm szépen a megtisztelő figyelmüket.", "Elnök úr!"],
    )
    [p] = placed_reactions(rec)
    assert p.closing


def test_clock_stamps_party_tags_and_non_reactions_are_skipped():
    rec = record(
        "NAGY JÁNOS (TISZA): Egy. (13.20) Kettő. (Elfoglalja helyét.) "
        "Három. (Derültség.)",
        ["Egy.", "Kettő.", "Három."],
        prefix="NAGY JÁNOS (TISZA)",
    )
    [p] = placed_reactions(rec)
    assert p.sentence_index == 2 and p.kinds == frozenset({"laughter"})


def test_a_reaction_that_cannot_be_found_is_reported_not_guessed():
    rec = record("Valami egészen más. (Taps.)", ["Ez nem az a szöveg."])
    [p] = placed_reactions(rec)
    assert not p.aligned and p.sentence_index is None and p.relative_position is None


def test_sides_and_line_kinds():
    rec = record(
        "Egy. (Derültség a Fidesz soraiban.) Kettő. (Taps a TISZA soraiban.) Három.",
        ["Egy.", "Kettő.", "Három."],
    )
    laugh, clap = placed_reactions(rec)
    assert side_relation(laugh, "government", 43) == "other"
    assert line_kinds(laugh, "other") == ["laughter_at"]
    assert line_kinds(clap, side_relation(clap, "government", 43)) == ["applause"]
    # Applause from the other side is not an applause line.
    assert line_kinds(clap, side_relation(clap, "opposition", 43)) == []


def test_an_unknown_speaker_side_is_unattributed():
    rec = record("Egy. (Taps a TISZA soraiban.) Kettő.", ["Egy.", "Kettő."])
    [p] = placed_reactions(rec)
    assert side_relation(p, None, 43) == "unattributed"


def test_features_name_the_other_side_only():
    assert sentence_features([], "Orbán Viktor hazudott.", "government")[
        "names_opponent"
    ]
    assert not sentence_features([], "Orbán Viktor hazudott.", "opposition")[
        "names_opponent"
    ]
    assert not sentence_features([], "A magyar emberek.", "opposition")[
        "names_opponent"
    ]


@given(
    st.lists(
        st.text(alphabet="abcdefg ", min_size=1, max_size=12).map(
            lambda s: s.strip() + "."
        ),
        min_size=1,
        max_size=8,
    ),
    st.data(),
)
def test_placements_are_in_bounds_and_never_move_backwards(sentences, data):
    sentences = [s for s in sentences if s.strip(".")] or ["egy."]
    marks = data.draw(st.sets(st.integers(0, len(sentences) - 1)))
    parts = []
    for i, s in enumerate(sentences):
        parts.append(s)
        if i in marks:
            parts.append("(Taps.)")
    placed = placed_reactions(record(" ".join(parts), sentences))
    indices = [p.sentence_index for p in placed if p.aligned]
    assert all(0 <= i < len(sentences) for i in indices)
    assert indices == sorted(indices)


@given(
    st.lists(
        st.text(alphabet="abc .!", min_size=1, max_size=10), min_size=1, max_size=6
    )
)
def test_tokens_rebuild_their_sentences(sentences):
    tokens = [[w, w, "[/N]"] for s in sentences for w in s.split()]
    if not tokens:
        return
    mapped = tokens_by_sentence(sentences, [tokens])
    assert sum(len(t) for t in mapped) == len(tokens)


def test_a_bare_thank_you_is_ritual_too():
    rec = record(
        "Kezdjük. Köszönöm szépen. (Taps.) Folytatom.",
        ["Kezdjük.", "Köszönöm szépen.", "Folytatom."],
    )
    [p] = placed_reactions(rec)
    assert p.closing
