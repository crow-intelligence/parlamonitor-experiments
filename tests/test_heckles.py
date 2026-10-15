import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.heckles import (
    HECKLE_STOP_EXTRAS,
    bench_of,
    content_lemmas,
    party_group,
    quotes_in,
    quotes_in_line,
)
from parlamonitor.interactions import SpeakerRegistry
from parlamonitor.stopwords import SPACY_HU


@pytest.fixture
def registry():
    return SpeakerRegistry(
        [
            {
                "person_id": "s1",
                "label": "Rétvári Bence",
                "faction": "KDNP",
                "office": None,
                "is_mp": True,
            },
            {
                "person_id": "s2",
                "label": "Magyar Péter",
                "faction": "TISZA",
                "office": "miniszterelnök",
                "is_mp": True,
            },
        ],
        governing=frozenset({"TISZA"}),
    )


# --- quotes -----------------------------------------------------------------


def test_a_multi_sentence_heckle_is_kept_whole():
    [q] = quotes_in(
        "Dr. Varga László: Mit nevetgéltek ezen? "
        "Inkább szégyelljétek magatokat! Minden érvénytelen!"
    )
    assert (
        q.quote
        == "Mit nevetgéltek ezen? Inkább szégyelljétek magatokat! Minden érvénytelen!"
    )
    assert q.reassembled


def test_a_single_sentence_heckle_is_not_marked_reassembled():
    [q] = quotes_in("Vadai Ágnes: Nem hallom!")
    assert (q.speaker, q.quote, q.reassembled) == ("Vadai Ágnes", "Nem hallom!", False)


def test_a_following_reaction_ends_the_quote():
    [q] = quotes_in("Bana Tibor: Tolvajok! Derültség a kormánypártok soraiban.")
    assert q.quote == "Tolvajok!"


@pytest.mark.parametrize(
    "piece, speaker",
    [
        ("Dr. Kátai-Németh Vilmos: Igen!", "Kátai-Németh Vilmos"),
        ("Varga-Damm Andrea: Ez nem igaz!", "Varga-Damm Andrea"),
        ("Dr. Gaudi-Nagy Tamás: Szégyen!", "Gaudi-Nagy Tamás"),
    ],
)
def test_hyphenated_interjectors_are_recovered_and_flagged(piece, speaker):
    [q] = quotes_in(piece)
    assert q.speaker == speaker and q.hyphenated_recovered


def test_a_bench_heckle_has_no_speaker_but_keeps_its_party():
    [q] = quotes_in("Közbeszólás az MSZP soraiból: Hazudik!")
    assert q.speaker is None and q.parties == frozenset({"MSZP"})


def test_reactions_and_procedure_hold_no_quote():
    assert quotes_in("Taps a kormánypártok soraiban.") == []
    assert (
        quotes_in_line(
            "Szünet: 13.05 - 13.14 Elnök: Lezsák Sándor Jegyző: Göndör István"
        )
        == []
    )


def test_each_piece_of_a_line_is_read_separately():
    line = "Közbeszólás a Fidesz soraiból: Így van! - Arató Gergely: Nem igaz!"
    assert [q.quote for q in quotes_in_line(line)] == ["Így van!", "Nem igaz!"]


@given(st.text(max_size=120))
def test_reading_quotes_never_raises(text):
    for q in quotes_in_line(text):
        assert q.quote


@given(st.text(max_size=120))
def test_a_quote_is_always_text_from_its_line(text):
    # Whitespace aside, every quote is a contiguous stretch of the input.
    flat = " ".join(text.split())
    for q in quotes_in_line(text):
        assert " ".join(q.quote.split()) in flat


# --- benches ----------------------------------------------------------------


def test_a_bench_named_in_the_note_settles_side_and_party():
    [q] = quotes_in("Közbeszólás a Fidesz soraiból: Csak keményen!")
    assert bench_of(q, 43) == ("opposition", "Fidesz-KDNP", "bench in the note")
    assert bench_of(q, 42) == ("government", "Fidesz-KDNP", "bench in the note")


def test_a_named_heckler_without_a_registry_stays_unattributed():
    [q] = quotes_in("Rétvári Bence: Így van!")
    assert bench_of(q, 42) == (None, None, "named, no party data")


def test_a_named_heckler_resolves_through_the_registry(registry):
    [q] = quotes_in("Rétvári Bence: Így van!")
    assert bench_of(q, 43, registry) == ("opposition", "Fidesz-KDNP", "named, resolved")
    [q] = quotes_in("Magyar Péter: Szégyen!")
    assert bench_of(q, 43, registry) == ("government", "other", "named, resolved")


def test_an_unknown_name_is_unresolved_not_guessed(registry):
    [q] = quotes_in("Kovács Ödön: Halló!")
    assert bench_of(q, 43, registry) == (None, None, "named, unresolved")


def test_the_cycle_43_opposition_is_not_assumed_to_be_fidesz():
    assert party_group(frozenset(), "opposition", 43) is None
    assert party_group(frozenset({"Fidesz", "Mi Hazánk"}), "opposition", 43) is None


# --- lemmas -----------------------------------------------------------------


SENTENCE = [
    ["Így", "így", "[/Adv]"],
    ["van", "van", "[/V][Prs.NDef.3Sg]"],
    ["!", "!", "[Punct]"],
    ["Ő", "ő", "[/N|Pro][3Sg][Nom]"],
    ["hazudik", "hazudik", "[/V][Prs.NDef.3Sg]"],
]


def test_content_lemmas_drop_punctuation_and_pronouns():
    assert content_lemmas([SENTENCE]) == ["így", "van", "hazudik"]


def test_content_lemmas_drop_stopwords_after_folding():
    assert content_lemmas([SENTENCE], frozenset({"így", "van"})) == ["hazudik"]


def test_the_nltk_extras_are_not_already_in_spacy():
    assert not set(HECKLE_STOP_EXTRAS) & SPACY_HU
    assert all("õ" not in word for word in HECKLE_STOP_EXTRAS)
