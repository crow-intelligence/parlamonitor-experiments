import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.reactions import (
    GOVERNING_PARTIES,
    INTENSITY_WEIGHT,
    Audience,
    Intensity,
    Kind,
    classify,
    events_in,
    find_audiences,
    side_of,
    split_events,
)

# --- splitting --------------------------------------------------------------


def test_the_dash_separates_events():
    assert split_events("Taps a Jobbik soraiban. - Derültség.") == [
        "Taps a Jobbik soraiban.",
        "Derültség.",
    ]


def test_sentences_separate_events_too():
    assert split_events("Taps. Az elnök csenget.") == ["Taps.", "Az elnök csenget."]


@pytest.mark.parametrize(
    "line",
    [
        "Taps. Dr. Vadai Ágnes közbeszól.",
        "Taps. Ifj. Nagy Béla közbeszól.",
    ],
)
def test_an_abbreviation_does_not_end_a_sentence(line):
    # Without this "Dr." is the most common "segment" in the whole corpus.
    assert len(split_events(line)) == 2


def test_an_initial_does_not_end_a_sentence():
    assert split_events("Z. Kárpát Dániel: Nem támadjuk!") == [
        "Z. Kárpát Dániel: Nem támadjuk!"
    ]


def test_splitting_nothing_yields_nothing():
    assert split_events("") == []
    assert split_events("   ") == []


@given(st.text(alphabet="abc .-", max_size=40))
def test_segments_are_never_blank(line):
    assert all(segment.strip() for segment in split_events(line))


# --- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    ("segment", "kind"),
    [
        ("Taps.", Kind.APPLAUSE),
        ("Derültség.", Kind.LAUGHTER),
        ("Közbeszólások az MSZP soraiból.", Kind.HECKLING),
        ("Folyamatos sípolás.", Kind.WHISTLING),
        ("Zaj.", Kind.NOISE),
        ("Felzúdulás a Fidesz soraiban.", Kind.UPROAR),
        ("Az elnök csenget.", Kind.BELL),
        ("Pfújolás az ellenzéki oldalról.", Kind.BOOING),
        ("Nincs jelentkező.", Kind.OTHER),
        ("Szavazás.", Kind.OTHER),
        ("Megtörténik.", Kind.OTHER),
    ],
)
def test_kinds(segment, kind):
    assert classify(segment).kind is kind


@pytest.mark.parametrize(
    ("segment", "intensity"),
    [
        ("Taps.", Intensity.PLAIN),
        ("Nagy taps.", Intensity.GREAT),
        ("Hosszan tartó taps.", Intensity.LONG),
        ("Szórványos taps a Fidesz soraiból.", Intensity.SCATTERED),
        ("Folyamatos sípolás.", Intensity.CONTINUOUS),
        ("A TISZA-frakció tagjai felállva tapsolnak.", Intensity.STANDING),
    ],
)
def test_intensity(segment, intensity):
    assert classify(segment).intensity is intensity


def test_a_stronger_qualifier_wins():
    # "hosszan tartó nagy taps" is long, not merely great.
    assert classify("Hosszan tartó nagy taps.").intensity is Intensity.LONG


def test_procedure_carries_no_intensity():
    assert classify("Nagy nehezen megtörténik.").intensity is Intensity.PLAIN


def test_a_segment_naming_two_reactions_keeps_both():
    event = classify("Derültség és taps a kormánypárti padsorokban.")
    assert event.kinds == frozenset({Kind.LAUGHTER, Kind.APPLAUSE})


def test_weight_follows_intensity():
    assert classify("Taps.").weight == INTENSITY_WEIGHT[Intensity.PLAIN]
    assert classify("Nagy taps.").weight == INTENSITY_WEIGHT[Intensity.GREAT]


# --- named interjections ----------------------------------------------------


def test_a_named_interjection_carries_speaker_and_quote():
    event = classify("Vadai Ágnes: Nem hallom!")
    assert event.kind is Kind.INTERJECTION
    assert event.speaker == "Vadai Ágnes"
    assert event.quote == "Nem hallom!"


def test_a_title_is_stripped_from_the_interjector():
    assert classify("Dr. Vadai Ágnes: Nem hallom!").speaker == "Vadai Ágnes"


def test_an_interjection_is_not_reclassified_by_its_quote():
    # The quote contains "taps", but the event is an interjection, not applause.
    event = classify("Nacsa Lőrinc: Hol van a taps?")
    assert event.kind is Kind.INTERJECTION


def test_a_trailing_colon_is_not_an_interjection():
    # "Bóna Zoltán jelzésére:" is a procedural cue with nothing quoted.
    assert classify("Bóna Zoltán jelzésére:").kind is Kind.OTHER


# --- audiences --------------------------------------------------------------


@pytest.mark.parametrize(
    ("segment", "audience"),
    [
        ("Taps a kormánypártok soraiban.", Audience.GOVERNMENT),
        ("Taps a kormánypárti padsorokban.", Audience.GOVERNMENT),
        ("Taps a kormányzó pártok padsoraiból.", Audience.GOVERNMENT),
        ("Taps az ellenzéki padsorokból.", Audience.OPPOSITION),
    ],
)
def test_bench_terms(segment, audience):
    assert audience in classify(segment).audiences


def test_named_parties_are_reported_as_themselves():
    audiences, parties = find_audiences("Taps a Fidesz és a KDNP soraiból.")
    assert parties == frozenset({"Fidesz", "KDNP"})
    assert Audience.PARTY in audiences


def test_mi_hazank_is_matched_as_two_words():
    assert find_audiences("Taps a Mi Hazánk soraiban.")[1] == frozenset({"Mi Hazánk"})


def test_a_bare_reaction_credits_nobody():
    # Not "everyone" -- the note-taker simply did not say.
    assert classify("Taps.").audiences == frozenset({Audience.UNSPECIFIED})


def test_procedure_gets_no_unspecified_audience():
    assert classify("Szavazás.").audiences == frozenset()


# --- sides ------------------------------------------------------------------


def test_a_party_changes_side_between_cycles():
    assert side_of("Fidesz", 41) is Audience.GOVERNMENT
    assert side_of("Fidesz", 43) is Audience.OPPOSITION
    assert side_of("TISZA", 43) is Audience.GOVERNMENT


def test_every_cycle_declares_a_government():
    assert set(GOVERNING_PARTIES) == {39, 40, 41, 42, 43}
    assert all(parties for parties in GOVERNING_PARTIES.values())


def test_an_unknown_cycle_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        side_of("Fidesz", 99)


# --- end to end -------------------------------------------------------------


def test_a_real_compound_line():
    line = (
        "Hosszan tartó taps a kormánypártok soraiban. - Szórványos taps a "
        "Jobbik soraiban. - Dr. Vadai Ágnes: Nem igaz!"
    )
    events = events_in(line)
    assert [e.kind for e in events] == [
        Kind.APPLAUSE,
        Kind.APPLAUSE,
        Kind.INTERJECTION,
    ]
    assert events[0].intensity is Intensity.LONG
    assert events[1].parties == frozenset({"Jobbik"})
    assert events[2].speaker == "Vadai Ágnes"


@given(st.text(max_size=60))
def test_classifying_arbitrary_text_never_raises(text):
    for event in events_in(text):
        assert event.kind in Kind
        assert event.weight > 0


# --- false interjectors, all found in the corpus ----------------------------


@pytest.mark.parametrize(
    ("segment", "kind"),
    [
        # A bench, not a person -- 3,541 events before this was fixed.
        ("Közbeszólás az MSZP soraiból: Hazudik!", Kind.HECKLING),
        ("Közbeszólások az ellenzéki padsorokból: Nem igaz!", Kind.HECKLING),
        ("Egy hang a Fidesz soraiból: Úgy van!", Kind.OTHER),
        # Procedure, not a person -- 1,538 events.
        ("Szünet: 14.19", Kind.OTHER),
        ("Elnök: Igen.", Kind.OTHER),
        ("Jelenlét-ellenőrzés: jelen van 190, távol van 9.", Kind.OTHER),
    ],
)
def test_a_reaction_phrase_or_procedure_is_never_a_person(segment, kind):
    event = classify(segment)
    assert event.speaker is None
    assert event.kind is kind


def test_an_unnamed_heckle_still_keeps_what_was_shouted():
    assert classify("Közbeszólás az MSZP soraiból: Hazudik!").quote == "Hazudik!"


def test_an_unnamed_heckle_is_credited_to_the_bench_that_made_it():
    event = classify("Közbeszólás az MSZP soraiból: Hazudik!")
    assert event.parties == frozenset({"MSZP"})


@pytest.mark.parametrize(
    "segment",
    ["Vadai Ágnes: Nem igaz!", "Dr. Vadai Ágnes: Nem igaz!", "Z. Kárpát Dániel: Nem!"],
)
def test_full_names_are_still_recognised(segment):
    assert classify(segment).speaker is not None
    assert len(classify(segment).speaker.split()) >= 2
