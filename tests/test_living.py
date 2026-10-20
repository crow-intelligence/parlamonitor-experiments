from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.applause import placed_reactions, without_directions
from parlamonitor.living import bench, flow, reaction_seconds
from parlamonitor.virtues import load_virtues, sentence_virtues

ROOT = Path(__file__).resolve().parents[1]


def one(text, sentences):
    [p] = placed_reactions(
        {"speaker_prefix": None, "text": text, "sentences": sentences}
    )
    return p


@given(
    st.lists(st.text(alphabet="ab ", min_size=1, max_size=20), min_size=1, max_size=10),
    st.data(),
    st.floats(0, 50_000),
    st.floats(0, 5_000),
)
def test_a_reaction_time_always_falls_inside_its_speech(sentences, data, start, length):
    i = data.draw(st.integers(0, len(sentences) - 1))
    t = reaction_seconds(sentences, i, None, start, length)
    assert start <= t <= start + length + 1e-6


def test_later_sentences_come_later():
    sentences = ["Egy kettő.", "Három négy öt.", "Hat."]
    times = [reaction_seconds(sentences, i, None, 0.0, 60.0) for i in range(3)]
    assert times == sorted(times) and times[-1] == 60.0


def test_the_note_decides_the_bench_before_the_interjector():
    p = one("Egy. (Taps a TISZA soraiban.) Kettő.", ["Egy.", "Kettő."])
    assert bench(p, "opposition", 43) == "government"


def test_a_named_heckler_takes_their_own_side():
    p = one("Egy. (Rétvári Bence: Hazugság!) Kettő.", ["Egy.", "Kettő."])
    assert p.interjector == "Rétvári Bence"
    assert bench(p, "opposition", 43) == "opposition"
    assert bench(p, None, 43) is None


def test_both_benches_and_the_bell_stay_out_of_the_bars():
    both = one("Egy. (Taps a TISZA és a Fidesz soraiban.) Kettő.", ["Egy.", "Kettő."])
    assert bench(both, None, 43) == "both"
    assert flow(both, "both", "government") is None
    bell = one("Egy. (Az elnök csenget.) Kettő.", ["Egy.", "Kettő."])
    assert flow(bell, bench(bell, None, 43), "government") is None


def test_unattributed_applause_is_own_and_unattributed_heckling_other():
    clap = one("Egy. (Taps.) Kettő.", ["Egy.", "Kettő."])
    heckle = one("Egy. (Közbeszólások.) Kettő.", ["Egy.", "Kettő."])
    assert flow(clap, None, "opposition") == "own"
    assert flow(heckle, None, "opposition") == "other"


def test_the_closing_ritual_is_not_a_response():
    p = one(
        "Egy. Köszönöm a figyelmet. (Taps a TISZA soraiban.)",
        ["Egy.", "Köszönöm a figyelmet."],
    )
    assert flow(p, "government", "government") is None


def test_a_hostile_heckle_takes_its_own_lane():
    from parlamonitor.living import lanes

    p = one("Egy. (Közbeszólás a Fidesz soraiból: Hazudsz!) Kettő.", ["Egy.", "Kettő."])
    assert lanes(p.kinds, "other", True) == ["heckle_hostile"]
    assert lanes(p.kinds, "other", False) == ["heckle"]
    # From the speaker's own benches the same note is support, not a heckle.
    assert lanes(p.kinds, "own", True) == ["support"]


def test_noise_is_a_lane_wherever_it_came_from():
    from parlamonitor.living import lanes

    p = one("Egy. (Felzúdulás a TISZA soraiban.) Kettő.", ["Egy.", "Kettő."])
    assert "noise" in lanes(p.kinds, "own", False) and "noise" in lanes(
        p.kinds, None, False
    )


def test_an_unreviewed_hostile_list_is_refused(tmp_path):
    import json

    import pytest

    from parlamonitor.living import load_hostile_words

    path = tmp_path / "words.json"
    path.write_text(
        json.dumps({"checked_by_human": False, "words": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not been reviewed"):
        load_hostile_words(path)


def test_the_committed_hostile_list_loads():
    from pathlib import Path

    from parlamonitor.living import load_hostile_words

    words = load_hostile_words(
        Path(__file__).resolve().parents[1]
        / "data"
        / "labels"
        / "hostile_heckle_words.json"
    )
    assert {"hazudik", "szégyen", "tolvaj"} <= words and "fúj" not in words


# The sonification's base layer: virtue mentions per sentence.

VIRTUES = load_virtues(ROOT / "data" / "lexicons" / "virtues" / "hu_virtues.json")


def mentions(sentence, words):
    """Virtue mentions in a sentence, from (form, lemma, tag) tokens."""
    _, kept = without_directions(sentence, [list(w) for w in words])
    return sentence_virtues([w[1] for w in kept], [w[2] for w in kept], VIRTUES)


def test_one_sentence_can_affirm_one_virtue_and_accuse_with_another():
    words = [
        ("Bátorság", "bátorság", "[/N]"),
        ("kell", "kell", "[/V]"),
        (",", ",", "[Punct]"),
        ("nem", "nem", "[/Adv]"),
        ("hazugság", "hazugság", "[/N]"),
        (".", ".", "[Punct]"),
    ]
    assert mentions("Bátorság kell, nem hazugság.", words) == [
        ("courage", "affirming"),
        ("truthfulness", "accusing"),
    ]


def test_a_hecklers_words_inside_a_sentence_are_not_the_speakers_virtues():
    sentence = "(Pócs János: Hazudik!) Bátorság kell."
    words = [
        ("(", "(", "[Punct]"),
        ("Pócs", "Pócs", "[/N]"),
        ("János", "János", "[/N]"),
        (":", ":", "[Punct]"),
        ("Hazudik", "hazudik", "[/V]"),
        ("!", "!", "[Punct]"),
        (")", ")", "[Punct]"),
        ("Bátorság", "bátorság", "[/N]"),
        ("kell", "kell", "[/V]"),
        (".", ".", "[Punct]"),
    ]
    assert mentions(sentence, words) == [("courage", "affirming")]
