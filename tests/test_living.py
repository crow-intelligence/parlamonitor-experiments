from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.applause import placed_reactions
from parlamonitor.living import bench, flow, reaction_seconds


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
