import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.parentheticals import (
    CYCLES,
    SOFT_HYPHEN,
    LineStats,
    collapse_consecutive,
    file_sha256,
    line_stats,
    load_parentheticals,
    parentheticals_path,
    strip_soft_hyphens,
)

LINES = [
    "Taps a kormánypártok soraiban.",
    "Taps a kormánypártok soraiban.",
    "Derültség.",
    "Taps a kormánypártok soraiban.",
]


@pytest.fixture
def export(tmp_path):
    """A miniature data/raw tree holding one cycle's export."""
    directory = tmp_path / "parantheticals"
    directory.mkdir()
    (directory / "cycle43-parentheticals.txt").write_text(
        "\n".join(LINES) + "\n", encoding="utf-8"
    )
    return tmp_path


# --- loading ----------------------------------------------------------------


def test_loads_one_line_per_parenthetical(export):
    assert load_parentheticals(43, export) == (LINES, 0)


def test_limit_takes_the_first_lines(export):
    assert load_parentheticals(43, export, limit=2)[0] == LINES[:2]


def test_blank_lines_are_dropped(tmp_path):
    directory = tmp_path / "parantheticals"
    directory.mkdir()
    (directory / "cycle43-parentheticals.txt").write_text(
        "Taps.\n\n   \nDerültség.\n", encoding="utf-8"
    )
    assert load_parentheticals(43, tmp_path)[0] == ["Taps.", "Derültség."]


# --- soft hyphens -----------------------------------------------------------


def test_a_soft_hyphen_is_removed_and_the_word_rejoined():
    assert strip_soft_hyphens(f"hát{SOFT_HYPHEN}oldalán") == ("hátoldalán", 1)


def test_text_without_soft_hyphens_is_returned_unchanged():
    assert strip_soft_hyphens("Taps.") == ("Taps.", 0)


def test_every_soft_hyphen_is_counted():
    assert strip_soft_hyphens(f"a{SOFT_HYPHEN}b{SOFT_HYPHEN}c") == ("abc", 2)


def test_loading_strips_soft_hyphens_and_reports_how_many(tmp_path):
    directory = tmp_path / "parantheticals"
    directory.mkdir()
    (directory / "cycle40-parentheticals.txt").write_text(
        f"Köz{SOFT_HYPHEN}beszólás.\nTaps.\n", encoding="utf-8"
    )
    assert load_parentheticals(40, tmp_path) == (["Közbeszólás.", "Taps."], 1)


def test_normalise_false_reads_the_file_verbatim(tmp_path):
    directory = tmp_path / "parantheticals"
    directory.mkdir()
    line = f"Köz{SOFT_HYPHEN}beszólás."
    (directory / "cycle40-parentheticals.txt").write_text(line + "\n", encoding="utf-8")
    assert load_parentheticals(40, tmp_path, normalise=False) == ([line], 0)


def test_a_missing_cycle_raises_rather_than_returning_nothing(export):
    # Silently returning [] would let a typo drop a fifth of the corpus.
    with pytest.raises(FileNotFoundError, match="cycle 99"):
        load_parentheticals(99, export)


def test_path_follows_the_export_spelling():
    assert parentheticals_path(41, "/data").name == "cycle41-parentheticals.txt"
    assert parentheticals_path(41, "/data").parent.name == "parantheticals"


def test_all_five_cycles_are_declared():
    assert CYCLES == (39, 40, 41, 42, 43)


# --- collapsing -------------------------------------------------------------


def test_collapse_removes_only_adjacent_repeats():
    assert collapse_consecutive(LINES) == [LINES[0], "Derültség.", LINES[0]]


def test_collapse_of_nothing_is_nothing():
    assert collapse_consecutive([]) == []


@given(st.lists(st.sampled_from(["a", "b", "c"])))
def test_collapse_never_grows_the_corpus(lines):
    assert len(collapse_consecutive(lines)) <= len(lines)


@given(st.lists(st.sampled_from(["a", "b", "c"])))
def test_collapse_is_idempotent(lines):
    once = collapse_consecutive(lines)
    assert collapse_consecutive(once) == once


@given(st.lists(st.sampled_from(["a", "b", "c"])))
def test_collapse_preserves_the_set_of_lines(lines):
    assert set(collapse_consecutive(lines)) == set(lines)


# --- statistics -------------------------------------------------------------


def test_line_stats_counts_words_lines_and_repeats(export):
    path = parentheticals_path(43, export)
    stats = line_stats(43, LINES, path, sha256="deadbeef")
    assert stats.n_lines == 4
    assert stats.n_distinct == 2
    assert stats.n_consecutive_repeats == 1
    assert stats.n_words == sum(len(line.split()) for line in LINES)
    assert stats.sha256 == "deadbeef"


def test_repeat_fraction_of_an_empty_file_is_zero_not_a_crash():
    assert LineStats(43, parentheticals_path(43), "", 0, 0, 0, 0).repeat_fraction == 0.0


@given(st.lists(st.sampled_from(["a", "b"]), min_size=1))
def test_repeats_are_exactly_what_collapsing_removes(lines):
    stats = line_stats(43, lines, parentheticals_path(43))
    assert stats.n_consecutive_repeats == len(lines) - len(collapse_consecutive(lines))
    assert 0.0 <= stats.repeat_fraction < 1.0


def test_sha256_is_of_the_file_bytes(export):
    import hashlib

    path = parentheticals_path(43, export)
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
