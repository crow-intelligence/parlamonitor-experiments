import math
from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.frequency import count_tokens, frequency_frame, ngram_order, npmi

DOCUMENTS = [["taps", "taps", "sor"], ["taps", "derültség"], ["sor"]]


def frame_for(documents, cycle="43"):
    terms, docs, n = count_tokens(documents)
    return frequency_frame(terms, docs, n_documents=n, cycle=cycle)


# --- counting ---------------------------------------------------------------


def test_terms_count_occurrences_and_documents_count_presence():
    terms, docs, n = count_tokens(DOCUMENTS)
    assert terms["taps"] == 3
    assert docs["taps"] == 2  # twice in one document still counts once
    assert n == 3


def test_counting_nothing_yields_nothing():
    assert count_tokens([]) == (Counter(), Counter(), 0)


# --- the frame --------------------------------------------------------------


def test_rows_are_sorted_by_raw_descending():
    assert frame_for(DOCUMENTS)["token"].tolist() == ["taps", "sor", "derültség"]


def test_the_denominators_travel_with_the_rates():
    frame = frame_for(DOCUMENTS)
    assert (frame["cycle_total_tokens"] == 6).all()
    assert (frame["cycle_total_docs"] == 3).all()


def test_relative_frequency_is_per_million():
    frame = frame_for(DOCUMENTS).set_index("token")
    assert frame.loc["taps", "relative_per_million"] == pytest.approx(3 / 6 * 1e6)


def test_empty_corpus_raises_rather_than_producing_nan():
    with pytest.raises(ValueError, match="no tokens"):
        frequency_frame(Counter(), Counter(), n_documents=1, cycle="43")


def test_zero_documents_raises():
    with pytest.raises(ValueError, match="expected > 0"):
        frequency_frame(Counter({"taps": 1}), Counter(), n_documents=0, cycle="43")


@given(
    st.lists(
        st.lists(st.sampled_from(["a", "b", "c"]), min_size=1), min_size=1, max_size=20
    )
)
def test_proportions_sum_to_one(documents):
    assert frame_for(documents)["proportion"].sum() == pytest.approx(1.0)


@given(
    st.lists(
        st.lists(st.sampled_from(["a", "b", "c"]), min_size=1), min_size=1, max_size=20
    )
)
def test_raw_counts_sum_to_the_reported_total(documents):
    frame = frame_for(documents)
    assert frame["raw"].sum() == frame["cycle_total_tokens"].iloc[0]


@given(
    st.lists(
        st.lists(st.sampled_from(["a", "b", "c"]), min_size=1), min_size=1, max_size=10
    ),
    st.integers(min_value=2, max_value=4),
)
def test_relative_frequency_is_invariant_under_duplicating_the_corpus(documents, k):
    # The whole point of normalising: a corpus counted twice reads the same.
    one = frame_for(documents).set_index("token")["relative_per_million"]
    many = frame_for(documents * k).set_index("token")["relative_per_million"]
    for token in one.index:
        assert many[token] == pytest.approx(one[token])


@given(
    st.lists(st.lists(st.sampled_from(["a", "b"]), min_size=1), min_size=1, max_size=20)
)
def test_document_proportion_never_exceeds_one(documents):
    frame = frame_for(documents)
    assert (frame["doc_proportion"] <= 1.0).all()
    assert (frame["doc_count"] <= frame["raw"]).all()


# --- n-gram order -----------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [("taps", 1), ("a#b", 2), ("a#b#c", 3), ("taps#a#kormánypárt#sor", 4)],
)
def test_ngram_order_counts_components(token, expected):
    assert ngram_order(token, "#") == expected


def test_an_empty_delimiter_is_refused():
    with pytest.raises(ValueError, match="non-empty"):
        ngram_order("taps", "")


# --- npmi -------------------------------------------------------------------


def test_worked_example_by_hand():
    # p(ab)=0.1, p(a)=0.2, p(b)=0.25 -> ln(2) / -ln(0.1)
    assert npmi(10, [20, 25], 100) == pytest.approx(math.log(2) / -math.log(0.1))


def test_perfect_co_occurrence_scores_one():
    assert npmi(10, [10, 10], 100) == pytest.approx(1.0)
    assert npmi(10, [10, 10, 10], 100) == pytest.approx(1.0)


def test_independence_scores_zero():
    # p(ab) = p(a) * p(b): 100 * 0.1 * 0.1 = 1 occurrence in 100 tokens.
    assert npmi(1, [10, 10], 100) == pytest.approx(0.0, abs=1e-12)


@given(
    st.integers(min_value=1, max_value=500),
    st.integers(min_value=1, max_value=500),
    st.integers(min_value=1, max_value=500),
)
def test_npmi_is_bounded_above_by_one(joint, a, b):
    # A joint count can never exceed either component's count.
    joint = min(joint, a, b)
    total = a + b + 1000
    assert npmi(joint, [a, b], total) <= 1.0 + 1e-9


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100))
def test_npmi_is_symmetric_in_its_components(a, b):
    joint = min(a, b)
    total = a + b + 500
    assert npmi(joint, [a, b], total) == pytest.approx(npmi(joint, [b, a], total))


@pytest.mark.parametrize(
    ("joint", "components", "total"),
    [(0, [10, 10], 100), (10, [0, 10], 100), (10, [10, 10], 0), (200, [10, 10], 100)],
)
def test_impossible_counts_raise(joint, components, total):
    with pytest.raises(ValueError):
        npmi(joint, components, total)


def test_an_ngram_needs_at_least_two_components():
    with pytest.raises(ValueError, match="at least two"):
        npmi(10, [10], 100)
