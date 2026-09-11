import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.keywords import cooccurrence_graph, textrank


def test_co_occurrence_is_windowed_and_weighted():
    graph = cooccurrence_graph(["adó", "emelés", "adó"], window=2)
    assert sorted(graph.nodes) == ["adó", "emelés"]
    assert graph["adó"]["emelés"]["weight"] == 2


def test_a_token_never_co_occurs_with_itself():
    graph = cooccurrence_graph(["adó", "adó", "adó"], window=3)
    assert graph.number_of_edges() == 0


def test_a_wider_window_connects_more():
    tokens = ["a", "b", "c", "d"]
    narrow = cooccurrence_graph(tokens, window=2)
    wide = cooccurrence_graph(tokens, window=4)
    assert wide.number_of_edges() > narrow.number_of_edges()


def test_a_window_below_two_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        cooccurrence_graph(["a", "b"], window=1)


def test_centrality_beats_frequency():
    # "híd" occurs less often than "a" but sits between everything.
    tokens = ["út", "híd", "vasút", "híd", "repülő", "híd", "kikötő"]
    assert textrank(tokens, top_n=1)[0][0] == "híd"


def test_stopwords_are_dropped_before_the_graph_is_built():
    tokens = ["adó", "emelés", "adó", "emelés", "költségvetés"]
    terms = [t for t, _ in textrank(tokens, top_n=3, stopwords={"adó"})]
    assert "adó" not in terms


def test_an_empty_document_yields_no_keywords():
    assert textrank([]) == []


def test_a_document_that_is_all_stopwords_yields_no_keywords():
    assert textrank(["a", "az"], stopwords={"a", "az"}) == []


def test_a_single_repeated_term_falls_back_to_frequency():
    # PageRank on an edgeless graph is uniform and says nothing.
    result = textrank(["adó", "adó", "adó"], top_n=2)
    assert result == [("adó", 1.0)]


def test_top_n_is_respected():
    tokens = [f"w{i}" for i in range(30)]
    assert len(textrank(tokens, top_n=5)) == 5


def test_scores_are_descending():
    tokens = ["a", "b", "c", "a", "b", "a", "d", "e", "a"]
    scores = [score for _, score in textrank(tokens, top_n=5)]
    assert scores == sorted(scores, reverse=True)


@given(st.lists(st.sampled_from(["a", "b", "c", "d"]), max_size=40))
def test_ranking_arbitrary_input_never_raises(tokens):
    for term, score in textrank(tokens, top_n=3):
        assert term in tokens
        assert score > 0


@given(st.lists(st.sampled_from(["a", "b", "c"]), min_size=6, max_size=40))
def test_every_keyword_came_from_the_document(tokens):
    assert {term for term, _ in textrank(tokens)} <= set(tokens)
