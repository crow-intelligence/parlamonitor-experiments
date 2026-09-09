"""Unsupervised keyword extraction: TextRank, and KeyBERT if it is installed.

Two methods, because they disagree in ways worth seeing.

**TextRank** (Mihalcea & Tarau 2004) builds a co-occurrence graph over
content-word lemmas inside a sliding window and ranks it with PageRank. It is
graph-based and entirely local to the document: a term scores well by sitting
between many other terms of that speech. It needs no model, no embeddings and
no corpus, which also means it cannot know that ``egészségügy`` and
``kórház`` are related.

**KeyBERT** ranks candidate phrases by cosine similarity to an embedding of the
whole document. It is semantic and will surface a phrase that captures the
speech's subject even if it occurs once. It needs a sentence encoder; the
huBERT model already used elsewhere in this workspace is the right one for
Hungarian, and it is optional here so the base install stays light.

Both consume **lemmas**, not surface forms. Hungarian inflection would
otherwise scatter one term across a dozen graph nodes: *egészségügyben*,
*egészségügyet*, *egészségügyre* are one keyword, not three.

Neither method is a topic model. They rank terms within a single document, so
a term frequent across the whole corpus -- ``elnök``, ``képviselő`` -- will be
ranked highly in many speeches unless it has been filtered out beforehand. Pass
a stoplist.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx

DEFAULT_WINDOW = 4
"""Co-occurrence window for TextRank, in tokens.

Mihalcea & Tarau use 2; larger windows connect more of the graph and favour
terms that recur across a long text. 4 is a common compromise and is exposed
because it changes the ranking.
"""

DEFAULT_TOP_N = 10
"""How many keywords to return by default."""


def cooccurrence_graph(tokens: Sequence[str], window: int = DEFAULT_WINDOW) -> nx.Graph:
    """Build the undirected co-occurrence graph TextRank ranks.

    Every pair of tokens within ``window`` positions of each other gets an
    edge, weighted by how often that co-occurrence happens.

    Args:
        tokens: Content-word lemmas, in document order.
        window: How far apart two tokens may be. Defaults to 4.

    Returns:
        An undirected weighted graph. A token repeated in the text is one node.

    Raises:
        ValueError: If ``window`` is below 2 -- no token can co-occur with
            itself, so a window of 1 yields an empty graph.

    Example:
        >>> graph = cooccurrence_graph(["adó", "emelés", "adó"], window=2)
        >>> sorted(graph.nodes)
        ['adó', 'emelés']
        >>> graph["adó"]["emelés"]["weight"]
        2
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    graph = nx.Graph()
    graph.add_nodes_from(tokens)
    for index, token in enumerate(tokens):
        for other in tokens[index + 1 : index + window]:
            if token == other:
                continue
            if graph.has_edge(token, other):
                graph[token][other]["weight"] += 1
            else:
                graph.add_edge(token, other, weight=1)
    return graph


def textrank(
    tokens: Sequence[str],
    *,
    top_n: int = DEFAULT_TOP_N,
    window: int = DEFAULT_WINDOW,
    stopwords: Iterable[str] = (),
) -> list[tuple[str, float]]:
    """Rank a document's terms with PageRank over their co-occurrence graph.

    Args:
        tokens: Content-word lemmas, in document order.
        top_n: How many to return. Defaults to 10.
        window: Co-occurrence window. Defaults to 4.
        stopwords: Lemmas to drop **before** the graph is built, so they
            neither rank nor lend their centrality to neighbours.

    Returns:
        ``(lemma, score)`` pairs, highest first. Scores are PageRank values and
        sum to 1 across the whole graph, so they are comparable within a
        document and not across documents of different size.

        Empty if nothing survives the stoplist -- a short procedural remark
        legitimately has no keywords, and inventing one would be worse.

    Example:
        ``emelés`` outranks ``adó`` here because it neighbours both other
        terms while ``adó`` neighbours ``költségvetés`` only once -- centrality,
        not frequency, which is the whole point of the method:

        >>> tokens = ["adó", "emelés", "adó", "emelés", "költségvetés"]
        >>> [term for term, _ in textrank(tokens, top_n=3)]
        ['emelés', 'adó', 'költségvetés']

        The stoplist is applied before ranking, so a dropped term does not
        lend its centrality to its neighbours either:

        >>> [term for term, _ in textrank(tokens, top_n=2, stopwords={"adó"})]
        ['emelés', 'költségvetés']

        Nothing to rank is an empty list, not an error:

        >>> textrank([], top_n=5)
        []
    """
    drop = set(stopwords)
    kept = [token for token in tokens if token not in drop]
    if not kept:
        return []
    graph = cooccurrence_graph(kept, window=window)
    if graph.number_of_edges() == 0:
        # A single repeated term, or a text shorter than the window: PageRank
        # on an edgeless graph is uniform and says nothing, so fall back to
        # frequency, which at least reflects the text.
        counts = {token: kept.count(token) for token in set(kept)}
        total = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(term, count / total) for term, count in ranked[:top_n]]
    scores = nx.pagerank(graph, weight="weight")
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(term, round(score, 6)) for term, score in ranked[:top_n]]


def keybert_available() -> bool:
    """Report whether KeyBERT and its encoder can be imported.

    Returns:
        ``True`` if ``keybert`` is installed. The extraction functions raise
        rather than silently degrading, so a caller can check first and record
        which method actually ran.
    """
    try:
        import keybert  # noqa: F401
    except ImportError:
        return False
    return True
