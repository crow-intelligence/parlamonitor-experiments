"""Word-frequency tables that carry their own denominators.

A relative frequency is uninterpretable without the total it was divided by,
and two implementations that disagree about the total will quietly disagree
about every rate they publish. So every frame built here ships
``cycle_total_tokens`` and ``cycle_total_docs`` alongside the rates, and the
raw count is always present next to the normalised one.

Three normalisations are emitted because they answer different questions:

``relative_per_million``
    ``raw / total_tokens * 1e6``. The corpus-linguistics convention, and the
    one to use when comparing cycle 43 (35k words) against cycle 41 (675k).
``proportion``
    ``raw / total_tokens``. The same number unscaled; sums to 1 over a cycle.
``doc_proportion``
    ``doc_count / n_docs``, where a document is one parenthetical line. This
    is the one that resists burstiness: a term shouted 15,840 times in one
    protest has a huge ``relative_per_million`` and a modest
    ``doc_proportion``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import pandas as pd

COUNT_COLUMNS: tuple[str, ...] = (
    "raw",
    "relative_per_million",
    "proportion",
    "doc_count",
    "doc_proportion",
)
"""The count columns :func:`frequency_frame` produces, in order."""


def count_tokens(documents: Iterable[Sequence[str]]) -> tuple[Counter, Counter, int]:
    """Count tokens and the documents they appear in.

    Args:
        documents: One token sequence per document. For this corpus a
            document is one parenthetical line.

    Returns:
        A ``(term_counts, document_counts, n_documents)`` triple.
        ``document_counts`` counts each term at most once per document.

    Example:
        >>> terms, docs, n = count_tokens([["taps", "taps"], ["taps", "derültség"]])
        >>> terms["taps"], docs["taps"], n
        (3, 2, 2)
        >>> terms["derültség"], docs["derültség"]
        (1, 1)
    """
    terms: Counter = Counter()
    docs: Counter = Counter()
    n_documents = 0
    for document in documents:
        n_documents += 1
        terms.update(document)
        docs.update(set(document))
    return terms, docs, n_documents


def frequency_frame(
    term_counts: Counter,
    document_counts: Counter,
    *,
    n_documents: int,
    cycle: str,
) -> pd.DataFrame:
    """Turn raw counts into a tidy frequency table.

    Args:
        term_counts: Token to number of occurrences.
        document_counts: Token to number of documents containing it.
        n_documents: How many documents were counted.
        cycle: Label for the ``cycle`` column -- a cycle number as a string,
            or :data:`parlamonitor.parentheticals.AGGREGATE_LABEL`.

    Returns:
        One row per token, sorted by ``raw`` descending then token ascending,
        with columns ``cycle``, ``token``, :data:`COUNT_COLUMNS`,
        ``cycle_total_tokens`` and ``cycle_total_docs``.

    Raises:
        ValueError: If ``term_counts`` is empty, or if ``n_documents`` is not
            positive. An empty corpus has no frequencies, and saying so is
            better than emitting a table of ``NaN``.

    Example:
        Three tokens over two documents; ``taps`` occurs 3 of 4 times, so its
        proportion is 0.75 and its rate 750,000 per million:

        >>> terms, docs, n = count_tokens([["taps", "taps"], ["taps", "csenget"]])
        >>> frame = frequency_frame(terms, docs, n_documents=n, cycle="43")
        >>> frame["token"].tolist()
        ['taps', 'csenget']
        >>> frame["raw"].tolist()
        [3, 1]
        >>> frame["relative_per_million"].tolist()
        [750000.0, 250000.0]
        >>> frame["proportion"].tolist()
        [0.75, 0.25]
        >>> frame["doc_proportion"].tolist()
        [1.0, 0.5]
        >>> frame["cycle_total_tokens"].unique().tolist()
        [4]
    """
    if not term_counts:
        raise ValueError(f"cycle {cycle!r} has no tokens to count")
    if n_documents <= 0:
        raise ValueError(f"cycle {cycle!r} has {n_documents} documents, expected > 0")

    total = sum(term_counts.values())
    rows = [
        {
            "cycle": cycle,
            "token": token,
            "raw": raw,
            "relative_per_million": raw / total * 1e6,
            "proportion": raw / total,
            "doc_count": document_counts.get(token, 0),
            "doc_proportion": document_counts.get(token, 0) / n_documents,
            "cycle_total_tokens": total,
            "cycle_total_docs": n_documents,
        }
        for token, raw in term_counts.items()
    ]
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["raw", "token"], ascending=[False, True], ignore_index=True
    )


def ngram_order(token: str, delimiter: str) -> int:
    """Report how many words a possibly-fused token holds.

    Args:
        token: A token, perhaps a fused n-gram.
        delimiter: The string the phrase model joins components with.

    Returns:
        1 for a plain word, 2 for a bigram, 3 for a trigram, and so on.

    Raises:
        ValueError: If ``delimiter`` is empty -- every token would then be of
            infinite order.

    Example:
        >>> ngram_order("taps", "#")
        1
        >>> ngram_order("kormánypárt#sor", "#")
        2
        >>> ngram_order("taps#kormánypárt#sor", "#")
        3
    """
    if not delimiter:
        raise ValueError("delimiter must be a non-empty string")
    return token.count(delimiter) + 1


def npmi(joint_count: int, component_counts: Sequence[int], n_tokens: int) -> float:
    """Normalised pointwise mutual information for an n-gram.

    Computed from counts the caller can see, rather than taken from the phrase
    model. Gensim records the score each phrase had *in the pass that detected
    it*, and a second pass scores against a corpus the first pass has already
    fused, which puts those numbers outside NPMI's range and on a different
    scale from the first pass's. Recomputing against the unfused corpus gives
    one comparable, bounded number for every phrase regardless of pass.

    The n-ary generalisation is::

        NPMI = (log p(w1..wn) - sum_i log p(wi)) / (-(n - 1) * log p(w1..wn))

    which reduces to the familiar bigram formula at ``n = 2``. It reaches 1.0
    when the components occur only together, and 0.0 when they are exactly as
    frequent together as independence predicts.

    Args:
        joint_count: Occurrences of the n-gram.
        component_counts: Occurrences of each component word, counted in the
            **unfused** corpus.
        n_tokens: Total tokens in that unfused corpus.

    Returns:
        The score. Bounded above by 1.0. For ``n = 2`` it is bounded below by
        -1.0; for longer n-grams the lower tail can exceed that, which does not
        arise for phrases a detector actually selected.

    Raises:
        ValueError: If any count is not positive, if ``joint_count`` exceeds
            ``n_tokens``, or if fewer than two components are given.

    Example:
        A bigram occurring 10 times in 100 tokens, its parts 20 and 25 times.
        Independence would predict ``0.2 * 0.25 = 0.05``, so the ratio is 2:

        >>> round(npmi(10, [20, 25], 100), 6)
        0.30103

        Components that never occur apart score 1.0:

        >>> npmi(10, [10, 10], 100)
        1.0

        And a trigram works the same way:

        >>> round(npmi(10, [10, 10, 10], 100), 6)
        1.0
    """
    from math import log

    if len(component_counts) < 2:
        raise ValueError(
            f"an n-gram needs at least two components, got {len(component_counts)}"
        )
    if joint_count <= 0 or n_tokens <= 0 or any(c <= 0 for c in component_counts):
        raise ValueError(
            f"counts must be positive: joint={joint_count}, "
            f"components={list(component_counts)}, n_tokens={n_tokens}"
        )
    if joint_count > n_tokens:
        raise ValueError(
            f"joint count {joint_count} exceeds the corpus size {n_tokens}"
        )

    log_joint = log(joint_count / n_tokens)
    numerator = log_joint - sum(log(c / n_tokens) for c in component_counts)
    denominator = -(len(component_counts) - 1) * log_joint
    return numerator / denominator
