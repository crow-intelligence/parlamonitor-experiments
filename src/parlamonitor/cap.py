"""CAP topic classification of parliamentary speeches.

Wraps ``classla/ParlaCAP-Topic-Classifier``: an XLM-RoBERTa-large model
pre-trained on parliamentary proceedings and fine-tuned on 29 ParlaMint 4.1
datasets, predicting the 21 CAP major topics plus ``Other``.

Two facts about it govern everything here.

**The confidence rule is the authors', not ours.** The model card says: "we
annotated instances that were predicted with confidence below 0.60 as 'Mix'",
and reports Mix rates of 8.9% (English) to 11.4% (Croatian) under it. Our own
Mix rate is therefore comparable to a published figure, which is worth
protecting -- see :func:`apply_threshold`.

**512 tokens is about 279 Hungarian words.** Measured on this corpus, the
tokenizer produces a median 1.82 subword tokens per Hungarian word, so a
single forward pass covers roughly the first 279 words. Only 750 of the 1,693
cycle-43 speeches fit whole; the rest are judged on their opening. Hungarian is
among the model's languages and ParlaMint-HU is among the training datasets,
but the published F1 scores cover only English, Croatian, Serbian and Bosnian
-- there is no Hungarian evaluation to quote.

:func:`aggregate_scores` exists so a caller can classify a speech in windows
and combine the results, covering the whole text at the cost of comparability
with the authors' figures. Doing both and reporting the disagreement measures
what the truncation actually costs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

MODEL_ID = "classla/ParlaCAP-Topic-Classifier"
"""The Hugging Face model this module wraps."""

MIX_LABEL = "Mix"
"""Label assigned when confidence falls below the threshold.

Distinct from the model's own ``Other`` class. ``Other`` is a confident
judgement that a speech fits no CAP topic; ``Mix`` is our admission that the
model was unsure, and per the authors usually means the speech spans several
topics. Collapsing the two would throw that difference away.
"""

DEFAULT_THRESHOLD = 0.60
"""Confidence below which a prediction becomes :data:`MIX_LABEL`.

The value recommended on the model card, not a tuned one.
"""

MAX_TOKENS = 512
"""Model input limit. ``max_position_embeddings`` is 514, giving 512 usable."""

TOKENS_PER_WORD = 1.82
"""Median subword tokens per word, measured on the cycle-43 Hungarian text.

Hungarian is agglutinative, so its fertility is well above the ~1.3 typical of
English. This is what makes :data:`MAX_TOKENS` worth only ~279 words.
"""

DEFAULT_CHUNK_WORDS = 250
"""Words per window when classifying a speech in full.

Below the ~279 the token budget allows, so that a window still fits after
tokenisation even when its words are longer than the corpus median. Windows
that overflow would be silently truncated by the tokenizer, losing their tails
without saying so.
"""


def apply_threshold(
    label: str, score: float, *, threshold: float = DEFAULT_THRESHOLD
) -> str:
    """Return the label, or :data:`MIX_LABEL` if confidence is too low.

    The comparison is ``score >= threshold``, so a prediction exactly at the
    threshold is accepted.

    Args:
        label: The model's predicted label.
        score: Its confidence, in ``[0, 1]``.
        threshold: Minimum confidence to accept. Defaults to
            :data:`DEFAULT_THRESHOLD`.

    Returns:
        ``label`` when ``score >= threshold``, otherwise :data:`MIX_LABEL`.

    Raises:
        ValueError: If ``score`` is outside ``[0, 1]``. A probability that is
            not a probability means the caller passed a logit.

    Example:
        >>> apply_threshold("Health", 0.87)
        'Health'
        >>> apply_threshold("Health", 0.42)
        'Mix'

        The boundary is inclusive:

        >>> apply_threshold("Health", 0.60)
        'Health'

        ``Other`` is a real prediction and survives on its own confidence:

        >>> apply_threshold("Other", 0.95)
        'Other'
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be a probability in [0, 1], got {score}")
    return label if score >= threshold else MIX_LABEL


def aggregate_scores(
    chunk_scores: Sequence[Mapping[str, float]],
    weights: Sequence[float] | None = None,
) -> tuple[str, float, dict[str, float]]:
    """Combine per-window label distributions into one prediction.

    Averages the probability of each label across windows, optionally weighted,
    then takes the argmax. Averaging distributions rather than voting on
    argmaxes keeps a window's uncertainty in play: three windows that each
    weakly favour a different topic should not outvote one that is certain.

    Args:
        chunk_scores: One mapping of label to probability per window. Every
            mapping must cover the same labels.
        weights: Per-window weights, normally the window's word count so a
            trailing 12-word window does not count as much as a full one.
            Defaults to ``None``, meaning uniform.

    Returns:
        A ``(label, score, distribution)`` triple, where ``score`` is the
        winning label's averaged probability and ``distribution`` is the full
        averaged mapping -- the counts behind the answer, not just the answer.

    Raises:
        ValueError: If ``chunk_scores`` is empty, if ``weights`` has a
            different length, if the weights sum to zero, or if the windows
            disagree about which labels exist.

    Example:
        >>> label, score, _ = aggregate_scores(
        ...     [{"Health": 0.9, "Labor": 0.1}, {"Health": 0.3, "Labor": 0.7}]
        ... )
        >>> label, round(score, 2)
        ('Health', 0.6)

        Weighting by window length changes which side wins:

        >>> label, score, _ = aggregate_scores(
        ...     [{"Health": 0.9, "Labor": 0.1}, {"Health": 0.3, "Labor": 0.7}],
        ...     weights=[10, 250],
        ... )
        >>> label
        'Labor'

        A single window is returned unchanged:

        >>> aggregate_scores([{"Health": 0.8, "Labor": 0.2}])[:2]
        ('Health', 0.8)
    """
    if not chunk_scores:
        raise ValueError("cannot aggregate an empty sequence of window scores")

    labels = set(chunk_scores[0])
    for index, scores in enumerate(chunk_scores):
        if set(scores) != labels:
            raise ValueError(
                f"window {index} has labels {sorted(set(scores))}, "
                f"expected {sorted(labels)}"
            )

    if weights is None:
        weights = [1.0] * len(chunk_scores)
    elif len(weights) != len(chunk_scores):
        raise ValueError(f"got {len(weights)} weights for {len(chunk_scores)} windows")

    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"weights must sum to a positive number, got {total}")

    averaged = {
        label: sum(s[label] * w for s, w in zip(chunk_scores, weights, strict=True))
        / total
        for label in labels
    }
    best = max(averaged, key=lambda label: averaged[label])
    return best, averaged[best], averaged
