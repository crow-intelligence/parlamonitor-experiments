"""Pipeline pieces for unsupervised topic modeling of parliamentary speeches.

Each function here is pure and testable on its own; the orchestration lives in
``scripts/task2_bertopic.py``. Every parameter that changes the resulting
topics is exposed with its default stated, because these are the knobs that
make two runs disagree.

The one that is easy to miss is chunking.
``NYTK/sentence-transformers-experimental-hubert-hungarian`` ships
``max_seq_length: 128``, roughly 65-85 Hungarian words, against a 307-word
median speech. Encoding a speech directly would cluster its opening paragraph
and silently discard the rest, so :func:`embed_documents` splits each speech
into windows, encodes every window, and mean-pools them into one vector.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Container, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from gensim.models.phrases import FrozenPhrases, Phrases
from sklearn.feature_extraction.text import CountVectorizer

HU_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "az",
        "és",
        "hogy",
        "is",
        "egy",
        "nem",
        "de",
    }
)
"""The manual Hungarian stopword list from the task specification.

Deliberately short. After lemmatisation and the part-of-speech filter in
:mod:`parlamonitor.emtsv` almost none of these survive to the vectoriser
anyway; the list is a backstop for tokens that emtsv tagged as content words
by mistake, not the primary filter.
"""


class SentenceEncoder(Protocol):
    """The slice of ``SentenceTransformer`` that :func:`embed_documents` uses."""

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        """Encode a batch of strings into a 2-D array of vectors."""
        ...


@dataclass(frozen=True, slots=True)
class StopwordResult:
    """Stopwords generated from corpus frequency, with the counts behind them.

    Attributes:
        stopwords: The combined list, sorted, ready for a ``CountVectorizer``.
        n_dynamic: How many came from the frequency thresholds.
        n_manual: How many came from the manual list.
        n_overlap: How many appeared in both.
        vocabulary_size: Terms that survived the thresholds.
        max_df: The upper document-frequency threshold used.
        min_df: The lower document-frequency threshold used.
    """

    stopwords: list[str]
    n_dynamic: int
    n_manual: int
    n_overlap: int
    vocabulary_size: int
    max_df: float
    min_df: float


def drop_stopwords(
    tokenized_corpus: Iterable[Sequence[str]], stopwords: Container[str]
) -> list[list[str]]:
    """Remove stopword lemmas from every document.

    Applied **before** bigram detection, not after. Gensim scores adjacent
    pairs, so leaving ``tisztelt`` in the stream lets it fuse into
    ``tisztelt_elnök`` and consume a bigram slot that a real collocation could
    have used. Filtering first means only content words can pair up.

    Args:
        tokenized_corpus: One lemma list per document.
        stopwords: Anything supporting ``in`` -- normally the frozenset from
            :func:`parlamonitor.stopwords.hungarian_stopwords`.

    Returns:
        The filtered documents, in order. Documents can come back empty; the
        caller decides whether an empty document is droppable.

    Example:
        >>> drop_stopwords([["tisztelt", "költségvetés", "van"]], {"tisztelt", "van"})
        [['költségvetés']]
    """
    return [
        [token for token in document if token not in stopwords]
        for document in tokenized_corpus
    ]


def embedding_cache_key(**config: object) -> str:
    """Hash an embedding configuration into a short, stable cache key.

    Any input that would change the resulting vectors -- model, chunk size,
    text normalisation, the set of documents -- belongs in ``config``. Two runs
    that agree on all of it can reuse the same ``.npy``; two that differ get
    different keys and recompute, rather than silently reusing vectors built
    from other text.

    Args:
        **config: JSON-serialisable values identifying the configuration.

    Returns:
        The first 16 hex characters of the SHA-256 of the sorted JSON.

    Example:
        >>> a = embedding_cache_key(model="hubert", chunk_size=80, n_docs=1693)
        >>> b = embedding_cache_key(chunk_size=80, model="hubert", n_docs=1693)
        >>> a == b  # key order does not matter
        True
        >>> a == embedding_cache_key(model="hubert", chunk_size=64, n_docs=1693)
        False
        >>> len(a)
        16
    """
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chunk_tokens(tokens: Sequence[str], size: int = 80) -> list[list[str]]:
    """Split a token sequence into consecutive windows.

    The windows partition the input: concatenating them reproduces ``tokens``
    exactly, with no overlap and nothing dropped. That is the property that
    makes mean-pooling over the windows a statement about the whole document
    rather than about its first paragraph.

    Args:
        tokens: The tokens to split. Must be non-empty.
        size: Window length. The final window is short unless ``size`` divides
            the input evenly. Defaults to 80, which sits inside the 128
            wordpiece budget of the Hungarian huBERT encoder for typical
            parliamentary prose.

    Returns:
        The windows, in order.

    Raises:
        ValueError: If ``tokens`` is empty or ``size`` is below 1.

    Example:
        >>> chunk_tokens(["a", "b", "c", "d", "e"], size=2)
        [['a', 'b'], ['c', 'd'], ['e']]

        The partition property, which the tests assert over random inputs:

        >>> tokens = ["kormány", "költségvetés", "vita", "törvény"]
        >>> sum(chunk_tokens(tokens, size=3), []) == tokens
        True
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    if len(tokens) == 0:
        raise ValueError("cannot chunk an empty token sequence")
    return [list(tokens[i : i + size]) for i in range(0, len(tokens), size)]


def embed_documents(
    texts: Sequence[str],
    model: SentenceEncoder,
    *,
    chunk_size: int = 80,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Embed documents by chunking, encoding, and mean-pooling.

    Every chunk of every document is encoded in a single batched call, then
    the chunks belonging to one document are averaged and the result is
    L2-normalised so that a cosine metric downstream behaves.

    Args:
        texts: The documents. Each must contain at least one non-whitespace
            token; a document of pure whitespace has no meaningful vector and
            is the caller's job to filter out beforehand.
        model: Anything with a ``SentenceTransformer``-shaped ``encode``.
        chunk_size: Words per window, passed to :func:`chunk_tokens`.
            Defaults to 80.
        batch_size: Encoder batch size. Defaults to 32.
        show_progress_bar: Passed through to the encoder. Defaults to
            ``False``.

    Returns:
        An array of shape ``(len(texts), embedding_dim)``, each row unit
        length.

    Raises:
        ValueError: If ``texts`` is empty, or any document is whitespace only.

    Example:
        >>> embeddings = embed_documents(speeches, model)  # doctest: +SKIP
        >>> embeddings.shape  # doctest: +SKIP
        (1693, 768)
    """
    if len(texts) == 0:
        raise ValueError("cannot embed an empty document collection")

    chunks: list[str] = []
    offsets: list[tuple[int, int]] = []
    for index, text in enumerate(texts):
        words = text.split()
        if not words:
            raise ValueError(f"document {index} has no tokens to embed")
        start = len(chunks)
        chunks.extend(" ".join(window) for window in chunk_tokens(words, chunk_size))
        offsets.append((start, len(chunks)))

    encoded = np.asarray(
        model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    pooled = np.stack([encoded[start:stop].mean(axis=0) for start, stop in offsets])
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    # A zero vector cannot be normalised; leave it as-is rather than divide by
    # zero. It would take an encoder returning all zeros, but silence here
    # would turn that into NaNs much further downstream.
    norms[norms == 0] = 1.0
    return pooled / norms


def build_phrases(
    tokenized_corpus: Iterable[list[str]],
    *,
    min_count: int = 5,
    threshold: float = 10.0,
) -> FrozenPhrases:
    """Train a Gensim bigram model and return it frozen for application.

    Gensim joins detected phrases with an underscore, so ``költségvetési
    hiány`` becomes the single token ``költségvetési_hiány``. Scikit-learn's
    default ``token_pattern`` treats ``_`` as a word character, so the fused
    token survives vectorisation intact.

    No ``connector_words`` are configured. The usual reason to need them --
    bigrams like ``a_kormány`` forming around function words -- does not arise
    here because the corpus has already been reduced to content-word lemmas by
    :func:`parlamonitor.emtsv.lemmatize`.

    Args:
        tokenized_corpus: One token list per document. Consumed once, so pass
            a list rather than a generator.
        min_count: Ignore bigrams occurring fewer than this many times.
            Defaults to 5.
        threshold: Score threshold for joining a bigram; higher is stricter.
            Defaults to 10.0. Gensim's default scorer is
            ``(count(ab) - min_count) / count(a) / count(b) * len(vocab)``, so
            the score scales with **vocabulary size**. A threshold that is
            sensible on a corpus of tens of thousands of types will fuse
            nothing at all on a toy corpus of five.

    Returns:
        A :class:`~gensim.models.phrases.FrozenPhrases`, which applies faster
        and uses less memory than the trainable model.

    Example:
        >>> corpus = [["költségvetési", "hiány", f"tétel{i}"] for i in range(40)]
        >>> phrases = build_phrases(corpus, min_count=5, threshold=1.0)
        >>> phrases[["költségvetési", "hiány", "tétel0"]]
        ['költségvetési_hiány', 'tétel0']
    """
    model = Phrases(
        tokenized_corpus,
        min_count=min_count,
        threshold=threshold,
        connector_words=frozenset(),
    )
    return model.freeze()


def dynamic_stopwords(
    documents: Sequence[str],
    *,
    max_df: float = 0.85,
    min_df: float = 0.01,
    manual: Iterable[str] = HU_FUNCTION_WORDS,
) -> StopwordResult:
    """Derive stopwords from corpus frequency and union them with a manual list.

    The terms rejected by the document-frequency thresholds are recovered by
    fitting twice -- once without thresholds, once with -- and taking the
    difference. ``CountVectorizer.stop_words_``, which used to hold exactly
    this set and which the task specification names, was deprecated in
    scikit-learn 1.2 and is gone in the 1.9 installed here.

    Note that the default ``token_pattern`` requires **two or more** word
    characters, so single-letter tokens never enter the vocabulary and so
    never appear as stopwords either.

    Args:
        documents: The corpus, one string per document.
        max_df: Ignore terms appearing in more than this proportion of
            documents. Defaults to 0.85.
        min_df: Ignore terms appearing in fewer than this proportion of
            documents. Defaults to 0.01 -- on 1,693 speeches that is about 17
            documents, which is aggressive; raise it to keep rarer vocabulary.
        manual: Function words to add unconditionally. Defaults to
            :data:`HU_FUNCTION_WORDS`.

    Returns:
        A :class:`StopwordResult` carrying the list and the counts that
        produced it.

    Raises:
        ValueError: If ``documents`` is empty, if the corpus yields no
            vocabulary at all, or if the thresholds reject everything -- which
            means they are mutually exclusive for this corpus, not that the
            corpus is empty.

    Example:
        >>> corpus = ["kormány költségvetés", "kormány vita", "kormány adó"]
        >>> result = dynamic_stopwords(corpus, max_df=0.9, min_df=1, manual=[])
        >>> result.stopwords
        ['kormány']
        >>> result.vocabulary_size
        3
    """
    if len(documents) == 0:
        raise ValueError("cannot derive stopwords from an empty corpus")

    unfiltered = CountVectorizer()
    try:
        unfiltered.fit(documents)
    except ValueError as exc:
        raise ValueError(
            f"{len(documents)} documents yielded no vocabulary at all: {exc}"
        ) from exc

    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
    try:
        vectorizer.fit(documents)
    except ValueError as exc:
        raise ValueError(
            f"max_df={max_df} and min_df={min_df} left no vocabulary in "
            f"{len(documents)} documents: {exc}"
        ) from exc

    dynamic = set(unfiltered.vocabulary_) - set(vectorizer.vocabulary_)
    manual_set = set(manual)
    return StopwordResult(
        stopwords=sorted(dynamic | manual_set),
        n_dynamic=len(dynamic),
        n_manual=len(manual_set),
        n_overlap=len(dynamic & manual_set),
        vocabulary_size=len(vectorizer.vocabulary_),
        max_df=max_df,
        min_df=min_df,
    )
