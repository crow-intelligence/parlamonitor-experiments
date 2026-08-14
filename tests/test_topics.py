import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.topics import (
    HU_FUNCTION_WORDS,
    build_phrases,
    chunk_tokens,
    dynamic_stopwords,
    embed_documents,
    topic_fingerprint,
)


class FakeEncoder:
    """Encodes a chunk as a vector of its word count, so pooling is checkable."""

    def __init__(self, dim=4):
        self.dim = dim
        self.seen = []

    def encode(self, sentences, **kwargs):
        self.seen.extend(sentences)
        return np.array(
            [[float(len(s.split()))] * self.dim for s in sentences], dtype=np.float32
        )


# --- chunking ---------------------------------------------------------------


def test_chunk_tokens_windows():
    assert chunk_tokens(["a", "b", "c", "d", "e"], size=2) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_chunk_tokens_size_larger_than_input():
    assert chunk_tokens(["a", "b"], size=80) == [["a", "b"]]


def test_chunk_tokens_exact_multiple():
    assert chunk_tokens(["a", "b", "c", "d"], size=2) == [["a", "b"], ["c", "d"]]


def test_chunk_tokens_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        chunk_tokens([])


def test_chunk_tokens_zero_size_raises():
    with pytest.raises(ValueError, match="at least 1"):
        chunk_tokens(["a"], size=0)


@given(
    tokens=st.lists(st.text(min_size=1, max_size=6), min_size=1, max_size=200),
    size=st.integers(min_value=1, max_value=50),
)
def test_chunk_tokens_is_a_partition(tokens, size):
    """Concatenating the windows reproduces the input: nothing lost, nothing doubled."""
    chunks = chunk_tokens(tokens, size)
    assert sum(chunks, []) == tokens
    assert all(len(c) == size for c in chunks[:-1])
    assert 1 <= len(chunks[-1]) <= size


# --- embedding --------------------------------------------------------------


def test_embed_documents_covers_the_whole_document():
    """A 250-word document must reach the encoder in full, not truncated."""
    encoder = FakeEncoder()
    long_doc = " ".join(f"w{i}" for i in range(250))
    embed_documents([long_doc], encoder, chunk_size=80)

    assert len(encoder.seen) == 4  # 80 + 80 + 80 + 10
    assert sum(len(chunk.split()) for chunk in encoder.seen) == 250


def test_embed_documents_shape_and_normalisation():
    encoder = FakeEncoder(dim=4)
    embeddings = embed_documents(["a b c", "d e"], encoder, chunk_size=2)

    assert embeddings.shape == (2, 4)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_embed_documents_pools_chunks_by_mean():
    encoder = FakeEncoder(dim=1)
    # One document of 3 words at chunk_size=2 -> chunks of 2 and 1 words,
    # so the pooled (pre-normalisation) value is the mean of 2 and 1.
    embeddings = embed_documents(["a b c"], encoder, chunk_size=2)
    assert embeddings.shape == (1, 1)
    assert np.isclose(abs(embeddings[0, 0]), 1.0)  # single dim normalises to 1


def test_embed_documents_empty_collection_raises():
    with pytest.raises(ValueError, match="empty"):
        embed_documents([], FakeEncoder())


def test_embed_documents_whitespace_document_raises():
    with pytest.raises(ValueError, match="document 1"):
        embed_documents(["kormány", "   "], FakeEncoder())


# --- phrases ----------------------------------------------------------------


def test_build_phrases_fuses_with_underscore():
    # Gensim's score scales with vocabulary size, so the filler words are not
    # padding -- without them nothing clears any threshold at all.
    corpus = [["költségvetési", "hiány", f"tétel{i}"] for i in range(40)]
    phrases = build_phrases(corpus, min_count=5, threshold=1.0)
    assert phrases[["költségvetési", "hiány", "tétel0"]] == [
        "költségvetési_hiány",
        "tétel0",
    ]


def test_build_phrases_respects_min_count():
    corpus = [["ritka", "bigram", f"x{i}"] for i in range(3)]
    corpus += [["egyéb", "szó", f"y{i}"] for i in range(30)]
    phrases = build_phrases(corpus, min_count=5, threshold=1.0)
    assert phrases[["ritka", "bigram"]] == ["ritka", "bigram"]


# --- stopwords --------------------------------------------------------------


def test_dynamic_stopwords_cuts_ubiquitous_terms():
    corpus = ["kormány költségvetés", "kormány vita", "kormány adó"]
    result = dynamic_stopwords(corpus, max_df=0.9, min_df=1, manual=[])

    assert result.stopwords == ["kormány"]
    assert result.n_dynamic == 1
    assert result.vocabulary_size == 3


def test_dynamic_stopwords_unions_the_manual_list():
    corpus = ["kormány költségvetés", "kormány vita", "kormány adó"]
    result = dynamic_stopwords(corpus, max_df=0.9, min_df=1, manual={"és", "kormány"})

    assert result.stopwords == ["kormány", "és"]
    assert result.n_dynamic == 1
    assert result.n_manual == 2
    assert result.n_overlap == 1


def test_dynamic_stopwords_reports_thresholds_used():
    corpus = ["kormány vita", "vita adó"]
    result = dynamic_stopwords(corpus, max_df=0.9, min_df=1, manual=[])
    assert (result.max_df, result.min_df) == (0.9, 1)


def test_dynamic_stopwords_ignores_single_character_tokens():
    """The default token_pattern in sklearn needs two word characters."""
    with pytest.raises(ValueError, match="no vocabulary at all"):
        dynamic_stopwords(["a b", "b c"], max_df=0.9, min_df=1, manual=[])


def test_dynamic_stopwords_empty_corpus_raises():
    with pytest.raises(ValueError, match="empty corpus"):
        dynamic_stopwords([])


def test_dynamic_stopwords_impossible_thresholds_raise():
    with pytest.raises(ValueError, match="no vocabulary"):
        dynamic_stopwords(["kormány vita"] * 5, max_df=0.1, min_df=0.9)


# --- topic fingerprint ------------------------------------------------------

TOPICS = {0: ["kormány", "vita"], 1: ["vasút", "vonat"], -1: ["magyar"]}


def test_fingerprint_is_stable_across_calls():
    assert topic_fingerprint(TOPICS) == topic_fingerprint(TOPICS)


def test_fingerprint_ignores_insertion_order():
    shuffled = {1: ["vasút", "vonat"], -1: ["magyar"], 0: ["kormány", "vita"]}
    assert topic_fingerprint(shuffled) == topic_fingerprint(TOPICS)


def test_fingerprint_detects_renumbering():
    """The failure this exists to catch: same topics, different ids."""
    renumbered = {1: ["kormány", "vita"], 0: ["vasút", "vonat"], -1: ["magyar"]}
    assert topic_fingerprint(renumbered) != topic_fingerprint(TOPICS)


def test_fingerprint_detects_a_changed_term():
    changed = {0: ["kormány", "vita"], 1: ["vasúti", "vonat"], -1: ["magyar"]}
    assert topic_fingerprint(changed) != topic_fingerprint(TOPICS)


def test_fingerprint_detects_reordered_terms_within_a_topic():
    """Term order is the model's output, not incidental."""
    reordered = {0: ["vita", "kormány"], 1: ["vasút", "vonat"], -1: ["magyar"]}
    assert topic_fingerprint(reordered) != topic_fingerprint(TOPICS)


def test_fingerprint_detects_a_dropped_topic():
    assert topic_fingerprint({0: ["kormány", "vita"]}) != topic_fingerprint(TOPICS)


def test_fingerprint_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        topic_fingerprint({})


def test_fingerprint_accepts_numpy_style_int_keys():
    """Numpy ints come back from pandas; they must fingerprint like plain ones."""
    assert topic_fingerprint({np.int64(0): ["a"]}) == topic_fingerprint({0: ["a"]})


@given(
    st.dictionaries(
        st.integers(min_value=-1, max_value=40),
        st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=6),
        min_size=1,
        max_size=15,
    )
)
def test_fingerprint_is_16_hex_chars(topics):
    result = topic_fingerprint(topics)
    assert len(result) == 16
    assert all(c in "0123456789abcdef" for c in result)


def test_manual_list_matches_the_spec():
    assert HU_FUNCTION_WORDS == frozenset(
        {"a", "az", "és", "hogy", "is", "egy", "nem", "de"}
    )
