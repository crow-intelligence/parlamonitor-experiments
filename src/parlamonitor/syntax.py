"""Syntactic complexity: mean dependency distance and hierarchical distance.

LIX and MATTR describe a text's surface -- how long its words are, how varied
its vocabulary is. Neither says anything about how the words are *arranged*,
and arrangement is where a lot of parliamentary difficulty lives: a sentence
can be built entirely from short common words and still be hard to follow if
its dependents sit far from their heads.

**Mean dependency distance (MDD)** is the average distance, in tokens, between
a word and its governor (Liu 2008; Jing & Liu 2015). Longer distances mean
more material has to be held in working memory before a dependency closes, so
MDD is a memory-load measure rather than a vocabulary one.

**Mean hierarchical distance (MHD)** is the average depth of a token in the
parse tree. MDD and MHD trade off against each other -- a language or a writer
can buy shorter dependencies with a deeper tree, or the reverse -- which is why
Jing & Liu report the pair rather than either alone.

Both need a **parse**, which is why the emtsv chain here is
:data:`parlamonitor.emtsv.DEPENDENCY_MODULES` and not the default one. The
arithmetic is saphes'; this module only adapts emtsv's columns to saphes'
input contract and records which parser produced them.

**The parser is part of the measurement.** Two parsers can score the same text
differently because they follow different head conventions -- whether a
preposition governs its noun or the reverse, for instance. Every result here
carries ``parser`` for that reason, and it should be quoted alongside any
number taken from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from saphes import DepToken, mean_dependency_distance, mean_hierarchical_distance

from parlamonitor.emtsv import Token

PARSER = "emtsv tok/morph/pos/conv-morph/dep"
"""What produced the parses. Provenance, and it belongs beside every score."""

PUNCTUATION_POLICY = "collapse"
"""How punctuation is handled, as the dependency-distance literature does it.

Punctuation is removed and the distances recomputed over what remains, so a
comma between a word and its head does not count as a token of distance.
"""

MIN_SENTENCE_LENGTH = 3
"""Sentences shorter than this are discarded.

Jing & Liu (2015) use 3. A one- or two-word sentence has a degenerate MDD --
often exactly 1 -- and this corpus is full of them: ``Köszönöm.``, ``Igen.``,
``Taps.`` Left in, they pull every speaker's score toward 1 in proportion to
how often they said "thank you", which is not a fact about syntax.
"""


@dataclass(frozen=True, slots=True)
class SyntaxResult:
    """Syntactic complexity of one text, with the counts behind it.

    Attributes:
        mdd: Mean dependency distance, or ``None`` if no sentence qualified.
        mhd: Mean hierarchical distance, or ``None``.
        n_sentences: Sentences that entered the calculation.
        n_sentences_discarded: Sentences dropped for being shorter than
            :data:`MIN_SENTENCE_LENGTH`.
        n_tokens: Non-punctuation tokens counted.
        parser: :data:`PARSER`.
        min_sentence_length: The filter used.
        punctuation: The punctuation policy used.
    """

    mdd: float | None
    mhd: float | None
    n_sentences: int
    n_sentences_discarded: int
    n_tokens: int
    parser: str = PARSER
    min_sentence_length: int = MIN_SENTENCE_LENGTH
    punctuation: str = PUNCTUATION_POLICY


def to_dep_tokens(sentence: Sequence[Token]) -> list[DepToken]:
    """Adapt one emtsv sentence to saphes' input contract.

    saphes never guesses whether a token is punctuation -- it does not know a
    parser's tagset -- so that judgement is made here, from the universal tag
    ``conv-morph`` assigns.

    Args:
        sentence: Tokens of one sentence, parsed with
            :data:`parlamonitor.emtsv.DEPENDENCY_MODULES`.

    Returns:
        One :class:`saphes.DepToken` per token.

    Raises:
        ValueError: If the sentence carries no parse, which happens when the
            module chain omitted ``dep``. Returning an empty list instead
            would silently drop the text from the corpus.

    Example:
        >>> sentence = [
        ...     Token("A", "a", "[/Det]", upostag="DET", dep_id=1, head=2),
        ...     Token("kormány", "kormány", "[/N]", upostag="NOUN", dep_id=2, head=3),
        ...     Token("dönt", "dönt", "[/V]", upostag="VERB", dep_id=3, head=0),
        ...     Token(".", ".", "[Punct]", upostag="PUNCT", dep_id=4, head=0),
        ... ]
        >>> [(t.index, t.head, t.is_punct) for t in to_dep_tokens(sentence)]
        [(1, 2, False), (2, 3, False), (3, 0, False), (4, 0, True)]
    """
    if any(token.head < 0 for token in sentence):
        raise ValueError(
            "sentence has no dependency parse; use emtsv.DEPENDENCY_MODULES"
        )
    return [
        DepToken(
            index=token.dep_id,
            head=token.head,
            is_punct=token.upostag == "PUNCT",
            pos=token.upostag or None,
        )
        for token in sentence
    ]


def measure(
    sentences: Sequence[Sequence[Token]],
    *,
    min_sentence_length: int = MIN_SENTENCE_LENGTH,
) -> SyntaxResult:
    """Measure one text's dependency and hierarchical distance.

    Args:
        sentences: Parsed sentences of one text.
        min_sentence_length: Discard sentences shorter than this, counted in
            non-punctuation tokens. Defaults to 3.

    Returns:
        A :class:`SyntaxResult`. ``mdd`` and ``mhd`` are ``None`` when every
        sentence was discarded -- a speech consisting of ``Köszönöm.`` has no
        measurable syntax, and reporting 0.0 would put it at the easy end of a
        scale it is not on at all.

    Raises:
        ValueError: If ``sentences`` is empty, or if any sentence lacks a
            parse.

    Example:
        >>> sentence = [
        ...     Token("A", "a", "[/Det]", upostag="DET", dep_id=1, head=2),
        ...     Token("kormány", "kormány", "[/N]", upostag="NOUN", dep_id=2, head=3),
        ...     Token("dönt", "dönt", "[/V]", upostag="VERB", dep_id=3, head=0),
        ...     Token(".", ".", "[Punct]", upostag="PUNCT", dep_id=4, head=0),
        ... ]
        >>> result = measure([sentence])

        Distances are 1 for ``A``->``kormány`` and 1 for ``kormány``->``dönt``;
        the root and the punctuation contribute none, so MDD is 1.0:

        >>> result.mdd, result.n_sentences, result.n_tokens
        (1.0, 1, 3)
        >>> result.parser
        'emtsv tok/morph/pos/conv-morph/dep'

        A text with nothing long enough is null, not zero:

        >>> short = [[Token("Igen", "igen", "[/Adv]", upostag="ADV", dep_id=1, head=0),
        ...           Token(".", ".", "[Punct]", upostag="PUNCT", dep_id=2, head=0)]]
        >>> measure(short).mdd is None
        True
    """
    if not sentences:
        raise ValueError("cannot measure syntax of a text with no sentences")

    parses = [to_dep_tokens(sentence) for sentence in sentences]
    qualifying = [
        parse
        for parse in parses
        if sum(1 for token in parse if not token.is_punct) >= min_sentence_length
    ]
    if not qualifying:
        return SyntaxResult(
            mdd=None,
            mhd=None,
            n_sentences=0,
            n_sentences_discarded=len(parses),
            n_tokens=0,
            min_sentence_length=min_sentence_length,
        )

    shared = {
        "punctuation": PUNCTUATION_POLICY,
        "aggregation": "macro",
        "min_sentence_length": min_sentence_length,
        "parser": PARSER,
    }
    mdd = mean_dependency_distance(parses, **shared)
    mhd = mean_hierarchical_distance(parses, **shared)
    return SyntaxResult(
        mdd=round(float(mdd.mdd), 6),
        mhd=round(float(mhd.mhd), 6),
        n_sentences=len(qualifying),
        n_sentences_discarded=len(parses) - len(qualifying),
        n_tokens=sum(1 for parse in qualifying for t in parse if not t.is_punct),
        min_sentence_length=min_sentence_length,
    )
