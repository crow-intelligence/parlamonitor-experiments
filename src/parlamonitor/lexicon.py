"""Dictionary-based sentiment and emotion, with Hungarian negation.

Replaces the transformer scoring. The trade is deliberate: a transformer reads
context and a word list does not, but a word list is auditable to the token —
every score here decomposes into the exact lemmas that produced it — it costs
nothing to run, it cannot drift, and it does not have to be revalidated when a
model is updated. On this corpus a bare count already correlated r = 0.54 with
the fine-tuned huBERT it replaces.

**Negation follows the reference implementation** in
``crow-intelligence/growth-hacking-sentiment``: mark tokens inside a negation's
scope, then flip their polarity, so a negated negative counts as positive.
``nem probléma`` is not a complaint.

Hungarian needs two things English does not.

**Direction.** ``nem``, ``sem``, ``se``, ``ne``, ``nincs``, ``sincs`` scope
*forwards*, as English "not" does. But ``nélkül`` is a **postposition** — in
``pénz nélkül`` the negated word comes *before* the cue — so it scopes
backwards. A forward-only implementation would negate the wrong half of the
sentence.

**Nothing to strip.** Privative forms are already lexicalised
(``haszontalan``, ``tehetetlen``, ``boldogtalan``), so the ``-talan/-telen``
suffix needs no morphological rule; the list carries those words itself.

**Scoring is per sentence, then averaged**, again following the reference. A
score is the lexicon-hit density of a sentence, not of a document, so one long
speech does not dilute a short sharp one. Emotion uses the same machinery with
one count per category rather than two poles.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

NEGATION_FORWARD: frozenset[str] = frozenset(
    {"nem", "sem", "se", "ne", "nincs", "nincsen", "sincs", "sincsen"}
)
"""Cues whose scope runs forwards from the cue, as English ``not`` does."""

NEGATION_BACKWARD: frozenset[str] = frozenset({"nélkül"})
"""Cues whose scope runs backwards.

``nélkül`` is a postposition: ``pénz nélkül``, "money without". The negated
word precedes the cue, so a forward-only rule would mark the wrong tokens.
"""

SCOPE_END = re.compile(r"^[,;:.!?()\[\]—–-]+$")
"""A token that closes a negation's scope. Clause punctuation, as in NLTK's
``mark_negation``."""

NEG_SUFFIX = "_NEG"
"""Marker appended to a token inside a negation's scope."""


@dataclass(frozen=True, slots=True)
class Lexicon:
    """A loaded word list, with the provenance to reproduce a number.

    Attributes:
        name: What it measures -- ``"positive"``, ``"anger"``, and so on.
        single: Single-word entries, case-folded.
        multi: Multiword entries as lemma tuples, so ``áldását adja`` can be
            matched against a lemma stream rather than discarded.
        path: Where it was read from.
        sha256: Hex digest of the file.
    """

    name: str
    single: frozenset[str]
    multi: frozenset[tuple[str, ...]]
    path: Path
    sha256: str

    @property
    def size(self) -> int:
        """Total entries, single and multiword."""
        return len(self.single) + len(self.multi)

    @property
    def lexicon_id(self) -> str:
        """Identifier recorded on every result."""
        return f"{self.path.name}@{self.sha256[:12]}"


def load_lexicon(path: Path | str, name: str) -> Lexicon:
    """Read a newline-separated word list.

    Blank lines and ``#`` comments are skipped. Entries are case-folded, and
    multiword entries are split into lemma tuples.

    Args:
        path: The word list.
        name: What the list measures.

    Returns:
        A :class:`Lexicon`.

    Raises:
        FileNotFoundError: If the file is missing. Named rather than falling
            back to an empty set, which would score every text as neutral and
            look like a result.
        ValueError: If the file holds no usable entries.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no {name} lexicon at {path}; see data/lexicons/README.md"
        )
    raw = path.read_bytes()
    single: set[str] = set()
    multi: set[tuple[str, ...]] = set()
    for line in raw.decode("utf-8").splitlines():
        entry = line.strip().casefold()
        if not entry or entry.startswith("#"):
            continue
        parts = entry.split()
        if len(parts) == 1:
            single.add(parts[0])
        else:
            multi.add(tuple(parts))
    if not single and not multi:
        raise ValueError(f"{name} lexicon at {path} is empty")
    return Lexicon(
        name=name,
        single=frozenset(single),
        multi=frozenset(multi),
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def fold(lemma: str) -> str:
    """Case-fold a lemma for lexicon lookup.

    Accents are **kept**: ``őr`` and ``or`` are different words, and stripping
    them would merge pairs the lists deliberately separate. This is the
    opposite of the loanword module's key, and for the opposite reason.

    Example:
        >>> fold("Bátorság")
        'bátorság'
        >>> fold("ŐSZINTE")
        'őszinte'
    """
    return unicodedata.normalize("NFC", lemma).casefold()


def mark_negation(
    lemmas: Sequence[str],
    *,
    forward: Iterable[str] = NEGATION_FORWARD,
    backward: Iterable[str] = NEGATION_BACKWARD,
) -> list[str]:
    r"""Append ``_NEG`` to every lemma inside a negation's scope.

    Forward cues mark from the cue to the next clause boundary. Backward cues
    mark from the previous boundary up to the cue, because a Hungarian
    postposition follows what it negates.

    Args:
        lemmas: One sentence's lemmas, punctuation included -- the punctuation
            is what closes a scope.
        forward: Forward-scoping cues. Defaults to :data:`NEGATION_FORWARD`.
        backward: Backward-scoping cues. Defaults to :data:`NEGATION_BACKWARD`.

    Returns:
        The lemmas, with marked tokens suffixed. The cue itself is not marked.

    Example:
        >>> mark_negation(["ez", "nem", "probléma", "."])
        ['ez', 'nem', 'probléma_NEG', '.']

        Scope stops at a clause boundary, so the second half is untouched:

        >>> mark_negation(["nem", "jó", ",", "hanem", "rossz"])
        ['nem', 'jó_NEG', ',', 'hanem', 'rossz']

        ``nélkül`` is a postposition and scopes backwards:

        >>> mark_negation(["siker", "nélkül", "zárult"])
        ['siker_NEG', 'nélkül', 'zárult']
    """
    forward, backward = frozenset(forward), frozenset(backward)
    marked = list(lemmas)
    folded = [fold(lemma) for lemma in lemmas]

    for index, lemma in enumerate(folded):
        if lemma in forward:
            for j in range(index + 1, len(folded)):
                if SCOPE_END.match(folded[j]) or folded[j] in forward:
                    break
                marked[j] = marked[j] + NEG_SUFFIX
        elif lemma in backward:
            for j in range(index - 1, -1, -1):
                if SCOPE_END.match(folded[j]):
                    break
                marked[j] = marked[j] + NEG_SUFFIX
    return marked


def count_hits(lemmas: Sequence[str], lexicon: Lexicon) -> tuple[int, list[str]]:
    """Count a lexicon's entries in a lemma sequence, multiword included.

    Multiword entries are matched as consecutive lemma n-grams, longest first,
    and a matched span is consumed so ``áldását adja`` is not also counted as
    two single words.

    Args:
        lemmas: Lemmas, already case-folded and negation-marked if wanted.
        lexicon: The list to match.

    Returns:
        A ``(count, matched)`` pair.

    Example:
        >>> lex = Lexicon("joy", frozenset({"öröm"}),
        ...               frozenset({("jó", "hír")}), Path("x"), "ab")
        >>> count_hits(["ez", "jó", "hír", "és", "öröm"], lex)
        (2, ['jó hír', 'öröm'])
    """
    matched: list[str] = []
    used = [False] * len(lemmas)
    widths = sorted({len(entry) for entry in lexicon.multi}, reverse=True)
    for width in widths:
        for start in range(len(lemmas) - width + 1):
            if any(used[start : start + width]):
                continue
            window = tuple(lemmas[start : start + width])
            if window in lexicon.multi:
                matched.append(" ".join(window))
                for offset in range(width):
                    used[start + offset] = True
    for index, lemma in enumerate(lemmas):
        if not used[index] and lemma in lexicon.single:
            matched.append(lemma)
            used[index] = True
    return len(matched), matched


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Polarity of one text, with the counts behind it.

    Attributes:
        score: Mean over sentences of ``(pos - neg) / tokens``. Bounded by
            ``[-1, 1]`` and in practice far inside it, because the denominator
            is every token and not only the hits.
        positive: Positive hits, after negation flipping.
        negative: Negative hits, after negation flipping.
        flipped: Hits whose polarity a negation reversed.
        n_tokens: Tokens counted, punctuation excluded.
        n_sentences: Sentences scored.
        balanced: Whether inverse-class weighting was applied.
    """

    score: float
    positive: int
    negative: int
    flipped: int
    n_tokens: int
    n_sentences: int
    balanced: bool = False
    lexicon_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def polarity(self) -> float:
        """``(pos - neg) / (pos + neg)``, or 0.0 with no hits at all.

        The hit-only ratio, ignoring how much other text surrounded them. Use
        it to compare two texts' *balance* of sentiment words; use
        :attr:`score` to compare how *saturated* with them they are.
        """
        total = self.positive + self.negative
        return (self.positive - self.negative) / total if total else 0.0


EKMAN: tuple[str, ...] = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
"""Ekman's six basic emotions, the categories this project scores.

Putz Orsolya's collection also supplies ``feszültség`` and ``szeretet``. Both
are left aside: neither is an Ekman basic category, and *szeretet* in
particular measures something the other six do not.
"""

EKMAN_FILES: dict[str, str] = {
    "anger": "emo_duh_full.txt",
    "disgust": "emo_undor_full.txt",
    "fear": "emo_felelem_full.txt",
    "joy": "emo_orom_full.txt",
    "sadness": "emo_banat_full.txt",
    "surprise": "emo_meglepodes_full.txt",
}
"""Which file backs each Ekman category. ``_full`` carries the multiword
entries, which :func:`count_hits` matches rather than discards."""


@dataclass(frozen=True, slots=True)
class EmotionResult:
    """Emotion densities of one text.

    Attributes:
        rates: Hits per token, per category.
        counts: Raw hits per category.
        n_tokens: Tokens counted.
        n_hits: Total hits across categories. Categories overlap -- 48 entries
            appear in more than one list -- so this can exceed the number of
            distinct tokens matched.
        n_sentences: Sentences scored.
        sparse: Whether the text fell below the hit threshold. Emotion words
            are 2.9% of content lemmas in this corpus, so a short speech can
            produce a confident-looking 0.0 from no evidence at all.
    """

    rates: dict[str, float]
    counts: dict[str, int]
    n_tokens: int
    n_hits: int
    n_sentences: int
    sparse: bool
    lexicon_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dominant(self) -> str | None:
        """The highest-scoring category, or ``None`` when there are no hits."""
        if not self.n_hits:
            return None
        return max(self.counts, key=lambda k: self.counts[k])


MIN_EMOTION_HITS = 5
"""Below this many hits a text is flagged :attr:`EmotionResult.sparse`.

Only 61% of this corpus's speeches reach it. The flag exists because a lexicon
reports 0.0 both for "no anger" and for "no evidence either way", and those are
not the same claim.
"""


def _content(lemmas: Sequence[str], tags: Sequence[str]) -> list[str]:
    """Drop punctuation, keeping the lemma stream the scorers count over."""
    return [
        lemma
        for lemma, tag in zip(lemmas, tags, strict=True)
        if not (tag.startswith("[") and "Punct" in tag)
    ]


def score_sentiment(
    sentences: Sequence[tuple[Sequence[str], Sequence[str]]],
    positive: Lexicon,
    negative: Lexicon,
    *,
    balanced: bool = False,
) -> SentimentResult:
    """Score a text's polarity, sentence by sentence.

    Follows the reference implementation: mark negation, flip the polarity of
    marked hits, score each sentence as ``(pos - neg) / tokens``, and average
    over sentences.

    Args:
        sentences: One ``(lemmas, xpostags)`` pair per sentence. Punctuation
            must be present -- it is what closes a negation's scope.
        positive: The positive list.
        negative: The negative list.
        balanced: Weight each class by the inverse of its list size. The
            Precognox list is 3.4x larger on the negative side, so an
            unweighted count is biased negative by construction. Defaults to
            ``False`` so the raw count is the reported default and the
            correction is a visible choice.

    Returns:
        A :class:`SentimentResult`.

    Raises:
        ValueError: If there are no sentences.

    Example:
        >>> pos = Lexicon("positive", frozenset({"jó"}), frozenset(), Path("p"), "a")
        >>> neg = Lexicon(
        ...     "negative", frozenset({"probléma"}), frozenset(), Path("n"), "b"
        ... )
        >>> tags = ["[/Adj]", "[/N]", "[Punct]"]
        >>> score_sentiment([(["ez", "jó", "."], tags)], pos, neg).positive
        1

        A negated negative counts positive -- ``nem probléma`` is not a
        complaint:

        >>> result = score_sentiment(
        ...     [(["nem", "probléma", "."], ["[/Adv]", "[/N]", "[Punct]"])], pos, neg
        ... )
        >>> result.positive, result.negative, result.flipped
        (1, 0, 1)
    """
    if not sentences:
        raise ValueError("cannot score a text with no sentences")

    weight = 1.0
    if balanced and negative.size:
        weight = positive.size / negative.size

    scores: list[float] = []
    pos_total = neg_total = flipped = tokens_total = 0
    for lemmas, tags in sentences:
        marked = mark_negation([fold(lemma) for lemma in lemmas])
        content = _content(marked, tags)
        if not content:
            continue
        plain = [t for t in content if not t.endswith(NEG_SUFFIX)]
        negated = [t[: -len(NEG_SUFFIX)] for t in content if t.endswith(NEG_SUFFIX)]

        p_plain, _ = count_hits(plain, positive)
        n_plain, _ = count_hits(plain, negative)
        # Inside a negation the polarity reverses.
        p_flip, _ = count_hits(negated, negative)
        n_flip, _ = count_hits(negated, positive)

        pos = p_plain + p_flip
        neg = n_plain + n_flip
        pos_total += pos
        neg_total += neg
        flipped += p_flip + n_flip
        tokens_total += len(content)
        scores.append((pos - weight * neg) / len(content))

    return SentimentResult(
        score=round(sum(scores) / len(scores), 6) if scores else 0.0,
        positive=pos_total,
        negative=neg_total,
        flipped=flipped,
        n_tokens=tokens_total,
        n_sentences=len(scores),
        balanced=balanced,
        lexicon_ids=(positive.lexicon_id, negative.lexicon_id),
    )


def score_emotion(
    sentences: Sequence[tuple[Sequence[str], Sequence[str]]],
    lexicons: dict[str, Lexicon],
    *,
    min_hits: int = MIN_EMOTION_HITS,
) -> EmotionResult:
    """Score a text's emotion densities.

    Negation is **not** applied. A negated emotion word is not the opposite
    emotion -- ``nem félek`` is not joy -- so the polarity flip that makes
    sense for sentiment has no counterpart here, and marking without flipping
    would only drop hits.

    Args:
        sentences: One ``(lemmas, xpostags)`` pair per sentence.
        lexicons: Category name to its list.
        min_hits: Below this total, the result is flagged sparse. Defaults to
            :data:`MIN_EMOTION_HITS`.

    Returns:
        An :class:`EmotionResult`.

    Raises:
        ValueError: If there are no sentences.

    Example:
        >>> lex = {"joy": Lexicon("joy", frozenset({"öröm"}), frozenset(),
        ...                       Path("j"), "a")}
        >>> tags = ["[/N]", "[/N]", "[Punct]"]
        >>> result = score_emotion([(["nagy", "öröm", "."], tags)], lex, min_hits=1)
        >>> result.counts["joy"], result.dominant, result.sparse
        (1, 'joy', False)
    """
    if not sentences:
        raise ValueError("cannot score a text with no sentences")

    counts = dict.fromkeys(lexicons, 0)
    tokens_total = 0
    scored = 0
    for lemmas, tags in sentences:
        content = _content([fold(lemma) for lemma in lemmas], tags)
        if not content:
            continue
        scored += 1
        tokens_total += len(content)
        for name, lexicon in lexicons.items():
            hits, _ = count_hits(content, lexicon)
            counts[name] += hits

    total_hits = sum(counts.values())
    return EmotionResult(
        rates={
            name: round(n / tokens_total, 6) if tokens_total else 0.0
            for name, n in counts.items()
        },
        counts=counts,
        n_tokens=tokens_total,
        n_hits=total_hits,
        n_sentences=scored,
        sparse=total_hits < min_hits,
        lexicon_ids=tuple(lex.lexicon_id for lex in lexicons.values()),
    )


DEFAULT_REGISTER_PERCENTILE = 0.0
"""Frequency-based register filter. **Off by default, and superseded.**

It existed because the lists are built for general Hungarian while this corpus
is parliamentary: ``jó``, ``támogatás``, ``kedves`` and ``segít`` in the joy
list, ``vita`` in the anger list, between them made joy the dominant emotion in
62% of speeches.

Those entries have since been removed from the working copies of the
dictionaries by ``scripts/curate_emotion_lexicons.py``, word by word and with a
reason recorded for each — a better instrument than a frequency cut, because it
judges ``fél`` on being three words at once rather than on being common. With
the curated lists a 1% cut now removes nothing at all.

The mechanism is kept for a changed or uncurated list, but the default is 0 so
that there is exactly one place where entries are excluded: the dictionary and
its recorded curation, not a second filter behind it.
"""


def register_vocabulary(
    frequencies: dict[str, int], percentile: float = DEFAULT_REGISTER_PERCENTILE
) -> frozenset[str]:
    """Return the lemmas in the top ``percentile`` share of corpus frequency.

    Args:
        frequencies: Lemma to corpus count.
        percentile: Share of the ranked vocabulary to treat as register.
            Defaults to 1.0 (the top 1%). 0 disables the filter.

    Returns:
        The lemmas to exclude.

    Example:
        >>> freq = {"a": 100, "b": 50, "c": 10, "d": 1}
        >>> sorted(register_vocabulary(freq, percentile=50))
        ['a', 'b']
        >>> register_vocabulary(freq, percentile=0)
        frozenset()
    """
    if percentile <= 0:
        return frozenset()
    ranked = sorted(frequencies, key=lambda w: -frequencies[w])
    cut = max(1, int(len(ranked) * percentile / 100))
    return frozenset(ranked[:cut])


def without(lexicon: Lexicon, excluded: frozenset[str]) -> tuple[Lexicon, list[str]]:
    """Drop entries from a lexicon, reporting what went.

    Only single-word entries are filtered: a multiword expression is not
    register vocabulary however common its parts are.

    Args:
        lexicon: The list.
        excluded: Lemmas to remove.

    Returns:
        A ``(lexicon, dropped)`` pair. The returned lexicon keeps the original
        path and hash, so provenance still points at the file on disk; what was
        removed is in the second element and belongs in the manifest.

    Example:
        >>> lex = Lexicon("joy", frozenset({"jó", "öröm"}), frozenset(),
        ...               Path("j"), "abc")
        >>> kept, dropped = without(lex, frozenset({"jó"}))
        >>> sorted(kept.single), dropped
        (['öröm'], ['jó'])
    """
    dropped = sorted(lexicon.single & excluded)
    if not dropped:
        return lexicon, []
    return (
        Lexicon(
            name=lexicon.name,
            single=lexicon.single - excluded,
            multi=lexicon.multi,
            path=lexicon.path,
            sha256=lexicon.sha256,
        ),
        dropped,
    )
