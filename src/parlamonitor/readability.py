"""Readability and lexical diversity of parliamentary speech, via saphes.

Three choices here change every number, so all three are stated rather than
inherited from a default.

**The long-word threshold is 8, not 6.** LIX counts words longer than a
threshold, and the familiar value of 6 is Swedish. ``saphes.recommended_
threshold("hu")`` returns 8 for Hungarian, which is agglutinative and has
longer words at every percentile. Running LIX on Hungarian at 6 makes every
text look harder than it is, and makes the score incomparable with published
Hungarian figures.

**Length is counted in letters, not characters.** Hungarian digraphs -- ``cs``,
``gy``, ``ly``, ``ny``, ``sz``, ``ty``, ``zs`` and the trigraph ``dzs`` -- are
single letters. ``egészségügy`` is 11 characters and 9 letters;
``dzsungel`` is 8 and 6. :func:`saphes.hungarian_letter_count` implements this,
and using ``len()`` instead would inflate LIX for exactly the polysyllabic
policy vocabulary this corpus is full of.

**Sentences come from emtsv, not from punctuation.** A regex splitter breaks on
``dr.``, on ``2026.`` and on the section marks that fill legislative debate,
and LIX is words-per-sentence, so a bad split moves the score directly.

What LIX measures is surface complexity -- long words in long sentences. It is
not comprehension, and a fluent speaker of dense policy language scores as
"difficult" while saying something perfectly clear to the chamber. The bands
are Björnsson's, calibrated on Swedish prose, and are reported because they are
conventional, not because they are authoritative for Hungarian parliamentary
speech.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from saphes import (
    hungarian_letter_count,
    interpret_lix,
    lexical_diversity,
    lix_from_counts,
    recommended_threshold,
)

HUNGARIAN_LONG_WORD_THRESHOLD: int = recommended_threshold("hu").threshold
"""Words longer than this count as long, for LIX and RIX. 8 for Hungarian."""

MAX_PLAUSIBLE_WORDS_PER_SENTENCE = 60
"""Above this, the text almost certainly lacks sentence punctuation.

Not every record in the export is prose. The notary's roll-call is 590 words of
names with no full stop, which emtsv reads as one sentence and LIX scores at
594 -- ten times the "very difficult" band. Hungarian parliamentary prose runs
15-25 words per sentence, and the longest genuine speeches here stay under 30,
so 60 is far above anything real and well below the artefacts.

Texts over the line are flagged, not dropped: the count is still true, and
which records are prose is the reader's call.
"""

MATTR_WINDOW = 100
"""Window for the moving-average type-token ratio.

MATTR is used rather than a bare type-token ratio because TTR falls as a text
grows, so it would rank speakers by how long they spoke. A 100-token window is
the usual choice and makes a two-minute intervention comparable with a
twenty-minute one. Speeches shorter than the window get a TTR instead, flagged
by :attr:`ReadabilityResult.mattr_windowed`.
"""


@dataclass(frozen=True, slots=True)
class ReadabilityResult:
    """Readability and diversity of one text, with the counts behind them.

    Attributes:
        n_words: Word tokens, punctuation excluded.
        n_sentences: Sentences, as emtsv segmented them.
        n_long_words: Words longer than :data:`HUNGARIAN_LONG_WORD_THRESHOLD`
            letters.
        lix: Björnsson's LIX.
        lix_band: The conventional interpretation band for that score.
        rix: Long words per sentence.
        words_per_sentence: Mean sentence length.
        mattr: Moving-average type-token ratio over lemmas.
        mattr_windowed: Whether the text was long enough for the window. When
            ``False``, ``mattr`` is a plain type-token ratio over the whole
            text and is not comparable with a windowed one.
        n_types: Distinct lemmas.
        readability_reliable: Whether ``lix`` and ``rix`` are trustworthy.
            ``False`` when the text exceeds
            :data:`MAX_PLAUSIBLE_WORDS_PER_SENTENCE`, which means it has no
            sentence punctuation -- a roll-call, a list of names, a table.
        long_word_threshold: The threshold used, carried so a score is never
            read without it.
        mattr_window: The window used.
    """

    n_words: int
    n_sentences: int
    n_long_words: int
    lix: float
    lix_band: str
    rix: float
    words_per_sentence: float
    mattr: float
    mattr_windowed: bool
    n_types: int
    readability_reliable: bool = True
    long_word_threshold: int = HUNGARIAN_LONG_WORD_THRESHOLD
    mattr_window: int = MATTR_WINDOW
    notes: tuple[str, ...] = field(default_factory=tuple)


def is_long_word(word: str, threshold: int = HUNGARIAN_LONG_WORD_THRESHOLD) -> bool:
    """Report whether a word is long, counting Hungarian letters.

    Args:
        word: A word token.
        threshold: Letters a word must exceed. Defaults to 8.

    Returns:
        ``True`` if the word has more than ``threshold`` letters.

    Example:
        ``egészségügy`` is 11 characters but 9 letters, so it is long:

        >>> is_long_word("egészségügy")
        True

        ``asszony`` is 7 characters and 5 letters, so it is not -- a
        character count would have made it 7 and much closer to the line:

        >>> is_long_word("asszony")
        False
        >>> hungarian_letter_count("asszony")
        5
    """
    return hungarian_letter_count(word) > threshold


def measure(
    words: list[str],
    lemmas: list[str],
    n_sentences: int,
    *,
    threshold: int = HUNGARIAN_LONG_WORD_THRESHOLD,
    window: int = MATTR_WINDOW,
) -> ReadabilityResult:
    """Measure one text from pre-tokenised input.

    Takes tokens rather than a string so that emtsv's segmentation is used
    throughout and nothing is re-tokenised with a naive regex.

    Args:
        words: Word tokens, punctuation already removed.
        lemmas: Lemmas for the diversity measures. Usually the same tokens
            lemmatised; pass a content-word-filtered stream to measure
            diversity of content vocabulary alone.
        n_sentences: Sentence count from emtsv.
        threshold: Long-word threshold in letters. Defaults to 8.
        window: MATTR window. Defaults to 100.

    Returns:
        A :class:`ReadabilityResult`.

    Raises:
        ValueError: If there are no words, or no sentences. An empty text has
            no readability, and dividing by zero to produce one would be worse
            than saying so.

    Example:
        >>> words = ["A", "kormány", "benyújtotta", "a", "törvényjavaslatot"]
        >>> lemmas = ["a", "kormány", "benyújt", "a", "törvényjavaslat"]
        >>> result = measure(words, lemmas, n_sentences=1)

        Two words pass the 8-letter threshold: ``benyújtotta`` (10 letters,
        11 characters) and ``törvényjavaslatot`` (16 letters, 17 characters).
        ``kormány`` does not -- ``ny`` is one letter, so it is 6, not 7:

        >>> result.n_words, result.n_long_words
        (5, 2)

        LIX is ``words/sentences + long_words/words*100`` = ``5/1 + 2/5*100``:

        >>> result.lix
        45.0
        >>> result.lix_band
        'standard'

        Five words is far below the MATTR window, so a plain TTR is reported
        and flagged:

        >>> result.mattr_windowed
        False
        >>> round(result.mattr, 2), result.n_types
        (0.8, 4)

        A roll-call of names has no sentence punctuation, so its LIX is an
        artefact and is flagged as one:

        >>> roll = measure(["Kovács"] * 200, ["kovács"] * 200, n_sentences=1)
        >>> roll.readability_reliable
        False
        >>> roll.notes[0].endswith("LIX and RIX are not meaningful")
        True
    """
    if not words:
        raise ValueError("cannot measure a text with no words")
    if n_sentences <= 0:
        raise ValueError(f"n_sentences must be positive, got {n_sentences}")

    n_long = sum(1 for word in words if is_long_word(word, threshold))
    score = lix_from_counts(words=len(words), sentences=n_sentences, long_words=n_long)

    notes: list[str] = []
    windowed = len(lemmas) >= window
    if lemmas:
        diversity = lexical_diversity(
            lemmas, unit="lemma", window=window if windowed else None
        )
        # saphes reports both; `mattr` is populated only when a window was
        # given, so short texts fall back to the plain ratio. `or` rather than a
        # conditional because a windowed fit on a degenerate text can still
        # return None, and a silent None would become NaN three steps later.
        raw = diversity.mattr if windowed else None
        mattr = float(raw if raw is not None else diversity.ttr)
        n_types = diversity.types
    else:
        mattr, n_types = 0.0, 0
        notes.append("no lemmas; diversity reported as zero")
    if not windowed and lemmas:
        notes.append(f"only {len(lemmas)} lemmas; TTR reported, not windowed MATTR")

    words_per_sentence = len(words) / n_sentences
    reliable = words_per_sentence <= MAX_PLAUSIBLE_WORDS_PER_SENTENCE
    if not reliable:
        notes.append(
            f"{words_per_sentence:.0f} words per sentence; text has no sentence "
            f"punctuation, so LIX and RIX are not meaningful"
        )

    return ReadabilityResult(
        n_words=len(words),
        n_sentences=n_sentences,
        n_long_words=n_long,
        lix=round(score, 4),
        lix_band=interpret_lix(score),
        rix=round(n_long / n_sentences, 4),
        words_per_sentence=round(words_per_sentence, 4),
        mattr=round(mattr, 6),
        mattr_windowed=windowed,
        n_types=n_types,
        readability_reliable=reliable,
        long_word_threshold=threshold,
        mattr_window=window,
        notes=tuple(notes),
    )
