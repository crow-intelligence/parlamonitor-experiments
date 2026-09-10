"""Share of foreign-origin lemmas -- *idegenszó-arány* -- per speech.

A register measure rather than a difficulty measure: it counts how much of a
text is drawn from vocabulary a Hungarian reader may have to have learned
separately. Political speech uses it heavily and unevenly, which is what makes
it worth counting per speaker.

**The lexicon is an argument, never a default.** saphes requires it and ships
none, and that is the right design: the number is a property of the word list
as much as of the text, so the list has to be named. :func:`load_lexicon`
reads one from disk and returns it with its hash, and every result carries
``lexicon_id``.

Two properties of the list this project uses, both of which shape the number:

* It was built by **frequency rank**, not by dictionary inclusion. The top 2%
  by corpus frequency was dropped on the reading that a word used that heavily
  is ordinary vocabulary rather than an *idegen szó*, and so were hapaxes. So
  this measures **uncommon** foreign vocabulary; ``probléma`` and ``program``
  are foreign in origin and are not in it.
* Proper nouns were filtered out of the list, but not out of a text. A foreign
  politician's surname would otherwise count as a loanword every time it is
  said, which in this corpus is often.

**Lemmas, not surface forms.** Hungarian morphology hides the root, so an
inflected stream misses the lexicon and returns a plausible, low ratio with no
error at all. That is the module's silent failure and the reason
:func:`measure` refuses a bare string.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from saphes import LoanwordResult, loanword_ratio

PROPER_NOUN_TAG = "PROPN"
"""Tag saphes excludes by default, and what the proxy below emits."""


@dataclass(frozen=True, slots=True)
class Lexicon:
    """A loaded word list, with enough provenance to reproduce a number.

    Attributes:
        entries: The lemmas, case-folded.
        path: Where it was read from.
        sha256: Hex digest of the file.
        size: How many entries it holds.
    """

    entries: frozenset[str]
    path: Path
    sha256: str
    size: int

    @property
    def lexicon_id(self) -> str:
        """Short identifier recorded on every result."""
        return f"{self.path.name}@{self.sha256[:12]}"


def load_lexicon(path: Path | str) -> Lexicon:
    """Read a newline-separated lemma list.

    Blank lines and ``#`` comments are skipped; entries are case-folded, which
    matches what :func:`saphes.loanword_ratio` does to the lemmas.

    Args:
        path: The word list.

    Returns:
        A :class:`Lexicon`.

    Raises:
        FileNotFoundError: If the file is missing. Named explicitly rather than
            falling back to an empty set, which would report every text as 0%
            foreign and look like a finding.
        ValueError: If the file holds no usable entries.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no loanword lexicon at {path}; saphes ships none and this "
            f"measure cannot be computed without one"
        )
    raw = path.read_bytes()
    entries = frozenset(
        line.strip().casefold()
        for line in raw.decode("utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    if not entries:
        raise ValueError(f"loanword lexicon at {path} is empty")
    return Lexicon(
        entries=entries,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(entries),
    )


def proper_noun_tags(lemmas: Sequence[str]) -> list[str]:
    """Guess which lemmas are proper nouns, from emtsv's lemma casing.

    A **proxy, not named-entity recognition.** emtsv lowercases the lemma of a
    sentence-initial common noun -- ``Taps`` becomes ``taps``, ``Az`` becomes
    ``az`` -- while leaving a proper noun's case alone, so a capitalised lemma
    is a usable signal for one. It will misfile a genuinely capitalised common
    noun and any proper noun the analyser lowercased.

    The alternative is worse: without it, every occurrence of a foreign
    surname counts as a loan word, and in a parliamentary corpus that is a
    large and systematic error.

    Args:
        lemmas: The lemma stream.

    Returns:
        One tag per lemma: :data:`PROPER_NOUN_TAG` for a capitalised lemma,
        ``"X"`` otherwise.

    Example:
        >>> proper_noun_tags(["taps", "Fidesz", "kormány", "Magyar"])
        ['X', 'PROPN', 'X', 'PROPN']
    """
    return [PROPER_NOUN_TAG if lemma[:1].isupper() else "X" for lemma in lemmas]


def measure(lemmas: Sequence[str], lexicon: Lexicon) -> LoanwordResult | None:
    """Measure one text's share of foreign-origin lemmas.

    Args:
        lemmas: The lemma stream, from emtsv. Not surface forms.
        lexicon: A loaded :class:`Lexicon`.

    Returns:
        The :class:`saphes.LoanwordResult`, carrying the ratio, the matched
        lemmas and every parameter that moved it. ``None`` when the text has no
        lemmas left after exclusions -- a speech that is one proper noun has no
        measurable ratio, and 0.0 would place it at the native end of a scale
        it is not on.

    Raises:
        TypeError: If ``lemmas`` is a raw string.

    Example:
        >>> lexicon = Lexicon(frozenset({"abszurd"}), Path("x"), "deadbeef", 1)
        >>> result = measure(["a", "javaslat", "abszurd"], lexicon)
        >>> round(result.ratio, 4), result.matched, sorted(result.matches)
        (0.3333, 1, ['abszurd'])

        A capitalised lemma is treated as a proper noun and excluded rather
        than counted, so a foreign surname does not inflate the ratio:

        >>> result = measure(["Abszurd", "javaslat"], lexicon)
        >>> result.ratio, result.matched, result.excluded
        (0.0, 0, 1)
    """
    if isinstance(lemmas, str):
        raise TypeError("loanword ratio needs a lemma sequence, not a string")
    if not lemmas:
        return None
    tags = proper_noun_tags(lemmas)
    if all(tag == PROPER_NOUN_TAG for tag in tags):
        return None
    return loanword_ratio(
        list(lemmas),
        lexicon=lexicon.entries,
        pos_tags=tags,
        lexicon_id=lexicon.lexicon_id,
    )
