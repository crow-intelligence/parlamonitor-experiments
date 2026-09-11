"""Virtue **salience**: which moral vocabulary a speaker reaches for.

Not whether they have the virtue. That distinction is the whole design, and it
is forced by the data rather than chosen for modesty.

Counting virtue words cannot measure virtue, for three reasons this corpus
demonstrates:

* **The sign inverts.** 60% of truthfulness vocabulary here is the accusing
  pole — ``hazugság``, ``megtéveszt``, ``félrevezet``. An MP who repeatedly
  calls opponents liars would score *highest* on truthfulness under a count.
* **Irony is invisible to a word list.** ``a Fidesz-KDNP rendkívül
  nagyvonalúan járt el a politikai vezetők bérének megállapításakor`` is
  sarcasm about politicians' pay, and reads to a counter as magnanimity.
* **Attribution is unrecoverable.** A speaker may claim a virtue, deny it of an
  opponent, or demand it of the house. ``próbáljon meg egy picit
  emelkedettebben hozzáállni`` is an instruction, not a display.

What survives all three is the weaker, honest claim: **this speaker keeps
returning to this virtue, so it is part of how they argue.** Accusing an
opponent of lying still invokes truthfulness as a norm; calling someone a
coward presupposes that courage counts. Sincere praise and sarcastic attack are
both engagement with the virtue, which is what is being measured.

The stance is nevertheless recorded. Every virtue has an ``affirming`` and an
``accusing`` list, and the two are always reported separately — a single number
would hide that truthfulness talk in this chamber is mostly accusation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from parlamonitor.lexicon import Lexicon, count_hits, fold

POLES = ("affirming", "accusing")
"""The two stances a virtue word can take. Both are salience."""


@dataclass(frozen=True, slots=True)
class Virtue:
    """One virtue's two word lists, with its labels.

    Attributes:
        key: Identifier, e.g. ``"courage"``.
        label: English name.
        label_hu: Hungarian name, for display.
        greek: The classical term, or empty.
        gloss: One-line definition.
        affirming: Words invoking the virtue positively.
        accusing: Words invoking its absence or opposite.
        note: Anything a reader must know before using the numbers.
    """

    key: str
    label: str
    label_hu: str
    greek: str
    gloss: str
    affirming: Lexicon
    accusing: Lexicon
    note: str = ""


@dataclass(frozen=True, slots=True)
class SalienceResult:
    """How much of a text engages with each virtue.

    Attributes:
        counts: Virtue to ``{"affirming": n, "accusing": n}``.
        rates: Virtue to total mentions per 1,000 tokens.
        n_tokens: Tokens counted, punctuation excluded.
        n_mentions: Total virtue words found.
        sparse: Whether the text is below the mention threshold. A lexicon
            reports 0.0 both for "does not talk about courage" and for "too
            short to tell", and those are different claims.
    """

    counts: dict[str, dict[str, int]]
    rates: dict[str, float]
    n_tokens: int
    n_mentions: int
    sparse: bool

    def stance(self, virtue: str) -> float | None:
        """Share of a virtue's mentions that affirm rather than accuse.

        Returns:
            ``1.0`` when every mention affirms, ``0.0`` when every mention
            accuses, ``None`` when the virtue is not mentioned at all.

        Example:
            >>> result = SalienceResult(
            ...     {"courage": {"affirming": 3, "accusing": 1}},
            ...     {"courage": 4.0}, 1000, 4, False,
            ... )
            >>> result.stance("courage")
            0.75
            >>> result.stance("justice") is None
            True
        """
        counts = self.counts.get(virtue)
        if not counts:
            return None
        total = counts["affirming"] + counts["accusing"]
        return counts["affirming"] / total if total else None


MIN_MENTIONS = 3
"""Below this many virtue words a text is flagged sparse.

Virtue vocabulary is rarer than emotion vocabulary, so this is lower than
:data:`parlamonitor.lexicon.MIN_EMOTION_HITS`. Salience is meant to be read at
speaker and topic level regardless; the per-speech flag exists so a short
procedural remark is not reported as caring about nothing.
"""


def load_virtues(path: Path | str) -> dict[str, Virtue]:
    """Read the virtue lexicon.

    Args:
        path: The JSON lexicon.

    Returns:
        Virtue key to :class:`Virtue`, in file order.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If a virtue lacks either pole — the design requires both,
            since a single list is what makes the sign invert.

    Example:
        >>> virtues = load_virtues("data/lexicons/virtues/hu_virtues.json")
        >>> sorted(virtues)[:3]
        ['courage', 'justice', 'magnanimity']
        >>> "hazugság" in virtues["truthfulness"].accusing.single
        True
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no virtue lexicon at {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))

    virtues: dict[str, Virtue] = {}
    for key, entry in data.items():
        if key.startswith("_"):
            continue
        for pole in POLES:
            if not entry.get(pole):
                raise ValueError(
                    f"virtue {key!r} has no {pole!r} list; both poles are "
                    f"required, because one list alone is what lets the sign invert"
                )
        virtues[key] = Virtue(
            key=key,
            label=entry.get("label", key),
            label_hu=entry.get("label_hu", ""),
            greek=entry.get("greek", ""),
            gloss=entry.get("gloss", ""),
            note=entry.get("note", ""),
            **{
                pole: Lexicon(
                    name=f"{key}.{pole}",
                    single=frozenset(fold(w) for w in entry[pole] if " " not in w),
                    multi=frozenset(
                        tuple(fold(w).split()) for w in entry[pole] if " " in w
                    ),
                    path=path,
                    sha256=digest,
                )
                for pole in POLES
            },
        )
    return virtues


def score_salience(
    sentences: Sequence[tuple[Sequence[str], Sequence[str]]],
    virtues: dict[str, Virtue],
    *,
    min_mentions: int = MIN_MENTIONS,
) -> SalienceResult:
    """Count each virtue's vocabulary in a text.

    Negation is **not** applied. ``nem bátor`` is still talk about courage,
    and under a salience construct that is the point: the polarity flip that
    makes sense for sentiment would here erase the mention.

    Args:
        sentences: One ``(lemmas, xpostags)`` pair per sentence.
        virtues: As :func:`load_virtues` returns them.
        min_mentions: Below this total the result is flagged sparse.

    Returns:
        A :class:`SalienceResult`.

    Raises:
        ValueError: If there are no sentences.

    Example:
        >>> virtues = load_virtues("data/lexicons/virtues/hu_virtues.json")
        >>> tags = ["[/N][Nom]", "[/N][Nom]", "[Punct]"]
        >>> result = score_salience(
        ...     [(["bátorság", "hazugság", "."], tags)], virtues, min_mentions=1
        ... )
        >>> result.counts["courage"]["affirming"]
        1
        >>> result.counts["truthfulness"]["accusing"]
        1

        An accusation of lying counts as engagement with truthfulness, and the
        stance records which way:

        >>> result.stance("truthfulness")
        0.0
    """
    if not sentences:
        raise ValueError("cannot score a text with no sentences")

    counts = {key: dict.fromkeys(POLES, 0) for key in virtues}
    n_tokens = 0
    for lemmas, tags in sentences:
        content = [
            fold(lemma)
            for lemma, tag in zip(lemmas, tags, strict=True)
            if "Punct" not in tag
        ]
        if not content:
            continue
        n_tokens += len(content)
        for key, virtue in virtues.items():
            for pole in POLES:
                hits, _ = count_hits(content, getattr(virtue, pole))
                counts[key][pole] += hits

    total = sum(c["affirming"] + c["accusing"] for c in counts.values())
    return SalienceResult(
        counts=counts,
        rates={
            key: round((c["affirming"] + c["accusing"]) / n_tokens * 1000, 4)
            if n_tokens
            else 0.0
            for key, c in counts.items()
        },
        n_tokens=n_tokens,
        n_mentions=total,
        sparse=total < min_mentions,
    )
