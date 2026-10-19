"""A sitting day as a conversation: when each reaction happened, and from where.

The rules behind the dashboard's "Élő parlament" tab, kept here so they are
tested rather than buried in JavaScript:

* :func:`reaction_seconds` -- an estimated time for a reaction, from the
  speech's start and length and how far through its words the reaction fell;
* :func:`bench` -- which benches a reaction came from, which decides the
  margin it is drawn in;
* :func:`flow` -- whether it is the speaker's own side responding or the other
  side answering back, which decides whether it is drawn above or below the
  day's timeline.
"""

from __future__ import annotations

from collections.abc import Sequence

from parlamonitor.applause import Placed
from parlamonitor.reactions import Audience, side_of


def _words(text: str) -> int:
    return len(text.split())


def reaction_seconds(
    sentences: Sequence[str],
    sentence_index: int,
    cut: str | None,
    start_s: float,
    duration_s: float,
) -> float:
    """Estimate when a reaction happened, in seconds from the sitting's start.

    Assumes the speaker spoke at an even pace: the reaction falls at the share
    of the speech's words said by the end of its sentence (or by the cut, when
    it interrupted the sentence). The export has no timing finer than the
    speech, so this is an estimate, and it is labelled as one where shown.

    Args:
        sentences: The speech's sentences.
        sentence_index: The sentence the reaction followed.
        cut: For an interruption, the sentence up to the cut; else ``None``.
        start_s: When the speech began, in seconds from the sitting's start.
        duration_s: How long it lasted.

    Returns:
        A time inside ``[start_s, start_s + duration_s]``.

    Example:
        >>> reaction_seconds(["Egy két há.", "Négy öt."], 0, None, 100.0, 50.0)
        130.0
        >>> reaction_seconds(["Egy két há.", "Négy öt."], 1, "Négy", 100.0, 50.0)
        140.0
    """
    total = sum(_words(s) for s in sentences) or 1
    said = sum(_words(s) for s in sentences[:sentence_index])
    said += _words(cut) if cut is not None else _words(sentences[sentence_index])
    return start_s + duration_s * min(1.0, said / total)


def bench(placed: Placed, interjector_side: str | None, cycle: int) -> str | None:
    """Which benches a reaction came from.

    One of ``government``, ``opposition``, ``both``, or ``None``.

    The note's own attribution wins (*Taps a TISZA soraiból*). A named
    interjector with no bench in the note takes their own side, resolved by
    the caller from the speeches export. A note that names no bench and no
    person -- *Taps.*, *Derültség.* -- is ``None``: it is not guessed.

    Args:
        placed: The reaction.
        interjector_side: The named interjector's side, if one was resolved.
        cycle: The parliamentary cycle, for mapping parties to sides.

    Example:
        >>> from parlamonitor.applause import placed_reactions
        >>> p = placed_reactions({"speaker_prefix": None, "text":
        ...     "Egy. (Taps a Fidesz soraiban.) Kettő.",
        ...     "sentences": ["Egy.", "Kettő."]})[0]
        >>> bench(p, None, 43)
        'opposition'
        >>> q = placed_reactions({"speaker_prefix": None, "text":
        ...     "Egy. (Taps.) Kettő.", "sentences": ["Egy.", "Kettő."]})[0]
        >>> bench(q, None, 43) is None
        True
    """
    sides: set[str] = set()
    if Audience.GOVERNMENT.value in placed.audiences:
        sides.add("government")
    if Audience.OPPOSITION.value in placed.audiences:
        sides.add("opposition")
    sides.update(side_of(p, cycle).value for p in placed.parties)
    if not sides and interjector_side in {"government", "opposition"}:
        sides.add(interjector_side)
    if not sides:
        return None
    return sides.pop() if len(sides) == 1 else "both"


def flow(
    placed: Placed, reaction_bench: str | None, speaker_side: str | None
) -> str | None:
    """Whether a reaction is the speaker's own side responding, or the other answering.

    * ``"own"`` -- from the speaker's own benches, or applause or laughter that
      names no bench (a bare *Taps.* almost always comes from the speaker's
      side);
    * ``"other"`` -- from the other benches, or a heckle, interjection, noise or
      uproar that names no bench;
    * ``None`` -- the chair's bell, the closing ritual, or a reaction from both
      sides at once: drawn in the transcript, not in the timeline's bars.

    Example:
        >>> from parlamonitor.applause import placed_reactions
        >>> p = placed_reactions({"speaker_prefix": None, "text":
        ...     "Egy. (Közbeszólás a Fidesz soraiból: Hazugság!) Kettő.",
        ...     "sentences": ["Egy.", "Kettő."]})[0]
        >>> flow(p, "opposition", "government"), flow(p, "opposition", "opposition")
        ('other', 'own')
    """
    if placed.closing or placed.kinds <= {"bell"} or reaction_bench == "both":
        return None
    if reaction_bench is not None and speaker_side is not None:
        return "own" if reaction_bench == speaker_side else "other"
    if placed.kinds & {"applause", "laughter"} and not placed.kinds & {
        "heckling",
        "interjection",
    }:
        return "own"
    if placed.kinds & {
        "heckling",
        "interjection",
        "noise",
        "uproar",
        "booing",
        "whistling",
    }:
        return "other"
    return None
