"""Text normalisation applied on top of the export's own ``text_clean``.

The export already removes the speaker attribution and the editorial stage
directions. What it leaves in is the chamber's opening salutation --
``Tisztelt Elnök Asszony! Tisztelt Képviselőtársaim!`` -- which opens 67% of
cycle-43 speeches and accounts for 9,016 words.

That formula is the same regardless of what the speech is about, so for short
speeches it can dominate the sentence embedding and cluster them together by
politeness rather than subject. The first topic model built on this corpus
produced exactly that: a 66-speech topic whose top terms were ``köszön_szép``,
``köszön_elnök_asszony``, ``köszön_szó``.

Stripping it changes the numbers. :data:`NORMALISATION_VERSION` exists so that
caches keyed on it are invalidated when the rule changes.
"""

from __future__ import annotations

import re

NORMALISATION_VERSION = "strip-salutation-1"
"""Identifier for the current normalisation, for cache keys.

Bump this whenever :func:`strip_salutation` changes behaviour, so that derived
caches (lemmas, embeddings) are recomputed instead of silently mixing rules.
"""

_SALUTATION = re.compile(
    r"^\s*(?:(?:köszönöm|köszönjük|köszönet)[^.!?]*[.!?]\s*"
    r"|tisztelt[^.!?]*[.!?]\s*)+",
    re.IGNORECASE,
)


def strip_salutation(text: str, *, max_words: int = 60) -> tuple[str, int]:
    """Remove the opening salutation from a speech.

    Matches a run of leading clauses that begin with ``köszönöm``/``köszönjük``
    or ``tisztelt``, in any order and any number, each ending at the first
    sentence-final punctuation.

    Two guards keep it from eating content. The match is abandoned if it would
    remove more than ``max_words`` words, and again if it would leave nothing
    at all -- one cycle-43 speech consists of a salutation and nothing else.
    In both cases the original text is returned unchanged and the reported
    count is zero, so a caller summing the counts always gets the truth.

    Args:
        text: A speech, normally the export's ``text_clean``.
        max_words: Refuse to strip more than this many words. Defaults to 60,
            comfortably above the longest genuine salutation observed (a
            four-clause address is about 20 words) and well below the
            307-word median speech.

    Returns:
        A ``(stripped_text, n_words_removed)`` pair.

    Example:
        >>> strip_salutation("Tisztelt Elnök Asszony! A költségvetés hiánya nő.")
        ('A költségvetés hiánya nő.', 3)

        Several clauses, in any combination:

        >>> text = "Köszönöm a szót. Tisztelt Ház! Az egészségügyről szólnék."
        >>> strip_salutation(text)
        ('Az egészségügyről szólnék.', 5)

        Substantive text is left alone:

        >>> strip_salutation("A kormány benyújtotta a javaslatot.")
        ('A kormány benyújtotta a javaslatot.', 0)

        And a speech that is nothing but salutation survives intact:

        >>> strip_salutation("Tisztelt Elnök Úr!")
        ('Tisztelt Elnök Úr!', 0)
    """
    match = _SALUTATION.match(text)
    if match is None:
        return text, 0

    removed = len(match.group(0).split())
    if removed > max_words:
        return text, 0

    stripped = text[match.end() :]
    if not stripped.strip():
        return text, 0
    return stripped, removed
