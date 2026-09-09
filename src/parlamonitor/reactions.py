"""Classifying what the chamber did: applause, laughter, heckling, disorder.

The note-takers' parentheticals are formulaic, which is what makes them
countable. A remark names an event and, usually, who produced it::

    Taps a kormánypártok soraiban.        applause, government benches
    Derültség az ellenzéki padsorokban.   laughter, opposition benches
    Szórványos taps a Fidesz soraiból.    applause, scattered, Fidesz
    Az elnök csenget.                     the chair's bell -- disorder
    Vadai Ágnes: Nem hallom!              a named interjection

This module turns those strings into records. Everything is rule-based and
every rule is a named constant, because the alternative -- a classifier -- would
put a model's guess between the transcript and the count, and there is nothing
here a model would do better. The vocabulary was read off the corpus's own
frequency table rather than invented: see ``REACTION_CUES``.

Three things it deliberately does not do:

* It does not guess who *caused* a reaction. The parenthetical says who
  reacted, never who provoked it; that link lives in the speeches export and is
  made in ``scripts/reaction_scores.py``.
* It does not resolve a bare ``Taps.`` to any bench. That is
  :data:`Audience.UNSPECIFIED`, not a default to "everyone".
* It does not map a party to government or opposition. That flips between
  cycles -- Fidesz-KDNP governed in 39-42, TISZA governs in 43 -- so it is a
  per-cycle input, in :data:`GOVERNING_PARTIES`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class Kind(StrEnum):
    """What sort of event a segment records."""

    APPLAUSE = "applause"
    LAUGHTER = "laughter"
    HECKLING = "heckling"
    NOISE = "noise"
    UPROAR = "uproar"
    WHISTLING = "whistling"
    BOOING = "booing"
    BELL = "bell"
    INTERJECTION = "interjection"
    OTHER = "other"


class Intensity(StrEnum):
    """How the note-taker qualified the event.

    Ordered from weakest to strongest as :data:`INTENSITY_WEIGHT` scores them,
    except ``PLAIN``, which is the unqualified default.
    """

    SCATTERED = "scattered"
    PLAIN = "plain"
    CONTINUOUS = "continuous"
    GREAT = "great"
    LONG = "long"
    STANDING = "standing"


class Audience(StrEnum):
    """Which benches the note-taker credited, when they credited any."""

    GOVERNMENT = "government"
    OPPOSITION = "opposition"
    PARTY = "party"
    HOUSE = "house"
    UNSPECIFIED = "unspecified"


REACTION_CUES: dict[Kind, str] = {
    # Ordered by specificity: the first match wins, so `whistling` must be
    # tried before `noise` (a whistle is also a noise) and `bell` before
    # `other`.
    Kind.WHISTLING: r"sípol|fütty|füttyög",
    Kind.BOOING: r"pfúj|pfuj",
    Kind.APPLAUSE: r"\btaps|tapsol|megtapsol",
    Kind.LAUGHTER: r"derülts|nevet|kacag|kuncog",
    Kind.HECKLING: r"közbeszól|közbekiált|bekiabál|kiabál|közbevet|bekiált",
    Kind.UPROAR: r"felzúdul|felháborod|háborg",
    Kind.NOISE: r"\bzaj\b|\bzajong|moraj|mormog|dörömb|pisszeg",
    Kind.BELL: r"cseng",
}
"""Regex cue per event kind, tried in this order against the lowercased text.

Read off the corpus frequency table, not invented. A segment may match more
than one cue -- ``Derültség és taps a kormánypártok soraiban.`` is both -- so
:attr:`Event.kinds` holds every match and
:attr:`Event.kind` the first. 1.7% of segments match more than one, which means
per-kind totals sum to slightly more than the number of events.

Counts across all five cycles: applause 90,038, heckling 27,480, whistling
21,198, the bell 17,240, noise 12,038, laughter 9,580, uproar 1,581, booing 946.
"""

INTENSITY_CUES: dict[Intensity, str] = {
    # Again first-match-wins, strongest first: "hosszan tartó nagy taps" is
    # long, not merely great.
    Intensity.STANDING: r"felállva|állva\s+taps|feláll\w*\s+és\s+taps",
    Intensity.LONG: r"hosszan\s+tartó|hosszú|kitartó",
    Intensity.GREAT: r"\bnagy\b|élénk|hatalmas|viharos|óriási|erős",
    Intensity.CONTINUOUS: r"folyamatos|szűnni\s+nem\s+akaró",
    Intensity.SCATTERED: r"szórványos|gyér|halk|néhány",
}
"""Regex cue per intensity, tried strongest first."""

INTENSITY_WEIGHT: dict[Intensity, float] = {
    Intensity.SCATTERED: 0.5,
    Intensity.PLAIN: 1.0,
    Intensity.CONTINUOUS: 1.5,
    Intensity.GREAT: 1.5,
    Intensity.LONG: 2.0,
    Intensity.STANDING: 3.0,
}
"""Weights for an intensity-weighted score.

These are a judgement, not a measurement: nothing in the transcript says a
standing ovation is worth three scattered claps. They are exposed so that any
score built on them can be rebuilt without them -- every table that uses a
weighted score also reports the unweighted count beside it.
"""

GOVERNMENT_TERMS = r"kormánypárt|kormányzó\s+párt|kormányoldal|kormánypárti"
"""How the note-takers name the government benches."""

OPPOSITION_TERMS = r"ellenzék|ellenzéki"
"""How the note-takers name the opposition benches."""

HOUSE_TERMS = (
    r"teremben\s+lévők|jelenlévők|egész\s+ház|valamennyi\s+képviselő|minden\s+oldal"
)
"""Phrases crediting the whole chamber rather than a side."""

PARTY_PATTERNS: dict[str, str] = {
    "Fidesz": r"\bFidesz\b",
    "KDNP": r"\bKDNP\b",
    "MSZP": r"\bMSZP\b",
    "Jobbik": r"\bJobbik\b",
    "LMP": r"\bLMP\b",
    "DK": r"\bDK\b",
    "Momentum": r"\bMomentum\b",
    "Párbeszéd": r"\bPárbeszéd\b",
    "Mi Hazánk": r"Mi\s+Hazánk",
    "TISZA": r"\bTISZA\b|\bTisza\b",
    "Együtt": r"\bEgyütt\b",
    "SZDSZ": r"\bSZDSZ\b",
    "MDF": r"\bMDF\b",
}
"""Party names as they appear in the parentheticals.

``Mi Hazánk`` and ``TISZA`` need their own patterns because a word-boundary
match on a single token would miss the first and, for ``Tisza``, also catch the
river. In this corpus the river does not come up.
"""

GOVERNING_PARTIES: dict[int, frozenset[str]] = {
    39: frozenset({"Fidesz", "KDNP"}),
    40: frozenset({"Fidesz", "KDNP"}),
    41: frozenset({"Fidesz", "KDNP"}),
    42: frozenset({"Fidesz", "KDNP"}),
    43: frozenset({"TISZA"}),
}
"""Which parties held government in each cycle.

Cycle 43 is derived from the export rather than assumed: the speeches file
gives ``miniszterelnök`` and every ministerial office to TISZA members, Magyar
Péter among them. Cycles 39-42 are the Fidesz-KDNP governments, consistent with
the parentheticals themselves, where ``Taps a kormánypártok soraiban`` and
``Taps a Fidesz és a KDNP soraiban`` are used interchangeably.

This mapping decides every own-side/other-side figure, so it is stated here
rather than buried in a script.
"""

# Abbreviations that end in a period without ending a sentence. Without these
# "Taps. Dr. Vadai Ágnes közbeszól." splits into three, and "Dr." becomes the
# 27,290-times-most-common "segment" in the corpus.
_ABBREVIATIONS = (
    "dr",
    "id",
    "ifj",
    "özv",
    "stb",
    "ún",
    "pl",
    "ill",
    "kb",
    "sz",
    "vö",
    "min",
    "áll",
    "ny",
)

_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ])",
)
_ABBREVIATION_END = re.compile(
    r"(?:^|\s)(?:" + "|".join(_ABBREVIATIONS) + r"|[A-ZÁÉÍÓÖŐÚÜŰ])\.$",
    re.IGNORECASE,
)
_EVENT_SPLIT = re.compile(r"\s+[-–—]\s+")

# "Vadai Ágnes: Nem hallom!" -- a named interjection. Up to four name tokens,
# each capitalised, followed by a colon. Anchored to the segment start so that
# a colon inside quoted speech cannot produce a spurious speaker.
#
# Case sensitivity is load-bearing and is scoped to the honorific rather than
# applied to the whole pattern: with a global IGNORECASE the capitalised-token
# classes also match lowercase, and "Közbeszólás az MSZP soraiból: Hazudik!"
# is read as a person called "Közbeszólás az MSZP soraiból" -- 3,541 events.
_NAMED_INTERJECTION = re.compile(
    r"^\s*(?:(?i:dr|id|ifj|özv)\.\s*)?"
    r"(?:[A-ZÁÉÍÓÖŐÚÜŰ]\.\s*)?"
    r"((?:[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű-]+\s+){1,3}[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű-]+)"
    r":\s+(.+)$",
)
# At least two name tokens: the transcripts record an interjector by full name,
# so a single capitalised word before a colon is procedure -- "Szünet: 14.19",
# "Elnök: Igen.", "Jelenlét-ellenőrzés: jelen van 190" -- 1,538 events of it.
# The cost is the handful of times a member is named by surname alone (5 events
# across five cycles), which is the better side of that trade.

# A capitalised phrase before a colon is only a person if it is not one of the
# ways the note-takers name a reaction or a bench: "Közbeszólás:", "Egy hang
# a Fidesz soraiból:", "Többen:".
_NOT_A_PERSON = re.compile(
    r"közbeszól|közbekiált|bekiabál|taps|derülts|moraj|\bzaj|felzúdul|sípol"
    r"|pfúj|hang\b|sorai|padsor|oldalról|frakció|képvisel|többen|ugyanonnan"
    r"|ugyanott|helyről|padsorokból|padsorokban",
    re.IGNORECASE,
)

# "Közbeszólás az MSZP soraiból: Hazudik!" -- what was shouted is worth keeping
# even when the note-taker credited a bench rather than a person.
_UNNAMED_QUOTE = re.compile(r"^(?P<prefix>[^:]{1,90}):\s+(?P<quote>.+)$")


@dataclass(frozen=True, slots=True)
class Event:
    """One classified segment of a parenthetical.

    Attributes:
        text: The segment, verbatim.
        kind: The highest-priority kind matched, for when one label is needed.
        kinds: Every kind the segment matched. Prefer this when counting --
            ``Derültség és taps`` is both a laugh and a round of applause, and
            dropping either would understate one of them.
        intensity: How the note-taker qualified it.
        audiences: The bench categories credited, in no particular order.
        parties: Party names credited, if any.
        speaker: For :attr:`Kind.INTERJECTION`, who was quoted; otherwise
            ``None``. Not the person who caused the reaction -- the person who
            made the noise.
        quote: For :attr:`Kind.INTERJECTION`, what they said.
    """

    text: str
    kind: Kind
    kinds: frozenset[Kind] = field(default_factory=frozenset)
    intensity: Intensity = Intensity.PLAIN
    audiences: frozenset[Audience] = field(default_factory=frozenset)
    parties: frozenset[str] = field(default_factory=frozenset)
    speaker: str | None = None
    quote: str | None = None

    @property
    def weight(self) -> float:
        """The intensity weight, from :data:`INTENSITY_WEIGHT`."""
        return INTENSITY_WEIGHT[self.intensity]


def split_events(line: str) -> list[str]:
    r"""Split one parenthetical into the events it records.

    The note-takers separate events with `` - `` and, within that, with
    sentence-final punctuation. Abbreviations and initials are protected, so
    ``Dr.`` and the ``Z.`` of ``Z. Kárpát Dániel`` do not end a sentence.

    Args:
        line: One parenthetical, as the file holds it.

    Returns:
        The segments, stripped, in order. Empty segments are dropped.

    Example:
        >>> split_events("Taps a kormánypártok soraiban. - Derültség.")
        ['Taps a kormánypártok soraiban.', 'Derültség.']

        Sentences split too:

        >>> split_events("Taps. Az elnök csenget.")
        ['Taps.', 'Az elnök csenget.']

        But an abbreviation does not end one:

        >>> split_events("Taps. Dr. Vadai Ágnes közbeszól.")
        ['Taps.', 'Dr. Vadai Ágnes közbeszól.']

        >>> split_events("Z. Kárpát Dániel: Nem támadjuk!")
        ['Z. Kárpát Dániel: Nem támadjuk!']
    """
    segments: list[str] = []
    for chunk in _EVENT_SPLIT.split(line.strip()):
        pieces = _SENTENCE_SPLIT.split(chunk)
        merged: list[str] = []
        for piece in pieces:
            if merged and _ABBREVIATION_END.search(merged[-1]):
                merged[-1] = f"{merged[-1]} {piece}"
            else:
                merged.append(piece)
        segments.extend(piece.strip() for piece in merged if piece.strip())
    return segments


def find_audiences(text: str) -> tuple[frozenset[Audience], frozenset[str]]:
    """Work out which benches a segment credits.

    Args:
        text: One segment.

    Returns:
        An ``(audiences, parties)`` pair. Both are empty when the segment names
        nobody, which is reported as :attr:`Audience.UNSPECIFIED` by
        :func:`classify` rather than silently treated as the whole house.

    Example:
        >>> audiences, parties = find_audiences("Taps a kormánypártok soraiban.")
        >>> [a.value for a in sorted(audiences)], sorted(parties)
        (['government'], [])

        A named party is reported both ways, so it can be counted either as
        itself or, given :data:`GOVERNING_PARTIES`, as a side:

        >>> audiences, parties = find_audiences("Taps a Fidesz és a KDNP soraiból.")
        >>> [a.value for a in sorted(audiences)], sorted(parties)
        (['party'], ['Fidesz', 'KDNP'])
    """
    audiences: set[Audience] = set()
    if re.search(GOVERNMENT_TERMS, text, re.IGNORECASE):
        audiences.add(Audience.GOVERNMENT)
    if re.search(OPPOSITION_TERMS, text, re.IGNORECASE):
        audiences.add(Audience.OPPOSITION)
    if re.search(HOUSE_TERMS, text, re.IGNORECASE):
        audiences.add(Audience.HOUSE)
    parties = {
        name for name, pattern in PARTY_PATTERNS.items() if re.search(pattern, text)
    }
    if parties:
        audiences.add(Audience.PARTY)
    return frozenset(audiences), frozenset(parties)


def classify(segment: str) -> Event:
    r"""Classify one segment of a parenthetical.

    A named interjection is recognised first, because ``Vadai Ágnes: Nem
    hallom, mert nagyon beszélnek…`` is an interjection by Vadai Ágnes and not
    a heckling event credited to nobody, even though it may contain any word at
    all inside the quote.

    Args:
        segment: One segment, as :func:`split_events` produces it.

    Returns:
        An :class:`Event`. Anything with no reaction cue is
        :attr:`Kind.OTHER` -- the procedural majority of the corpus, ``Nincs
        jelentkező.``, ``Szavazás.``, ``Megtörténik.``

    Example:
        >>> event = classify("Szórványos taps a Fidesz soraiból.")
        >>> event.kind, event.intensity, sorted(event.parties)
        (<Kind.APPLAUSE: 'applause'>, <Intensity.SCATTERED: 'scattered'>, ['Fidesz'])

        >>> classify("Derültség az ellenzéki padsorokban.").kind
        <Kind.LAUGHTER: 'laughter'>

        A standing ovation:

        >>> classify("A TISZA-frakció tagjai felállva tapsolnak.").intensity
        <Intensity.STANDING: 'standing'>

        A named interjection carries its speaker and what they said:

        >>> event = classify("Vadai Ágnes: Nem hallom!")
        >>> event.kind, event.speaker, event.quote
        (<Kind.INTERJECTION: 'interjection'>, 'Vadai Ágnes', 'Nem hallom!')

        An unnamed one keeps the quote but invents no speaker:

        >>> event = classify("Közbeszólás az MSZP soraiból: Hazudik!")
        >>> event.kind, event.speaker, event.quote
        (<Kind.HECKLING: 'heckling'>, None, 'Hazudik!')

        Procedure is not a reaction:

        >>> classify("Nincs jelentkező.").kind
        <Kind.OTHER: 'other'>

        A segment naming two reactions keeps both:

        >>> event = classify("Derültség és taps a kormánypárti padsorokban.")
        >>> sorted(k.value for k in event.kinds)
        ['applause', 'laughter']
    """
    audiences, parties = find_audiences(segment)

    match = _NAMED_INTERJECTION.match(segment)
    quote = None
    if match is not None:
        candidate = match.group(1).strip()
        if not _NOT_A_PERSON.search(candidate):
            return Event(
                text=segment,
                kind=Kind.INTERJECTION,
                kinds=frozenset({Kind.INTERJECTION}),
                audiences=audiences,
                parties=parties,
                speaker=candidate,
                quote=match.group(2).strip(),
            )
        # Not a person: keep what was shouted, drop the false speaker, and let
        # the cue matching below name the kind.
        quote = match.group(2).strip()

    if quote is None:
        unnamed = _UNNAMED_QUOTE.match(segment)
        if unnamed is not None and _NOT_A_PERSON.search(unnamed.group("prefix")):
            quote = unnamed.group("quote").strip()

    lowered = segment.lower()
    matched = [
        candidate
        for candidate, pattern in REACTION_CUES.items()
        if re.search(pattern, lowered)
    ]
    kind = matched[0] if matched else Kind.OTHER

    intensity = Intensity.PLAIN
    if kind is not Kind.OTHER:
        for candidate, pattern in INTENSITY_CUES.items():
            if re.search(pattern, lowered):
                intensity = candidate
                break

    if not audiences and kind is not Kind.OTHER:
        audiences = frozenset({Audience.UNSPECIFIED})

    return Event(
        text=segment,
        kind=kind,
        kinds=frozenset(matched),
        intensity=intensity,
        audiences=audiences,
        parties=parties,
        quote=quote,
    )


def events_in(line: str) -> list[Event]:
    """Split a parenthetical and classify every segment.

    Args:
        line: One parenthetical.

    Returns:
        One :class:`Event` per segment, in order.

    Example:
        >>> [e.kind for e in events_in("Taps a Jobbik soraiban. - Derültség.")]
        [<Kind.APPLAUSE: 'applause'>, <Kind.LAUGHTER: 'laughter'>]
    """
    return [classify(segment) for segment in split_events(line)]


def side_of(party: str, cycle: int) -> Audience:
    """Report whether a party sat in government or opposition in a cycle.

    Args:
        party: A party name as :data:`PARTY_PATTERNS` spells it.
        cycle: The parliamentary cycle.

    Returns:
        :attr:`Audience.GOVERNMENT` or :attr:`Audience.OPPOSITION`.

    Raises:
        KeyError: If the cycle has no entry in :data:`GOVERNING_PARTIES`.
            Guessing a side would silently invent the central fact of every
            cross-bench figure.

    Example:
        >>> side_of("Fidesz", 41)
        <Audience.GOVERNMENT: 'government'>
        >>> side_of("Fidesz", 43)
        <Audience.OPPOSITION: 'opposition'>
        >>> side_of("TISZA", 43)
        <Audience.GOVERNMENT: 'government'>
    """
    governing = GOVERNING_PARTIES[cycle]
    return Audience.GOVERNMENT if party in governing else Audience.OPPOSITION
