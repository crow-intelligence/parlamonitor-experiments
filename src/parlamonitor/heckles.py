"""What the hecklers say: full quotes, their bench, and their content words.

Task 4 classified the parentheticals into reactions and counted them. This
module reads the *words* inside the interjections -- ``Vadai Ágnes: Nem
hallom!``, ``Közbeszólás az MSZP soraiból: Hazudik!`` -- and does three things
the reaction classifier was not built for:

* **Whole quotes.** :attr:`parlamonitor.reactions.Event.quote` stops at the
  first sentence end, so ``Mit nevetgéltek ezen? Inkább szégyelljétek
  magatokat!`` keeps only its first sentence and the rest becomes an ``OTHER``
  segment. :func:`quotes_in` re-attaches it. About one quote in ten is affected.
* **Hyphenated surnames.** The reaction classifier's name pattern has no
  hyphen in it, so ``Varga-Damm Andrea: …`` is never read as an interjection --
  1,038 interjections by 22 members across five cycles. :func:`quotes_in`
  recovers them and flags each one.
* **A bench.** :func:`bench_of` attributes a quote to the government or the
  opposition, and to Fidesz-KDNP or not, where the note settles it.

Nothing here changes :mod:`parlamonitor.reactions` or any Task 4 output; the
two bugs are fixed for this module's purposes only, and listed for a decision.

**What cannot be attributed.** The parentheticals name an interjector but
never their party, so a *named* heckle can be placed on a bench only through a
speeches export -- and only cycle 43 has one. Named heckles in cycles 39-42
stay unattributed. Inferring a member's party from the benches mentioned around
them would be inventing data, so it is not done.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from parlamonitor.emtsv import is_content_word
from parlamonitor.interactions import SpeakerRegistry
from parlamonitor.lexicon import fold
from parlamonitor.reactions import _EVENT_SPLIT, Audience, Kind, events_in, side_of

FIDESZ_KDNP: frozenset[str] = frozenset({"Fidesz", "KDNP"})
"""The party group whose move from government (39-42) to opposition (43) the
role-flip comparison follows."""

HECKLE_STOP_EXTRAS: tuple[str, ...] = ("első", "elő", "először", "ők")
"""The words NLTK's Hungarian stoplist adds to spaCy's, once its encoding is repaired.

NLTK's list (198 words) is almost a subset of spaCy's 219. Its seven extra
entries are all spelled with ``õ`` (U+00F5, o-tilde) where ``ő`` (U+0151)
belongs -- ``elõtt``, ``õk`` -- a Latin-1 encoding error. Repaired, three of
them (``előtt``, ``ő``, ``őket``) are already in spaCy's list and these four
are not. They are added to :func:`parlamonitor.stopwords.hungarian_stopwords`
for heckle content; the list is fixed here rather than read from NLTK because
NLTK's stopword corpus is downloaded data, absent on a clean install.
"""

_UPPER = "A-ZÁÉÍÓÖŐÚÜŰ"
_HYPHENATED = re.compile(
    rf"^(?:(?:[Dd]r|[Ii]d|[Ii]fj|[Öö]zv)\.\s+)?(?:[{_UPPER}][a-záéíóöőúüű]?\.\s+)?"
    rf"(?P<name>(?:[{_UPPER}][\w'’]+\s+)*[{_UPPER}][\w'’]*-[{_UPPER}]?[\w'’]+"
    rf"(?:\s+[{_UPPER}][\w'’-]+){{0,3}}):\s+(?P<quote>\S.*)$"
)


@dataclass(frozen=True, slots=True)
class Quote:
    """One quoted utterance inside a parenthetical.

    Attributes:
        speaker: The named interjector, or ``None`` when the note names a bench
            rather than a person (``Közbeszólás a Fidesz soraiból: …``).
        addressee: Who the remark was aimed at, for ``X Y-nak: …``; else ``None``.
        kind: The reaction kind of the event that carried the quote, as a
            string (``"interjection"``, ``"heckling"``, ...).
        audiences: The benches the note mentions.
        parties: The parties the note mentions.
        quote: The full utterance, continuation sentences re-attached.
        reassembled: Whether continuation sentences were re-attached.
        hyphenated_recovered: Whether the interjector has a hyphenated surname
            the reaction classifier misses, recovered here.
    """

    speaker: str | None
    addressee: str | None
    kind: str
    audiences: frozenset[Audience]
    parties: frozenset[str]
    quote: str
    reassembled: bool
    hyphenated_recovered: bool


def quotes_in(piece: str) -> list[Quote]:
    """Find the quoted utterance in one `` - ``-separated piece of a parenthetical.

    A quote is whatever follows ``Name:`` or ``Közbeszólás a … soraiból:``. It
    runs to the end of the piece, or until the next segment that is itself a
    reaction (``Taps.``, ``Derültség.``) or a new speaker.

    Args:
        piece: One piece of a line, as split by the reaction classifier's event
            separator. Use :func:`quotes_in_line` for whole lines.

    Returns:
        A list of at most one :class:`Quote`. A second speaker always starts a
        new piece, so a piece never holds two.

    Example:
        Continuation sentences are re-attached:

        >>> [q] = quotes_in("Dr. Varga László: Mit nevettek? Sírni kellene!")
        >>> q.speaker, q.quote, q.reassembled
        ('Varga László', 'Mit nevettek? Sírni kellene!', True)

        A hyphenated surname is recovered and flagged:

        >>> [q] = quotes_in("Dr. Kátai-Németh Vilmos: Igen!")
        >>> q.speaker, q.quote, q.hyphenated_recovered
        ('Kátai-Németh Vilmos', 'Igen!', True)

        A pure reaction holds no quote:

        >>> quotes_in("Taps a kormánypártok soraiban.")
        []
    """
    events = events_in(piece)
    for j, event in enumerate(events):
        recovered = None
        if event.quote is None and event.kind is Kind.OTHER:
            m = _HYPHENATED.match(event.text.strip())
            if m and " " in m.group("name"):
                recovered = m
        if event.quote is None and recovered is None:
            continue
        parts = [event.quote if recovered is None else recovered.group("quote")]
        for later in events[j + 1 :]:
            if later.kind is not Kind.OTHER or _HYPHENATED.match(later.text.strip()):
                break
            parts.append(later.text.strip())
        return [
            Quote(
                speaker=event.speaker if recovered is None else recovered.group("name"),
                addressee=event.addressee,
                kind=event.kind.value,
                audiences=frozenset(event.audiences),
                parties=frozenset(event.parties),
                quote=" ".join(part for part in parts if part),
                reassembled=len(parts) > 1,
                hyphenated_recovered=recovered is not None,
            )
        ]
    return []


def quotes_in_line(line: str) -> list[Quote]:
    """Every quote in a whole parenthetical line, one per `` - ``-piece at most.

    Args:
        line: One line of a parentheticals file.

    Returns:
        The quotes, in order.

    Example:
        >>> line = "Közbeszólás az MSZP soraiból: Hazudik! - Bana Tibor: Tolvajok!"
        >>> [(q.speaker, q.quote) for q in quotes_in_line(line)]
        [(None, 'Hazudik!'), ('Bana Tibor', 'Tolvajok!')]
    """
    return [quote for piece in _EVENT_SPLIT.split(line) for quote in quotes_in(piece)]


def party_group(parties: frozenset[str], side: str | None, cycle: int) -> str | None:
    """``"Fidesz-KDNP"``, ``"other"``, or ``None`` where the note does not settle it.

    A note that names parties settles it directly. One that names only a side
    settles it where that side had a single composition: the government was
    Fidesz-KDNP in cycles 39-42 and TISZA in 43, and the opposition of 39-42
    did not include Fidesz-KDNP. The cycle-43 opposition did -- alongside Mi
    Hazánk -- so a bare "from the opposition" there stays ``None``.

    Args:
        parties: Parties the note mentions.
        side: ``"government"``, ``"opposition"``, or ``None``.
        cycle: The parliamentary cycle.

    Returns:
        The group, or ``None``.

    Example:
        >>> party_group(frozenset({"Fidesz"}), "opposition", 43)
        'Fidesz-KDNP'
        >>> party_group(frozenset(), "government", 42)
        'Fidesz-KDNP'
        >>> party_group(frozenset(), "government", 43)
        'other'
        >>> party_group(frozenset(), "opposition", 43) is None
        True
    """
    if parties:
        if parties <= FIDESZ_KDNP:
            return "Fidesz-KDNP"
        return "other" if not parties & FIDESZ_KDNP else None
    if side == "government":
        return "Fidesz-KDNP" if cycle <= 42 else "other"
    if side == "opposition" and cycle <= 42:
        return "other"
    return None


def bench_of(
    quote: Quote, cycle: int, registry: SpeakerRegistry | None = None
) -> tuple[str | None, str | None, str]:
    """Attribute a quote to a side and a party group, saying how.

    Args:
        quote: The quote.
        cycle: Its parliamentary cycle.
        registry: The cycle's speaker registry, built from its speeches export.
            Only cycle 43 has one; without it, named quotes stay unattributed.
            Defaults to ``None``.

    Returns:
        ``(side, party_group, route)``. ``side`` is ``"government"``,
        ``"opposition"`` or ``None``; ``route`` names the evidence:
        ``"bench in the note"``, ``"bench not stated"``, ``"named, resolved"``,
        ``"named, unresolved"`` or ``"named, no party data"``.

    Example:
        >>> [q] = quotes_in("Közbeszólás a Fidesz soraiból: Csak keményen!")
        >>> bench_of(q, 43)
        ('opposition', 'Fidesz-KDNP', 'bench in the note')
        >>> [q] = quotes_in("Vadai Ágnes: Nem hallom!")
        >>> bench_of(q, 41)
        (None, None, 'named, no party data')
    """
    if quote.speaker is None:
        if Audience.GOVERNMENT in quote.audiences:
            side: str | None = "government"
        elif Audience.OPPOSITION in quote.audiences:
            side = "opposition"
        elif quote.parties:
            sides = {side_of(p, cycle).value for p in quote.parties}
            side = sides.pop() if len(sides) == 1 else None
        else:
            return None, None, "bench not stated"
        return side, party_group(quote.parties, side, cycle), "bench in the note"
    if registry is None:
        return None, None, "named, no party data"
    person = registry.resolve(quote.speaker)
    if person.resolved and person.side:
        group = "Fidesz-KDNP" if person.faction in FIDESZ_KDNP else "other"
        return person.side, group, "named, resolved"
    return None, None, "named, unresolved"


def content_lemmas(
    sentences: Iterable[Sequence[Sequence[str]]],
    stopwords: frozenset[str] = frozenset(),
    *,
    content_only: bool = True,
) -> list[str]:
    """The case-folded content lemmas of an emtsv analysis, stopwords removed.

    Args:
        sentences: Sentences of ``(form, lemma, xpostag)`` triples, as the
            lemma caches store them.
        stopwords: Lemmas to drop, compared after :func:`parlamonitor.lexicon.fold`.
            Defaults to none.
        content_only: Keep only nouns, verbs, adjectives and adverbs, pronouns
            excluded (:func:`parlamonitor.emtsv.is_content_word`). Defaults to
            ``True``.

    Returns:
        The lemmas, in order.

    Example:
        >>> sents = [[["Ne", "ne", "[/Adv]"],
        ...           ["hazudj", "hazudik", "[/V][Sbjv.Def.2Sg]"],
        ...           ["!", "!", "[Punct]"]]]
        >>> content_lemmas(sents)
        ['ne', 'hazudik']
        >>> content_lemmas(sents, frozenset({"ne"}))
        ['hazudik']
    """
    out = []
    for sentence in sentences:
        for _, lemma, tag in sentence:
            if content_only and not is_content_word(tag):
                continue
            key = fold(lemma)
            if key not in stopwords:
                out.append(key)
    return out
