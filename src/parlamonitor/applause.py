"""Applause lines: the sentence a reaction answered.

The speeches export keeps its stage directions inline in ``text`` -- ``…egy
szabadabb, emberségesebb ország korszakát. (Taps a TISZA soraiból.)`` -- each
right after the words that drew it. This module puts every reaction back on
the sentence it followed, in the export's own sentence segmentation
(``record["sentences"]``), so that a sentence can be asked what it earned.

**Placement.** ``text`` carries the speaker attribution, paragraph breaks,
clock stamps and the directions; ``sentences`` carries none of them. A
direction is placed by taking the 40 non-space characters of text before it
(directions removed) and finding them in the joined, whitespace-free
sentences, scanning forward from the previous reaction. The end of the match
is a character offset, and the sentence holding that offset is the trigger:

* at a sentence's end, the reaction answered the whole sentence;
* inside a sentence, the reaction **interrupted** it -- mostly heckles -- and
  the trigger is the sentence up to the cut.

Textual parentheses that ``text_clean`` keeps (``13. § (2) bekezdés``) are
stripped from the search key along with the directions, which can break the
match; a second attempt uses only the last 15 characters. A reaction that
still cannot be found is reported as unaligned, never guessed. On cycle 43
that is 19 of 4,597.

**The closing ritual.** A third of all reactions follow a speech's final
sentence -- usually *Köszönöm a figyelmet.* -- and are the chamber's routine
applause for whoever just spoke. They are flagged ``closing`` and kept out of
the applause lines, because they say nothing about what was said.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from parlamonitor.reactions import Audience, Kind, events_in, side_of

_SPAN = re.compile(r"\(([^()]*)\)")
_CLOCK = re.compile(r"^\d{1,2}\.\d{2}$")
_SPACE = re.compile(r"\s+")

KEY_LENGTH = 40
"""Non-space characters before a direction used to find it."""

RETRY_LENGTH = 15
"""Shorter key for the second attempt, after textual parentheses broke the first."""

LOOKBACK = 200
"""How far before the previous match a search may start, in characters."""

CLOSING_FORMULA = re.compile(
    r"^(?:köszönöm|köszönjük)(?:\s+szépen)?(?:"
    r"(?:\s+(?:a|az))?(?:\s+\w+)?\s+(?:figyelm\w*|szót|lehetőséget|türelm\w*)"
    r"|\s*[.!]?$)",
    re.IGNORECASE,
)
"""Ritual thanks, whose reactions answer the speech rather than the sentence,
wherever they fall: *Köszönöm a figyelmet*, *Köszönöm szépen a megtisztelő
figyelmüket*, *Köszönöm a szót*, and a sentence that is nothing but
*Köszönöm (szépen).*"""

SALUTATION = re.compile(
    r"^(?:Igen\s+|Nagyon\s+)?(?:[Tt]isztelt|[Kk]edves)\b[^.!?]{0,80}!$"
)
"""A sentence that only addresses the chamber: *Tisztelt Elnök Úr!*, *Tisztelt
Ház!*, *Kedves Képviselőtársaim!*

There are 3,228 in cycle 43 and none is ever followed by an applause line. Left
in the baseline they make "quiet" sentences look exclamatory -- they end in
``!`` -- and turn exclamations from twice as likely before applause into less
likely. :func:`is_salutation` finds them so the baseline can leave them out.
"""

NEGATION: frozenset[str] = frozenset(
    {"nem", "sem", "se", "ne", "nincs", "nincsen", "sincs", "sincsen"}
)
"""Negation lemmas, as :mod:`parlamonitor.lexicon` scopes them."""

OPPONENT_NAMES: dict[str, re.Pattern[str]] = {
    # Who a cycle-43 speaker on each side names when they name the other side.
    "government": re.compile(
        r"\b(?:Fidesz\w*|KDNP\w*|Orbán\w*|Mi Hazánk\w*|Toroczkai\w*)"
    ),
    "opposition": re.compile(r"\b(?:TISZA\w*|Tisza-\w+|Magyar Péter\w*)"),
}
"""Opposing parties and their leaders, by the speaker's side in cycle 43.

Matched on the sentence text, case-sensitively, so ``magyar`` the adjective
and ``Tisza`` the river in a place name (no hyphen, not capitalised as a
party) are not hits. A deliberately short list: it measures naming the other
side's leadership, not every reference to it.
"""

LINE_KINDS: tuple[str, ...] = ("applause", "laughter_with", "laughter_at", "heckle")
"""The reactions a sentence can be a "line" for, in display order."""


@dataclass(frozen=True, slots=True)
class Placed:
    """One reaction, put back on the sentence it followed.

    Attributes:
        sentence_index: Index into ``record["sentences"]``; ``None`` if unaligned.
        interrupted: The reaction fell inside the sentence, not after it.
        cut: For an interrupted sentence, its characters before the cut.
        kinds: Every reaction kind in the direction, as strings.
        weight: The strongest intensity weight among its events.
        parties: Parties the note credits.
        audiences: Bench categories the note credits, as strings.
        quote: An interjection's words, when the direction quotes one.
        span: The direction verbatim, without its parentheses.
        offset: Where the direction's ``(`` sits in ``record["text"]``.
        relative_position: Sentence index over the last index, in [0, 1];
            ``None`` if unaligned.
        closing: Ritual closing applause -- after the final sentence, or after a
            closing formula.
    """

    sentence_index: int | None
    interrupted: bool
    cut: str | None
    kinds: frozenset[str]
    weight: float
    parties: frozenset[str]
    audiences: frozenset[str]
    quote: str | None
    span: str
    offset: int
    relative_position: float | None
    closing: bool

    @property
    def aligned(self) -> bool:
        """Whether the reaction was placed on a sentence."""
        return self.sentence_index is not None


def _nonspace(text: str) -> str:
    return _SPACE.sub("", text)


def _body_start(record: Mapping[str, Any]) -> int:
    """Where the speech proper begins: after ``NAGY JÁNOS (TISZA):``."""
    text = record.get("text") or ""
    return text.find(":") + 1 if record.get("speaker_prefix") else 0


def _cut(sentence: str, n_chars: int) -> str:
    """The prefix of ``sentence`` holding its first ``n_chars`` non-space characters."""
    seen = 0
    for i, ch in enumerate(sentence):
        if not ch.isspace():
            seen += 1
            if seen == n_chars:
                return sentence[: i + 1]
    return sentence


def placed_reactions(record: Mapping[str, Any]) -> list[Placed]:
    """Every reaction in one speech, on the sentence it followed.

    Args:
        record: A speeches-export record with ``text``, ``sentences`` and
            ``speaker_prefix``.

    Returns:
        One :class:`Placed` per direction that holds a reaction, in text order.
        Clock stamps (``(13.20)``), the attribution's party tag, and
        directions that are not reactions (``Elfoglalja helyét…``, ``sic!``)
        are skipped. Placements never move backwards through the speech.

    Example:
        >>> rec = {"speaker_prefix": "X (TISZA)", "text":
        ...        "X (TISZA): Ez a mi ügyünk. (Taps a TISZA soraiból.) Köszönöm a "
        ...        "figyelmet. (Taps.)",
        ...        "sentences": ["Ez a mi ügyünk.", "Köszönöm a figyelmet."]}
        >>> [(p.sentence_index, p.interrupted, p.closing)
        ...  for p in placed_reactions(rec)]
        [(0, False, False), (1, False, True)]

        A heckle that cuts a sentence off is placed inside it:

        >>> rec = {"speaker_prefix": None, "text":
        ...        "A javaslat (Közbeszólás a Fidesz soraiból: Hazugság!) jó.",
        ...        "sentences": ["A javaslat jó."]}
        >>> [p] = placed_reactions(rec)
        >>> p.interrupted, p.cut, sorted(p.kinds)
        (True, 'A javaslat', ['heckling'])
    """
    text = record.get("text") or ""
    sentences: Sequence[str] = record.get("sentences") or []
    body = _body_start(record)
    joined = _nonspace("".join(sentences))
    ends: list[int] = []
    total = 0
    for sentence in sentences:
        total += len(_nonspace(sentence))
        ends.append(total)
    last = len(sentences) - 1

    out: list[Placed] = []
    pointer = 0
    for m in _SPAN.finditer(text):
        if m.start() < body:
            continue
        inner = m.group(1).strip()
        if _CLOCK.match(inner):
            continue
        events = [e for e in events_in(inner) if e.kind is not Kind.OTHER]
        if not events:
            continue

        before = _nonspace(_SPAN.sub("", text[body : m.start()]))
        index: int | None = None
        offset = None
        for length in (KEY_LENGTH, RETRY_LENGTH):
            key = before[-length:]
            if not key:
                break
            found = joined.find(key, max(0, pointer - LOOKBACK))
            if found >= 0:
                offset = found + len(key)
                break
        interrupted, cut = False, None
        if offset is not None and ends:
            pointer = offset
            index = min(bisect_left(ends, offset), last)
            if ends[index] != offset:
                interrupted = True
                start = ends[index - 1] if index else 0
                cut = _cut(sentences[index], offset - start)

        trigger = sentences[index] if index is not None else ""
        closing = (
            index is not None
            and not interrupted
            and (index == last or bool(CLOSING_FORMULA.match(trigger.strip())))
        )
        quotes = [e.quote for e in events if e.quote]
        out.append(
            Placed(
                sentence_index=index,
                interrupted=interrupted,
                cut=cut,
                kinds=frozenset(k.value for e in events for k in e.kinds),
                weight=max(e.weight for e in events),
                parties=frozenset(p for e in events for p in e.parties),
                audiences=frozenset(a.value for e in events for a in e.audiences),
                quote=quotes[0] if quotes else None,
                span=inner,
                offset=m.start(),
                relative_position=(
                    None if index is None else (index / last if last > 0 else 1.0)
                ),
                closing=closing,
            )
        )
    return out


def is_salutation(sentence: str) -> bool:
    """Whether a sentence only addresses the chamber (:data:`SALUTATION`).

    Example:
        >>> is_salutation("Tisztelt Elnök Úr!"), is_salutation("Tisztelt Ház!")
        (True, True)
        >>> is_salutation("Tisztelt képviselőtársaim, ez a törvény rossz.")
        False
    """
    return bool(SALUTATION.match(sentence.strip()))


def side_relation(placed: Placed, speaker_side: str | None, cycle: int) -> str:
    """Whether a reaction came from the speaker's own side, the other, or both.

    Args:
        placed: The reaction.
        speaker_side: ``"government"`` or ``"opposition"``; ``None`` if unknown.
        cycle: The parliamentary cycle, for mapping parties to sides.

    Returns:
        ``"own"``, ``"other"``, ``"mixed"``, or ``"unattributed"`` when the note
        names no bench or the speaker's side is unknown.

    Example:
        >>> p = placed_reactions({"speaker_prefix": None, "text":
        ...     "Szép. (Derültség a Fidesz soraiban.)", "sentences": ["Szép."]})[0]
        >>> side_relation(p, "government", 43), side_relation(p, "opposition", 43)
        ('other', 'own')
    """
    sides: set[str] = set()
    if Audience.GOVERNMENT.value in placed.audiences:
        sides.add("government")
    if Audience.OPPOSITION.value in placed.audiences:
        sides.add("opposition")
    sides.update(side_of(p, cycle).value for p in placed.parties)
    if not sides or speaker_side is None:
        return "unattributed"
    if sides == {speaker_side}:
        return "own"
    return "other" if speaker_side not in sides else "mixed"


def line_kinds(placed: Placed, relation: str) -> list[str]:
    """Which kinds of line the reaction's sentence is, if any.

    * ``applause`` -- applause from the speaker's own side or no named side,
      except the closing ritual;
    * ``laughter_with`` / ``laughter_at`` -- laughter from the speaker's own
      side or from the other side;
    * ``heckle`` -- a heckle or a named interjection.

    The chair's bell, noise and uproar are kept by the caller but are not lines.

    Example:
        >>> p = placed_reactions({"speaker_prefix": None, "text":
        ...     "Szép. (Derültség és taps a TISZA soraiban.) Tovább.",
        ...     "sentences": ["Szép.", "Tovább."]})[0]
        >>> line_kinds(p, "own")
        ['applause', 'laughter_with']

        After the last sentence, the same applause is the closing ritual:

        >>> p = placed_reactions({"speaker_prefix": None, "text":
        ...     "Szép. (Taps a TISZA soraiban.)", "sentences": ["Szép."]})[0]
        >>> line_kinds(p, "own")
        []
    """
    out = []
    if (
        "applause" in placed.kinds
        and relation in {"own", "unattributed"}
        and not placed.closing
    ):
        out.append("applause")
    if "laughter" in placed.kinds and relation in {"own", "other"}:
        out.append("laughter_with" if relation == "own" else "laughter_at")
    if placed.kinds & {"heckling", "interjection"}:
        out.append("heckle")
    return out


def tokens_by_sentence(
    sentences: Sequence[str], analysis: Iterable[Sequence[Sequence[str]]]
) -> list[list[Sequence[str]]]:
    """Assign an emtsv analysis's tokens to the export's sentences.

    emtsv splits sentences its own way, but its tokens rebuild the export's
    ``sentences`` character for character once whitespace is removed (true for
    every cycle-43 speech), so each token goes to the sentence holding its first
    character.

    Args:
        sentences: ``record["sentences"]``.
        analysis: The emtsv sentences of ``(form, lemma, xpostag)`` triples.

    Returns:
        One token list per export sentence.

    Raises:
        ValueError: If the tokens do not rebuild the sentences, which would
            make every assignment wrong.

    Example:
        >>> tokens_by_sentence(["Igen.", "Nem!"],
        ...     [[["Igen", "igen", "[/Adv]"], [".", ".", "[Punct]"],
        ...       ["Nem", "nem", "[/Adv]"], ["!", "!", "[Punct]"]]])[1][0][0]
        'Nem'
    """
    ends = []
    total = 0
    for sentence in sentences:
        total += len(_nonspace(sentence))
        ends.append(total)
    out: list[list[Sequence[str]]] = [[] for _ in sentences]
    seen = 0
    rebuilt = []
    for sentence in analysis:
        for token in sentence:
            index = min(bisect_left(ends, seen + 1), len(ends) - 1) if ends else 0
            if out:
                out[index].append(token)
            form = _nonspace(token[0])
            rebuilt.append(form)
            seen += len(form)
    if _nonspace("".join(sentences)) != "".join(rebuilt):
        raise ValueError("emtsv tokens do not rebuild the export's sentences")
    return out


def sentence_features(
    tokens: Sequence[Sequence[str]], text: str, speaker_side: str | None
) -> dict[str, bool | int]:
    """Interpretable rhetorical features of one sentence.

    Args:
        tokens: Its emtsv tokens, ``(form, lemma, xpostag)``.
        text: The sentence verbatim.
        speaker_side: For naming the other side, the speaker's side.

    Returns:
        ``words`` (count of word tokens) and booleans ``exclamation``,
        ``question``, ``number``, ``we`` (a first-person-plural form or
        possessive), ``negation`` and ``names_opponent``.

    Example:
        >>> toks = [["Nem", "nem", "[/Adv]"], ["fogjuk", "fog", "[/V][Prs.Def.1Pl]"],
        ...         ["hagyni", "hagy", "[/V][Inf]"], ["!", "!", "[Punct]"]]
        >>> f = sentence_features(toks, "Nem fogjuk hagyni!", "government")
        >>> f["words"], f["exclamation"], f["we"], f["negation"], f["names_opponent"]
        (3, True, True, True, False)
    """
    stripped = text.rstrip()
    pattern = OPPONENT_NAMES.get(speaker_side or "")
    return {
        "words": sum(1 for _, _, tag in tokens if "Punct" not in tag),
        "exclamation": stripped.endswith("!"),
        "question": stripped.endswith("?"),
        "number": any(
            "[/Num" in tag or any(c.isdigit() for c in form) for form, _, tag in tokens
        ),
        "we": any("1Pl" in tag for _, _, tag in tokens),
        "negation": any(lemma.casefold() in NEGATION for _, lemma, _ in tokens),
        "names_opponent": bool(pattern and pattern.search(text)),
    }
