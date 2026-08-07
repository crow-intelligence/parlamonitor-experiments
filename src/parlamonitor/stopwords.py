"""Hungarian stopwords for parliamentary text, in three auditable groups.

The groups are kept separate rather than merged into one opaque blob so that
each can be inspected, edited and argued with on its own. Only
:func:`hungarian_stopwords` unions them.

Everything here is a **lemma**, because the list is applied after
:func:`parlamonitor.emtsv.lemmatize`. Inflected forms in :data:`SPACY_HU` are
harmless -- they simply never match.

This whole module changes the numbers. Every topic label depends on it.
"""

from __future__ import annotations

from collections.abc import Iterable

from spacy.lang.hu.stop_words import STOP_WORDS as _SPACY_HU

SPACY_HU: frozenset[str] = frozenset(_SPACY_HU)
"""spaCy's Hungarian stopword list -- 219 entries, no model download needed.

Covers the closed classes and a few light verbs (``van``, ``kell``, ``lesz``)
but **not** ``tud``, ``mond``, ``beszél`` or ``gondol``, which is exactly why
:data:`LIGHT_VERBS` exists.
"""

LIGHT_VERBS: frozenset[str] = frozenset(
    {
        # Existential, modal and auxiliary
        "van",
        "volt",
        "lesz",
        "kell",
        "lehet",
        "szabad",
        "fog",
        "akar",
        "szeret",
        "kíván",
        # Saying and thinking -- the verbs a parliament uses to frame anything
        "mond",
        "elmond",
        "beszél",
        "szól",
        "gondol",
        "hisz",
        "vél",
        "ért",
        "érez",
        "tud",
        "ismer",
        "lát",
        "néz",
        "említ",
        "jelent",
        "tekint",
        "kérdez",
        "válaszol",
        "köszön",
        # Support verbs: carry almost no meaning without their object
        "tesz",
        "vesz",
        "ad",
        "kap",
        "kér",
        "hoz",
        "visz",
        "jön",
        "megy",
        "áll",
        "tart",
        "marad",
        "kerül",
        "csinál",
        "hagy",
        "kezd",
        "folytat",
        "próbál",
        "történik",
        # Procedural verbs the user asked to drop: frequent in every speech
        # regardless of subject, so they distinguish nothing.
        "benyújt",
        "felszólal",
    }
)
"""Light, modal and support verbs, as lemmas.

Nouns and adjectives survive the part-of-speech filter on their own merits;
verbs mostly do not. These are the ones that appear across every topic and so
separate none of them. Deliberately *not* included: ``szavaz``, ``elfogad``,
``elutasít``, ``módosít``, ``támogat`` -- speech-act verbs that genuinely mark
what a speech is doing.
"""

PARLIAMENTARY: frozenset[str] = frozenset(
    {
        "tisztelt",
        "képviselőtárs",
        "képviselő",
        "honfitárs",
        "hölgy",
        "úr",
        "uram",
        "asszony",
        "elnök_asszony",
        "elnök_úr",
        "képviselő_úr",
        "képviselő_asszony",
        "szép",
        "köszönöm",
    }
)
"""Chamber address terms.

Most of these disappear with :func:`parlamonitor.text.strip_salutation`; this
list catches the residue that appears mid-speech ("tisztelt képviselőtársaim,
ahogy említettem"). ``elnök`` alone is **not** here -- it carries real topical
weight in debates about the ``köztársasági elnök``.
"""


def hungarian_stopwords(
    *,
    spacy: bool = True,
    light_verbs: bool = True,
    parliamentary: bool = True,
    extra: Iterable[str] = (),
) -> frozenset[str]:
    """Union the stopword groups, each switchable.

    Args:
        spacy: Include :data:`SPACY_HU`. Defaults to ``True``.
        light_verbs: Include :data:`LIGHT_VERBS`. Defaults to ``True``.
        parliamentary: Include :data:`PARLIAMENTARY`. Defaults to ``True``.
        extra: Any further lemmas to drop.

    Returns:
        The union, as a frozenset of lemmas.

    Example:
        >>> stops = hungarian_stopwords()
        >>> {"tud", "tesz", "benyújt", "van"} <= stops
        True
        >>> "költségvetés" in stops
        False

        Speech-act verbs are kept on purpose:

        >>> any(v in stops for v in ("szavaz", "elutasít", "módosít"))
        False

        Groups can be switched off individually:

        >>> "tud" in hungarian_stopwords(light_verbs=False)
        False
    """
    result: set[str] = set(extra)
    if spacy:
        result |= SPACY_HU
    if light_verbs:
        result |= LIGHT_VERBS
    if parliamentary:
        result |= PARLIAMENTARY
    return frozenset(result)
