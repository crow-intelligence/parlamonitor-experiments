"""Sentiment and emotion scoring for Hungarian parliamentary speech.

Three models, chosen so that no single one is trusted alone:

``NYTK/sentiment-hts5-hubert-hungarian``
    Hungarian, huBERT, five-point sentiment.
``visegradmedia-emotion/Emotion_RoBERTa_hungarian6``
    Hungarian, XLM-R, six emotions, multi-label.
``MilaNLProc/xlm-emo-t``
    Multilingual, four emotions, single-label. Runs as a cross-check on the
    Hungarian emotion model, which needs one -- see below.

**None of these models declares its labels.** Every one ships ``LABEL_0``,
``LABEL_1``, ... in its config, so the index-to-name mapping had to be
established by probing with texts of known valence and recorded here as
:data:`EMOTION_LABELS` and :data:`SENTIMENT_LABELS`. :data:`CALIBRATION_PROBES`
holds the probes, so the mapping is a reproducible claim rather than a guess
copied from a model card.

**Short probes are a weak test, and disagreed with the corpus.** Probing
flagged the Hungarian model's anger channel: four unambiguous Hungarian anger
sentences averaged 0.08 on index 0 while scoring 0.38 on disgust and 0.27 on
fear. On the full 1,693 speeches that channel behaves far better than the
probes suggested -- it correlates -0.40 with sentiment valence and +0.42 with
the multilingual model's anger. A four-sentence probe simply does not activate
it; a speech does.

The channel that actually fails is the Hungarian model's **fear**. It dominates
the corpus at a mean of 0.64, and carries no valence signal at all: r = -0.04
against sentiment and r = +0.02 against the other model's anger. A channel that
fires on two thirds of parliamentary prose while correlating with nothing is
measuring register, not emotion.

So both tests are kept and both are reported. ``scripts/validate_affect.py``
computes the corpus-level convergent validity, and it -- not the probe run --
is what :data:`UNRELIABLE_CHANNELS` reflects.

**Scores are not measurements of a speaker's feelings.** These models were
trained on social media and news, not parliamentary oratory, which is
performative and frequently scripted. A high "anger" score means the text
resembles text labelled angry in the training data.
"""

from __future__ import annotations

from dataclasses import dataclass

SENTIMENT_MODEL = "NYTK/sentiment-hts5-hubert-hungarian"
"""Five-point Hungarian sentiment, from the huBERT family."""

EMOTION_MODEL_HU = "visegradmedia-emotion/Emotion_RoBERTa_hungarian6"
"""Six-way Hungarian emotion, multi-label (sigmoid, not softmax)."""

EMOTION_MODEL_XLM = "MilaNLProc/xlm-emo-t"
"""Four-way multilingual emotion, single-label. Declares its own labels."""

EMOTION_LABELS: tuple[str, ...] = (
    "anger",
    "fear",
    "disgust",
    "sadness",
    "joy",
    "none",
)
"""Index-to-emotion for :data:`EMOTION_MODEL_HU`, established by probing.

Indices 1-5 are unambiguous: a probe of the named emotion scores 0.74-0.99 on
its index and near zero elsewhere. Index 0 is anger by elimination and by the
model card's stated ordering, but see :data:`UNRELIABLE_CHANNELS`.
"""

XLM_EMOTION_LABELS: tuple[str, ...] = ("anger", "fear", "joy", "sadness")
"""Index-to-emotion for :data:`EMOTION_MODEL_XLM`. Declared in its own config."""

UNRELIABLE_CHANNELS: frozenset[str] = frozenset(
    {"emotion_hu_fear", "emotion_hu_sadness"}
)
"""Columns that do not measure what their label says, on this corpus.

Both are Hungarian-model channels, and both were found by
``scripts/validate_affect.py`` rather than by probing:

``emotion_hu_fear``
    Mean 0.64 -- it fires on two thirds of everything -- while correlating
    -0.04 with sentiment valence and +0.02 with ``emotion_xlm_anger``. A
    channel that saturates and correlates with nothing is tracking register,
    not emotion.
``emotion_hu_sadness``
    Correlates **+0.11** with valence. Sadness should move against it. Its
    agreement with the other model's sadness is r = +0.01, which is to say
    none.

Prefer ``emotion_xlm_*``, every channel of which behaves as expected on this
corpus: anger -0.60 against valence, joy +0.68, sadness -0.32, fear -0.17.
"""

CALIBRATION_PROBES: dict[str, tuple[str, ...]] = {
    "anger": (
        "Dühös vagyok rád! Haragszom, mérges vagyok.",
        "Felháborító, ez tűrhetetlen, dühít.",
        "Haragszom rátok, mérges vagyok mindenkire.",
        "Bosszant és dühít ez az egész ügy.",
    ),
    "fear": (
        "Félek attól, ami történni fog. Rettegek.",
        "Nagyon ijesztő, félelmetes, szorongok tőle.",
        "Rettegés fog el, ha erre gondolok.",
        "Aggódom és félek a következményektől.",
    ),
    "disgust": (
        "Undorodom tőle. Gusztustalan.",
        "Visszataszító és undorító dolog.",
        "Ocsmány, undorító, gusztustalan viselkedés.",
        "Felfordul a gyomrom tőle, undorító.",
    ),
    "sadness": (
        "Nagyon szomorú vagyok, sírok.",
        "Elszomorító veszteség, gyászolunk.",
        "Bánatos vagyok, fáj a szívem.",
        "Szomorúság tölt el, elkeserítő.",
    ),
    "joy": (
        "Nagyon örülök! Boldog vagyok!",
        "Csodálatos hír, öröm tölt el!",
        "Boldogság és öröm, nagyszerű!",
        "Örömteli nap, nagyon boldogok vagyunk.",
    ),
    "none": (
        "A bizottság ülése kedden 10 órakor kezdődik.",
        "A törvényjavaslat 3. paragrafusa módosul.",
        "Az ülés a 2. számú teremben lesz megtartva.",
        "A napirendi pont tárgyalása folytatódik.",
    ),
}
"""The probes that established :data:`EMOTION_LABELS`.

Kept in the package so the mapping can be re-derived after a model update
rather than trusted on faith. ``scripts/calibrate_affect.py`` runs them.
"""

CHUNK_TOKENS = 400
"""Subword tokens per chunk.

All three models cap at 512 positions. Hungarian runs about 1.82 subword
tokens per word, so 512 is roughly 280 words against a 307-word median speech:
most speeches do not fit and must be chunked. 400 leaves room for the special
tokens without wasting much of the window.
"""


@dataclass(frozen=True, slots=True)
class AffectScores:
    """Scores for one text, with the counts that produced them.

    Attributes:
        sentiment: Probability per class, five-point scale, mean over chunks.
        sentiment_label: The highest-probability class name.
        sentiment_valence: The five-point scale mapped onto ``[-1, 1]``, so a
            single ordinal number is available. Uses the class *expectation*,
            not the argmax, so a text split between neighbouring classes lands
            between them.
        emotion_hu: Probability per Hungarian emotion, mean over chunks.
        emotion_xlm: Probability per multilingual emotion, mean over chunks.
        n_chunks: How many chunks the text was split into.
        n_subword_tokens: Total subword tokens.
    """

    sentiment: dict[str, float]
    sentiment_label: str
    sentiment_valence: float
    emotion_hu: dict[str, float]
    emotion_xlm: dict[str, float]
    n_chunks: int
    n_subword_tokens: int


def valence(probabilities: list[float]) -> float:
    """Collapse an ordinal sentiment distribution to one number in ``[-1, 1]``.

    Takes the expectation over evenly spaced class positions rather than the
    argmax, so a text the model splits between "negative" and "neutral" scores
    between them instead of snapping to one.

    Args:
        probabilities: Class probabilities, ordered most negative first. Must
            sum to approximately 1.

    Returns:
        The expected position on ``[-1, 1]``.

    Raises:
        ValueError: If fewer than two classes are given, or they do not sum to
            approximately 1.

    Example:
        All mass on the most negative class:

        >>> valence([1.0, 0.0, 0.0, 0.0, 0.0])
        -1.0

        All mass on the most positive:

        >>> valence([0.0, 0.0, 0.0, 0.0, 1.0])
        1.0

        Dead centre:

        >>> valence([0.0, 0.0, 1.0, 0.0, 0.0])
        0.0

        Split evenly between the two middle classes:

        >>> valence([0.0, 0.5, 0.5, 0.0, 0.0])
        -0.25
    """
    n = len(probabilities)
    if n < 2:
        raise ValueError(f"need at least two classes, got {n}")
    total = sum(probabilities)
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"probabilities sum to {total}, expected 1")
    positions = [-1.0 + 2.0 * i / (n - 1) for i in range(n)]
    return round(sum(p * x for p, x in zip(probabilities, positions, strict=True)), 6)


SENTIMENT_PROBES: dict[str, tuple[str, ...]] = {
    "very negative": (
        "Ez borzalmas, katasztrofális, teljesen elfogadhatatlan és szörnyű.",
        "Rettenetes döntés, óriási kudarc, minden szempontból káros.",
    ),
    "negative": (
        "Nem értek egyet ezzel a javaslattal, rossz irányba megy.",
        "Sajnos ez nem elég jó, csalódás.",
    ),
    "neutral": (
        "A bizottság ülése kedden 10 órakor kezdődik a 2. teremben.",
        "A törvényjavaslat harmadik paragrafusa módosul.",
    ),
    "positive": (
        "Jó javaslat, támogatom, hasznos lépés.",
        "Örvendetes fejlemény, ezt jónak tartom.",
    ),
    "very positive": (
        "Kiváló, nagyszerű, fantasztikus eredmény, rendkívül örülök!",
        "Csodálatos siker, minden várakozást felülmúlt, kitűnő!",
    ),
}
"""Ordinal probes for :data:`SENTIMENT_MODEL`, most negative first.

The model declares ``LABEL_0`` ... ``LABEL_4`` and says nothing about which end
is which, so the direction of the scale is derived rather than assumed. See
:func:`derive_ordinal_mapping`.
"""


@dataclass(frozen=True, slots=True)
class Calibration:
    """What a probe run established about a model's unnamed labels.

    Attributes:
        mapping: Index to label name.
        confusion: Probe label to the mean probability vector it produced.
        weak: Labels whose own probe did not put them first. A label listed
            here does not discriminate what it claims to.
        reversed_scale: For an ordinal model, whether index 0 turned out to be
            the positive end.
    """

    mapping: dict[int, str]
    confusion: dict[str, list[float]]
    weak: tuple[str, ...]
    reversed_scale: bool = False


def derive_ordinal_mapping(
    confusion: dict[str, list[float]],
) -> tuple[bool, tuple[str, ...]]:
    """Work out which end of an unnamed ordinal scale is negative.

    Compares where the most-negative probe puts its mass against where the
    most-positive probe puts its. If the negative probe peaks at a higher index
    than the positive one, the scale runs positive-to-negative.

    Args:
        confusion: Probe label to mean probability vector, ordered as
            :data:`SENTIMENT_PROBES` is -- most negative first.

    Returns:
        A ``(reversed, weak)`` pair. ``weak`` names probes whose distribution
        peaked somewhere other than an end, which would undermine the reading.

    Raises:
        ValueError: If fewer than two probe classes are given.

    Example:
        A scale running negative to positive:

        >>> confusion = {
        ...     "very negative": [0.9, 0.1, 0.0],
        ...     "neutral": [0.1, 0.8, 0.1],
        ...     "very positive": [0.0, 0.1, 0.9],
        ... }
        >>> derive_ordinal_mapping(confusion)[0]
        False

        And the same model with its labels the other way round:

        >>> flipped = {k: list(reversed(v)) for k, v in confusion.items()}
        >>> derive_ordinal_mapping(flipped)[0]
        True
    """
    labels = list(confusion)
    if len(labels) < 2:
        raise ValueError(f"need at least two probe classes, got {len(labels)}")
    negative_peak = confusion[labels[0]].index(max(confusion[labels[0]]))
    positive_peak = confusion[labels[-1]].index(max(confusion[labels[-1]]))
    weak = tuple(
        label for label in (labels[0], labels[-1]) if max(confusion[label]) < 0.4
    )
    return negative_peak > positive_peak, weak
