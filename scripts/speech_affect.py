"""Dictionary-based sentiment and emotion for every speech.

Replaces the transformer scoring entirely. Two Hungarian word lists do the
work, and the whole corpus scores in seconds from the cached lemmas — no
models, no downloads, no GPU, and every number decomposes into the exact
lemmas that produced it.

**Sentiment** — Precognox Hungarian Sentiment Lexicon (CC BY-NC 4.0), 1,748
positive and 5,940 negative lemmas. Negation follows the reference
implementation in ``crow-intelligence/growth-hacking-sentiment``: mark the
tokens inside a negation's scope, then flip their polarity, so ``nem probléma``
counts positive. Hungarian adds the postposition ``nélkül``, which scopes
backwards.

**Emotion** — Ekman's six basic categories, from Putz Orsolya's collection.
Used with permission and not redistributable; any output derived from it must
credit her.

Two limits are reported rather than hidden. The sentiment list is 3.4x larger
on the negative side, so ``--balanced`` exists and the unweighted count is the
default. Emotion words are 2.9% of content lemmas, so a speech below the hit
threshold is flagged ``emotion_sparse`` — a lexicon reports 0.0 both for "no
anger" and for "no evidence", and those are different claims.

Usage::

    uv run python scripts/speech_affect.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from parlamonitor.lexicon import (
    DEFAULT_REGISTER_PERCENTILE,
    EKMAN,
    EKMAN_FILES,
    MIN_EMOTION_HITS,
    fold,
    load_lexicon,
    register_vocabulary,
    score_emotion,
    score_sentiment,
    without,
)
from parlamonitor.loading import DATA_RAW

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "metrics"
LEXICON_DIR = ROOT / "data" / "lexicons"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--lexicon-dir", type=Path, default=LEXICON_DIR)
    parser.add_argument("--cycle", type=int, default=43)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-emotion-hits", type=int, default=MIN_EMOTION_HITS)
    parser.add_argument(
        "--register-percentile",
        type=float,
        default=DEFAULT_REGISTER_PERCENTILE,
        help="drop lexicon entries in this top share of corpus frequency as "
        "register vocabulary; 0 disables. Without it 'jó', 'támogatás' and "
        "'vita' make joy the dominant emotion in 62%% of speeches",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="weight each sentiment class by the inverse of its list size, "
        "correcting the 3.4:1 negative bias built into the lexicon",
    )
    return parser.parse_args(argv)


def load_sentences(path: Path) -> dict[str, list[tuple[list[str], list[str]]]]:
    """Read the cached emtsv output as (lemmas, tags) per sentence."""
    cache: dict[str, list[tuple[list[str], list[str]]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in handle:
            if not row.strip():
                continue
            record = json.loads(row)
            cache[record["uid"]] = [
                ([lemma for _, lemma, _ in sentence], [tag for _, _, tag in sentence])
                for sentence in record["sentences"]
            ]
    return cache


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cache_path = out / "speech_lemmas.jsonl"
    if not cache_path.is_file():
        print(
            f"No lemma cache at {cache_path}; run scripts/speech_metrics.py first",
            file=sys.stderr,
        )
        return 1

    root = DATA_RAW if args.data_dir is None else Path(args.data_dir)
    speeches_path = root / f"cycle{args.cycle}-speeches.jsonl"
    with speeches_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    records = [r for r in records if (r.get("text_clean") or "").strip()]
    if args.limit:
        records = records[: args.limit]

    print("Loading lexicons ...")
    positive = load_lexicon(args.lexicon_dir / "sentiment" / "PrecoPos.txt", "positive")
    negative = load_lexicon(args.lexicon_dir / "sentiment" / "PrecoNeg.txt", "negative")
    print(
        f"  sentiment: {positive.size:,} positive, {negative.size:,} negative "
        f"(1:{negative.size / positive.size:.1f})"
    )
    emotions = {
        name: load_lexicon(args.lexicon_dir / "emotion" / EKMAN_FILES[name], name)
        for name in EKMAN
    }
    for name, lex in emotions.items():
        print(f"  {name:9s} {lex.size:4d} entries ({len(lex.multi)} multiword)")

    cache = load_sentences(cache_path)

    # Register filter. The lists were built for general Hungarian; this corpus
    # is parliamentary, and a handful of policy words sit in them.
    frequencies: dict[str, int] = {}
    for sentences in cache.values():
        for lemmas, tags in sentences:
            for lemma, tag in zip(lemmas, tags, strict=True):
                if "Punct" not in tag:
                    key = fold(lemma)
                    frequencies[key] = frequencies.get(key, 0) + 1
    excluded = register_vocabulary(frequencies, args.register_percentile)
    dropped_by: dict[str, list[str]] = {}
    if excluded:
        for name, lex in list(emotions.items()):
            emotions[name], dropped = without(lex, excluded)
            if dropped:
                dropped_by[name] = dropped
        print(
            f"\nRegister filter (top {args.register_percentile}% of "
            f"{len(frequencies):,} lemmas): "
            + (
                "; ".join(f"{k} -{', '.join(v)}" for k, v in dropped_by.items())
                or "nothing dropped"
            )
        )

    print(f"\nScoring {len(records):,} speeches ...")

    rows = []
    for record in records:
        sentences = cache.get(record["uid"])
        if not sentences:
            continue
        sentiment = score_sentiment(
            sentences, positive, negative, balanced=args.balanced
        )
        emotion = score_emotion(sentences, emotions, min_hits=args.min_emotion_hits)
        speaker = record.get("speaker") or {}
        row = {
            "uid": record["uid"],
            "cycle": args.cycle,
            "date": record.get("date"),
            "speaker_id": speaker.get("person_id"),
            "speaker": speaker.get("label"),
            "faction": speaker.get("faction"),
            "speech_type": record.get("speech_type"),
            "n_words": record.get("n_words"),
            "sentiment_score": sentiment.score,
            "sentiment_polarity": round(sentiment.polarity, 6),
            "sentiment_positive": sentiment.positive,
            "sentiment_negative": sentiment.negative,
            "sentiment_flipped": sentiment.flipped,
            "sentiment_tokens": sentiment.n_tokens,
            "emotion_hits": emotion.n_hits,
            "emotion_sparse": emotion.sparse,
            "emotion_dominant": emotion.dominant,
        }
        for name in EKMAN:
            row[f"emotion_{name}"] = emotion.rates[name]
            row[f"emotion_{name}_count"] = emotion.counts[name]
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "speech_affect.csv", index=False, encoding="utf-8")
    print(f"  scored {len(frame):,} speeches")
    print(
        f"  sentiment: mean {frame['sentiment_score'].mean():+.4f}, "
        f"polarity {frame['sentiment_polarity'].mean():+.4f}, "
        f"{int(frame['sentiment_flipped'].sum()):,} hits flipped by negation"
    )
    sparse = int(frame["emotion_sparse"].sum())
    print(f"  emotion: {sparse:,} of {len(frame):,} speeches below the hit threshold")
    print(f"  dominant emotion: {frame['emotion_dominant'].value_counts().to_dict()}")

    # Per speaker. Emotion is aggregated from counts, not by averaging rates:
    # a speaker's rate is their total hits over their total tokens, so a short
    # speech does not weigh as much as a long one.
    grouped = frame.groupby(["speaker_id", "speaker", "faction"], dropna=False)
    by_mp = grouped.agg(
        speeches=("uid", "size"),
        words=("n_words", "sum"),
        sentiment_score=("sentiment_score", "mean"),
        sentiment_positive=("sentiment_positive", "sum"),
        sentiment_negative=("sentiment_negative", "sum"),
        emotion_hits=("emotion_hits", "sum"),
        emotion_tokens=("sentiment_tokens", "sum"),
        **{f"emotion_{n}_count": (f"emotion_{n}_count", "sum") for n in EKMAN},
    ).reset_index()
    # `.where(x != 0)` rather than a sentinel: a speaker with no hits has no
    # polarity, and NaN says exactly that where 0.0 would claim neutrality.
    total = by_mp["sentiment_positive"] + by_mp["sentiment_negative"]
    by_mp["sentiment_polarity"] = (
        (by_mp["sentiment_positive"] - by_mp["sentiment_negative"])
        / total.where(total != 0)
    ).round(6)
    tokens = by_mp["emotion_tokens"]
    for name in EKMAN:
        by_mp[f"emotion_{name}"] = (
            by_mp[f"emotion_{name}_count"] / tokens.where(tokens != 0)
        ).round(6)
    by_mp["emotion_sparse"] = by_mp["emotion_hits"] < args.min_emotion_hits
    by_mp = by_mp.sort_values("speeches", ascending=False, ignore_index=True)
    by_mp.to_csv(out / "mp_affect.csv", index=False, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "method": "dictionary-based; replaces the transformer scoring",
        "lexicons": {
            "sentiment_positive": {
                "id": positive.lexicon_id,
                "entries": positive.size,
                "source": "Precognox Hungarian Sentiment Lexicon, CC BY-NC 4.0",
            },
            "sentiment_negative": {
                "id": negative.lexicon_id,
                "entries": negative.size,
                "source": "Precognox Hungarian Sentiment Lexicon, CC BY-NC 4.0",
            },
            "emotion": {
                name: {
                    "id": lex.lexicon_id,
                    "entries": lex.size,
                    "multiword": len(lex.multi),
                }
                for name, lex in emotions.items()
            },
            "emotion_source": (
                "Putz Orsolya's own collection, used with permission, not "
                "redistributable. Any output derived from it must credit her."
            ),
        },
        "parameters": {
            "emotion_scheme": "Ekman six basic emotions",
            "emotion_categories": list(EKMAN),
            "emotion_excluded": ["feszültség", "szeretet"],
            "negation_forward": [
                "nem",
                "sem",
                "se",
                "ne",
                "nincs",
                "nincsen",
                "sincs",
                "sincsen",
            ],
            "negation_backward": ["nélkül"],
            "negation_scope": (
                "to the next clause punctuation, as in NLTK mark_negation"
            ),
            "negation_applied_to": (
                "sentiment only; a negated emotion word is not the opposite "
                "emotion, so there is nothing to flip"
            ),
            "sentiment_scoring": (
                "per sentence, (pos - neg) / tokens, averaged over sentences"
            ),
            "balanced": args.balanced,
            "min_emotion_hits": args.min_emotion_hits,
            "register_percentile": args.register_percentile,
            "register_dropped": dropped_by,
            "register_rationale": (
                "the lists were built for general Hungarian; in this corpus "
                "'jó', 'támogatás', 'kedves', 'segít' (joy) and 'vita' (anger) "
                "are policy vocabulary, not emotion markers. Unfiltered, joy "
                "was dominant in 62% of speeches."
            ),
            "mp_aggregation": "counts summed then divided, not a mean of rates",
        },
        "counts": {
            "speeches": len(frame),
            "speakers": int(by_mp["speaker_id"].nunique()),
            "sentiment_hits_flipped_by_negation": int(frame["sentiment_flipped"].sum()),
            "emotion_sparse_speeches": sparse,
            "dominant_emotion": {
                str(k): int(v)
                for k, v in frame["emotion_dominant"].value_counts().items()
            },
        },
    }
    (out / "affect_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
