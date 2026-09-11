"""Virtue salience per speech, speaker and topic.

Measures which moral vocabulary a speaker reaches for -- **not** whether they
have the virtue. See :mod:`parlamonitor.virtues` for why that distinction is
forced by the data rather than chosen for modesty.

Every virtue is reported as two numbers, never one: how often it is invoked
approvingly and how often as an accusation. In this corpus 60% of truthfulness
vocabulary is accusation, so a single score would rank the MP who most often
calls opponents liars as the most truthful.

Usage::

    uv run python scripts/speech_virtues.py
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

from parlamonitor.loading import DATA_RAW
from parlamonitor.virtues import MIN_MENTIONS, POLES, load_virtues, score_salience

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "metrics"
LEXICON = ROOT / "data" / "lexicons" / "virtues" / "hu_virtues.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--lexicon", type=Path, default=LEXICON)
    parser.add_argument("--cycle", type=int, default=43)
    parser.add_argument("--min-mentions", type=int, default=MIN_MENTIONS)
    return parser.parse_args(argv)


def load_sentences(path: Path):
    cache = {}
    with path.open(encoding="utf-8") as handle:
        for row in handle:
            if not row.strip():
                continue
            record = json.loads(row)
            cache[record["uid"]] = [
                ([lemma for _, lemma, _ in s], [tag for _, _, tag in s])
                for s in record["sentences"]
            ]
    return cache


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cache_path = out / "speech_lemmas.jsonl"
    if not cache_path.is_file():
        print(f"No lemma cache at {cache_path}", file=sys.stderr)
        return 1

    virtues = load_virtues(args.lexicon)
    print(f"Virtue lexicon: {len(virtues)} virtues")
    for key, virtue in virtues.items():
        print(
            f"  {key:14s} {virtue.affirming.size:3d} affirming + "
            f"{virtue.accusing.size:2d} accusing   {virtue.label_hu}"
        )

    root = DATA_RAW if args.data_dir is None else Path(args.data_dir)
    with (root / f"cycle{args.cycle}-speeches.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    records = [r for r in records if (r.get("text_clean") or "").strip()]
    cache = load_sentences(cache_path)

    rows = []
    for record in records:
        sentences = cache.get(record["uid"])
        if not sentences:
            continue
        result = score_salience(sentences, virtues, min_mentions=args.min_mentions)
        speaker = record.get("speaker") or {}
        row = {
            "uid": record["uid"],
            "cycle": args.cycle,
            "speaker_id": speaker.get("person_id"),
            "speaker": speaker.get("label"),
            "faction": speaker.get("faction"),
            "n_words": record.get("n_words"),
            "virtue_tokens": result.n_tokens,
            "virtue_mentions": result.n_mentions,
            "virtue_sparse": result.sparse,
        }
        for key in virtues:
            row[f"virtue_{key}"] = result.rates[key]
            for pole in POLES:
                row[f"virtue_{key}_{pole}"] = result.counts[key][pole]
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "speech_virtues.csv", index=False, encoding="utf-8")
    print(
        f"\n{len(frame):,} speeches, {int(frame['virtue_mentions'].sum()):,} mentions"
    )
    print(f"  {int(frame['virtue_sparse'].sum()):,} below the mention threshold")
    print("\n  virtue        affirming  accusing  accusing share")
    for key in virtues:
        a = int(frame[f"virtue_{key}_affirming"].sum())
        x = int(frame[f"virtue_{key}_accusing"].sum())
        share = x / (a + x) if a + x else 0
        print(f"  {key:14s} {a:9d} {x:9d} {share:14.0%}")

    # Per speaker: counts summed then divided, so a long speech is not weighed
    # like a short one.
    grouped = frame.groupby(["speaker_id", "speaker", "faction"], dropna=False)
    by_mp = grouped.agg(
        speeches=("uid", "size"),
        virtue_tokens=("virtue_tokens", "sum"),
        virtue_mentions=("virtue_mentions", "sum"),
        **{
            f"virtue_{k}_{p}": (f"virtue_{k}_{p}", "sum")
            for k in virtues
            for p in POLES
        },
    ).reset_index()
    tokens = by_mp["virtue_tokens"]
    for key in virtues:
        total = by_mp[f"virtue_{key}_affirming"] + by_mp[f"virtue_{key}_accusing"]
        by_mp[f"virtue_{key}"] = (total / tokens.where(tokens != 0) * 1000).round(4)
        # Stance is NaN, not 0.5, where the virtue is never mentioned.
        by_mp[f"virtue_{key}_stance"] = (
            by_mp[f"virtue_{key}_affirming"] / total.where(total != 0)
        ).round(4)
    by_mp["virtue_sparse"] = by_mp["virtue_mentions"] < args.min_mentions
    by_mp = by_mp.sort_values("speeches", ascending=False, ignore_index=True)
    by_mp.to_csv(out / "mp_virtues.csv", index=False, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "construct": "virtue salience — which moral vocabulary a speaker uses",
        "not_measured": (
            "virtue possession, cultivation or sincerity. A speaker may claim a "
            "virtue, deny it of an opponent, demand it of the house, or use it "
            "ironically, and the counts cannot tell these apart."
        ),
        "lexicon": {
            "path": str(args.lexicon),
            "sha256": next(iter(virtues.values())).affirming.sha256,
            "virtues": {
                k: {
                    "affirming": v.affirming.size,
                    "accusing": v.accusing.size,
                    "note": v.note,
                }
                for k, v in virtues.items()
            },
        },
        "parameters": {
            "negation": "not applied — 'nem bátor' is still talk about courage",
            "min_mentions": args.min_mentions,
            "mp_aggregation": "counts summed then divided, not a mean of rates",
            "rate_unit": "mentions per 1,000 tokens",
        },
        "counts": {
            "speeches": len(frame),
            "mentions": int(frame["virtue_mentions"].sum()),
            "sparse_speeches": int(frame["virtue_sparse"].sum()),
            "by_virtue": {
                k: {
                    "affirming": int(frame[f"virtue_{k}_affirming"].sum()),
                    "accusing": int(frame[f"virtue_{k}_accusing"].sum()),
                }
                for k in virtues
            },
        },
    }
    (out / "virtues_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
