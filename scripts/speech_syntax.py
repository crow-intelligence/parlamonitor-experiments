"""Mean dependency distance and hierarchical distance for every speech.

Needs a **parse**, not just a lemmatiser, so this runs emtsv's
``tok/morph/pos/conv-morph/dep`` chain rather than the default one -- roughly
three times slower, and cached to its own file so the cheaper chain's cache is
not disturbed.

MDD measures how far a word sits from its governor: memory load rather than
vocabulary. MHD measures how deep the parse tree is. The two trade off, which
is why Jing & Liu (2015) report the pair, and why both are written here.

Usage::

    uv run python scripts/speech_syntax.py --limit 20   # smoke run
    uv run python scripts/speech_syntax.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from parlamonitor.emtsv import (
    DEFAULT_BASE_URL,
    DEPENDENCY_MODULES,
    Token,
    analyse_lines,
)
from parlamonitor.loading import DATA_RAW
from parlamonitor.syntax import MIN_SENTENCE_LENGTH, PARSER, measure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "metrics"
CHECKPOINT = 100


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cycle", type=int, default=43)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--modules", default=DEPENDENCY_MODULES)
    parser.add_argument("--min-sentence-length", type=int, default=MIN_SENTENCE_LENGTH)
    parser.add_argument("--batch-lines", type=int, default=40)
    parser.add_argument("--batch-words", type=int, default=1200)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args(argv)


def load_cache(path: Path, modules: str) -> dict[str, list[list[list]]]:
    if not path.is_file():
        return {}
    cache = {}
    with path.open(encoding="utf-8") as handle:
        for row in handle:
            if not row.strip():
                continue
            record = json.loads(row)
            if record.get("modules") == modules:
                cache[record["uid"]] = record["sentences"]
    return cache


def rebuild(serialised: list[list[list]]) -> list[list[Token]]:
    """Rebuild tokens from the cache, keeping only what the metrics read."""
    return [
        [
            Token(
                form="",
                lemma="",
                xpostag="",
                upostag=upostag,
                dep_id=dep_id,
                head=head,
            )
            for dep_id, head, upostag in sentence
        ]
        for sentence in serialised
    ]


def analyse_missing(records, cache, cache_path: Path, args) -> None:
    todo = [r for r in records if r["uid"] not in cache]
    if not todo:
        print(f"  cache hit for all {len(records):,} speeches")
        return
    print(f"  {len(todo):,} of {len(records):,} speeches need a parse")
    started = time.time()
    session = requests.Session()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    for start in range(0, len(todo), CHECKPOINT):
        chunk = todo[start : start + CHECKPOINT]
        texts = [" ".join((r.get("text_clean") or "").split()) for r in chunk]
        analysed = analyse_lines(
            texts,
            base_url=args.base_url,
            modules=args.modules,
            batch_lines=args.batch_lines,
            batch_words=args.batch_words,
            timeout=900.0,
            session=session,
        )
        with cache_path.open("a", encoding="utf-8") as handle:
            for record, sentences in zip(chunk, analysed, strict=True):
                # Only the three fields the metrics read are cached: the parse
                # of 826k words with every column would be hundreds of MB.
                serialised = [
                    [[t.dep_id, t.head, t.upostag] for t in sentence]
                    for sentence in sentences
                ]
                cache[record["uid"]] = serialised
                handle.write(
                    json.dumps(
                        {
                            "uid": record["uid"],
                            "modules": args.modules,
                            "sentences": serialised,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        done += len(chunk)
        rate = done / max(time.time() - started, 1e-9)
        remaining = (len(todo) - done) / max(rate, 1e-9) / 60
        print(
            f"    {done:,}/{len(todo):,} speeches ({rate:.2f}/s, "
            f"~{remaining:.0f} min left)",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "speech_parses.jsonl"
    if args.refresh_cache and cache_path.exists():
        cache_path.unlink()

    root = DATA_RAW if args.data_dir is None else Path(args.data_dir)
    path = root / f"cycle{args.cycle}-speeches.jsonl"
    if not path.is_file():
        print(f"No speeches export at {path}", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    records = [r for r in records if (r.get("text_clean") or "").strip()]
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records):,} speeches with text")

    print(f"Parsing with emtsv ({args.modules}) ...")
    cache = load_cache(cache_path, args.modules)
    analyse_missing(records, cache, cache_path, args)

    print("Measuring syntactic complexity ...")
    rows, null_mdd = [], 0
    for record in records:
        sentences = rebuild(cache[record["uid"]])
        if not sentences:
            continue
        result = measure(sentences, min_sentence_length=args.min_sentence_length)
        null_mdd += result.mdd is None
        speaker = record.get("speaker") or {}
        rows.append(
            {
                "uid": record["uid"],
                "cycle": args.cycle,
                "speaker_id": speaker.get("person_id"),
                "speaker": speaker.get("label"),
                "faction": speaker.get("faction"),
                "speech_type": record.get("speech_type"),
                "mdd": result.mdd,
                "mhd": result.mhd,
                "n_parsed_sentences": result.n_sentences,
                "n_sentences_discarded": result.n_sentences_discarded,
                "n_parsed_tokens": result.n_tokens,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "speech_syntax.csv", index=False, encoding="utf-8")
    print(f"  {len(frame):,} speeches; {null_mdd} with no sentence long enough")
    print(f"  MDD mean {frame['mdd'].mean():.3f} · MHD mean {frame['mhd'].mean():.3f}")

    by_mp = (
        frame.dropna(subset=["mdd"])
        .groupby(["speaker_id", "speaker", "faction"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "speeches_parsed": len(g),
                    "parsed_tokens": g["n_parsed_tokens"].sum(),
                    # Token-weighted: a two-sentence intervention should not
                    # count as much as a twenty-minute address.
                    "mdd": (g["mdd"] * g["n_parsed_tokens"]).sum()
                    / g["n_parsed_tokens"].sum(),
                    "mhd": (g["mhd"] * g["n_parsed_tokens"]).sum()
                    / g["n_parsed_tokens"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("parsed_tokens", ascending=False, ignore_index=True)
    )
    by_mp.to_csv(out / "mp_syntax.csv", index=False, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "source": str(path),
        "parser": PARSER,
        "parameters": {
            "emtsv_modules": args.modules,
            "min_sentence_length": args.min_sentence_length,
            "punctuation": "collapse",
            "aggregation": "macro",
            "mp_aggregation": "token-weighted mean",
        },
        "counts": {
            "speeches": len(frame),
            "speeches_without_measurable_syntax": int(null_mdd),
            "speakers": int(by_mp["speaker_id"].nunique()),
            "mdd_mean": round(float(frame["mdd"].mean()), 4),
            "mhd_mean": round(float(frame["mhd"].mean()), 4),
        },
        "note": (
            "A parser carries the head conventions of the treebank it was "
            "trained on, and those conventions decide every distance here. "
            "Quote `parser` alongside any figure taken from this file."
        ),
    }
    (out / "syntax_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
