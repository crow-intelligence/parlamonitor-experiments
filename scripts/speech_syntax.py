"""Mean dependency distance and hierarchical distance for every speech.

Needs a **parse**, not just a lemmatiser, and the parser is HuSpaCy rather
than emtsv. emtsv can parse, but its cost grows super-linearly with document
length -- 80 words in 0.4 s, 633 in 14 s, a 3,000-word request not returning
inside 100 s -- which on this corpus is hours. HuSpaCy runs in-process at
~2,100 words/s, linearly, so the whole corpus takes minutes and the 5,281-word
longest speech needs no chunking: spaCy's ``max_length`` is 1,000,000
characters against its 39,103.

Chunking would have been the alternative, and for this metric it has a trap
worth stating. MDD aggregates over **sentences**, not over texts, so a chunked
text must be recombined by concatenating its sentences and scoring once --
*not* by averaging per-chunk scores, which would weight a three-sentence chunk
the same as a forty-sentence one.

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

from parlamonitor.loading import DATA_RAW
from parlamonitor.syntax import (
    MAX_SENTENCE_LENGTH,
    MIN_SENTENCE_LENGTH,
    PARSER,
    measure_doc,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "metrics"
DEFAULT_MODEL = "hu_core_news_md"
BATCH_SIZE = 32


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cycle", type=int, default=43)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--min-sentence-length", type=int, default=MIN_SENTENCE_LENGTH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

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

    import spacy

    print(f"Loading {args.model} ...")
    # `ner` is excluded: it is a third of the pipeline's time and nothing here
    # reads an entity.
    nlp = spacy.load(args.model, exclude=["ner"])
    print(f"  pipeline: {', '.join(nlp.pipe_names)}")

    print("Parsing ...")
    started = time.time()
    rows, null_mdd, words, repaired_heads, too_long = [], 0, 0, 0, 0
    texts = [r["text_clean"] for r in records]
    docs = nlp.pipe(texts, batch_size=args.batch_size, n_process=args.processes)
    for index, (record, doc) in enumerate(zip(records, docs, strict=True), start=1):
        result = measure_doc(doc, min_sentence_length=args.min_sentence_length)
        null_mdd += result.mdd is None
        repaired_heads += result.n_heads_repaired
        too_long += result.n_sentences_too_long
        words += record.get("n_words") or 0
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
                "n_sentences_too_long": result.n_sentences_too_long,
                "n_parsed_tokens": result.n_tokens,
                "n_heads_repaired": result.n_heads_repaired,
            }
        )
        if index % 250 == 0 or index == len(records):
            rate = words / max(time.time() - started, 1e-9)
            print(
                f"  {index:,}/{len(records):,} speeches ({rate:,.0f} words/s)",
                flush=True,
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
            "model": args.model,
            "pipeline_excludes": ["ner"],
            "min_sentence_length": args.min_sentence_length,
            "max_sentence_length": MAX_SENTENCE_LENGTH,
            "punctuation": "collapse",
            "aggregation": "macro",
            "mp_aggregation": "token-weighted mean",
        },
        "counts": {
            "speeches": len(frame),
            "speeches_without_measurable_syntax": int(null_mdd),
            "sentences_too_long": int(too_long),
            "sentences_too_long_note": (
                "sentences over the length cap, i.e. text with no sentence "
                "punctuation such as the notary's roll-call"
            ),
            "heads_repaired": int(repaired_heads),
            "heads_repaired_note": (
                "tokens whose governor lay outside their own sentence, made "
                "local roots; the segmenter and parser disagreed there"
            ),
            "speakers": int(by_mp["speaker_id"].nunique()),
            "mdd_mean": round(float(frame["mdd"].mean()), 4),
            "mhd_mean": round(float(frame["mhd"].mean()), 4),
        },
        "note": (
            "A parser carries the head conventions of the treebank it was "
            "trained on, and those conventions decide every distance here. "
            "Quote `parser` alongside any figure taken from this file, and do "
            "not compare these numbers with an MDD from a different parser: "
            "HuSpaCy follows Universal Dependencies, emtsv's `dep` does not."
        ),
        "model_licence": "CC BY-SA 4.0 (hu_core_news_md, SzegedAI/MILAB)",
    }
    (out / "syntax_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
