"""Sentiment and emotion scores for every speech, with the label mapping derived.

None of the three models declares its labels -- all ship ``LABEL_0``,
``LABEL_1``, ... -- so this script **calibrates before it scores**. It runs
probes of known valence through each model, derives the index-to-label mapping
from the result, and writes both the mapping and the confusion matrix that
produced it into the manifest. A model that changes upstream will change the
calibration rather than silently mislabel a column.

The calibration also reports which channels do not work. The Hungarian emotion
model's anger head averages 0.08 on unambiguous Hungarian anger text, which is
why the multilingual model runs alongside it.

Speeches longer than the models' 512-token window are chunked and averaged, the
same approach ``scripts/task1_classifier.py`` uses.

Usage::

    uv run python scripts/speech_affect.py --limit 40   # smoke run
    uv run python scripts/speech_affect.py
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

from parlamonitor.affect import (
    CALIBRATION_PROBES,
    CHUNK_TOKENS,
    EMOTION_LABELS,
    EMOTION_MODEL_HU,
    EMOTION_MODEL_XLM,
    SENTIMENT_MODEL,
    SENTIMENT_PROBES,
    UNRELIABLE_CHANNELS,
    XLM_EMOTION_LABELS,
    derive_ordinal_mapping,
    valence,
)
from parlamonitor.loading import DATA_RAW

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "metrics"

SENTIMENT_SCALE = ("very negative", "negative", "neutral", "positive", "very positive")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cycle", type=int, default=43)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


def load_model(model_id: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    return tokenizer, model


def score_texts(
    tokenizer, model, texts: Sequence[str], *, multi_label: bool, batch: int
):
    """Return one probability vector per text."""
    import torch

    out = []
    for start in range(0, len(texts), batch):
        window = list(texts[start : start + batch])
        encoded = tokenizer(
            window, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=-1)
        out.extend(probs.tolist())
    return out


def calibrate(tokenizer, model, probes: dict, *, multi_label: bool, batch: int):
    """Run probes of known class and return the mean vector each produced."""
    confusion = {}
    for label, texts in probes.items():
        vectors = score_texts(
            tokenizer, model, list(texts), multi_label=multi_label, batch=batch
        )
        confusion[label] = [
            round(sum(col) / len(col), 4) for col in zip(*vectors, strict=True)
        ]
    return confusion


def check_mapping(confusion: dict, labels: Sequence[str]) -> dict:
    """Check that each probe peaks on the index its label claims."""
    report = {}
    for probe_label, vector in confusion.items():
        if probe_label not in labels:
            continue
        expected = labels.index(probe_label)
        peak = vector.index(max(vector))
        report[probe_label] = {
            "expected_index": expected,
            "observed_peak": peak,
            "own_score": vector[expected],
            "agrees": peak == expected,
        }
    return report


def chunk_text(tokenizer, text: str, size: int) -> list[str]:
    """Split a text into chunks that fit the model window."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= size:
        return [text]
    return [
        tokenizer.decode(ids[i : i + size], skip_special_tokens=True)
        for i in range(0, len(ids), size)
    ]


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    return [sum(col) / len(col) for col in zip(*vectors, strict=True)]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)

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

    print("Loading models ...")
    sent_tok, sent_model = load_model(SENTIMENT_MODEL)
    hu_tok, hu_model = load_model(EMOTION_MODEL_HU)
    xlm_tok, xlm_model = load_model(EMOTION_MODEL_XLM)

    print("Calibrating unnamed labels ...")
    sent_conf = calibrate(
        sent_tok, sent_model, SENTIMENT_PROBES, multi_label=False, batch=args.batch_size
    )
    reversed_scale, weak_ends = derive_ordinal_mapping(sent_conf)
    scale = tuple(reversed(SENTIMENT_SCALE)) if reversed_scale else SENTIMENT_SCALE
    print(f"  sentiment scale: index 0 = {scale[0]!r} (reversed={reversed_scale})")
    if weak_ends:
        print(f"  warning: weak sentiment probes at {weak_ends}")

    hu_conf = calibrate(
        hu_tok, hu_model, CALIBRATION_PROBES, multi_label=True, batch=args.batch_size
    )
    hu_check = check_mapping(hu_conf, EMOTION_LABELS)
    disagree = [k for k, v in hu_check.items() if not v["agrees"]]
    agree = len(hu_check) - len(disagree)
    print(f"  hungarian emotion: {agree}/{len(hu_check)} probes agree")
    if disagree:
        print(f"  warning: these do not peak on their own index: {disagree}")

    xlm_probes = {
        k: v for k, v in CALIBRATION_PROBES.items() if k in XLM_EMOTION_LABELS
    }
    xlm_conf = calibrate(
        xlm_tok, xlm_model, xlm_probes, multi_label=False, batch=args.batch_size
    )
    xlm_check = check_mapping(xlm_conf, XLM_EMOTION_LABELS)
    print(
        f"  multilingual emotion: "
        f"{sum(v['agrees'] for v in xlm_check.values())}/{len(xlm_check)} probes agree"
    )

    print("Scoring ...")
    rows = [
        {
            "uid": record["uid"],
            "cycle": args.cycle,
            "date": record.get("date"),
            "speaker_id": (record.get("speaker") or {}).get("person_id"),
            "speaker": (record.get("speaker") or {}).get("label"),
            "faction": (record.get("speaker") or {}).get("faction"),
            "speech_type": record.get("speech_type"),
            "n_words": record.get("n_words"),
        }
        for record in records
    ]

    # Every chunk of every speech goes through a model in one pass. Batching
    # within a single speech wastes the batch: most speeches are one or two
    # chunks, so a batch of 16 ran at 2/16 capacity and the corpus would have
    # taken hours.
    for tok, model, labels, prefix, multi in (
        (sent_tok, sent_model, scale, "sentiment", False),
        (hu_tok, hu_model, EMOTION_LABELS, "emotion_hu", True),
        (xlm_tok, xlm_model, XLM_EMOTION_LABELS, "emotion_xlm", False),
    ):
        started = time.time()
        chunks: list[str] = []
        offsets: list[tuple[int, int]] = []
        for record in records:
            pieces = chunk_text(tok, record["text_clean"], args.chunk_tokens)
            offsets.append((len(chunks), len(chunks) + len(pieces)))
            chunks.extend(pieces)
        print(f"  {prefix}: {len(chunks):,} chunks from {len(records):,} speeches")

        vectors: list[list[float]] = []
        for start in range(0, len(chunks), args.batch_size):
            vectors.extend(
                score_texts(
                    tok,
                    model,
                    chunks[start : start + args.batch_size],
                    multi_label=multi,
                    batch=args.batch_size,
                )
            )
            done = min(start + args.batch_size, len(chunks))
            if done % (args.batch_size * 20) < args.batch_size or done == len(chunks):
                rate = done / max(time.time() - started, 1e-9)
                print(f"    {done:,}/{len(chunks):,} chunks ({rate:.1f}/s)", flush=True)

        for row, (a, b) in zip(rows, offsets, strict=True):
            averaged = mean_vector(vectors[a:b])
            for label, value in zip(labels, averaged, strict=True):
                row[f"{prefix}_{label.replace(' ', '_')}"] = round(value, 6)
            if prefix == "sentiment":
                row["sentiment_label"] = labels[averaged.index(max(averaged))]
                # Expectation over the ordinal scale, always oriented negative
                # to positive regardless of the model's own index order.
                ordered = list(reversed(averaged)) if reversed_scale else averaged
                total = sum(ordered)
                row["sentiment_valence"] = valence([p / total for p in ordered])
                row["n_chunks"] = b - a

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "speech_affect.csv", index=False, encoding="utf-8")

    score_columns = [
        c
        for c in frame.columns
        if c.startswith(("sentiment_", "emotion_")) and c != "sentiment_label"
    ]
    by_mp = (
        frame.groupby(["speaker_id", "speaker", "faction"], dropna=False)
        .agg(
            speeches=("uid", "size"),
            words=("n_words", "sum"),
            **{c: (c, "mean") for c in score_columns},
        )
        .reset_index()
        .sort_values("speeches", ascending=False, ignore_index=True)
    )
    by_mp.to_csv(out / "mp_affect.csv", index=False, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "source": str(path),
        "models": {
            "sentiment": SENTIMENT_MODEL,
            "emotion_hu": EMOTION_MODEL_HU,
            "emotion_xlm": EMOTION_MODEL_XLM,
        },
        "parameters": {
            "chunk_tokens": args.chunk_tokens,
            "aggregation": "mean over chunks",
        },
        "calibration": {
            "note": (
                "No model declares its labels; every mapping below was derived "
                "by probing and is reproducible from parlamonitor.affect"
            ),
            "sentiment_scale": list(scale),
            "sentiment_reversed": reversed_scale,
            "sentiment_confusion": sent_conf,
            "emotion_hu_labels": list(EMOTION_LABELS),
            "emotion_hu_confusion": hu_conf,
            "emotion_hu_check": hu_check,
            "emotion_xlm_labels": list(XLM_EMOTION_LABELS),
            "emotion_xlm_confusion": xlm_conf,
            "emotion_xlm_check": xlm_check,
            "unreliable_channels": sorted(UNRELIABLE_CHANNELS),
        },
        "counts": {
            "speeches": len(frame),
            "speakers": int(by_mp["speaker_id"].nunique()),
            "sentiment_label_counts": frame["sentiment_label"].value_counts().to_dict(),
            "chunked": int((frame["n_chunks"] > 1).sum()),
        },
    }
    (out / "affect_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
