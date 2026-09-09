"""Readability, lexical diversity and keywords for every speech.

Runs the cycle-43 speeches through emtsv once -- cached -- and then measures:

* **Readability**: LIX and RIX at the Hungarian long-word threshold of 8,
  counting Hungarian letters rather than characters, over emtsv's sentence
  segmentation. See :mod:`parlamonitor.readability` for why each of those
  three choices matters.
* **Lexical diversity**: MATTR over content-word lemmas, with a plain TTR and
  a flag for speeches shorter than the window.
* **Keywords**: TextRank over the co-occurrence graph of content lemmas, and
  KeyBERT over huBERT embeddings when the ``metrics`` extra is installed.

Emits one row per speech and one per speaker, the latter carrying the
denominators so any rate can be recomputed.

Usage::

    uv run python scripts/speech_metrics.py --limit 50   # smoke run
    uv run python scripts/speech_metrics.py
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
    DEFAULT_MODULES,
    analyse_lines,
    is_content_word,
    parse_xpostag,
)
from parlamonitor.keywords import DEFAULT_TOP_N, DEFAULT_WINDOW, textrank
from parlamonitor.loading import DATA_RAW
from parlamonitor.readability import (
    HUNGARIAN_LONG_WORD_THRESHOLD,
    MATTR_WINDOW,
    measure,
)
from parlamonitor.stopwords import hungarian_stopwords

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "metrics"
CHECKPOINT = 200


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cycle", type=int, default=43)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--textrank-window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--mattr-window", type=int, default=MATTR_WINDOW)
    parser.add_argument(
        "--long-word-threshold", type=int, default=HUNGARIAN_LONG_WORD_THRESHOLD
    )
    parser.add_argument("--no-keybert", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args(argv)


def load_speeches(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records if limit is None else records[:limit]


def load_cache(path: Path, modules: str) -> dict[str, list[list[list[str]]]]:
    if not path.is_file():
        return {}
    cache: dict[str, list[list[list[str]]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in handle:
            if not row.strip():
                continue
            record = json.loads(row)
            if record.get("modules") == modules:
                cache[record["uid"]] = record["sentences"]
    return cache


def analyse_speeches(
    records: Sequence[dict], cache: dict, cache_path: Path, args: argparse.Namespace
) -> None:
    """Send every uncached speech through emtsv, checkpointing as it goes."""
    todo = [r for r in records if r["uid"] not in cache]
    if not todo:
        print(f"  cache hit for all {len(records):,} speeches")
        return
    print(f"  {len(todo):,} of {len(records):,} speeches need emtsv")
    started = time.time()
    session = requests.Session()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    for start in range(0, len(todo), CHECKPOINT):
        chunk = todo[start : start + CHECKPOINT]
        # One speech per input line: newlines inside a speech would otherwise
        # split it into several, and `analyse_lines` aligns on newlines.
        texts = [" ".join((r.get("text_clean") or "").split()) for r in chunk]
        analysed = analyse_lines(
            texts,
            base_url=args.base_url,
            modules=args.modules,
            session=session,
        )
        with cache_path.open("a", encoding="utf-8") as handle:
            for record, sentences in zip(chunk, analysed, strict=True):
                serialised = [
                    [[t.form, t.lemma, t.xpostag] for t in sentence]
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
        print(f"    {done:,}/{len(todo):,} speeches ({rate:.1f}/s)", flush=True)


def streams(sentences: list[list[list[str]]]) -> tuple[list[str], list[str], int]:
    """Split a cached speech into word forms, content lemmas and a sentence count."""
    words: list[str] = []
    content: list[str] = []
    for sentence in sentences:
        for form, lemma, xpostag in sentence:
            category, _ = parse_xpostag(xpostag)
            if category is None and "Punct" in xpostag:
                continue
            if not form.strip():
                continue
            words.append(form)
            if is_content_word(xpostag):
                content.append(lemma)
    return words, content, len(sentences)


def keybert_keywords(
    documents: Sequence[str], top_n: int, stopwords: list[str]
) -> list[list[tuple[str, float]]]:
    """Rank phrases by similarity to the document embedding, via huBERT.

    Candidates are drawn from the **content-lemma stream**, not the original
    text, so that Hungarian inflection does not scatter one term across a dozen
    candidates. The consequence is that a returned bigram is two lemmas
    adjacent *after* function words were removed, which need not be a
    contiguous phrase in the speech -- ``felesküdött magyar`` is a real pairing
    of two content words, not a quotation. Read them as themes, not as phrases.
    """
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        "NYTK/sentence-transformers-experimental-hubert-hungarian"
    )
    model = KeyBERT(model=encoder)
    return model.extract_keywords(
        list(documents),
        keyphrase_ngram_range=(1, 2),
        stop_words=stopwords,
        top_n=top_n,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "speech_lemmas.jsonl"
    if args.refresh_cache and cache_path.exists():
        cache_path.unlink()

    root = DATA_RAW if args.data_dir is None else Path(args.data_dir)
    path = root / f"cycle{args.cycle}-speeches.jsonl"
    if not path.is_file():
        print(f"No speeches export at {path}", file=sys.stderr)
        return 1

    print(f"Reading {path.name} ...")
    records = load_speeches(path, args.limit)
    records = [r for r in records if (r.get("text_clean") or "").strip()]
    print(f"  {len(records):,} speeches with text")

    print("Lemmatising with emtsv ...")
    cache = load_cache(cache_path, args.modules)
    analyse_speeches(records, cache, cache_path, args)

    stoplist = hungarian_stopwords()
    print("Measuring ...")
    rows = []
    skipped = []
    for record in records:
        words, content, n_sentences = streams(cache[record["uid"]])
        if not words or n_sentences == 0:
            skipped.append(record["uid"])
            continue
        result = measure(
            words,
            content,
            n_sentences,
            threshold=args.long_word_threshold,
            window=args.mattr_window,
        )
        ranked = textrank(
            content,
            top_n=args.top_n,
            window=args.textrank_window,
            stopwords=stoplist,
        )
        speaker = record.get("speaker") or {}
        rows.append(
            {
                "uid": record["uid"],
                "cycle": args.cycle,
                "date": record.get("date"),
                "speaker_id": speaker.get("person_id"),
                "speaker": speaker.get("label"),
                "faction": speaker.get("faction"),
                "speech_type": record.get("speech_type"),
                "duration_s": record.get("duration_s"),
                "n_words": result.n_words,
                "n_sentences": result.n_sentences,
                "n_long_words": result.n_long_words,
                "words_per_sentence": result.words_per_sentence,
                "lix": result.lix,
                "lix_band": result.lix_band,
                "rix": result.rix,
                "mattr": result.mattr,
                "mattr_windowed": result.mattr_windowed,
                "readability_reliable": result.readability_reliable,
                "n_content_lemmas": len(content),
                "n_types": result.n_types,
                "long_word_threshold": result.long_word_threshold,
                "mattr_window": result.mattr_window,
                "textrank_keywords": ";".join(term for term, _ in ranked),
                "textrank_scores": ";".join(f"{score:.6f}" for _, score in ranked),
                "notes": ";".join(result.notes),
            }
        )
    frame = pd.DataFrame(rows)
    print(f"  measured {len(frame):,} speeches; skipped {len(skipped)} with no words")

    if not args.no_keybert:
        try:
            print("Extracting KeyBERT keywords ...")
            documents = [" ".join(streams(cache[uid])[1]) for uid in frame["uid"]]
            extracted = keybert_keywords(documents, args.top_n, sorted(stoplist))
            frame["keybert_keywords"] = [
                ";".join(term for term, _ in kws) for kws in extracted
            ]
            frame["keybert_scores"] = [
                ";".join(f"{score:.4f}" for _, score in kws) for kws in extracted
            ]
        except ImportError as exc:
            print(f"  KeyBERT unavailable ({exc}); TextRank only")
            frame["keybert_keywords"] = None
            frame["keybert_scores"] = None

    frame.to_csv(out / "speech_metrics.csv", index=False, encoding="utf-8")

    # Per speaker, carrying the denominators. Records with no sentence
    # punctuation are excluded from the readability means -- a 594-LIX
    # roll-call would swamp any speaker unlucky enough to have read one -- but
    # they remain in speech_metrics.csv, flagged.
    prose = frame[frame["readability_reliable"]]
    by_mp = (
        prose.groupby(["speaker_id", "speaker", "faction"], dropna=False)
        .agg(
            speeches=("uid", "size"),
            words=("n_words", "sum"),
            sentences=("n_sentences", "sum"),
            lix_mean=("lix", "mean"),
            lix_weighted=(
                "lix",
                lambda s: None,
            ),
            rix_mean=("rix", "mean"),
            mattr_mean=("mattr", "mean"),
            words_per_sentence_mean=("words_per_sentence", "mean"),
        )
        .reset_index()
    )
    # A mean of per-speech LIX weights a two-sentence intervention like a
    # twenty-minute address. The word-weighted mean is the one to compare.
    weighted = (
        prose.assign(_w=prose["n_words"])
        .groupby("speaker_id", dropna=False)
        .apply(
            lambda g: pd.Series(
                {"lix_word_weighted": (g["lix"] * g["_w"]).sum() / g["_w"].sum()}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    by_mp = by_mp.drop(columns=["lix_weighted"]).merge(
        weighted, on="speaker_id", how="left"
    )
    by_mp = by_mp.sort_values("speeches", ascending=False, ignore_index=True)
    by_mp.to_csv(out / "mp_metrics.csv", index=False, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "source": str(path),
        "cycle": args.cycle,
        "parameters": {
            "emtsv_modules": args.modules,
            "long_word_threshold": args.long_word_threshold,
            "long_word_threshold_source": "saphes.recommended_threshold('hu')",
            "letter_counting": "saphes.hungarian_letter_count (digraph-aware)",
            "sentences_from": "emtsv tok segmentation",
            "mattr_window": args.mattr_window,
            "diversity_unit": "lemma, content words only",
            "textrank_window": args.textrank_window,
            "top_n": args.top_n,
            "keybert": not args.no_keybert,
        },
        "counts": {
            "speeches": len(frame),
            "skipped_no_words": skipped,
            "speakers": int(by_mp["speaker_id"].nunique()),
            "lix_band_counts": frame["lix_band"].value_counts().to_dict(),
            "mattr_not_windowed": int((~frame["mattr_windowed"]).sum()),
            "readability_unreliable": int((~frame["readability_reliable"]).sum()),
            "readability_unreliable_by_speech_type": (
                frame[~frame["readability_reliable"]]["speech_type"]
                .value_counts()
                .to_dict()
            ),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
