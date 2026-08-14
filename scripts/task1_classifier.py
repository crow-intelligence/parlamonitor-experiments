"""Task 1: supervised CAP topic classification of cycle-43 speeches.

Runs ``classla/ParlaCAP-Topic-Classifier`` over the corpus twice.

**Truncated** is the specification's pass, and the model authors' own: one
forward pass per speech, ``max_length=512, truncation=True``. It is what
produced the published Mix rates of 8.9-11.4%, so our rate is comparable to
theirs. It also means 56% of these speeches are judged on their opening ~279
words, because Hungarian costs a median 1.82 subword tokens per word.

**Chunked** splits each speech into 250-word windows, classifies every one, and
averages the label distributions weighted by window length. It sees the whole
speech, at the cost of that comparability.

Both are written, along with whether they agree. The disagreement rate is the
measurement of what truncation costs on this corpus -- a number the spec cannot
provide from one pass.

Raw per-window scores are cached to JSONL, so a re-run or an interrupted run
costs nothing it has already paid for.

Usage::

    uv run python scripts/task1_classifier.py --limit 40   # smoke run
    uv run python scripts/task1_classifier.py              # ~115 min on CPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from parlamonitor.cap import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_THRESHOLD,
    MAX_TOKENS,
    MODEL_ID,
    aggregate_scores,
    apply_threshold,
)
from parlamonitor.loading import load_speeches, provenance
from parlamonitor.roles import discourse_role
from parlamonitor.topics import chunk_tokens

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "derived" / "task1"

# The specification's pre-truncation. Measured against this corpus it changes
# no prediction -- 400 words is ~728 tokens, so the tokenizer's 512 limit binds
# first either way -- but it does what it claims, keeping long speeches out of
# the tokenizer, and it is cheap. Kept.
PRETRUNCATE_WORDS = 400


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="first N speeches only")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    parser.add_argument(
        "--no-chunked",
        action="store_true",
        help="run only the specification's truncated pass",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args(argv)


def log(message):
    print(f"[task1] {message}", flush=True)


def read_cache(path):
    """Map cache key to its stored label distribution."""
    if not path.exists():
        return {}
    cached = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                cached[record["key"]] = record
    return cached


def model_revision(model_id):
    """Return the pinned HF commit for the model, or None if offline."""
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


def classify(classifier, texts, *, batch_size):
    """Return a full label distribution for each text."""
    outputs = classifier(list(texts), batch_size=batch_size, top_k=None)
    return [{row["label"]: float(row["score"]) for row in out} for out in outputs]


def run_pass(classifier, jobs, cache, cache_handle, *, batch_size, label):
    """Classify every uncached job, appending results to the cache as it goes.

    Args:
        classifier: A transformers text-classification pipeline.
        jobs: ``(key, text)`` pairs.
        cache: Already-known results, keyed the same way; mutated in place.
        cache_handle: Open file to append new records to.
        batch_size: Pipeline batch size.
        label: Name of this pass, for progress messages.
    """
    todo = [(key, text) for key, text in jobs if key not in cache]
    if not todo:
        log(f"{label}: all {len(jobs)} already cached")
        return
    log(f"{label}: classifying {len(todo)} of {len(jobs)}")

    started = time.monotonic()
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        for (key, _), scores in zip(
            batch,
            classify(classifier, [text for _, text in batch], batch_size=batch_size),
            strict=True,
        ):
            record = {"key": key, "scores": scores}
            cache_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            cache[key] = record
        cache_handle.flush()
        done = min(start + batch_size, len(todo))
        if done % (batch_size * 10) == 0 or done == len(todo):
            rate = done / max(time.monotonic() - started, 1e-9)
            left = (len(todo) - done) / max(rate, 1e-9) / 60
            log(f"  {label}: {done}/{len(todo)} ({rate:.2f}/s, ~{left:.0f} min left)")


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    speeches = load_speeches()
    if args.limit is not None:
        speeches = speeches[: args.limit]
    log(f"{len(speeches)} speeches, {sum(s['n_words'] for s in speeches):,} words")

    from transformers import pipeline

    log(f"loading {args.model} (XLM-R large, CPU, {args.threads} threads)")
    started = time.monotonic()
    classifier = pipeline(
        "text-classification",
        model=args.model,
        max_length=MAX_TOKENS,
        truncation=True,
    )
    log(f"  loaded in {time.monotonic() - started:.0f}s")

    cache_path = args.output_dir / "cap_scores.jsonl"
    if args.refresh_cache and cache_path.exists():
        cache_path.unlink()
    cache = read_cache(cache_path)
    if cache:
        log(f"reusing {len(cache)} cached classifications")

    # --- build the jobs for both passes ------------------------------------
    truncated_jobs, chunk_jobs, chunk_index = [], [], {}
    for speech in speeches:
        words = speech["text_clean"].split()
        truncated_jobs.append(
            (f"trunc:{speech['uid']}", " ".join(words[:PRETRUNCATE_WORDS]))
        )
        if not args.no_chunked:
            windows = chunk_tokens(words, args.chunk_words) if words else [[""]]
            keys = []
            for position, window in enumerate(windows):
                key = f"chunk:{speech['uid']}:{position}"
                chunk_jobs.append((key, " ".join(window)))
                keys.append((key, len(window)))
            chunk_index[speech["uid"]] = keys

    with cache_path.open("a", encoding="utf-8") as handle:
        run_pass(
            classifier,
            truncated_jobs,
            cache,
            handle,
            batch_size=args.batch_size,
            label="truncated",
        )
        if chunk_jobs:
            log(f"chunked: {len(chunk_jobs)} windows of <={args.chunk_words} words")
            run_pass(
                classifier,
                chunk_jobs,
                cache,
                handle,
                batch_size=args.batch_size,
                label="chunked",
            )

    # Real token counts, not a words-times-1.82 estimate: this is what decides
    # whether a speech was actually truncated, and it is cheap next to a
    # forward pass.
    tokenizer = classifier.tokenizer
    token_counts = {
        speech["uid"]: len(tokenizer(speech["text_clean"])["input_ids"])
        for speech in speeches
    }
    n_truncated = sum(1 for n in token_counts.values() if n > MAX_TOKENS)
    log(
        f"{n_truncated}/{len(speeches)} speeches exceed {MAX_TOKENS} tokens "
        f"({n_truncated / len(speeches):.1%})"
    )

    # --- assemble ----------------------------------------------------------
    rows = []
    for speech in speeches:
        scores = cache[f"trunc:{speech['uid']}"]["scores"]
        raw_label = max(scores, key=lambda label: scores[label])
        raw_score = scores[raw_label]
        final = apply_threshold(raw_label, raw_score, threshold=args.threshold)

        row = {
            # The three columns the specification asks for, first and by name.
            "Original_Text": speech["text_clean"],
            "Predicted_CAP_Topic": final,
            "Confidence_Score": raw_score,
            # Added so the output can be joined back to the corpus and to
            # Task 2. Three anonymous columns cannot be.
            "uid": speech["uid"],
            "date": speech["date"],
            "speaker_label": speech["speaker"]["label"],
            "faction": speech["speaker"]["faction"],
            "speech_type": speech["speech_type"],
            "discourse_role": discourse_role(speech),
            "n_words": speech["n_words"],
            # Kept so the Mix override stays auditable: a confident `Other` and
            # a low-confidence anything both mean "no single topic", but only
            # one of them is the model's judgement.
            "raw_label": raw_label,
            "n_tokens": token_counts[speech["uid"]],
            "truncated": token_counts[speech["uid"]] > MAX_TOKENS,
        }

        if not args.no_chunked:
            windows = chunk_index[speech["uid"]]
            distributions = [cache[key]["scores"] for key, _ in windows]
            weights = [max(size, 1) for _, size in windows]
            chunk_label, chunk_score, _ = aggregate_scores(distributions, weights)
            row["chunked_CAP_Topic"] = apply_threshold(
                chunk_label, chunk_score, threshold=args.threshold
            )
            row["chunked_Confidence"] = chunk_score
            row["chunked_raw_label"] = chunk_label
            row["n_windows"] = len(windows)
            row["passes_agree"] = row["chunked_CAP_Topic"] == final
        rows.append(row)

    frame = pd.DataFrame(rows)
    output = args.output_dir / "parlacap_classifications.csv"
    frame.to_csv(output, index=False, encoding="utf-8")

    # --- report ------------------------------------------------------------
    mix_rate = (frame.Predicted_CAP_Topic == "Mix").mean()
    log(f"Mix rate (truncated): {mix_rate:.1%} -- authors saw 8.9-11.4%")
    log(f"labels used: {frame.Predicted_CAP_Topic.nunique()}")
    counts = {"truncated_mix_rate": float(mix_rate)}
    if not args.no_chunked:
        agree = frame.passes_agree.mean()
        chunk_mix = (frame.chunked_CAP_Topic == "Mix").mean()
        multi = frame.n_windows > 1
        log(f"Mix rate (chunked): {chunk_mix:.1%}")
        log(f"passes agree: {agree:.1%} overall")
        log(f"  on speeches needing >1 window: {frame[multi].passes_agree.mean():.1%}")
        log(f"  on speeches fitting in one:    {frame[~multi].passes_agree.mean():.1%}")
        counts |= {
            "chunked_mix_rate": float(chunk_mix),
            "agreement": float(agree),
            "agreement_multiwindow": float(frame[multi].passes_agree.mean()),
            "speeches_needing_multiple_windows": int(multi.sum()),
            "windows_total": int(frame.n_windows.sum()),
        }

    manifest = {
        "source": provenance(),
        "parameters": {
            "model": args.model,
            "revision": model_revision(args.model),
            "threshold": args.threshold,
            "threshold_source": "model card: below 0.60 annotated as Mix",
            "max_length": MAX_TOKENS,
            "pretruncate_words": PRETRUNCATE_WORDS,
            "pretruncate_effect": (
                "none on predictions; 400 words is ~728 Hungarian tokens, so "
                "the 512-token limit binds first either way"
            ),
            "chunk_words": args.chunk_words,
            "chunk_aggregation": "length-weighted mean of label distributions",
            "limit": args.limit,
        },
        "counts": counts
        | {
            "speeches": len(frame),
            "label_distribution": frame.Predicted_CAP_Topic.value_counts().to_dict(),
        },
        "caveat": (
            "The model card reports F1 for English, Croatian, Serbian and "
            "Bosnian only. Hungarian is among its languages and ParlaMint-HU "
            "is among the 29 training datasets, but there is no published "
            "Hungarian evaluation, so no accuracy figure can be quoted here."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
