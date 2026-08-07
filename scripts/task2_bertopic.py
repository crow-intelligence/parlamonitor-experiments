"""Task 2: unsupervised topic modeling of cycle-43 parliamentary speeches.

Pipeline: emtsv lemmatisation with a part-of-speech filter, Gensim bigram
fusion, frequency-derived stopwords, then BERTopic over chunked, mean-pooled
huBERT embeddings.

The lemmatisation pass is the slow one (~15-35 minutes for 826,775 words) and
is cached to JSONL keyed by speech uid, so re-running only redoes what is
missing. Every parameter that moves the topics is a flag, and all of them are
written to ``run_manifest.json`` next to the results.

Usage::

    uv run python scripts/task2_bertopic.py --limit 25    # smoke run
    uv run python scripts/task2_bertopic.py               # full corpus
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from parlamonitor.emtsv import (
    CONTENT_CATEGORIES,
    DEFAULT_BASE_URL,
    DEFAULT_MODULES,
    EmtsvError,
    analyse,
    is_content_word,
)
from parlamonitor.loading import load_speeches, provenance
from parlamonitor.topics import (
    HU_FUNCTION_WORDS,
    build_phrases,
    dynamic_stopwords,
    embed_documents,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "derived" / "task2"
DEFAULT_MODEL = "NYTK/sentence-transformers-experimental-hubert-hungarian"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="first N speeches only")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-count", type=int, default=5, help="gensim min_count")
    parser.add_argument(
        "--threshold", type=float, default=10.0, help="gensim threshold"
    )
    parser.add_argument("--max-df", type=float, default=0.85)
    parser.add_argument("--min-df", type=float, default=0.01)
    parser.add_argument("--min-cluster-size", type=int, default=10, help="HDBSCAN")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random_state")
    parser.add_argument(
        "--embed-source",
        choices=("text_clean", "phrased"),
        default="text_clean",
        help=(
            "what the encoder reads. text_clean is natural Hungarian, which is "
            "what huBERT was trained on; phrased is the lemma bag. Topic words "
            "always come from the phrased text either way."
        ),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="re-lemmatise everything instead of reusing lemmatized.jsonl",
    )
    return parser.parse_args(argv)


def log(message):
    print(f"[task2] {message}", flush=True)


# --- Step 2A: lemmatisation, cached ----------------------------------------


def read_cache(path):
    if not path.exists():
        return {}
    cached = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                cached[record["uid"]] = record
    return cached


def lemmatize_corpus(speeches, cache_path, *, base_url, modules, refresh):
    """Lemmatise every speech, appending to the JSONL cache as it goes."""
    cached = {} if refresh else read_cache(cache_path)
    if cached:
        log(f"reusing {len(cached)} cached analyses from {cache_path.name}")

    todo = [s for s in speeches if s["uid"] not in cached]
    if todo:
        log(f"lemmatising {len(todo)} speeches via {base_url}/{modules}")

    session = requests.Session()
    started = time.monotonic()
    with cache_path.open("a", encoding="utf-8") as handle:
        for index, speech in enumerate(todo, start=1):
            record = analyse_one(speech, session, base_url=base_url, modules=modules)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            cached[speech["uid"]] = record
            if index % 25 == 0 or index == len(todo):
                rate = index / max(time.monotonic() - started, 1e-9)
                remaining = (len(todo) - index) / max(rate, 1e-9)
                log(
                    f"  {index}/{len(todo)} speeches "
                    f"({rate:.1f}/s, ~{remaining / 60:.1f} min left)"
                )
    return [cached[s["uid"]] for s in speeches]


def analyse_one(speech, session, *, base_url, modules):
    try:
        tokens = analyse(
            speech["text_clean"], base_url=base_url, modules=modules, session=session
        )
    except (EmtsvError, ValueError) as exc:
        # Never fall back to the raw text: a speech that silently skipped
        # lemmatisation still produces plausible-looking topics.
        return {
            "uid": speech["uid"],
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "lemmas": None,
            "n_tokens": None,
            "n_content": None,
        }
    kept = [t.lemma for t in tokens if is_content_word(t.xpostag)]
    return {
        "uid": speech["uid"],
        "ok": True,
        "error": None,
        "lemmas": kept,
        "n_tokens": len(tokens),
        "n_content": len(kept),
    }


# --- output -----------------------------------------------------------------


def docker_image_digest(container="emtsv"):
    """Return the emtsv image digest, or None if docker cannot tell us."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Image}}", container],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def model_revision(model_id):
    """Return the pinned HF commit for the embedding model, or None."""
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "lemmatized.jsonl"

    # --- load -------------------------------------------------------------
    speeches = load_speeches()
    if args.limit is not None:
        speeches = speeches[: args.limit]
    log(f"{len(speeches)} speeches, {sum(s['n_words'] for s in speeches):,} words")

    # --- Step 2A ----------------------------------------------------------
    records = lemmatize_corpus(
        speeches,
        cache_path,
        base_url=args.base_url,
        modules=args.modules,
        refresh=args.refresh_cache,
    )

    failed = [r for r in records if not r["ok"]]
    empty = [r for r in records if r["ok"] and not r["lemmas"]]
    usable = [
        (s, r)
        for s, r in zip(speeches, records, strict=True)
        if r["ok"] and r["lemmas"]
    ]
    log(
        f"lemmatised ok: {len(records) - len(failed)}, failed: {len(failed)}, "
        f"no content words: {len(empty)}, usable: {len(usable)}"
    )
    if failed:
        log(f"  first failure: {failed[0]['uid']} -- {failed[0]['error']}")
    if not usable:
        log("no usable speeches; stopping")
        return 1

    kept_speeches = [s for s, _ in usable]
    tokenized = [r["lemmas"] for _, r in usable]
    n_tokens = sum(r["n_tokens"] for _, r in usable)
    n_content = sum(r["n_content"] for _, r in usable)
    log(
        f"tokens: {n_tokens:,} analysed, {n_content:,} kept as content words "
        f"({n_content / n_tokens:.1%})"
    )

    # --- Step 2B: bigrams -------------------------------------------------
    phrases = build_phrases(
        tokenized, min_count=args.min_count, threshold=args.threshold
    )
    phrased_tokens = [phrases[doc] for doc in tokenized]
    phrased_speeches = [" ".join(doc) for doc in phrased_tokens]
    n_bigrams = len(phrases.phrasegrams)
    log(f"gensim fused {n_bigrams} bigram types")

    # --- Step 2C: dynamic stopwords ---------------------------------------
    stopwords = dynamic_stopwords(
        phrased_speeches,
        max_df=args.max_df,
        min_df=args.min_df,
        manual=HU_FUNCTION_WORDS,
    )
    log(
        f"stopwords: {stopwords.n_dynamic} from frequency + {stopwords.n_manual} "
        f"manual = {len(stopwords.stopwords)} total; "
        f"{stopwords.vocabulary_size} terms survive"
    )

    # --- Step 2D: embeddings then BERTopic --------------------------------
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    log(f"loading encoder {args.model}")
    encoder = SentenceTransformer(args.model)
    max_seq = getattr(encoder, "max_seq_length", None)
    log(f"  max_seq_length={max_seq}; chunking at {args.chunk_size} words")

    embed_texts = (
        [s["text_clean"] for s in kept_speeches]
        if args.embed_source == "text_clean"
        else phrased_speeches
    )
    started = time.monotonic()
    embeddings = embed_documents(
        embed_texts,
        encoder,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    log(
        f"embedded {embeddings.shape[0]} documents into {embeddings.shape[1]} dims "
        f"in {time.monotonic() - started:.0f}s"
    )

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    topic_model = BERTopic(
        embedding_model=encoder,
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=args.seed,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(stop_words=stopwords.stopwords),
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probabilities = topic_model.fit_transform(
        phrased_speeches, embeddings=embeddings
    )

    probabilities = np.asarray(probabilities)
    assigned = probabilities.max(axis=1) if probabilities.ndim == 2 else probabilities

    topic_info = topic_model.get_topic_info()
    n_outliers = sum(1 for t in topics if t == -1)
    log(
        f"{len(topic_info) - 1} topics + outlier group; "
        f"{n_outliers} of {len(topics)} documents are outliers "
        f"({n_outliers / len(topics):.1%})"
    )

    # --- outputs ----------------------------------------------------------
    documents = pd.DataFrame(
        {
            "uid": [s["uid"] for s in kept_speeches],
            "date": [s["date"] for s in kept_speeches],
            "session_id": [s["session_id"] for s in kept_speeches],
            "speaker_label": [s["speaker"]["label"] for s in kept_speeches],
            "faction": [s["speaker"]["faction"] for s in kept_speeches],
            "speech_type": [s["speech_type"] for s in kept_speeches],
            "n_words": [s["n_words"] for s in kept_speeches],
            "topic": topics,
            "probability": assigned,
        }
    )
    documents_path = args.output_dir / "bertopic_documents.csv"
    topics_path = args.output_dir / "bertopic_topics.csv"
    documents.to_csv(documents_path, index=False, encoding="utf-8")
    topic_info.to_csv(topics_path, index=False, encoding="utf-8")

    manifest = {
        "source": provenance(),
        "parameters": {
            "limit": args.limit,
            "emtsv": {
                "base_url": args.base_url,
                "modules": args.modules,
                "image": docker_image_digest(),
                "content_categories": list(CONTENT_CATEGORIES),
                "drop_pronouns": True,
            },
            "phrases": {"min_count": args.min_count, "threshold": args.threshold},
            "stopwords": {
                "max_df": stopwords.max_df,
                "min_df": stopwords.min_df,
                "manual": sorted(HU_FUNCTION_WORDS),
            },
            "embedding": {
                "model": args.model,
                "revision": model_revision(args.model),
                "max_seq_length": max_seq,
                "chunk_size": args.chunk_size,
                "pooling": "mean over chunks, L2-normalised",
                "source": args.embed_source,
            },
            "bertopic": {
                "min_cluster_size": args.min_cluster_size,
                "umap_random_state": args.seed,
                "calculate_probabilities": True,
            },
        },
        "counts": {
            "speeches_loaded": len(speeches),
            "lemmatised_ok": len(records) - len(failed),
            "lemmatised_failed": len(failed),
            "no_content_words": len(empty),
            "speeches_modelled": len(usable),
            "tokens_analysed": n_tokens,
            "tokens_kept_as_content": n_content,
            "bigram_types": n_bigrams,
            "stopwords_dynamic": stopwords.n_dynamic,
            "stopwords_manual": stopwords.n_manual,
            "stopwords_total": len(stopwords.stopwords),
            "vocabulary_size": stopwords.vocabulary_size,
            "topics": len(topic_info) - 1,
            "outliers": n_outliers,
        },
        "failures": [{"uid": r["uid"], "error": r["error"]} for r in failed],
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log(f"wrote {documents_path}")
    log(f"wrote {topics_path}")
    log(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
