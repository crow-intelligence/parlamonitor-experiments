"""Task 2: unsupervised topic modeling of cycle-43 parliamentary speeches.

Pipeline: strip the opening salutation, lemmatise with emtsv under a
part-of-speech filter, drop Hungarian stopwords, fuse bigrams with Gensim,
derive further stopwords from corpus frequency, then cluster with BERTopic over
chunked, mean-pooled huBERT embeddings.

Two caches make iteration cheap. Lemmas go to JSONL keyed by
``(uid, normalisation)``; embeddings go to ``.npy`` keyed by a hash of
everything that would change them. Neither is reused across a change of rule.

The corpus is partitioned by discourse role -- question, answer, rejoinder,
reaction, debate -- so question time can be compared against ordinary debate.
The Q&A export is not a separate corpus: 622 of its 659 turns are speeches this
same file already contains.

Usage::

    uv run python scripts/task2_bertopic.py --limit 60    # smoke run
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
from parlamonitor.loading import load_qa, load_speeches, provenance
from parlamonitor.roles import compare_role_sources, discourse_role, qa_turn_uids
from parlamonitor.stopwords import hungarian_stopwords
from parlamonitor.text import NORMALISATION_VERSION, strip_salutation
from parlamonitor.topics import (
    build_phrases,
    drop_stopwords,
    dynamic_stopwords,
    embed_documents,
    embedding_cache_key,
    topic_fingerprint,
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
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help=(
            "absolute document count. Round 1 used the spec's 0.01 (~17 docs), "
            "which rejected 90%% of the vocabulary and so left only the most "
            "frequent, least distinctive terms"
        ),
    )
    parser.add_argument("--min-cluster-size", type=int, default=10, help="HDBSCAN")
    parser.add_argument("--mmr-diversity", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42, help="UMAP random_state")
    parser.add_argument(
        "--keep-pos",
        nargs="+",
        default=list(CONTENT_CATEGORIES),
        help="emMorph main categories to keep",
    )
    parser.add_argument("--no-strip-salutation", action="store_true")
    parser.add_argument("--no-stoplist", action="store_true")
    parser.add_argument(
        "--embed-source", choices=("text_clean", "phrased"), default="text_clean"
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "task2_bertopic",
        help="committed safetensors model; the pickle goes next to the outputs",
    )
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args(argv)


def log(message):
    print(f"[task2] {message}", flush=True)


# --- Step 2A: lemmatisation, cached by (uid, normalisation) -----------------


def read_cache(path, normalisation):
    """Return cached records whose normalisation matches the current rule."""
    if not path.exists():
        return {}
    cached, stale = {}, 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("normalisation") == normalisation:
                cached[record["uid"]] = record
            else:
                stale += 1
    if stale:
        log(f"ignoring {stale} cache entries written under an older normalisation")
    return cached


def lemmatize_corpus(
    speeches, cache_path, *, base_url, modules, keep, normalisation, refresh
):
    cached = {} if refresh else read_cache(cache_path, normalisation)
    if cached:
        log(f"reusing {len(cached)} cached analyses")

    todo = [s for s in speeches if s["uid"] not in cached]
    if todo:
        log(f"lemmatising {len(todo)} speeches via {base_url}/{modules}")

    session = requests.Session()
    started = time.monotonic()
    with cache_path.open("a", encoding="utf-8") as handle:
        for index, speech in enumerate(todo, start=1):
            record = analyse_one(
                speech, session, base_url=base_url, modules=modules, keep=keep
            )
            record["normalisation"] = normalisation
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            cached[speech["uid"]] = record
            if index % 50 == 0 or index == len(todo):
                rate = index / max(time.monotonic() - started, 1e-9)
                left = (len(todo) - index) / max(rate, 1e-9) / 60
                log(f"  {index}/{len(todo)} ({rate:.1f}/s, ~{left:.1f} min left)")
    return [cached[s["uid"]] for s in speeches]


def analyse_one(speech, session, *, base_url, modules, keep):
    try:
        tokens = analyse(
            speech["normalised_text"],
            base_url=base_url,
            modules=modules,
            session=session,
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
    kept = [t.lemma for t in tokens if is_content_word(t.xpostag, keep=tuple(keep))]
    return {
        "uid": speech["uid"],
        "ok": True,
        "error": None,
        "lemmas": kept,
        "n_tokens": len(tokens),
        "n_content": len(kept),
    }


# --- provenance helpers -----------------------------------------------------


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


def library_versions():
    """Versions the saved model was written under.

    BERTopic's own documentation says a model saved under one version should
    not be loaded under another, so the versions belong beside the artifact
    rather than in a lockfile the model may travel without.
    """
    import importlib.metadata as md

    versions = {}
    for name in ("bertopic", "umap-learn", "hdbscan", "scikit-learn", "numpy", "torch"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def save_model(topic_model, topic_info, *, model_dir, pickle_path, embedding_model):
    """Save both serializations and prove the committed one reloads.

    Returns the sizes on disk. A save that cannot be loaded is worse than no
    save at all -- it looks like insurance and is not -- so the safetensors
    copy is reloaded and checked against the in-memory model before the run is
    allowed to report success.
    """
    from bertopic import BERTopic

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(
        str(model_dir),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=embedding_model,
    )
    topic_model.save(str(pickle_path), serialization="pickle")

    reloaded = BERTopic.load(str(model_dir))
    reloaded_info = reloaded.get_topic_info()
    if len(reloaded_info) != len(topic_info):
        raise RuntimeError(
            f"saved model reloads with {len(reloaded_info)} topics, "
            f"expected {len(topic_info)}"
        )
    if list(reloaded_info.Topic) != list(topic_info.Topic):
        raise RuntimeError("saved model reloads with different topic ids")

    directory_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    return directory_size, pickle_path.stat().st_size


def normalise(speeches, *, strip):
    """Attach `normalised_text` to each speech; return the words removed."""
    removed_total = 0
    for speech in speeches:
        if strip:
            stripped, removed = strip_salutation(speech["text_clean"])
            speech["normalised_text"] = stripped
            removed_total += removed
        else:
            speech["normalised_text"] = speech["text_clean"]
    return removed_total


def main(argv=None):  # noqa: PLR0915 - a linear pipeline reads better in one piece
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "lemmatized.jsonl"

    # --- load and normalise ------------------------------------------------
    speeches = load_speeches()
    if args.limit is not None:
        speeches = speeches[: args.limit]

    strip = not args.no_strip_salutation
    normalisation = NORMALISATION_VERSION if strip else "none"
    salutation_words = normalise(speeches, strip=strip)
    log(
        f"{len(speeches)} speeches, {sum(s['n_words'] for s in speeches):,} words; "
        f"salutation stripped: {salutation_words:,} words ({normalisation})"
    )

    # --- discourse roles, derived twice and cross-checked -------------------
    role_check = compare_role_sources(speeches, qa_turn_uids(load_qa()))
    log(f"roles: {role_check['by_role']}")
    if role_check["agree"]:
        log("  speech_type and the Q&A turn list agree on the partition")
    else:
        log(
            f"  MISMATCH: {role_check['n_typed_qa_not_in_qa_file']} typed Q&A but "
            f"absent from the Q&A file; {role_check['n_in_qa_file_not_typed_qa']} "
            "the other way round"
        )

    # --- Step 2A: lemmatise ------------------------------------------------
    records = lemmatize_corpus(
        speeches,
        cache_path,
        base_url=args.base_url,
        modules=args.modules,
        keep=args.keep_pos,
        normalisation=normalisation,
        refresh=args.refresh_cache,
    )

    failed = [r for r in records if not r["ok"]]
    usable = [
        (s, r)
        for s, r in zip(speeches, records, strict=True)
        if r["ok"] and r["lemmas"]
    ]
    log(f"lemmatised ok: {len(records) - len(failed)}, failed: {len(failed)}")
    if failed:
        log(f"  first failure: {failed[0]['uid']} -- {failed[0]['error']}")
    if not usable:
        log("no usable speeches; stopping")
        return 1

    kept_speeches = [s for s, _ in usable]
    tokenized = [r["lemmas"] for _, r in usable]
    n_tokens = sum(r["n_tokens"] for _, r in usable)
    n_content = sum(len(t) for t in tokenized)

    # --- stoplist, applied before bigram detection -------------------------
    stoplist = frozenset() if args.no_stoplist else hungarian_stopwords()
    tokenized = drop_stopwords(tokenized, stoplist)
    n_after_stops = sum(len(t) for t in tokenized)
    log(
        f"tokens: {n_tokens:,} analysed -> {n_content:,} content words -> "
        f"{n_after_stops:,} after {len(stoplist)} stopwords "
        f"({n_after_stops / n_tokens:.1%} of all tokens)"
    )

    surviving = [i for i, tokens in enumerate(tokenized) if tokens]
    if len(surviving) < len(tokenized):
        log(f"  {len(tokenized) - len(surviving)} speeches emptied by filtering")
        kept_speeches = [kept_speeches[i] for i in surviving]
        tokenized = [tokenized[i] for i in surviving]

    # --- Step 2B: bigrams --------------------------------------------------
    phrases = build_phrases(
        tokenized, min_count=args.min_count, threshold=args.threshold
    )
    phrased_speeches = [" ".join(phrases[doc]) for doc in tokenized]
    n_bigrams = len(phrases.phrasegrams)
    log(f"gensim fused {n_bigrams} bigram types")

    # --- Step 2C: stopwords from corpus frequency --------------------------
    stopword_result = dynamic_stopwords(
        phrased_speeches, max_df=args.max_df, min_df=args.min_df, manual=stoplist
    )
    log(
        f"stopwords: {stopword_result.n_dynamic} from frequency + "
        f"{stopword_result.n_manual} listed = {len(stopword_result.stopwords)}; "
        f"{stopword_result.vocabulary_size} terms survive"
    )

    # --- Step 2D: embeddings, cached ---------------------------------------
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    embed_texts = (
        [s["normalised_text"] for s in kept_speeches]
        if args.embed_source == "text_clean"
        else phrased_speeches
    )
    key = embedding_cache_key(
        model=args.model,
        chunk_size=args.chunk_size,
        source=args.embed_source,
        normalisation=normalisation,
        uids=[s["uid"] for s in kept_speeches],
    )
    embeddings_path = args.output_dir / f"embeddings-{key}.npy"

    log(f"loading encoder {args.model}")
    encoder = SentenceTransformer(args.model)
    max_seq = getattr(encoder, "max_seq_length", None)

    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        log(f"reused cached embeddings {embeddings_path.name} {embeddings.shape}")
    else:
        log(f"  max_seq_length={max_seq}; chunking at {args.chunk_size} words")
        started = time.monotonic()
        embeddings = embed_documents(
            embed_texts,
            encoder,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            show_progress_bar=True,
        )
        np.save(embeddings_path, embeddings)
        log(
            f"embedded {embeddings.shape[0]} docs into {embeddings.shape[1]} dims "
            f"in {time.monotonic() - started:.0f}s -> {embeddings_path.name}"
        )

    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
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
        vectorizer_model=CountVectorizer(stop_words=stopword_result.stopwords),
        representation_model=MaximalMarginalRelevance(diversity=args.mmr_diversity),
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
        f"{len(topic_info) - 1} topics; {n_outliers}/{len(topics)} outliers "
        f"({n_outliers / len(topics):.1%})"
    )

    # Both assignments are written. Reduction places documents in topics they
    # only loosely fit, so the raw column stays alongside it.
    reduced = topic_model.reduce_outliers(
        phrased_speeches, topics, strategy="embeddings", embeddings=embeddings
    )
    n_reduced = sum(1 for t in reduced if t == -1)
    log(f"after reduce_outliers: {n_reduced} outliers ({n_reduced / len(reduced):.1%})")

    # --- persist the fitted model ------------------------------------------
    fingerprint = topic_fingerprint(
        {
            int(t): [term for term, _ in topic_model.get_topic(int(t))]
            for t in topic_info.Topic
        }
    )
    log(f"topic fingerprint: {fingerprint}")
    model_info = {"fingerprint": fingerprint, "versions": library_versions()}
    if not args.no_save_model:
        pickle_path = args.output_dir / "model.pkl"
        safetensors_bytes, pickle_bytes = save_model(
            topic_model,
            topic_info,
            model_dir=args.model_dir,
            pickle_path=pickle_path,
            embedding_model=args.model,
        )
        log(
            f"saved model: {args.model_dir} ({safetensors_bytes / 1e6:.1f} MB, "
            f"safetensors, reload verified) + "
            f"{pickle_path.name} ({pickle_bytes / 1e6:.1f} MB, pickle)"
        )
        model_info |= {
            "safetensors_dir": str(args.model_dir),
            "safetensors_bytes": safetensors_bytes,
            "pickle_path": str(pickle_path),
            "pickle_bytes": pickle_bytes,
        }

    # --- question time versus debate ---------------------------------------
    roles = [discourse_role(s) for s in kept_speeches]
    per_class = topic_model.topics_per_class(phrased_speeches, classes=roles)
    per_class.to_csv(
        args.output_dir / "topics_by_role.csv", index=False, encoding="utf-8"
    )

    # --- outputs -----------------------------------------------------------
    documents = pd.DataFrame(
        {
            "uid": [s["uid"] for s in kept_speeches],
            "date": [s["date"] for s in kept_speeches],
            "session_id": [s["session_id"] for s in kept_speeches],
            "speaker_label": [s["speaker"]["label"] for s in kept_speeches],
            "faction": [s["speaker"]["faction"] for s in kept_speeches],
            "speech_type": [s["speech_type"] for s in kept_speeches],
            "discourse_role": roles,
            "n_words": [s["n_words"] for s in kept_speeches],
            "topic": topics,
            "topic_reduced": reduced,
            "probability": assigned,
        }
    )
    documents.to_csv(
        args.output_dir / "bertopic_documents.csv", index=False, encoding="utf-8"
    )
    topic_info.to_csv(
        args.output_dir / "bertopic_topics.csv", index=False, encoding="utf-8"
    )

    manifest = {
        "source": provenance(),
        "model": model_info,
        "roles": role_check,
        "parameters": {
            "limit": args.limit,
            "normalisation": {
                "strip_salutation": strip,
                "version": normalisation,
                "words_removed": salutation_words,
            },
            "emtsv": {
                "base_url": args.base_url,
                "modules": args.modules,
                "image": docker_image_digest(),
                "keep_pos": list(args.keep_pos),
                "drop_pronouns": True,
            },
            "stoplist": {
                "enabled": not args.no_stoplist,
                "size": len(stoplist),
                "groups": "spacy_hu | light_verbs | parliamentary",
            },
            "phrases": {"min_count": args.min_count, "threshold": args.threshold},
            "frequency_stopwords": {
                "max_df": stopword_result.max_df,
                "min_df": stopword_result.min_df,
            },
            "embedding": {
                "model": args.model,
                "revision": model_revision(args.model),
                "max_seq_length": max_seq,
                "chunk_size": args.chunk_size,
                "pooling": "mean over chunks, L2-normalised",
                "source": args.embed_source,
                "cache_key": key,
            },
            "bertopic": {
                "min_cluster_size": args.min_cluster_size,
                "umap_random_state": args.seed,
                "mmr_diversity": args.mmr_diversity,
                "calculate_probabilities": True,
            },
        },
        "counts": {
            "speeches_loaded": len(speeches),
            "lemmatised_ok": len(records) - len(failed),
            "lemmatised_failed": len(failed),
            "speeches_modelled": len(kept_speeches),
            "tokens_analysed": n_tokens,
            "tokens_content": n_content,
            "tokens_after_stoplist": n_after_stops,
            "bigram_types": n_bigrams,
            "stopwords_dynamic": stopword_result.n_dynamic,
            "stopwords_listed": stopword_result.n_manual,
            "stopwords_total": len(stopword_result.stopwords),
            "vocabulary_size": stopword_result.vocabulary_size,
            "topics": len(topic_info) - 1,
            "outliers": n_outliers,
            "outliers_after_reduction": n_reduced,
        },
        "failures": [{"uid": r["uid"], "error": r["error"]} for r in failed],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
