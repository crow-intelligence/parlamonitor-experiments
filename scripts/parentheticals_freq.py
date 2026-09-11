"""Lemmatised word frequencies, with fused n-grams, for the parentheticals corpus.

Pipeline:

1. Read the five cycle files (39-43) exactly as they are; record line, word,
   distinct-line and repeat counts per cycle.
2. Send every **distinct** line through emtsv once, batched and line-aligned,
   and cache the result. 64,052 of 261,935 lines are distinct, so this is a
   4x saving on the analysis and makes re-runs free.
3. Repair the party names emMorph decomposes into common words -- ``Jobbik``
   would otherwise be counted as ``jó`` 20,705 times.
4. Drop punctuation, fit one NPMI phrase model on all cycles pooled, and fuse
   bigrams then trigrams with ``#``.
5. Count per cycle and pooled, twice: as the files read, and with consecutive
   duplicate lines collapsed.

Usage::

    uv run python scripts/parentheticals_freq.py --limit 2000   # smoke run
    uv run python scripts/parentheticals_freq.py                # full corpus
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from parlamonitor.emtsv import (
    DEFAULT_BASE_URL,
    DEFAULT_MODULES,
    Token,
    analyse_lines,
    is_content_word,
    parse_xpostag,
)
from parlamonitor.frequency import count_tokens, frequency_frame, ngram_order, npmi
from parlamonitor.parentheticals import (
    AGGREGATE_LABEL,
    CYCLES,
    NORMALISATION_VERSION,
    collapse_consecutive,
    file_sha256,
    line_stats,
    load_parentheticals,
    parentheticals_path,
)
from parlamonitor.propernouns import PROPER_NOUN_LEMMAS, repair_sentences
from parlamonitor.stopwords import hungarian_stopwords
from parlamonitor.topics import HU_CONNECTOR_WORDS, build_phrases

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "parentheticals"

DELIMITER = "#"
"""What fused n-grams are joined with, as the task specifies."""

CHECKPOINT = 2000
"""Lines analysed between flushes of the lemma cache."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    parser.add_argument(
        "--cycles", type=int, nargs="+", default=list(CYCLES), help="cycles to analyse"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="first N lines of each cycle only"
    )
    parser.add_argument("--batch-lines", type=int, default=250)
    parser.add_argument("--batch-words", type=int, default=3000)
    parser.add_argument(
        "--min-count", type=int, default=5, help="gensim Phrases min_count"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="NPMI score threshold"
    )
    parser.add_argument("--scoring", default="npmi", choices=["npmi", "default"])
    parser.add_argument("--delimiter", default=DELIMITER)
    parser.add_argument("--no-trigrams", action="store_true", help="bigram pass only")
    parser.add_argument(
        "--no-connector-words",
        action="store_true",
        help="allow phrases to begin or end with an article or conjunction",
    )
    parser.add_argument(
        "--no-proper-noun-repair",
        action="store_true",
        help="leave emMorph's party-name analyses as they are",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# emtsv analysis, cached by line
# --------------------------------------------------------------------------


def load_cache(path: Path, modules: str) -> dict[str, list[list[list[str]]]]:
    """Read the lemma cache, keeping only entries built by the same chain."""
    if not path.is_file():
        return {}
    cache: dict[str, list[list[list[str]]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in handle:
            if not row.strip():
                continue
            record = json.loads(row)
            if record.get("modules") != modules:
                continue
            cache[record["line"]] = record["sentences"]
    return cache


def analyse_missing(
    lines: Sequence[str],
    cache: dict[str, list[list[list[str]]]],
    cache_path: Path,
    args: argparse.Namespace,
) -> None:
    """Analyse the lines absent from the cache and append them to it."""
    missing = [line for line in lines if line not in cache]
    if not missing:
        print(f"  cache hit for all {len(lines):,} distinct lines")
        return

    print(f"  {len(missing):,} of {len(lines):,} distinct lines need emtsv")
    started = time.time()
    session = requests.Session()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Flushed every CHECKPOINT lines rather than once at the end: a run over
    # 64,052 lines takes half an hour, and a failure at line 60,000 should cost
    # the last chunk, not the whole thing. A resumed run reads the cache back
    # and asks emtsv only for what is still missing.
    done = 0
    for start in range(0, len(missing), CHECKPOINT):
        chunk = missing[start : start + CHECKPOINT]
        analysed = analyse_lines(
            chunk,
            base_url=args.base_url,
            modules=args.modules,
            batch_lines=args.batch_lines,
            batch_words=args.batch_words,
            session=session,
        )
        with cache_path.open("a", encoding="utf-8") as handle:
            for line, sentences in zip(chunk, analysed, strict=True):
                serialised = [
                    [[t.form, t.lemma, t.xpostag] for t in sentence]
                    for sentence in sentences
                ]
                cache[line] = serialised
                handle.write(
                    json.dumps(
                        {
                            "line": line,
                            "modules": args.modules,
                            "sentences": serialised,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        done += len(chunk)
        rate = done / max(time.time() - started, 1e-9)
        print(f"    {done:,}/{len(missing):,} lines ({rate:.0f}/s)", flush=True)


# --------------------------------------------------------------------------
# Token streams
# --------------------------------------------------------------------------


def is_punctuation(xpostag: str) -> bool:
    """Report whether a tag is punctuation and carries no word."""
    category, _ = parse_xpostag(xpostag)
    return category is None and "Punct" in xpostag


def line_sentences(
    serialised: list[list[list[str]]], *, repair: bool
) -> tuple[list[list[str]], int, int]:
    """Turn one cached line into lemma sentences.

    Returns the sentences, how many tokens the proper-noun repair touched, and
    how many punctuation tokens were dropped.
    """
    sentences = [
        [Token(form=f, lemma=lemma, xpostag=tag) for f, lemma, tag in sentence]
        for sentence in serialised
    ]
    n_repaired = 0
    if repair:
        sentences, n_repaired = repair_sentences(sentences)

    out: list[list[str]] = []
    n_punct = 0
    for sentence in sentences:
        kept = []
        for token in sentence:
            if is_punctuation(token.xpostag) or not token.lemma.strip():
                n_punct += 1
                continue
            kept.append(token.lemma)
        if kept:
            out.append(kept)
    return out, n_repaired, n_punct


def tag_index(
    cache: dict[str, list[list[list[str]]]], lines: set[str]
) -> dict[str, str]:
    """Map each lemma to the emMorph main category it most often carried."""
    seen: dict[str, Counter] = {}
    for line in lines:
        for sentence in cache.get(line, []):
            for _, lemma, xpostag in sentence:
                if is_punctuation(xpostag) or not lemma.strip():
                    continue
                category, _ = parse_xpostag(xpostag)
                seen.setdefault(lemma, Counter())[category or "?"] += 1
    return {lemma: counts.most_common(1)[0][0] for lemma, counts in seen.items()}


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def annotate(
    frame: pd.DataFrame,
    *,
    delimiter: str,
    categories: dict[str, str],
    stopwords: frozenset[str],
) -> pd.DataFrame:
    """Add the n-gram order and the part-of-speech / stopword flags.

    For a fused n-gram the flags are derived from its components: ``is_content``
    is true only if **every** component is a content word, ``is_stopword`` only
    if every component is a stopword, and ``pos_category`` is the components'
    categories joined by the same delimiter.
    """
    tokens = frame["token"]
    parts = tokens.map(lambda t: t.split(delimiter))
    frame = frame.copy()
    frame["ngram_order"] = tokens.map(lambda t: ngram_order(t, delimiter))
    frame["pos_category"] = parts.map(
        lambda ps: delimiter.join(categories.get(p, "?") for p in ps)
    )
    frame["is_content"] = parts.map(
        lambda ps: all(is_content_word(f"[/{categories.get(p, '?')}]") for p in ps)
    )
    frame["is_stopword"] = parts.map(lambda ps: all(p in stopwords for p in ps))
    return frame


COLUMN_ORDER = [
    "cycle",
    "token",
    "ngram_order",
    "pos_category",
    "is_content",
    "is_stopword",
    "raw",
    "relative_per_million",
    "proportion",
    "doc_count",
    "doc_proportion",
    "raw_collapsed",
    "relative_per_million_collapsed",
    "doc_count_collapsed",
    "cycle_total_tokens",
    "cycle_total_docs",
    "cycle_total_tokens_collapsed",
    "cycle_total_docs_collapsed",
]


def build_cycle_frame(
    label: str,
    primary: tuple[Counter, Counter, int],
    collapsed: tuple[Counter, Counter, int],
) -> pd.DataFrame:
    """Merge the as-is and collapsed counts for one cycle into one frame."""
    frame = frequency_frame(primary[0], primary[1], n_documents=primary[2], cycle=label)
    other = frequency_frame(
        collapsed[0], collapsed[1], n_documents=collapsed[2], cycle=label
    )
    other = other[
        [
            "token",
            "raw",
            "relative_per_million",
            "doc_count",
            "cycle_total_tokens",
            "cycle_total_docs",
        ]
    ].rename(
        columns={
            "raw": "raw_collapsed",
            "relative_per_million": "relative_per_million_collapsed",
            "doc_count": "doc_count_collapsed",
            "cycle_total_tokens": "cycle_total_tokens_collapsed",
            "cycle_total_docs": "cycle_total_docs_collapsed",
        }
    )
    merged = frame.merge(other, on="token", how="left")
    # A token can only be missing from the collapsed view if every one of its
    # occurrences was on a repeated line; that is a real zero, not unknown.
    merged["raw_collapsed"] = merged["raw_collapsed"].fillna(0).astype(int)
    merged["doc_count_collapsed"] = merged["doc_count_collapsed"].fillna(0).astype(int)
    merged["relative_per_million_collapsed"] = merged[
        "relative_per_million_collapsed"
    ].fillna(0.0)
    merged["cycle_total_tokens_collapsed"] = collapsed[0].total()
    merged["cycle_total_docs_collapsed"] = collapsed[2]
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "lemma_cache.jsonl"
    if args.refresh_cache and cache_path.exists():
        cache_path.unlink()

    # 1. Read the raw files.
    print("Reading parentheticals ...")
    corpus: dict[int, list[str]] = {}
    stats = []
    soft_hyphens = {}
    for cycle in args.cycles:
        lines, soft_hyphens[cycle] = load_parentheticals(
            cycle, args.data_dir, limit=args.limit
        )
        path = parentheticals_path(cycle, args.data_dir)
        corpus[cycle] = lines
        stats.append(line_stats(cycle, lines, path, sha256=file_sha256(path)))
        print(
            f"  cycle {cycle}: {stats[-1].n_lines:,} lines, "
            f"{stats[-1].n_words:,} words, {stats[-1].n_distinct:,} distinct, "
            f"{stats[-1].repeat_fraction:.1%} consecutive repeats, "
            f"{soft_hyphens[cycle]:,} soft hyphens stripped"
        )

    distinct = sorted({line for lines in corpus.values() for line in lines})
    print(f"  {len(distinct):,} distinct lines corpus-wide")

    # 2. Lemmatise the distinct lines, cached.
    print("Lemmatising with emtsv ...")
    cache = load_cache(cache_path, args.modules)
    analyse_missing(distinct, cache, cache_path, args)

    # 3. Repair party names, drop punctuation.
    repair = not args.no_proper_noun_repair
    sentences_by_line: dict[str, list[list[str]]] = {}
    repairs_by_line: dict[str, int] = {}
    punct_by_line: dict[str, int] = {}
    for line in distinct:
        sents, n_repaired, n_punct = line_sentences(cache[line], repair=repair)
        sentences_by_line[line] = sents
        repairs_by_line[line] = n_repaired
        punct_by_line[line] = n_punct

    # A lemma that already contains the delimiter is indistinguishable from a
    # fused n-gram afterwards. The corpus has two such lines -- hashtags on
    # protest badges, "#I stand with CEU" -- so this is reported, not fatal.
    collisions = sum(
        1
        for line in distinct
        for sentence in sentences_by_line[line]
        for lemma in sentence
        if args.delimiter in lemma
    )
    if collisions:
        print(
            f"  warning: {collisions} lemma(s) already contain "
            f"{args.delimiter!r} and cannot be told from a fused n-gram"
        )

    n_repairs = sum(
        repairs_by_line[line] for lines in corpus.values() for line in lines
    )
    n_punct = sum(punct_by_line[line] for lines in corpus.values() for line in lines)
    print(f"  proper-noun repairs: {n_repairs:,} tokens")
    print(f"  punctuation dropped: {n_punct:,} tokens")

    # 4. Fit one phrase model on all cycles pooled, in the as-is view.
    print(
        f"Fitting phrase model (scoring={args.scoring}, "
        f"threshold={args.threshold}, min_count={args.min_count}) ..."
    )
    pooled_sentences = [
        sentence
        for lines in corpus.values()
        for line in lines
        for sentence in sentences_by_line[line]
    ]
    print(f"  {len(pooled_sentences):,} sentences")

    connectors = frozenset() if args.no_connector_words else HU_CONNECTOR_WORDS
    bigrams = build_phrases(
        pooled_sentences,
        min_count=args.min_count,
        threshold=args.threshold,
        scoring=args.scoring,
        delimiter=args.delimiter,
        connector_words=connectors,
    )
    models = [bigrams]
    fused = [bigrams[sentence] for sentence in pooled_sentences]
    print(f"  bigrams detected: {len(bigrams.phrasegrams):,}")
    if not args.no_trigrams:
        trigrams = build_phrases(
            fused,
            min_count=args.min_count,
            threshold=args.threshold,
            scoring=args.scoring,
            delimiter=args.delimiter,
            connector_words=connectors,
        )
        models.append(trigrams)
        print(f"  higher-order phrases detected: {len(trigrams.phrasegrams):,}")

    with (output_dir / "phrases.pkl").open("wb") as handle:
        pickle.dump(models, handle)

    def apply_phrases(sentence: list[str]) -> list[str]:
        for model in models:
            sentence = model[sentence]
        return sentence

    fused_by_line = {
        line: [token for sentence in sents for token in apply_phrases(sentence)]
        for line, sents in sentences_by_line.items()
    }

    # 5. Count, per cycle and pooled, in both views.
    print("Counting ...")
    frames = []
    pooled_primary: tuple[Counter, Counter, int] = (Counter(), Counter(), 0)
    pooled_collapsed: tuple[Counter, Counter, int] = (Counter(), Counter(), 0)
    per_cycle_totals = {}
    for cycle in args.cycles:
        lines = corpus[cycle]
        primary = count_tokens(fused_by_line[line] for line in lines)
        collapsed = count_tokens(
            fused_by_line[line] for line in collapse_consecutive(lines)
        )
        frames.append(build_cycle_frame(str(cycle), primary, collapsed))
        per_cycle_totals[cycle] = {
            "tokens": primary[0].total(),
            "documents": primary[2],
            "types": len(primary[0]),
            "tokens_collapsed": collapsed[0].total(),
            "documents_collapsed": collapsed[2],
            "types_collapsed": len(collapsed[0]),
        }
        for accumulator, counted in (
            (pooled_primary, primary),
            (pooled_collapsed, collapsed),
        ):
            accumulator[0].update(counted[0])
            accumulator[1].update(counted[1])
        pooled_primary = (
            pooled_primary[0],
            pooled_primary[1],
            pooled_primary[2] + primary[2],
        )
        pooled_collapsed = (
            pooled_collapsed[0],
            pooled_collapsed[1],
            pooled_collapsed[2] + collapsed[2],
        )
        print(
            f"  cycle {cycle}: {primary[0].total():,} tokens, {len(primary[0]):,} types"
        )

    frames.append(build_cycle_frame(AGGREGATE_LABEL, pooled_primary, pooled_collapsed))
    print(
        f"  {AGGREGATE_LABEL}: {pooled_primary[0].total():,} tokens, "
        f"{len(pooled_primary[0]):,} types"
    )

    combined = pd.concat(frames, ignore_index=True)
    categories = tag_index(cache, set(distinct))
    combined = annotate(
        combined,
        delimiter=args.delimiter,
        categories=categories,
        stopwords=hungarian_stopwords(),
    )
    combined = combined[COLUMN_ORDER]

    # 6. Verification, asserted rather than assumed.
    checks = verify(combined, per_cycle_totals, args.cycles)

    # 7. Write everything out.
    combined.to_csv(output_dir / "token_frequencies.csv", index=False, encoding="utf-8")
    write_wide(combined, args.cycles, output_dir)
    unfused_counts: Counter = Counter()
    for lines in corpus.values():
        for line in lines:
            for sentence in sentences_by_line[line]:
                unfused_counts.update(sentence)
    write_ngrams(combined, models, args, output_dir, unfused_counts)
    write_line_stats(stats, output_dir)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "command": " ".join(sys.argv),
        "sources": [
            {
                "cycle": s.cycle,
                "path": str(s.path),
                "sha256": s.sha256,
                "n_lines": s.n_lines,
                "n_words": s.n_words,
                "n_distinct_lines": s.n_distinct,
                "n_consecutive_repeats": s.n_consecutive_repeats,
                "repeat_fraction": round(s.repeat_fraction, 4),
            }
            for s in stats
        ],
        "parameters": {
            "emtsv_modules": args.modules,
            "emtsv_image": "mtaril/emtsv",
            "limit": args.limit,
            "phrase_scoring": args.scoring,
            "phrase_threshold": args.threshold,
            "phrase_min_count": args.min_count,
            "phrase_delimiter": args.delimiter,
            "phrase_passes": len(models),
            "phrase_connector_words": sorted(connectors),
            "phrase_fit_view": "as-is (primary), all cycles pooled",
            "punctuation": "dropped before phrase detection and counting",
            "proper_noun_repair": repair,
            "proper_noun_entries": sorted(PROPER_NOUN_LEMMAS),
            "case": "lemmas keep the case emtsv produced",
            "normalisation": NORMALISATION_VERSION,
        },
        "counts": {
            "distinct_lines_analysed": len(distinct),
            "soft_hyphens_stripped": sum(soft_hyphens.values()),
            "delimiter_collisions": collisions,
            "proper_noun_repairs": n_repairs,
            "punctuation_tokens_dropped": n_punct,
            "bigrams_detected": len(models[0].phrasegrams),
            "higher_order_detected": (
                len(models[1].phrasegrams) if len(models) > 1 else 0
            ),
            "per_cycle": per_cycle_totals,
        },
        "verification": checks,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {output_dir}")
    for name, ok in checks.items():
        print(f"  check {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(checks.values()) else 1


def verify(
    combined: pd.DataFrame, totals: dict[int, dict[str, int]], cycles: Sequence[int]
) -> dict[str, bool]:
    """Check the invariants that would catch a mis-built table."""
    checks: dict[str, bool] = {}
    for cycle in cycles:
        rows = combined[combined["cycle"] == str(cycle)]
        checks[f"cycle{cycle}_raw_sums_to_total"] = bool(
            rows["raw"].sum() == totals[cycle]["tokens"]
        )
        checks[f"cycle{cycle}_proportion_sums_to_one"] = bool(
            abs(rows["proportion"].sum() - 1.0) < 1e-9
        )
    aggregate = combined[combined["cycle"] == AGGREGATE_LABEL].set_index("token")["raw"]
    per_cycle = (
        combined[combined["cycle"] != AGGREGATE_LABEL].groupby("token")["raw"].sum()
    )
    checks["aggregate_equals_sum_of_cycles"] = bool(
        aggregate.sort_index().equals(per_cycle.sort_index())
    )
    return checks


def write_wide(combined: pd.DataFrame, cycles: Sequence[int], out: Path) -> None:
    """Pivot to one row per token, with a column pair per cycle."""
    wide = combined[combined["cycle"] == AGGREGATE_LABEL][
        ["token", "ngram_order", "pos_category", "is_content", "is_stopword"]
    ].copy()
    for label in [str(c) for c in cycles] + [AGGREGATE_LABEL]:
        rows = combined[combined["cycle"] == label].set_index("token")
        suffix = f"c{label}" if label != AGGREGATE_LABEL else "all"
        wide[f"raw_{suffix}"] = wide["token"].map(rows["raw"]).fillna(0).astype(int)
        wide[f"rel_{suffix}"] = (
            wide["token"].map(rows["relative_per_million"]).fillna(0.0)
        )
    wide = wide.sort_values(["raw_all", "token"], ascending=[False, True])
    wide.to_csv(out / "token_frequencies_wide.csv", index=False, encoding="utf-8")


def write_ngrams(
    combined: pd.DataFrame,
    models: list,
    args: argparse.Namespace,
    out: Path,
    unfused_counts: Counter,
) -> None:
    """Write every fused phrase with its score and its per-cycle counts."""
    scores: dict[str, float] = {}
    passes: dict[str, int] = {}
    for pass_number, model in enumerate(models, start=1):
        for phrase, score in model.phrasegrams.items():
            key = phrase if isinstance(phrase, str) else args.delimiter.join(phrase)
            scores[key] = float(score)
            passes[key] = pass_number

    ngrams = combined[
        (combined["cycle"] == AGGREGATE_LABEL) & (combined["ngram_order"] > 1)
    ].copy()
    # A literal delimiter in the source text splits into empty components; that
    # is a collision, not a detected phrase, so it does not belong in this table.
    ngrams = ngrams[
        ngrams["token"].map(lambda t: all(part for part in t.split(args.delimiter)))
    ]
    ngrams["gensim_score"] = ngrams["token"].map(scores)
    # Pass 2 scores are computed over the corpus pass 1 already fused, so they
    # are comparable within a pass and not across the two, and can fall outside
    # NPMI's range. `npmi` below is recomputed against the unfused corpus and is
    # the column to compare phrases on.
    ngrams["detected_in_pass"] = ngrams["token"].map(passes)
    ngrams["scoring"] = args.scoring
    ngrams["components"] = ngrams["token"].map(
        lambda t: " ".join(t.split(args.delimiter))
    )

    n_unfused = unfused_counts.total()
    ngrams["component_counts"] = ngrams["token"].map(
        lambda t: ";".join(
            str(unfused_counts.get(p, 0)) for p in t.split(args.delimiter)
        )
    )
    ngrams["unfused_corpus_tokens"] = n_unfused

    def recompute(token: str, joint: int) -> float | None:
        parts = [unfused_counts.get(p, 0) for p in token.split(args.delimiter)]
        # A component fused away entirely by an earlier pass has no unfused
        # count of its own. Missing is null, not zero.
        if joint <= 0 or any(c <= 0 for c in parts):
            return None
        return npmi(joint, parts, n_unfused)

    ngrams["npmi"] = [
        recompute(token, joint)
        for token, joint in zip(ngrams["token"], ngrams["raw"], strict=True)
    ]
    columns = [
        "token",
        "components",
        "ngram_order",
        "pos_category",
        "scoring",
        "npmi",
        "gensim_score",
        "detected_in_pass",
        "component_counts",
        "unfused_corpus_tokens",
        "raw",
        "relative_per_million",
        "doc_count",
        "raw_collapsed",
    ]
    for cycle in args.cycles:
        rows = combined[combined["cycle"] == str(cycle)].set_index("token")
        ngrams[f"raw_c{cycle}"] = ngrams["token"].map(rows["raw"]).fillna(0).astype(int)
        columns.append(f"raw_c{cycle}")
    ngrams.sort_values(["raw", "token"], ascending=[False, True])[columns].to_csv(
        out / "significant_ngrams.csv", index=False, encoding="utf-8"
    )


def write_line_stats(stats: list, out: Path) -> None:
    """Write the per-cycle provenance table."""
    pd.DataFrame(
        [
            {
                "cycle": s.cycle,
                "path": str(s.path),
                "sha256": s.sha256,
                "n_lines": s.n_lines,
                "n_words": s.n_words,
                "n_distinct_lines": s.n_distinct,
                "n_consecutive_repeats": s.n_consecutive_repeats,
                "repeat_fraction": round(s.repeat_fraction, 4),
            }
            for s in stats
        ]
    ).to_csv(out / "line_stats.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
