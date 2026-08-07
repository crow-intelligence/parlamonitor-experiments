"""Does question time differ from ordinary debate, and askers from answerers?

The topic model answers this distributionally, via ``topics_by_role.csv``. This
script answers it lexically, with keyflux: log-likelihood keyness for
significance, log ratio for effect size, and rank-turbulence divergence with an
allotaxonograph for which exact words drove the difference.

It reads the lemma cache written by ``task2_bertopic.py`` rather than calling
emtsv again, so it runs in seconds and cannot disagree with the topic model
about what the corpus is.

Two contrasts:

* **question vs answer** -- the opposition MP asking against the minister
  replying, within the same exchange.
* **question time vs debate** -- the whole Q&A partition against the rest.

Usage::

    uv run python scripts/task2_compare.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from keyflux import Keyness, RankedList, allotaxonograph, rtd

from parlamonitor.loading import load_speeches
from parlamonitor.roles import QA_ROLES, discourse_role

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "derived" / "task2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--top", type=int, default=60, help="keywords per side")
    parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
        help="minimum count in each corpus for a type to be scored",
    )
    parser.add_argument("--alpha", type=float, default=1 / 3, help="RTD alpha")
    return parser.parse_args(argv)


def log(message):
    print(f"[compare] {message}", flush=True)


def load_lemmas(cache_path):
    """Map uid to its content-word lemmas, for successful analyses only."""
    lemmas = {}
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("ok") and record.get("lemmas"):
                lemmas[record["uid"]] = record["lemmas"]
    return lemmas


def write_keyness(focus, reference, *, labels, path, top, min_freq):
    """Score one contrast and write it, reproducibility record included."""
    analysis = Keyness(
        focus,
        reference,
        measure="log_likelihood",
        min_focus_freq=min_freq,
        min_reference_freq=min_freq,
        reference_id=labels[1],
    )
    table = analysis.keywords(top=top)
    frame = pd.DataFrame([row.__dict__ for row in table.rows])
    if not frame.empty:
        frame.insert(0, "focus", labels[0])
        frame.insert(1, "reference", labels[1])
    frame.to_csv(path, index=False, encoding="utf-8")

    repro = table.repro.__dict__
    log(
        f"{labels[0]} vs {labels[1]}: {len(table.positive)} keywords for "
        f"{labels[0]}, {len(table.negative)} for {labels[1]} "
        f"({repro['focus_total']:,} vs {repro['reference_total']:,} tokens)"
    )
    return {"labels": list(labels), "repro": repro, "n_rows": len(frame)}


def main(argv=None):
    args = parse_args(argv)
    cache_path = args.output_dir / "lemmatized.jsonl"
    if not cache_path.exists():
        log(f"no lemma cache at {cache_path}; run task2_bertopic.py first")
        return 1

    lemmas = load_lemmas(cache_path)
    speeches = load_speeches()
    log(f"{len(lemmas)} analysed speeches in the cache")

    counts: dict[str, Counter[str]] = {}
    speeches_per_group: Counter[str] = Counter()
    for speech in speeches:
        tokens = lemmas.get(speech["uid"])
        if not tokens:
            continue
        role = discourse_role(speech)
        groups = [role, "qa" if role in QA_ROLES else "debate"]
        for group in groups:
            counts.setdefault(group, Counter()).update(tokens)
            speeches_per_group[group] += 1

    log(
        "tokens per group: "
        + ", ".join(
            f"{g}={sum(c.values()):,} ({speeches_per_group[g]} speeches)"
            for g, c in sorted(counts.items())
        )
    )

    for required in ("question", "answer", "qa", "debate"):
        if required not in counts:
            log(f"group {required!r} is empty; nothing to compare")
            return 1

    summary = {
        "question_vs_answer": write_keyness(
            counts["question"],
            counts["answer"],
            labels=("question", "answer"),
            path=args.output_dir / "keyness_question_vs_answer.csv",
            top=args.top,
            min_freq=args.min_freq,
        ),
        "qa_vs_debate": write_keyness(
            counts["qa"],
            counts["debate"],
            labels=("qa", "debate"),
            path=args.output_dir / "keyness_qa_vs_debate.csv",
            top=args.top,
            min_freq=args.min_freq,
        ),
    }

    # Rank-turbulence divergence: one number for how far apart the two
    # vocabularies are, plus the allotaxonograph showing which words moved.
    for name, (left, right) in {
        "question_answer": ("question", "answer"),
        "qa_debate": ("qa", "debate"),
    }.items():
        list1 = RankedList.from_counts(counts[left], label=left)
        list2 = RankedList.from_counts(counts[right], label=right)
        result = rtd(list1, list2, alpha=args.alpha)
        summary[f"rtd_{name}"] = {
            "divergence": float(result.divergence),
            "alpha": float(result.alpha),
            "labels": list(result.labels),
        }
        log(f"RTD {left} vs {right}: {result.divergence:.4f} (alpha={result.alpha})")

        figure = allotaxonograph(list1, list2, alpha=args.alpha, labels=(left, right))
        figure_path = args.output_dir / f"allotaxonograph_{name}.png"
        figure.savefig(figure_path, dpi=150, bbox_inches="tight")
        log(f"wrote {figure_path.name}")

    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    log("wrote comparison_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
