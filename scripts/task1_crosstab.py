"""Cross-tabulate the supervised CAP labels against Task 2's unsupervised work.

Two questions the two tasks can only answer together:

* **Do the discovered topics line up with the policy schema?** BERTopic found
  28 clusters with no supervision; CAP is a fixed 21-topic codebook. Where one
  BERTopic topic maps cleanly onto one CAP label, both methods found the same
  thing independently. Where a topic scatters across CAP labels, they are
  cutting the corpus differently, and it is worth knowing which.
* **Does question time cover different policy areas than debate?** Task 2
  answered this in its own topic space; CAP answers it in a schema that is
  comparable across parliaments and across the agenda-setting literature.

Needs both tasks to have run. Joins on ``uid``, which is why the Task 1 output
carries one.

Usage::

    uv run python scripts/task1_crosstab.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK1 = ROOT / "data" / "derived" / "task1"
DEFAULT_TASK2 = ROOT / "data" / "derived" / "task2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task1-dir", type=Path, default=DEFAULT_TASK1)
    parser.add_argument("--task2-dir", type=Path, default=DEFAULT_TASK2)
    parser.add_argument(
        "--label-column",
        default="Predicted_CAP_Topic",
        help="which CAP column to tabulate; the chunked pass is the alternative",
    )
    return parser.parse_args(argv)


def log(message):
    print(f"[crosstab] {message}", flush=True)


def purity(table):
    """Share of each row that falls in its single largest column.

    A BERTopic topic whose speeches nearly all carry one CAP label has high
    purity: the two methods agree about that part of the corpus. Low purity
    means the unsupervised cluster cuts across the codebook.
    """
    totals = table.sum(axis=1)
    return (table.max(axis=1) / totals.where(totals > 0)).rename("purity")


def main(argv=None):
    args = parse_args(argv)
    cap_path = args.task1_dir / "parlacap_classifications.csv"
    topics_path = args.task2_dir / "bertopic_documents.csv"
    labels_path = args.task2_dir / "topic_labels.csv"

    for path in (cap_path, topics_path):
        if not path.exists():
            log(f"missing {path}; run both tasks first")
            return 1

    cap = pd.read_csv(cap_path, encoding="utf-8")
    topics = pd.read_csv(topics_path, encoding="utf-8")
    merged = cap.merge(topics[["uid", "topic", "topic_reduced"]], on="uid", how="inner")
    log(
        f"joined {len(merged)} speeches "
        f"({len(cap)} classified, {len(topics)} in the topic model)"
    )
    if len(merged) < len(topics):
        log(f"  {len(topics) - len(merged)} topic-model speeches have no CAP label")

    label = args.label_column
    args.task1_dir.mkdir(parents=True, exist_ok=True)

    # --- CAP by discourse role --------------------------------------------
    by_role = pd.crosstab(merged[label], merged.discourse_role)
    by_role["total"] = by_role.sum(axis=1)
    by_role = by_role.sort_values("total", ascending=False)
    by_role.to_csv(args.task1_dir / "cap_by_discourse_role.csv", encoding="utf-8")

    qa_mask = merged.discourse_role != "debate"
    shares = pd.DataFrame(
        {
            "qa": merged[qa_mask][label].value_counts(normalize=True),
            "debate": merged[~qa_mask][label].value_counts(normalize=True),
        }
    ).fillna(0.0)
    shares["difference"] = shares.qa - shares.debate
    shares["n"] = merged[label].value_counts()
    shares.sort_values("difference", ascending=False).to_csv(
        args.task1_dir / "cap_qa_vs_debate_share.csv", encoding="utf-8"
    )

    log("CAP areas most over-represented in question time:")
    for name, row in shares.nlargest(5, "difference").iterrows():
        log(
            f"  {name:24s} qa {row.qa:6.1%} vs debate {row.debate:6.1%} (n={row.n:.0f})"
        )
    log("...and in debate:")
    for name, row in shares.nsmallest(5, "difference").iterrows():
        log(
            f"  {name:24s} qa {row.qa:6.1%} vs debate {row.debate:6.1%} (n={row.n:.0f})"
        )

    # --- CAP by BERTopic topic --------------------------------------------
    by_topic = pd.crosstab(merged.topic, merged[label])
    scores = purity(by_topic)
    dominant = by_topic.idxmax(axis=1).rename("dominant_cap")
    summary = pd.concat([by_topic.sum(axis=1).rename("n"), dominant, scores], axis=1)
    if labels_path.exists():
        names = pd.read_csv(labels_path, encoding="utf-8").set_index("topic")
        summary = summary.join(names[["label_en"]], how="left")
    summary = summary.sort_values("purity", ascending=False)
    summary.to_csv(args.task1_dir / "cap_by_bertopic_topic.csv", encoding="utf-8")
    by_topic.to_csv(args.task1_dir / "cap_bertopic_matrix.csv", encoding="utf-8")

    named = summary[summary.index >= 0]
    log(f"\nmedian purity of a BERTopic topic against CAP: {named.purity.median():.1%}")
    log("cleanest agreement between the two methods:")
    for topic, row in named.head(6).iterrows():
        name = row.get("label_en", "")
        log(f"  T{topic:<3} {row.purity:5.1%} -> {row.dominant_cap:22s} {name}")
    log("least:")
    for topic, row in named.tail(4).iterrows():
        name = row.get("label_en", "")
        log(f"  T{topic:<3} {row.purity:5.1%} -> {row.dominant_cap:22s} {name}")

    summary_json = {
        "label_column": label,
        "n_joined": int(len(merged)),
        "median_purity": float(named.purity.median()),
        "qa_over_represented": shares.nlargest(5, "difference").index.tolist(),
        "debate_over_represented": shares.nsmallest(5, "difference").index.tolist(),
    }
    (args.task1_dir / "crosstab_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\nwrote 5 files to {args.task1_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
