"""Join hand-authored topic names to the model output, with evidence attached.

The names in ``data/labels/task2_topic_names.json`` are a draft: someone read
the c-TF-IDF terms and the representative documents and wrote a name. That is
exactly the kind of content that has to be checkable, so this script grounds
every one of them in a **verbatim quote** -- the sentence from a member speech
that contains the most of the topic's own top terms. Verifying a label is then
a string match against ``data/raw/``, not a reread of the corpus.

``checked_by_human`` is written as ``false``. Nothing here is a finding until
that column says otherwise.

Usage::

    uv run python scripts/task2_label_topics.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

from parlamonitor.loading import load_speeches
from parlamonitor.text import strip_salutation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "derived" / "task2"
DEFAULT_NAMES = ROOT / "data" / "labels" / "task2_topic_names.json"

# Hungarian is agglutinative, so a top term rarely appears in its lemma form.
# Matching on a prefix catches the inflected surface form without a second
# round-trip through emtsv.
_STEM = 6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES)
    parser.add_argument("--candidates", type=int, default=8, help="docs to search")
    return parser.parse_args(argv)


def sentences(text):
    """Split on sentence-final punctuation; good enough to quote from."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def best_quote(texts, terms, *, max_chars=240):
    """Return the sentence matching the most topic terms, and how many.

    Args:
        texts: Candidate documents, most representative first.
        terms: The topic's top terms; underscores are split back into words.
        max_chars: Truncate the quote at this length. Defaults to 240.

    Returns:
        A ``(quote, n_terms_matched)`` pair. Falls back to the opening of the
        first document when nothing matches, with a count of zero, so the
        caller can see that the evidence is weak.
    """
    stems = {
        part[:_STEM].lower()
        for term in terms
        for part in term.split("_")
        if len(part) >= _STEM
    }
    best, best_hits = "", -1
    for text in texts:
        for sentence in sentences(text):
            lowered = sentence.lower()
            hits = sum(1 for stem in stems if stem in lowered)
            if hits > best_hits and 40 <= len(sentence) <= 400:
                best, best_hits = sentence, hits
    if best_hits <= 0:
        fallback = " ".join(texts[0].split())[:max_chars] if texts else ""
        return fallback, 0
    return " ".join(best.split())[:max_chars], best_hits


def main(argv=None):
    args = parse_args(argv)
    documents = pd.read_csv(
        args.output_dir / "bertopic_documents.csv", encoding="utf-8"
    )
    topics = pd.read_csv(args.output_dir / "bertopic_topics.csv", encoding="utf-8")
    names = json.loads(args.names.read_text(encoding="utf-8"))
    speeches = {s["uid"]: s for s in load_speeches()}

    rows = []
    for record in topics.itertuples():
        topic = int(record.Topic)
        terms = ast.literal_eval(record.Representation)
        members = documents[documents.topic == topic].nlargest(
            args.candidates, "probability"
        )
        texts = [
            strip_salutation(speeches[uid]["text_clean"])[0]
            for uid in members.uid
            if uid in speeches
        ]
        quote, hits = best_quote(texts, terms)
        evidence_uid = next(
            (
                uid
                for uid, text in zip(members.uid, texts, strict=False)
                if quote[:60] in " ".join(text.split())
            ),
            members.uid.iloc[0] if len(members) else "",
        )
        name = names.get(str(topic), {})
        qa = members_role_share(documents, topic)
        rows.append(
            {
                "topic": topic,
                "count": int(record.Count),
                "label_hu": name.get("hu", ""),
                "label_en": name.get("en", ""),
                "qa_share": round(qa, 4),
                "top_terms": ", ".join(terms),
                "evidence_uid": evidence_uid,
                "evidence_terms_matched": hits,
                "evidence_quote": quote,
                "comment": name.get("comment", ""),
                "checked_by_human": False,
            }
        )

    frame = pd.DataFrame(rows).sort_values("topic")
    unnamed = frame[frame.label_en == ""].topic.tolist()
    if unnamed:
        print(f"[label] WARNING: {len(unnamed)} topics have no name: {unnamed}")
    weak = frame[frame.evidence_terms_matched == 0].topic.tolist()
    if weak:
        print(f"[label] {len(weak)} topics have no term-matched quote: {weak}")

    path = args.output_dir / "topic_labels.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    print(f"[label] wrote {path} -- {len(frame)} topics, all checked_by_human=False")
    return 0


def members_role_share(documents, topic):
    """Share of a topic's speeches that come from question time."""
    members = documents[documents.topic == topic]
    if members.empty:
        return 0.0
    return float((members.discourse_role != "debate").mean())


if __name__ == "__main__":
    sys.exit(main())
