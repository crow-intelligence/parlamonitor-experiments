"""Check that the saved topic model still matches the hand-authored names.

``data/labels/task2_topic_names.json`` is keyed by topic id, and topic ids are
positional. Change the stoplist, the seed or the corpus and HDBSCAN renumbers
everything, at which point every name in that file quietly describes a
different topic. Nothing about the file's appearance would change.

So the names record the fingerprint of the model they were written against,
and this script recomputes it from the saved model and compares. It is wired
into ``make ci``, so drift fails the build instead of surviving into a report.

Exit codes: 0 match, 1 mismatch or missing model.

Usage::

    uv run python scripts/verify_topic_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parlamonitor.topics import topic_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "task2_bertopic"
DEFAULT_NAMES = ROOT / "data" / "labels" / "task2_topic_names.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES)
    return parser.parse_args(argv)


def log(message):
    print(f"[verify] {message}", flush=True)


def fingerprint_of(model_dir):
    """Load the saved model and fingerprint its topic -> terms mapping."""
    from bertopic import BERTopic

    model = BERTopic.load(str(model_dir))
    info = model.get_topic_info()
    topics = {int(t): [term for term, _ in model.get_topic(int(t))] for t in info.Topic}
    return topic_fingerprint(topics), info


def main(argv=None):
    args = parse_args(argv)

    if not args.model_dir.exists():
        log(f"no saved model at {args.model_dir}")
        log("  run: uv run python scripts/task2_bertopic.py")
        return 1

    names = json.loads(args.names.read_text(encoding="utf-8"))
    expected = names.get("_about", {}).get("fingerprint")
    if not expected:
        log(f"{args.names.name} records no fingerprint; cannot verify")
        return 1

    actual, info = fingerprint_of(args.model_dir)
    named = {k for k in names if not k.startswith("_")}
    model_topics = {str(int(t)) for t in info.Topic}

    log(f"model:  {len(info)} topics, fingerprint {actual}")
    log(f"names:  {len(named)} entries, fingerprint {expected}")

    if actual != expected:
        log("MISMATCH — the saved model is not the one these names describe.")
        log("  Every label may now point at a different topic.")
        log("  Regenerate the names against this model before trusting them.")
        return 1

    missing = sorted(model_topics - named, key=int)
    extra = sorted(named - model_topics, key=int)
    if missing:
        log(f"WARNING: {len(missing)} topics have no name: {missing}")
    if extra:
        log(f"WARNING: {len(extra)} names refer to absent topics: {extra}")

    log("OK — fingerprints match; the names describe this model.")
    return 1 if (missing or extra) else 0


if __name__ == "__main__":
    sys.exit(main())
