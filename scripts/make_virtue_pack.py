"""Annotation pack for the Hungarian virtue lexicon.

The English virtue lexicon in ``personalityPolitics`` cannot be translated into
Hungarian word by word. Three reasons this corpus demonstrates:

* **Ritual address swamps the vocabulary.** ``tisztelt`` occurs 3,592 times and
  is almost entirely salutation — *Tisztelt Ház!* Any honour-word list that
  includes it makes every speaker maximally magnanimous for saying hello.
* **Hungarian cuts the concepts differently.** ``igazság`` means *both* truth
  and justice, where English has two words. That has to be a decision, not an
  accident.
* **Polysemy hides in the frequent words.** ``igaz`` is also a discourse
  particle, ``kiáll`` is also standing up, ``méltóság`` is a constitutional
  term rather than greatness of soul.

So candidates are proposed, grounded, and **verified by a human** — they are not
shipped on my say-so. This script writes the sheet: every candidate with its
corpus frequency and real KWIC lines, so a verifier reads actual usage rather
than judging a word in the abstract. Same pattern as the topic-name pack.

Emits ``virtue_candidates.csv`` (one row per candidate, an empty ``verdict``
column to fill in) and ``virtue_pack.md`` (the same, readable, with context).

Usage::

    uv run python scripts/make_virtue_pack.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from parlamonitor.lexicon import fold
from parlamonitor.virtues import POLES, load_virtues

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "annotation" / "virtues"
LEMMA_CACHE = ROOT / "data" / "derived" / "metrics" / "speech_lemmas.jsonl"

LEXICON = ROOT / "data" / "lexicons" / "virtues" / "hu_virtues.json"

# Candidates come from the shipped lexicon, not a second copy: a pack that can
# drift from what is actually scored is worse than no pack.


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lemma-cache", type=Path, default=LEMMA_CACHE)
    parser.add_argument("--lexicon", type=Path, default=LEXICON)
    parser.add_argument(
        "--examples", type=int, default=3, help="KWIC lines per candidate"
    )
    parser.add_argument("--window", type=int, default=7, help="words either side")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def build_index(path: Path, wanted: set[str], window: int):
    """Corpus frequency and KWIC lines for every candidate, in one pass."""
    freq: collections.Counter = collections.Counter()
    kwic: dict[str, list[str]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in handle:
            if not row.strip():
                continue
            record = json.loads(row)
            for sentence in record["sentences"]:
                forms = [tok[0] for tok in sentence]
                lemmas = [fold(tok[1]) for tok in sentence]
                for index, lemma in enumerate(lemmas):
                    freq[lemma] += 1
                    if lemma not in wanted or len(kwic[lemma]) >= 60:
                        continue
                    left = " ".join(forms[max(0, index - window) : index])
                    right = " ".join(forms[index + 1 : index + 1 + window])
                    kwic[lemma].append(f"{left} **{forms[index]}** {right}".strip())
    return freq, kwic


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    virtues = load_virtues(args.lexicon)
    wanted = {
        word
        for virtue in virtues.values()
        for pole in POLES
        for word in getattr(virtue, pole).single
    }
    freq, kwic = build_index(args.lemma_cache, wanted, args.window)

    rows = []
    for key, virtue in virtues.items():
        for pole in POLES:
            for word in sorted(getattr(virtue, pole).single):
                examples = kwic.get(word, [])
                sample = random.sample(examples, min(args.examples, len(examples)))
                rows.append(
                    {
                        "virtue": key,
                        "pole": pole,
                        "candidate": word,
                        "corpus_frequency": freq.get(word, 0),
                        "attested": freq.get(word, 0) > 0,
                        "verdict": "",  # accept / reject / move:<virtue>.<pole>
                        "verifier_note": "",
                        "example_1": sample[0] if len(sample) > 0 else "",
                        "example_2": sample[1] if len(sample) > 1 else "",
                        "example_3": sample[2] if len(sample) > 2 else "",
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "virtue_candidates.csv", index=False, encoding="utf-8")

    meta = json.loads(args.lexicon.read_text(encoding="utf-8"))["_meta"]
    lines = [
        "# Hungarian virtue lexicon — for verification",
        "",
        f"Generated {datetime.now(UTC).date()} from "
        f"`{args.lexicon.relative_to(ROOT)}` by `scripts/make_virtue_pack.py`.",
        "",
        "## What is being measured",
        "",
        f"**{meta['construct']}.**",
        "",
        meta["what_this_measures"],
        "",
        f"**Not measured:** {meta['what_this_does_not_measure']}",
        "",
        meta["poles"],
        "",
        "## How to verify",
        "",
        "Fill the `verdict` column in `virtue_candidates.csv` with `accept`,",
        "`reject`, or `move:<virtue>.<pole>`. Matching is on **lemmas**, so one",
        "entry covers every inflection: `bátorság` catches *bátorságot,",
        "bátorságunk, bátorsággal*.",
        "",
        "## Words already excluded, and why",
        "",
    ]
    for word, reason in meta.get("abandoned", {}).items():
        lines.append(f"- **`{word}`** — {reason}")
    lines.append("")

    for key, virtue in virtues.items():
        total = sum(
            freq.get(w, 0) for pole in POLES for w in getattr(virtue, pole).single
        )
        lines += [
            f"## {virtue.label} — {virtue.label_hu}"
            + (f" ({virtue.greek})" if virtue.greek else ""),
            "",
            f"*{virtue.gloss}*",
            "",
            f"{total} hits in the corpus.",
            "",
        ]
        if virtue.note:
            lines += [f"> **Note.** {virtue.note}", ""]
        for pole in POLES:
            words = sorted(getattr(virtue, pole).single, key=lambda w: -freq.get(w, 0))
            lines += [f"### {pole}", ""]
            for word in words:
                n = freq.get(word, 0)
                if not n:
                    lines.append(f"- **`{word}`** — not attested")
                    continue
                lines.append(f"- **`{word}`** ({n} hits)")
                for example in random.sample(
                    kwic[word], min(args.examples, len(kwic[word]))
                ):
                    lines.append(f"  - …{example}…")
            lines.append("")

    (out / "virtue_pack.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(frame):,} candidates across {len(virtues)} virtues")
    print(f"  attested: {int(frame['attested'].sum())}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
