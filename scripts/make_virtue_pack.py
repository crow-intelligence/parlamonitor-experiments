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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "annotation" / "virtues"
LEMMA_CACHE = ROOT / "data" / "derived" / "metrics" / "speech_lemmas.jsonl"

# Six virtues: the four classical cardinals plus the two MacIntyrean
# practice-virtues, matching data/virtues/lexicon.json in personalityPolitics.
CANDIDATES: dict[str, list[str]] = {
    "prudence": [
        "bölcs",
        "bölcsesség",
        "okos",
        "okosság",
        "megfontolt",
        "megfontoltság",
        "megfontol",
        "belátás",
        "belát",
        "előrelátás",
        "előrelátó",
        "józan",
        "józanság",
        "körültekintő",
        "körültekintés",
        "mérlegel",
        "mérlegelés",
        "észszerű",
        "ésszerű",
        "átgondolt",
        "átgondol",
        "meggondolt",
        "meggondolatlan",
        "higgadt",
        "higgadtság",
        "tapasztalt",
        "szakszerű",
    ],
    "justice": [
        "igazságos",
        "igazságosság",
        "igazságtalan",
        "igazságtalanság",
        "méltányos",
        "méltányosság",
        "jogos",
        "jogosság",
        "jogszerű",
        "jogszerűség",
        "jogtalan",
        "egyenlő",
        "egyenlőség",
        "egyenlőtlenség",
        "pártatlan",
        "pártatlanság",
        "elfogulatlan",
        "elfogult",
        "tisztesség",
        "tisztességes",
        "tisztességtelen",
        "részrehajló",
        "részrehajlás",
        "arányos",
        "arányosság",
    ],
    "courage": [
        "bátor",
        "bátorság",
        "bátran",
        "merész",
        "merészség",
        "hős",
        "hősies",
        "hősiesség",
        "helytáll",
        "helytállás",
        "kiáll",
        "elszánt",
        "elszántság",
        "rettenthetetlen",
        "gyáva",
        "gyávaság",
        "megalkuvó",
        "megalkuvás",
        "kitart",
        "kitartás",
        "szilárd",
        "eltökélt",
        "eltökéltség",
        "felvállal",
    ],
    "temperance": [
        "mértékletes",
        "mértékletesség",
        "mérséklet",
        "mérsékelt",
        "önmérséklet",
        "önuralom",
        "önfegyelem",
        "fegyelem",
        "fegyelmezett",
        "türelem",
        "türelmes",
        "türelmetlen",
        "visszafogott",
        "visszafogottság",
        "szerény",
        "szerénység",
        "takarékos",
        "takarékosság",
        "arányérzék",
    ],
    "truthfulness": [
        "igaz",
        "igazmondás",
        "igazmondó",
        "őszinte",
        "őszinteség",
        "becsületes",
        "becsület",
        "becsületesség",
        "hiteles",
        "hitelesség",
        "hazug",
        "hazugság",
        "hazudik",
        "félrevezet",
        "félrevezető",
        "megtéveszt",
        "megtévesztő",
        "valótlan",
        "valótlanság",
        "nyílt",
        "egyenes",
        "átlátható",
        "átláthatóság",
        "titkol",
        "eltitkol",
        "elhallgat",
    ],
    "magnanimity": [
        "nagylelkű",
        "nagylelkűség",
        "nemeslelkű",
        "nemeslelkűség",
        "nemes",
        "nagyvonalú",
        "nagyvonalúság",
        "önzetlen",
        "önzetlenség",
        "önfeláldozó",
        "áldozatkész",
        "áldozatkészség",
        "méltóság",
        "méltó",
        "emelkedett",
        "bőkezű",
        "adakozó",
        "segítőkész",
        "szolidaritás",
        "szolidáris",
        "kisstílű",
        "kicsinyes",
    ],
}

# Candidates I would reject on the corpus evidence, with the reason. They are
# in the sheet anyway, pre-flagged: the verifier overrules me, not the reverse.
FLAGGED: dict[str, str] = {
    "igaz": (
        "also a discourse particle ('az igaz, hogy'); at 295 hits it would "
        "dominate truthfulness on its own"
    ),
    "méltóság": (
        "'emberi méltóság' is a constitutional term, not the speaker's "
        "greatness of soul"
    ),
    "méltó": "'méltó' is mostly 'worthy of' in a policy sense",
    "kiáll": "polysemous: 'kiáll a pulpitushoz' is physically standing up",
    "nemes": "often 'nemes cél', and a surname",
    "szilárd": "a given name, and a physical sense",
    "egyenes": "literal sense ('egyenes ág', 'egyenes arányban')",
    "átlátható": "institutional transparency, arguably not a personal virtue",
    "átláthatóság": "institutional transparency, arguably not a personal virtue",
}

# Hungarian collapses what English separates. Recorded so the verifier decides
# rather than inherits my guess.
NOTES = {
    "igazság": (
        "NOT proposed for either list. Hungarian 'igazság' means both truth "
        "and justice; assigning it to one would be arbitrary and to both would "
        "double-count. Its unambiguous derivatives are proposed instead."
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lemma-cache", type=Path, default=LEMMA_CACHE)
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

    wanted = {w for words in CANDIDATES.values() for w in words}
    freq, kwic = build_index(args.lemma_cache, wanted, args.window)

    rows = []
    for virtue, words in CANDIDATES.items():
        for word in words:
            examples = kwic.get(word, [])
            sample = random.sample(examples, min(args.examples, len(examples)))
            rows.append(
                {
                    "virtue": virtue,
                    "candidate": word,
                    "corpus_frequency": freq.get(word, 0),
                    "attested": freq.get(word, 0) > 0,
                    "proposer_flag": FLAGGED.get(word, ""),
                    "verdict": "",  # accept / reject / move:<virtue>
                    "verifier_note": "",
                    "example_1": sample[0] if len(sample) > 0 else "",
                    "example_2": sample[1] if len(sample) > 1 else "",
                    "example_3": sample[2] if len(sample) > 2 else "",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "virtue_candidates.csv", index=False, encoding="utf-8")

    lines = [
        "# Hungarian virtue lexicon — candidates for verification",
        "",
        f"Generated {datetime.now(UTC).date()} by `scripts/make_virtue_pack.py`.",
        "",
        "Six virtues: the four classical cardinals (Aristotle, *Nicomachean",
        "Ethics*) plus two MacIntyrean practice-virtues, matching the English",
        "lexicon in `personalityPolitics`.",
        "",
        "**Nothing here is decided.** Every candidate carries its corpus",
        "frequency and real usage from the cycle-43 speeches. Fill the",
        "`verdict` column in `virtue_candidates.csv` with `accept`, `reject`,",
        "or `move:<virtue>`. Rows I would reject are pre-flagged with a reason;",
        "the flag is a proposal, not a decision.",
        "",
        "Because matching is on **lemmas**, one entry covers every inflection:",
        "`bátorság` catches *bátorságot, bátorságunk, bátorsággal*. The",
        "Hungarian lists are therefore shorter than the English ones, not",
        "longer.",
        "",
        "## Known problems, for the verifier to rule on",
        "",
    ]
    for word, note in NOTES.items():
        lines += [f"**`{word}`** — {note}", ""]
    lines += [
        "The corpus also shows `tisztelt` 3,592 times, essentially all",
        "salutation (*Tisztelt Ház!*). It is deliberately not proposed for",
        "magnanimity. If honour-words are wanted there, the salutation must be",
        "stripped first — `parlamonitor.text.strip_salutation` exists for it.",
        "",
        "Strip the flagged terms from magnanimity and it retains roughly 60",
        "hits across 12 words, which may be too thin to score. That megalopsychia",
        "has little purchase in modern parliamentary Hungarian is itself a",
        "finding, and better reported than papered over with a weak list.",
        "",
    ]
    for virtue, words in CANDIDATES.items():
        attested = [w for w in words if freq.get(w, 0) > 0]
        total = sum(freq.get(w, 0) for w in words)
        lines += [
            f"## {virtue} — {len(attested)}/{len(words)} attested, {total} hits",
            "",
        ]
        for word in sorted(words, key=lambda w: -freq.get(w, 0)):
            n = freq.get(word, 0)
            if not n:
                lines.append(f"- **`{word}`** — not attested in the corpus")
                continue
            flag = f"  ⚠ {FLAGGED[word]}" if word in FLAGGED else ""
            lines.append(f"- **`{word}`** ({n} hits){flag}")
            for example in random.sample(
                kwic[word], min(args.examples, len(kwic[word]))
            ):
                lines.append(f"  - …{example}…")
        lines.append("")

    (out / "virtue_pack.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(frame):,} candidates across {len(CANDIDATES)} virtues")
    print(f"  attested: {int(frame['attested'].sum())}")
    print(f"  pre-flagged for rejection: {int((frame['proposer_flag'] != '').sum())}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
