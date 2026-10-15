"""Who speaks like whom: content-word counts per speaker, faction and side.

The speech counterpart of ``scripts/heckle_content.py``. It reads the emtsv
analyses that ``scripts/speech_metrics.py`` cached for every cycle-43 speech,
keeps content lemmas (nouns, verbs, adjectives, adverbs; no pronouns) minus
the project stoplist, and counts them per entity:

* each **speaker** with at least ``--min-tokens`` content words,
* each **faction**, and Fidesz-KDNP together,
* the **government** and **opposition** sides.

It writes the counts, not the comparisons: any two entities can be compared,
which is far too many pairs to precompute, so the dashboard computes
rank-turbulence divergence in the browser (``dashboard/rtd.js``, checked
against keyflux by ``tests/test_dashboard_rtd.py``). The keyflux values for a
few reference pairs are written to the manifest so the two can be compared.

Cycle 43 only: it is the only cycle with a speeches export.

Usage::

    uv run python scripts/speech_vocabulary.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from keyflux import RankedList, rtd

from parlamonitor.emtsv import is_content_word
from parlamonitor.heckles import HECKLE_STOP_EXTRAS, content_lemmas
from parlamonitor.interactions import SpeakerRegistry
from parlamonitor.lexicon import fold
from parlamonitor.loading import DATA_RAW, load_speeches
from parlamonitor.parentheticals import file_sha256
from parlamonitor.reactions import GOVERNING_PARTIES
from parlamonitor.stopwords import hungarian_stopwords

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "vocabulary"
DEFAULT_LEMMAS = ROOT / "data" / "derived" / "metrics" / "speech_lemmas.jsonl"
CYCLE = 43
FIDESZ_KDNP = frozenset({"Fidesz", "KDNP"})
SPEECH_STOP_EXTRAS: tuple[str, ...] = ("dr.",)
"""Honorifics the shared stoplist does not have. ``dr.`` is a title, not a
word, like ``úr`` and ``asszony`` which the address-term list already drops;
in speeches it is frequent enough to rank among the top contributors."""

REFERENCE_PAIRS: tuple[tuple[str, str], ...] = (
    ("faction:TISZA", "faction:Fidesz"),
    ("side:government", "side:opposition"),
    ("faction:Fidesz", "faction:KDNP"),
    ("group:Fidesz-KDNP", "faction:Mi Hazánk"),
)
"""Pairs whose keyflux RTD goes into the manifest, as a cross-check for the
browser implementation."""


def _repo_relative(path):
    """Path as written in the repo, for provenance records that get published.

    `str(path)` on a resolved path carries the developer's home directory into
    every artefact built from it. Falls back to the basename when the input
    genuinely lives outside the checkout.
    """
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--lemmas", type=Path, default=DEFAULT_LEMMAS)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=300,
        help="content words a speaker needs to be offered for comparison",
    )
    parser.add_argument("--alpha", type=float, default=1 / 3)
    parser.add_argument(
        "--no-light-verbs",
        action="store_true",
        help="keep light verbs (van, mond, kér ...) in the counted vocabulary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not args.lemmas.is_file():
        print(
            f"No lemma cache at {_repo_relative(args.lemmas)}; "
            "run scripts/speech_metrics.py first.",
            file=sys.stderr,
        )
        return 1

    speeches = {record["uid"]: record for record in load_speeches(args.data_dir)}
    registry = SpeakerRegistry(
        (record.get("speaker") or {} for record in speeches.values()),
        governing=GOVERNING_PARTIES[CYCLE],
    )
    stop = hungarian_stopwords(
        light_verbs=not args.no_light_verbs,
        extra=(*HECKLE_STOP_EXTRAS, *SPEECH_STOP_EXTRAS),
    )

    print("Counting content lemmas per speaker ...")
    per_person: dict[str, Counter] = {}
    person_meta: dict[str, dict] = {}
    spellings: dict[str, Counter] = {}
    missing = 0
    with args.lemmas.open(encoding="utf-8") as handle:
        for raw in handle:
            record = json.loads(raw)
            speech = speeches.get(record["uid"])
            if speech is None:
                missing += 1
                continue
            speaker = speech.get("speaker") or {}
            pid = speaker.get("person_id")
            if not pid:
                missing += 1
                continue
            lemmas = content_lemmas(record["sentences"], stop)
            per_person.setdefault(pid, Counter()).update(lemmas)
            meta = person_meta.setdefault(
                pid,
                {
                    "name": speaker.get("label"),
                    "faction": speaker.get("faction") or None,
                    "side": registry.resolve(speaker.get("label") or "").side,
                    "speeches": 0,
                },
            )
            meta["speeches"] += 1
            for sentence in record["sentences"]:
                for _, lemma, tag in sentence:
                    if is_content_word(tag):
                        spellings.setdefault(fold(lemma), Counter())[lemma] += 1
    print(f"  {len(per_person):,} speakers; {missing} lemma records without a speaker")

    def pooled(members: list[str]) -> Counter:
        total: Counter = Counter()
        for pid in members:
            total.update(per_person[pid])
        return total

    groups: dict[str, tuple[str, str, list[str]]] = {
        "side:government": (
            "side",
            "government",
            [p for p, m in person_meta.items() if m["side"] == "government"],
        ),
        "side:opposition": (
            "side",
            "opposition",
            [p for p, m in person_meta.items() if m["side"] == "opposition"],
        ),
        "group:Fidesz-KDNP": (
            "group",
            "Fidesz-KDNP",
            [p for p, m in person_meta.items() if m["faction"] in FIDESZ_KDNP],
        ),
    }
    for faction in sorted({m["faction"] for m in person_meta.values() if m["faction"]}):
        groups[f"faction:{faction}"] = (
            "faction",
            faction,
            [p for p, m in person_meta.items() if m["faction"] == faction],
        )

    counts: dict[str, Counter] = {}
    entities = []
    for eid, (kind, name, members) in groups.items():
        counts[eid] = pooled(members)
        entities.append(
            {
                "id": eid,
                "kind": kind,
                "name": name,
                "members": len(members),
                "speeches": sum(person_meta[p]["speeches"] for p in members),
            }
        )
    below = 0
    for pid, meta in sorted(person_meta.items(), key=lambda kv: kv[1]["name"] or ""):
        if per_person[pid].total() < args.min_tokens:
            below += 1
            continue
        eid = f"person:{pid}"
        counts[eid] = per_person[pid]
        entities.append({"id": eid, "kind": "person", **meta})
    print(
        f"  {sum(e['kind'] == 'person' for e in entities)} speakers offered; "
        f"{below} below {args.min_tokens} content words"
    )

    # One shared vocabulary, each entity as parallel (index, count) arrays.
    vocab = sorted(set().union(*counts.values()))
    index = {t: i for i, t in enumerate(vocab)}
    # Display spelling only where it differs from the folded lemma ("" = same),
    # which is most words; it saves a third of a megabyte.
    shown = {
        t: min(spellings[t].items(), key=lambda kv: (-kv[1], kv[0]))[0]
        for t in vocab
        if t in spellings
    }
    labels = ["" if shown.get(t, t) == t else shown[t] for t in vocab]
    for entity in entities:
        cnt = counts[entity["id"]]
        pairs = sorted((index[t], n) for t, n in cnt.items())
        entity["tokens"] = int(cnt.total())
        entity["types"] = len(cnt)
        # Indices are sorted, so they travel as gaps: small numbers, short JSON.
        entity["gaps"] = [
            i - j
            for i, j in zip(
                [i for i, _ in pairs], [0] + [i for i, _ in pairs][:-1], strict=True
            )
        ]
        entity["n"] = [n for _, n in pairs]

    checks = {}
    for left, right in REFERENCE_PAIRS:
        if left in counts and right in counts:
            result = rtd(
                RankedList.from_counts(counts[left]),
                RankedList.from_counts(counts[right]),
                alpha=args.alpha,
            )
            checks[f"{left}|{right}"] = round(float(result.divergence), 10)
            print(f"  keyflux RTD {left} vs {right}: {result.divergence:.4f}")

    generated = datetime.now(UTC).isoformat()
    params = {
        "cycle": CYCLE,
        "content_words": "N, V, Adj, Adv; pronouns dropped",
        "stopwords": {
            "spacy_hu": True,
            "light_verbs": not args.no_light_verbs,
            "parliamentary": True,
            "extras": [*HECKLE_STOP_EXTRAS, *SPEECH_STOP_EXTRAS],
            "size": len(stop),
        },
        "min_tokens": args.min_tokens,
        "alpha": args.alpha,
        "side_rule": "faction in the governing parties, or holding executive office",
    }
    bundle = {
        "generated_at": generated,
        "source": "scripts/speech_vocabulary.py",
        "params": params,
        "encoding": (
            "vocab[i] is a case-folded lemma, labels[i] its display spelling or '' "
            "when identical; entity.gaps are successive differences of sorted vocab "
            "indices (first one absolute), entity.n the matching counts"
        ),
        "vocab": vocab,
        "labels": labels,
        "entities": entities,
        "reference_rtd": checks,
    }
    (out / "vocabulary.json").write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    speeches_path = Path(args.data_dir or DATA_RAW) / f"cycle{CYCLE}-speeches.jsonl"
    manifest = {
        "generated_at": generated,
        "command": "scripts/speech_vocabulary.py",
        "sources": {
            "speeches": {
                "path": _repo_relative(speeches_path),
                "sha256": file_sha256(speeches_path),
            },
            "lemmas": {
                "path": _repo_relative(args.lemmas),
                "sha256": file_sha256(args.lemmas),
            },
        },
        "params": params,
        "counts": {
            "speakers": len(per_person),
            "speakers_offered": sum(e["kind"] == "person" for e in entities),
            "speakers_below_threshold": below,
            "vocabulary": len(vocab),
            "lemma_records_without_speaker": missing,
        },
        "reference_rtd": checks,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    size = (out / "vocabulary.json").stat().st_size / 1e6
    print(f"\nWrote {_repo_relative(out)} (vocabulary.json {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
