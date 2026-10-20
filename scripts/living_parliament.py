"""A sitting day as a conversation: per-day files for the "Élő parlament" tab.

For every cycle-43 sitting day, writes one JSON with each speech (when it
started, how long it ran, who spoke, their faction and side, the speech's
sentences) and each reaction placed on its sentence. Every reaction gets an
estimated time, the benches it came from, and whether it was the speaker's own
side responding or the other side answering back
(:mod:`parlamonitor.living`).

The days are separate files because all the transcripts together come to
about 6 MB; the dashboard loads one day at a time.

Usage::

    uv run python scripts/living_parliament.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from parlamonitor.applause import placed_reactions
from parlamonitor.emtsv import EmtsvError, analyse_lines
from parlamonitor.interactions import SpeakerRegistry
from parlamonitor.lexicon import fold
from parlamonitor.living import (
    LANES,
    bench,
    flow,
    hostile_words,
    lanes,
    load_hostile_words,
    reaction_seconds,
)
from parlamonitor.loading import DATA_RAW, load_speeches
from parlamonitor.parentheticals import file_sha256
from parlamonitor.reactions import GOVERNING_PARTIES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "living"
REACTION_LINES = ROOT / "data" / "derived" / "applause" / "reaction_lines.csv"
HOSTILE_WORDS = ROOT / "data" / "labels" / "hostile_heckle_words.json"
SEED_CACHE = ROOT / "data" / "derived" / "heckles" / "quote_lemma_cache.jsonl"
HECKLE_KINDS = {"heckling", "interjection"}
CYCLE = 43


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
    parser.add_argument("--hostile-words", type=Path, default=HOSTILE_WORDS)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    return parser.parse_args(argv)


def quote_lemmas(
    quotes: set[str], cache_path: Path, base_url: str
) -> dict[str, set[str]]:
    """Case-folded lemmas of each heckle quote, from a cache; emtsv for the rest.

    The cache starts from the heckle-content cache (the same quotes, mostly), so
    only quotes new to the in-speech view need the service.
    """
    cache: dict[str, list] = {}
    for path in (SEED_CACHE, cache_path):
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for raw in handle:
                    record = json.loads(raw)
                    cache[record["quote"]] = record["sentences"]
    todo = sorted(quotes - set(cache))
    if todo:
        with requests.Session() as session:
            analysed = analyse_lines(todo, base_url=base_url, session=session)
        with cache_path.open("a", encoding="utf-8") as handle:
            for quote, sentences in zip(todo, analysed, strict=True):
                serial = [[[t.form, t.lemma, t.xpostag] for t in s] for s in sentences]
                cache[quote] = serial
                handle.write(
                    json.dumps(
                        {"quote": quote, "sentences": serial}, ensure_ascii=False
                    )
                    + "\n"
                )
    return {q: {fold(t[1]) for s in cache[q] for t in s} for q in quotes}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    speeches = load_speeches(args.data_dir)
    registry = SpeakerRegistry(
        (r.get("speaker") or {} for r in speeches), governing=GOVERNING_PARTIES[CYCLE]
    )

    days: dict[str, list[dict]] = defaultdict(list)
    counted = Counter()
    for record in speeches:
        sentences = record.get("sentences") or []
        if not sentences:
            continue
        speaker = record.get("speaker") or {}
        side = registry.resolve(speaker.get("label") or "").side
        start, length = float(record["time_start_s"]), float(record["duration_s"])
        reactions = []
        for p in placed_reactions(record):
            counted["placed"] += 1
            if not p.aligned:
                counted["unaligned"] += 1
                continue
            who_side = registry.resolve(p.interjector).side if p.interjector else None
            where = bench(p, who_side, CYCLE)
            direction = flow(p, where, side)
            if p.cut is not None and p.cut not in record["text_clean"]:
                raise AssertionError(f"{record['uid']}: cut is not verbatim text")
            t = reaction_seconds(sentences, p.sentence_index, p.cut, start, length)
            if not start - 1e-6 <= t <= start + length + 1e-6:
                raise AssertionError(
                    f"{record['uid']}: reaction time outside its speech"
                )
            counted[f"flow_{direction}"] += 1
            reactions.append(
                {
                    "s": p.sentence_index,
                    "t": round(t, 1),
                    "kinds": sorted(p.kinds),
                    "w": p.weight,
                    "bench": where,
                    "flow": direction,
                    "closing": p.closing,
                    "cut": p.cut,
                    "span": p.span,
                    "quote": p.quote,
                    "who": p.interjector,
                }
            )
        agenda = record.get("agenda") or {}
        days[record["date"]].append(
            {
                "uid": record["uid"],
                "t0": round(start, 1),
                "dur": round(length, 1),
                "speaker_id": speaker.get("person_id"),
                "speaker": speaker.get("label"),
                "faction": speaker.get("faction"),
                "side": side,
                "type": record.get("speech_type"),
                "agenda": agenda.get("title"),
                "sentences": sentences,
                "reactions": reactions,
            }
        )

    # Hostile heckles and lanes need every quote's lemmas, in one batch.
    words = load_hostile_words(args.hostile_words)
    quotes = {
        r["quote"]
        for items in days.values()
        for sp in items
        for r in sp["reactions"]
        if r["quote"] and set(r["kinds"]) & HECKLE_KINDS
    }
    try:
        lemmas = quote_lemmas(quotes, out / "quote_lemma_cache.jsonl", args.base_url)
    except (EmtsvError, requests.ConnectionError) as error:
        print(
            f"emtsv is needed for uncached quotes and failed: {error}", file=sys.stderr
        )
        return 1
    for items in days.values():
        for sp in items:
            for r in sp["reactions"]:
                hits = (
                    hostile_words(lemmas[r["quote"]], words)
                    if r["quote"] and set(r["kinds"]) & HECKLE_KINDS
                    else []
                )
                r["hostile"] = hits
                r["lanes"] = lanes(frozenset(r["kinds"]), r["flow"], bool(hits))
                for lane in r["lanes"]:
                    counted[f"lane_{lane}"] += 1

    index = []
    generated = datetime.now(UTC).isoformat()
    for date, items in sorted(days.items()):
        items.sort(key=lambda s: s["t0"])
        n = sum(len(s["reactions"]) for s in items)
        busiest = max(items, key=lambda s: len(s["reactions"]))
        own = sum(r["flow"] == "own" for s in items for r in s["reactions"])
        other = sum(r["flow"] == "other" for s in items for r in s["reactions"])
        index.append(
            {
                "date": date,
                "file": f"{date}.json",
                "speeches": len(items),
                "reactions": n,
                "own": own,
                "other": other,
                "end_s": round(max(s["t0"] + s["dur"] for s in items), 1),
                "busiest": busiest["uid"],
                "lanes": {
                    lane: sum(lane in r["lanes"] for s in items for r in s["reactions"])
                    for lane, _ in LANES
                },
            }
        )
        (out / f"{date}.json").write_text(
            json.dumps(
                {"date": date, "speeches": items},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    # Reconcile with the applause-lines output: same placements, same days.
    if REACTION_LINES.is_file():
        lines = pd.read_csv(REACTION_LINES)
        expected = lines[lines["aligned"]].groupby("date").size().to_dict()
        got = {d["date"]: d["reactions"] for d in index}
        if expected != got:
            raise AssertionError(
                "per-day reaction counts differ from reaction_lines.csv"
            )
        reconciled = True
    else:
        reconciled = None

    params = {
        "lanes": [{"id": lane, "side": side} for lane, side in LANES],
        "hostile_words": {
            "path": _repo_relative(args.hostile_words),
            "sha256": file_sha256(args.hostile_words),
            "count": len(words),
        },
        "cycle": CYCLE,
        "time_estimate": (
            "speech start + duration x share of the speech's words said by the end of "
            "the reaction's sentence (or its cut)"
        ),
        "flow": (
            "own: the speaker's benches, or applause/laughter naming no bench; other: "
            "the other benches, or heckles/noise naming no bench; none: bell, closing "
            "ritual, both benches"
        ),
    }
    (out / "index.json").write_text(
        json.dumps(
            {"generated_at": generated, "params": params, "days": index},
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    speeches_path = Path(args.data_dir or DATA_RAW) / f"cycle{CYCLE}-speeches.jsonl"
    manifest = {
        "generated_at": generated,
        "command": "scripts/living_parliament.py",
        "sources": {
            "speeches": {
                "path": _repo_relative(speeches_path),
                "sha256": file_sha256(speeches_path),
            }
        },
        "params": params,
        "counts": dict(counted) | {"days": len(index)},
        "verification": {
            "reaction_times_inside_their_speech": True,
            "cuts_verbatim_in_text_clean": True,
            "per_day_counts_match_reaction_lines": reconciled,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    size = sum(f.stat().st_size for f in out.glob("20*.json")) / 1e6
    print(
        f"{len(index)} days, {counted['placed'] - counted['unaligned']:,} reactions "
        f"(own {counted['flow_own']:,}, other {counted['flow_other']:,}, "
        f"neither {counted['flow_None']:,}); {size:.1f} MB in day files"
    )
    if reconciled is None:
        print(
            "  (reaction_lines.csv not found; per-day counts not reconciled)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
