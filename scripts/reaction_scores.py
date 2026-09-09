"""Who made the chamber laugh, clap, and shout -- and from which benches.

Three tables, from two sources:

``reaction_events.csv``, ``reaction_summary.csv``
    Every classified reaction across cycles 39-43, from the parentheticals
    files. No speaker attribution: those files record who *reacted*, never who
    provoked it.

``heckler_scores.csv``
    Who interjects, across all five cycles. This one needs no speeches export
    because the heckler names themselves are inside the parenthetical --
    ``Vadai Ágnes: Nem hallom!``

``mp_reaction_scores.csv``
    Cycle 43 only: which MPs' speeches drew laughter and applause, and from
    whose benches. Reactions are read from the inline ``( ... )`` spans of
    ``cycle43-speeches.jsonl``, which is the only export carrying a speaker.
    Cycles 39-42 have no speeches export, so this table cannot be built for
    them; that is a gap in the data, not a choice.

The attribution rule is stated plainly because it is the load-bearing
assumption: **a reaction is credited to whoever held the floor when it was
recorded.** That is right for applause and laughter, which respond to the
speaker. It is wrong for a heckle, which is produced by someone else -- so
heckles are attributed to their named interjector instead, and never to the
floor-holder.

Usage::

    uv run python scripts/reaction_scores.py
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from parlamonitor.loading import DATA_RAW
from parlamonitor.parentheticals import (
    AGGREGATE_LABEL,
    CYCLES,
    file_sha256,
    load_parentheticals,
    parentheticals_path,
)
from parlamonitor.reactions import (
    GOVERNING_PARTIES,
    Audience,
    Kind,
    events_in,
    side_of,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "reactions"

SCORED_KINDS = (Kind.LAUGHTER, Kind.APPLAUSE, Kind.HECKLING)
"""The reactions the MP scores are built on."""

DISORDER_KINDS = (Kind.NOISE, Kind.UPROAR, Kind.WHISTLING, Kind.BOOING, Kind.BELL)
"""Reactions that measure trouble rather than approval."""

# Spans of `text` that are not stage directions: the faction tag in a speaker
# attribution, "(Fidesz):", and the sitting clock, "(10.20)".
_NOT_A_REMARK = re.compile(
    r"^\s*(?:[A-ZÁÉÍÓÖŐÚÜŰ][\w\s]{0,14}|\d{1,2}[.:]\d{2}|\d+)\s*$"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cycles", type=int, nargs="+", default=list(CYCLES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--speeches-cycle", type=int, default=43, help="cycle with a speeches export"
    )
    parser.add_argument(
        "--min-speeches",
        type=int,
        default=5,
        help="MPs below this are kept but flagged; rate columns are noisy there",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Corpus-level reaction events
# --------------------------------------------------------------------------


def audience_label(event, cycle: int) -> str:
    """Reduce an event's audiences to one label, resolving parties to a side."""
    if Audience.GOVERNMENT in event.audiences:
        return "government"
    if Audience.OPPOSITION in event.audiences:
        return "opposition"
    if event.parties:
        sides = {side_of(party, cycle) for party in event.parties}
        if len(sides) == 1:
            return next(iter(sides)).value
        return "both"
    if Audience.HOUSE in event.audiences:
        return "house"
    return "unspecified"


def collect_events(corpus: dict[int, list[str]]) -> pd.DataFrame:
    """Classify every parenthetical in every cycle."""
    rows = []
    for cycle, lines in corpus.items():
        for line_no, line in enumerate(lines):
            for event in events_in(line):
                if event.kind is Kind.OTHER:
                    continue
                rows.append(
                    {
                        "cycle": cycle,
                        "line_no": line_no,
                        "kind": event.kind.value,
                        "kinds": ";".join(sorted(k.value for k in event.kinds)),
                        "intensity": event.intensity.value,
                        "weight": event.weight,
                        "audience": audience_label(event, cycle),
                        "parties": ";".join(sorted(event.parties)),
                        "speaker": event.speaker or "",
                        "quote": event.quote or "",
                        "text": event.text,
                    }
                )
    return pd.DataFrame(rows)


def summarise(events: pd.DataFrame, corpus: dict[int, list[str]]) -> pd.DataFrame:
    """Count each kind per cycle and audience, raw and per 1,000 parentheticals."""
    rows = []
    for cycle, lines in corpus.items():
        n_lines = len(lines)
        subset = events[events["cycle"] == cycle]
        for kind in Kind:
            if kind is Kind.OTHER:
                continue
            # `kinds` rather than `kind`: an event that is both a laugh and a
            # round of applause counts once toward each.
            matching = subset[
                subset["kinds"].str.split(";").map(lambda ks, k=kind.value: k in ks)
            ]
            if matching.empty:
                continue
            for audience, group in matching.groupby("audience"):
                rows.append(
                    {
                        "cycle": cycle,
                        "kind": kind.value,
                        "audience": audience,
                        "events": len(group),
                        "weighted": round(group["weight"].sum(), 2),
                        "per_1000_parentheticals": len(group) / n_lines * 1000,
                        "cycle_parentheticals": n_lines,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["cycle", "kind", "events"], ascending=[True, True, False], ignore_index=True
    )


def heckler_scores(events: pd.DataFrame) -> pd.DataFrame:
    """Rank the named interjectors, per cycle and pooled."""
    named = events[
        (events["kind"] == Kind.INTERJECTION.value) & (events["speaker"] != "")
    ]
    frames = []
    for label, subset in [
        (str(c), named[named["cycle"] == c]) for c in sorted(events["cycle"].unique())
    ] + [(AGGREGATE_LABEL, named)]:
        counts = subset["speaker"].value_counts()
        if counts.empty:
            continue
        frame = counts.rename_axis("speaker").reset_index(name="interjections")
        frame.insert(0, "cycle", label)
        frame["share_of_cycle"] = frame["interjections"] / counts.sum()
        frame["cycle_interjections"] = counts.sum()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Cycle-43 MP scores, from the speeches export
# --------------------------------------------------------------------------


def load_speech_reactions(path: Path, cycle: int) -> pd.DataFrame:
    """Read the inline parentheticals of a speeches export, with their speaker."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = json.loads(raw)
            speaker = record.get("speaker") or {}
            text = record.get("text") or ""
            for span in re.findall(r"\(([^()]*)\)", text):
                if not span.strip() or _NOT_A_REMARK.match(span):
                    continue
                for event in events_in(span):
                    if event.kind is Kind.OTHER:
                        continue
                    rows.append(
                        {
                            "uid": record.get("uid"),
                            "speaker_id": speaker.get("person_id"),
                            "speaker": speaker.get("label"),
                            "faction": speaker.get("faction"),
                            "is_mp": speaker.get("is_mp"),
                            "speech_type": record.get("speech_type"),
                            "n_words": record.get("n_words") or 0,
                            "duration_s": record.get("duration_s") or 0.0,
                            "kinds": frozenset(event.kinds),
                            "intensity": event.intensity.value,
                            "weight": event.weight,
                            "audience": audience_label(event, cycle),
                            "interjector": event.speaker or "",
                            "text": event.text,
                        }
                    )
    return pd.DataFrame(rows)


def speech_totals(path: Path) -> pd.DataFrame:
    """Per-speaker denominators: speeches, words, and minutes on the floor."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = json.loads(raw)
            speaker = record.get("speaker") or {}
            rows.append(
                {
                    "speaker_id": speaker.get("person_id"),
                    "speaker": speaker.get("label"),
                    "faction": speaker.get("faction"),
                    "office": speaker.get("office") or "",
                    "is_mp": speaker.get("is_mp"),
                    "n_words": record.get("n_words") or 0,
                    "duration_s": record.get("duration_s") or 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    # `office` is aggregated rather than grouped on: a minister who also speaks
    # as a backbencher would otherwise be split into two rows and every rate
    # computed against half a denominator.
    return (
        frame.groupby(["speaker_id", "speaker", "faction", "is_mp"], dropna=False)
        .agg(
            speeches=("n_words", "size"),
            words=("n_words", "sum"),
            seconds=("duration_s", "sum"),
            office=("office", "max"),
        )
        .reset_index()
    )


def mp_scores(
    reactions: pd.DataFrame, totals: pd.DataFrame, cycle: int, min_speeches: int
) -> pd.DataFrame:
    """Score each speaker on the reactions their speeches drew.

    Heckles are excluded from a speaker's own reaction counts -- they are
    produced by someone else -- and reported separately as ``heckles_received``.
    """
    frame = totals.copy()
    governing = GOVERNING_PARTIES[cycle]

    def own_side(row: pd.Series) -> tuple[str | None, str]:
        """Which bench a speaker sits on, and how we know."""
        faction = row["faction"]
        if isinstance(faction, str) and faction:
            side = "government" if faction in governing else "opposition"
            return side, "faction"
        # Cycle 43's cabinet includes ministers who hold no party card, so
        # faction is null for them. Holding an executive office puts a speaker
        # on the government bench regardless -- recorded as `side_source` so
        # the inference is visible rather than assumed.
        office = row["office"]
        if isinstance(office, str) and office.strip():
            return "government", "office"
        return None, "none"

    sides = frame.apply(own_side, axis=1)
    frame["side"] = [side for side, _ in sides]
    frame["side_source"] = [source for _, source in sides]

    for kind in (*SCORED_KINDS, *DISORDER_KINDS):
        subset = reactions[reactions["kinds"].map(lambda ks, k=kind: k in ks)]
        counts = subset.groupby("speaker_id").size()
        weighted = subset.groupby("speaker_id")["weight"].sum()
        column = "heckles_received" if kind is Kind.HECKLING else kind.value
        frame[column] = frame["speaker_id"].map(counts).fillna(0).astype(int)
        frame[f"{column}_weighted"] = (
            frame["speaker_id"].map(weighted).fillna(0.0).round(2)
        )

    # Which benches responded, relative to the speaker's own side.
    for kind in (Kind.LAUGHTER, Kind.APPLAUSE):
        subset = reactions[reactions["kinds"].map(lambda ks, k=kind: k in ks)]
        by_audience = (
            subset.groupby(["speaker_id", "audience"]).size().unstack(fill_value=0)
        )
        for side in ("government", "opposition", "unspecified"):
            if side in by_audience:
                frame[f"{kind.value}_from_{side}"] = (
                    frame["speaker_id"].map(by_audience[side]).fillna(0).astype(int)
                )
            else:
                frame[f"{kind.value}_from_{side}"] = 0
        # A speaker with no faction -- the 96 non-MP guests and officials -- has
        # no own side, so these stay 0 rather than being assigned one.
        other = {"government": "opposition", "opposition": "government"}
        own, opposite = [], []
        for _, row in frame.iterrows():
            side = row["side"] if isinstance(row["side"], str) else None
            own.append(row[f"{kind.value}_from_{side}"] if side else 0)
            opposite.append(row[f"{kind.value}_from_{other[side]}"] if side else 0)
        frame[f"{kind.value}_own_side"] = own
        frame[f"{kind.value}_other_side"] = opposite
        # 11% of reactions name no bench ("Derültség."), so own + other does
        # not reach the total. The remainder is carried, not hidden.
        frame[f"{kind.value}_unattributed"] = (
            frame[kind.value]
            - frame[f"{kind.value}_own_side"]
            - frame[f"{kind.value}_other_side"]
        )

    frame["minutes"] = (frame["seconds"] / 60).round(2)
    for column in ("laughter", "applause"):
        frame[f"{column}_per_speech"] = (frame[column] / frame["speeches"]).round(4)
        frame[f"{column}_per_1000_words"] = (
            (frame[column] / frame["words"].replace(0, pd.NA) * 1000)
            .astype(float)
            .round(3)
        )
        frame[f"{column}_per_minute"] = (
            (frame[column] / frame["minutes"].replace(0, pd.NA)).astype(float).round(4)
        )

    frame["below_min_speeches"] = frame["speeches"] < min_speeches
    return frame.sort_values(
        ["laughter", "applause"], ascending=False, ignore_index=True
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Reading parentheticals ...")
    corpus: dict[int, list[str]] = {}
    sources = []
    for cycle in args.cycles:
        lines, _ = load_parentheticals(cycle, args.data_dir, limit=args.limit)
        corpus[cycle] = lines
        path = parentheticals_path(cycle, args.data_dir)
        sources.append(
            {
                "cycle": cycle,
                "path": str(path),
                "sha256": file_sha256(path),
                "n_lines": len(lines),
            }
        )
        print(f"  cycle {cycle}: {len(lines):,} parentheticals")

    print("Classifying reactions ...")
    events = collect_events(corpus)
    print(f"  {len(events):,} reaction events")
    for kind, n in events["kind"].value_counts().items():
        print(f"    {kind:14s} {n:8,d}")

    events.to_csv(out / "reaction_events.csv", index=False, encoding="utf-8")
    summary = summarise(events, corpus)
    summary.to_csv(out / "reaction_summary.csv", index=False, encoding="utf-8")

    hecklers = heckler_scores(events)
    hecklers.to_csv(out / "heckler_scores.csv", index=False, encoding="utf-8")
    top = hecklers[hecklers["cycle"] == AGGREGATE_LABEL].head(5)
    print("  top interjectors, all cycles:")
    for _, row in top.iterrows():
        print(f"    {row['speaker']:26s} {row['interjections']:6,d}")

    # Cycle-43 MP scores.
    speeches_path = (DATA_RAW if args.data_dir is None else Path(args.data_dir)) / (
        f"cycle{args.speeches_cycle}-speeches.jsonl"
    )
    mp_frame = None
    if speeches_path.is_file():
        print(f"Scoring MPs from {speeches_path.name} ...")
        reactions = load_speech_reactions(speeches_path, args.speeches_cycle)
        totals = speech_totals(speeches_path)
        mp_frame = mp_scores(reactions, totals, args.speeches_cycle, args.min_speeches)
        mp_frame.to_csv(out / "mp_reaction_scores.csv", index=False, encoding="utf-8")
        print(
            f"  {len(reactions):,} attributed reactions over {len(totals):,} speakers"
        )
        print("  most laughter drawn:")
        for _, row in mp_frame.head(5).iterrows():
            print(
                f"    {str(row['speaker']):26s} {row['laughter']:4d} laughs, "
                f"{row['applause']:4d} applause ({row['faction']})"
            )
    else:
        print(f"No speeches export at {speeches_path}; skipping MP scores.")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "sources": sources,
        "speeches_source": str(speeches_path) if speeches_path.is_file() else None,
        "parameters": {
            "governing_parties": {
                str(c): sorted(p) for c, p in GOVERNING_PARTIES.items()
            },
            "scored_kinds": [k.value for k in SCORED_KINDS],
            "disorder_kinds": [k.value for k in DISORDER_KINDS],
            "min_speeches": args.min_speeches,
            "attribution": (
                "a reaction is credited to whoever held the floor; heckles are "
                "credited to their named interjector instead"
            ),
        },
        "counts": {
            "reaction_events": len(events),
            "by_kind": {k: int(v) for k, v in events["kind"].value_counts().items()},
            "named_interjectors": int(
                events[events["speaker"] != ""]["speaker"].nunique()
            ),
            "mp_scores_rows": 0 if mp_frame is None else len(mp_frame),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
