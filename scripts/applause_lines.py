"""Applause lines: which sentences earn applause, laughter and heckling.

Puts every reaction recorded inline in the cycle-43 speeches back on the
sentence it followed (:mod:`parlamonitor.applause`), then asks what those
sentences have in common:

* **rates** -- lines per 100 sentences, per speaker, faction and speech type;
* **features** -- exclamation, question, numbers, "we", negation, naming the
  other side: how much more often each appears in a kind of line than in a
  sentence that drew no reaction, with both counts and a log-likelihood band;
* **position** -- where in a speech each kind of line falls;
* **vocabulary** -- content-lemma counts per kind of line and for the quiet
  sentences, for the dashboard to compare with rank-turbulence divergence;
* **best lines** -- each speaker's strongest applause, laughter and heckle
  lines, verbatim.

Every trigger is asserted to be a verbatim substring of the speech's
``text_clean``, so checking a line against the source is a string match.

Usage::

    uv run python scripts/applause_lines.py
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
from keyflux import RankedList, rtd
from keyflux.keyness.measures import log_likelihood, significance_band

from parlamonitor.applause import (
    LINE_KINDS,
    is_salutation,
    line_kinds,
    placed_reactions,
    sentence_features,
    side_relation,
    tokens_by_sentence,
    without_directions,
)
from parlamonitor.heckles import (
    HECKLE_STOP_EXTRAS,
    SPEECH_STOP_EXTRAS,
    content_lemmas,
    display_spellings,
)
from parlamonitor.interactions import SpeakerRegistry
from parlamonitor.loading import DATA_RAW, load_speeches
from parlamonitor.parentheticals import file_sha256
from parlamonitor.reactions import GOVERNING_PARTIES
from parlamonitor.stopwords import hungarian_stopwords

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "applause"
DEFAULT_LEMMAS = ROOT / "data" / "derived" / "metrics" / "speech_lemmas.jsonl"
CYCLE = 43
FEATURES = ("exclamation", "question", "number", "we", "negation", "names_opponent")
COMPARISONS = (
    ("applause", "quiet"),
    ("heckle", "quiet"),
    ("laughter_with", "laughter_at"),
)
"""(left, right) group pairs the dashboard compares; ``quiet`` is every
sentence that drew no reaction at all."""


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
        "--min-sentences",
        type=int,
        default=100,
        help="sentences a speaker needs for their rates to be shown unflagged",
    )
    parser.add_argument("--alpha", type=float, default=1 / 3)
    parser.add_argument(
        "--keep-salutations",
        action="store_true",
        help="keep 'Tisztelt Elnök Úr!'-type sentences in the quiet baseline",
    )
    parser.add_argument(
        "--best", type=int, default=5, help="applause lines per speaker"
    )
    parser.add_argument("--seed", type=int, default=0, help="for the hand-check sample")
    return parser.parse_args(argv)


def load_lemmas(path: Path) -> dict[str, list]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            record = json.loads(raw)
            out[record["uid"]] = record["sentences"]
    return out


def feature_table(sentences: pd.DataFrame) -> pd.DataFrame:
    """Each feature's rate in each kind of line against quiet sentences."""
    quiet = sentences[sentences["quiet"]]
    rows = []
    for kind in LINE_KINDS:
        lines = sentences[sentences[f"is_{kind}"]]
        for feature in FEATURES:
            a, n1 = int(lines[feature].sum()), len(lines)
            b, n2 = int(quiet[feature].sum()), len(quiet)
            rate1, rate2 = (a / n1 if n1 else None), (b / n2 if n2 else None)
            statistic = log_likelihood(a, b, n1, n2) if n1 and n2 else None
            rows.append(
                {
                    "line_kind": kind,
                    "feature": feature,
                    "lines_with": a,
                    "lines": n1,
                    "quiet_with": b,
                    "quiet": n2,
                    "rate_lines": rate1,
                    "rate_quiet": rate2,
                    "rate_ratio": (rate1 / rate2)
                    if rate1 is not None and rate2
                    else None,
                    "log_likelihood": statistic,
                    "significance": significance_band(statistic)
                    if statistic is not None
                    else None,
                }
            )
        rows.append(
            {
                "line_kind": kind,
                "feature": "median_words",
                "lines_with": None,
                "lines": len(lines),
                "quiet_with": None,
                "quiet": len(quiet),
                "rate_lines": float(lines["words"].median()) if len(lines) else None,
                "rate_quiet": float(quiet["words"].median()),
                "rate_ratio": None,
                "log_likelihood": None,
                "significance": None,
            }
        )
    return pd.DataFrame(rows)


def rate_table(
    sentences: pd.DataFrame, by: list[str], min_sentences: int
) -> pd.DataFrame:
    """Lines per 100 sentences for each group, counts beside the rates."""
    agg = {"sentences": ("uid", "size")}
    agg.update({kind: (f"is_{kind}", "sum") for kind in LINE_KINDS})
    frame = sentences.groupby(by, dropna=False).agg(**agg).reset_index()
    for kind in LINE_KINDS:
        frame[f"{kind}_per_100"] = (frame[kind] / frame["sentences"] * 100).round(3)
    frame["below_min_sentences"] = frame["sentences"] < min_sentences
    return frame


def gap_encode(counts: Counter, index: dict[str, int]) -> dict:
    pairs = sorted((index[t], n) for t, n in counts.items())
    idx = [i for i, _ in pairs]
    return {
        "gaps": [i - j for i, j in zip(idx, [0, *idx[:-1]], strict=True)],
        "n": [n for _, n in pairs],
        "tokens": int(counts.total()),
        "types": len(counts),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not args.lemmas.is_file():
        print(
            f"No lemma cache at {_repo_relative(args.lemmas)}; run speech_metrics.py.",
            file=sys.stderr,
        )
        return 1

    speeches = load_speeches(args.data_dir)
    lemmas = load_lemmas(args.lemmas)
    registry = SpeakerRegistry(
        (r.get("speaker") or {} for r in speeches), governing=GOVERNING_PARTIES[CYCLE]
    )
    stop = hungarian_stopwords(extra=(*HECKLE_STOP_EXTRAS, *SPEECH_STOP_EXTRAS))

    print("Placing reactions on sentences ...")
    event_rows, sentence_rows = [], []
    sentence_lemmas: dict[tuple[str, int], list[str]] = {}
    counted = Counter()
    for record in speeches:
        sentences = record.get("sentences") or []
        if not sentences:
            continue
        speaker = record.get("speaker") or {}
        side = registry.resolve(speaker.get("label") or "").side
        tokens = tokens_by_sentence(sentences, lemmas.get(record["uid"], []))
        placed = placed_reactions(record)
        followed: dict[int, set[str]] = defaultdict(set)
        any_reaction: set[int] = set()
        for p in placed:
            counted["placed"] += 1
            counted["aligned" if p.aligned else "unaligned"] += 1
            counted["closing"] += p.closing
            relation = side_relation(p, side, CYCLE)
            kinds = line_kinds(p, relation) if p.aligned else []
            if p.aligned:
                any_reaction.add(p.sentence_index)
                followed[p.sentence_index].update(kinds)
                trigger = p.cut if p.interrupted else sentences[p.sentence_index]
                if trigger not in record["text_clean"]:
                    raise AssertionError(
                        f"{record['uid']}: trigger is not verbatim text"
                    )
            else:
                trigger = None
            for kind in kinds:
                counted[f"line_{kind}"] += 1
            event_rows.append(
                {
                    "uid": record["uid"],
                    "date": record.get("date"),
                    "speaker_id": speaker.get("person_id"),
                    "speaker": speaker.get("label"),
                    "faction": speaker.get("faction"),
                    "side": side,
                    "speech_type": record.get("speech_type"),
                    "sentence_index": p.sentence_index,
                    "n_sentences": len(sentences),
                    "relative_position": p.relative_position,
                    "kinds": ";".join(sorted(p.kinds)),
                    "weight": p.weight,
                    "relation": relation,
                    "line_kinds": ";".join(kinds),
                    "closing": p.closing,
                    "interrupted": p.interrupted,
                    "aligned": p.aligned,
                    "trigger": trigger,
                    "previous": sentences[p.sentence_index - 1]
                    if p.aligned and p.sentence_index > 0
                    else None,
                    "quote": p.quote,
                    "span": p.span,
                    "context_after": " ".join(
                        record["text"][p.offset :].split()[: 12 + len(p.span.split())]
                    ),
                }
            )
        for i, sentence in enumerate(sentences):
            # The export leaves some directions inside sentences; measure the
            # speaker's words only, not a heckler's.
            clean, words = without_directions(sentence, tokens[i])
            counted["sentences_with_direction"] += clean != sentence
            features = sentence_features(words, clean, side)
            row = {
                "uid": record["uid"],
                "sentence_index": i,
                "speaker_id": speaker.get("person_id"),
                "speaker": speaker.get("label"),
                "faction": speaker.get("faction"),
                "side": side,
                "speech_type": record.get("speech_type"),
                "relative_position": i / (len(sentences) - 1)
                if len(sentences) > 1
                else 1.0,
                "salutation": is_salutation(clean),
                "quiet": i not in any_reaction
                and (args.keep_salutations or not is_salutation(clean)),
                **features,
            }
            for kind in LINE_KINDS:
                row[f"is_{kind}"] = kind in followed.get(i, ())
            sentence_rows.append(row)
            sentence_lemmas[(record["uid"], i)] = content_lemmas([words], stop)

    events = pd.DataFrame(event_rows)
    sentences = pd.DataFrame(sentence_rows)
    print(
        f"  {counted['placed']:,} reactions; {counted['aligned']:,} placed, "
        f"{counted['unaligned']} unaligned, {counted['closing']:,} closing ritual"
    )
    print("  lines: " + ", ".join(f"{k} {counted[f'line_{k}']:,}" for k in LINE_KINDS))

    # Vocabulary per group, against one shared vocabulary.
    groups: dict[str, Counter] = {kind: Counter() for kind in (*LINE_KINDS, "quiet")}
    for row in sentences.itertuples(index=False):
        words = sentence_lemmas[(row.uid, row.sentence_index)]
        if row.quiet:
            groups["quiet"].update(words)
        for kind in LINE_KINDS:
            if getattr(row, f"is_{kind}"):
                groups[kind].update(words)
    vocab = sorted(set().union(*groups.values()))
    index = {t: i for i, t in enumerate(vocab)}
    shown = display_spellings(lemmas[r["uid"]] for r in speeches if r["uid"] in lemmas)
    labels = ["" if shown.get(t, t) == t else shown[t] for t in vocab]
    reference = {}
    for left, right in COMPARISONS:
        result = rtd(
            RankedList.from_counts(groups[left]),
            RankedList.from_counts(groups[right]),
            alpha=args.alpha,
        )
        reference[f"{left}|{right}"] = round(float(result.divergence), 10)
        print(f"  keyflux RTD {left} vs {right}: {result.divergence:.4f}")

    features = feature_table(sentences)
    rates_speaker = rate_table(
        sentences, ["speaker_id", "speaker", "faction"], args.min_sentences
    )
    rates_faction = rate_table(sentences, ["faction"], 0)
    rates_type = rate_table(sentences, ["speech_type"], 0)

    # Where in a speech each kind of line falls, in deciles of relative position.
    positions = {}
    for kind in LINE_KINDS:
        hits = sentences[sentences[f"is_{kind}"]]["relative_position"]
        deciles = (
            (hits * 10)
            .clip(upper=9.999)
            .astype(int)
            .value_counts()
            .reindex(range(10), fill_value=0)
        )
        base = (
            (sentences["relative_position"] * 10)
            .clip(upper=9.999)
            .astype(int)
            .value_counts()
            .reindex(range(10), fill_value=0)
        )
        positions[kind] = {
            "lines": deciles.tolist(),
            "sentences": base.tolist(),
            "per_100": [
                round(a / b * 100, 3) if b else None
                for a, b in zip(deciles, base, strict=True)
            ],
        }

    # Best lines per eligible speaker, verbatim.
    eligible = set(
        rates_speaker.loc[~rates_speaker["below_min_sentences"], "speaker_id"]
    )
    best: dict[str, dict] = {}
    for sid, group in events[
        events["aligned"] & events["speaker_id"].isin(eligible)
    ].groupby("speaker_id"):
        entry = {}
        for kind, limit in (
            ("applause", args.best),
            ("laughter_with", 3),
            ("laughter_at", 3),
            ("heckle", 3),
        ):
            rows = group[
                group["line_kinds"].str.split(";").map(lambda ks, k=kind: k in ks)
            ]
            rows = rows.sort_values(
                ["weight", "relative_position"], ascending=[False, True]
            )
            entry[kind] = [
                {
                    "text": r.trigger,
                    "span": r.span,
                    # pandas fills a missing quote with NaN, which is not JSON.
                    "quote": None if pd.isna(r.quote) else r.quote,
                    "interrupted": bool(r.interrupted),
                    "date": r.date,
                    "speech_type": r.speech_type,
                }
                for r in rows.head(limit).itertuples(index=False)
            ]
        best[sid] = entry

    # A seeded sample for a human to check by eye -- trigger against the raw text.
    aligned = events[events["aligned"]]
    sample = aligned.sample(n=min(40, len(aligned)), random_state=args.seed)[
        [
            "uid",
            "speaker",
            "trigger",
            "span",
            "context_after",
            "interrupted",
            "closing",
            "line_kinds",
        ]
    ].assign(checked_by_human=False)

    generated = datetime.now(UTC).isoformat()
    params = {
        "cycle": CYCLE,
        "closing_rule": (
            "after the final sentence, or after a closing formula "
            "(applause.CLOSING_FORMULA)"
        ),
        "applause_line": "own-side or unattributed applause, closing ritual excluded",
        "trigger_unit": (
            "the one sentence before the reaction; for an interruption, the "
            "sentence up to the cut (features and vocabulary use the whole sentence)"
        ),
        "quiet": (
            "sentences followed by no reaction of any kind"
            + ("" if args.keep_salutations else ", salutations excluded")
        ),
        "min_sentences": args.min_sentences,
        "alpha": args.alpha,
        "stopwords": {
            "extras": [*HECKLE_STOP_EXTRAS, *SPEECH_STOP_EXTRAS],
            "size": len(stop),
        },
        "features": list(FEATURES),
    }
    events.to_csv(out / "reaction_lines.csv", index=False, encoding="utf-8")
    sentences.to_csv(
        out / "sentence_table.csv.gz", index=False, encoding="utf-8", compression="gzip"
    )
    features.to_csv(out / "features.csv", index=False, encoding="utf-8")
    rates_speaker.to_csv(out / "line_rates_speaker.csv", index=False, encoding="utf-8")
    rates_faction.to_csv(out / "line_rates_faction.csv", index=False, encoding="utf-8")
    rates_type.to_csv(out / "line_rates_speech_type.csv", index=False, encoding="utf-8")
    sample.to_csv(out / "hand_check.csv", index=False, encoding="utf-8")

    def records(frame):
        return json.loads(frame.to_json(orient="records", force_ascii=False))

    summary = {
        "generated_at": generated,
        "source": "scripts/applause_lines.py",
        "params": params,
        "counts": {k: int(v) for k, v in counted.items()},
        "sentences": int(len(sentences)),
        "quiet_sentences": int(sentences["quiet"].sum()),
        "features": records(features),
        "positions": positions,
        "rates": {
            "faction": records(rates_faction),
            "speech_type": records(rates_type),
            "speaker": records(rates_speaker),
        },
        "reference_rtd": reference,
    }
    # allow_nan=False: a NaN in a published file is a bug, and a browser rejects
    # the whole file over it.
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    lines_bundle = {
        "generated_at": generated,
        "source": "scripts/applause_lines.py",
        "encoding": (
            "vocab[i] is a case-folded lemma, labels[i] its display spelling or '' "
            "when identical; groups[*].gaps are successive differences of sorted "
            "vocab indices, n the counts"
        ),
        "vocab": vocab,
        "labels": labels,
        "groups": {name: gap_encode(cnt, index) for name, cnt in groups.items()},
        "best": best,
    }
    (out / "lines.json").write_text(
        json.dumps(
            lines_bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ),
        encoding="utf-8",
    )
    speeches_path = Path(args.data_dir or DATA_RAW) / f"cycle{CYCLE}-speeches.jsonl"
    manifest = {
        "generated_at": generated,
        "command": "scripts/applause_lines.py",
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
        "counts": summary["counts"]
        | {
            "sentences": summary["sentences"],
            "quiet_sentences": summary["quiet_sentences"],
        },
        "reference_rtd": reference,
        "verification": {"every_trigger_verbatim_in_text_clean": True},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    size = (out / "lines.json").stat().st_size / 1e6
    print(f"\nWrote {_repo_relative(out)} (lines.json {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
