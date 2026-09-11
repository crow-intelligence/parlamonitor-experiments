"""Join every derived table into one JSON bundle for the dashboard.

The dashboard is a static page: no server, no database, so everything it needs
arrives as one file it can fetch. This builds that file.

Four tables join cleanly and this script is where that is asserted rather than
assumed:

* ``uid`` links a speech to its readability, affect, topic and reactions.
* ``speaker_id`` links a speaker to their metrics, affect, reaction scores and
  their node in the heckling network.

Everything here is cycle 43, because that is the only cycle with a speeches
export and therefore the only one where a score can be attached to a person.

Two things are deliberately *not* averaged away:

* Channels that failed validation (:data:`parlamonitor.affect.UNRELIABLE_CHANNELS`)
  are carried with a flag, so the dashboard can grey them out rather than
  quietly present a broken number as a real one.
* Records whose readability is an artefact (the notary's roll-call) keep their
  flag and are excluded from every mean.

Usage::

    uv run python scripts/build_dashboard_data.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from parlamonitor.lexicon import EKMAN as EKMAN_NAMES

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
DEFAULT_OUT = ROOT / "dashboard" / "data"

EKMAN = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
VIRTUES = (
    "prudence",
    "justice",
    "courage",
    "temperance",
    "truthfulness",
    "magnanimity",
)

# The metrics shown as percentile strips, in display order. Each is (column,
# whether higher is "more of the thing" for the axis direction).
PROFILE_METRICS: tuple[tuple[str, str], ...] = (
    ("lix_word_weighted", "readability_lix"),
    ("mdd", "syntax_mdd"),
    ("mhd", "syntax_mhd"),
    ("mattr_mean", "diversity_mattr"),
    ("loanword_ratio_mean", "loanword_ratio"),
    ("words_per_sentence_mean", "words_per_sentence"),
    ("sentiment_polarity", "sentiment_polarity"),
    ("emotion_anger", "emotion_anger"),
    ("emotion_disgust", "emotion_disgust"),
    ("emotion_fear", "emotion_fear"),
    ("emotion_joy", "emotion_joy"),
    ("emotion_sadness", "emotion_sadness"),
    ("emotion_surprise", "emotion_surprise"),
    *[(f"virtue_{v}", f"virtue_{v}") for v in VIRTUES],
    ("laughter_per_minute", "laughter_per_minute"),
    ("applause_per_minute", "applause_per_minute"),
    ("heckles_received", "heckles_received"),
    ("heckles_given", "heckles_given"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--derived", type=Path, default=DERIVED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-speeches", type=int, default=5)
    return parser.parse_args(argv)


def read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run the script that produces it before this one"
        )
    return pd.read_csv(path, low_memory=False)


def build_speeches(derived: Path) -> pd.DataFrame:
    """One row per speech, with readability, affect, topic and reaction counts."""
    metrics = read(derived / "metrics" / "speech_metrics.csv")
    affect = read(derived / "metrics" / "speech_affect.csv")
    topics = read(derived / "task2" / "bertopic_documents.csv")
    syntax_path = derived / "metrics" / "speech_syntax.csv"

    affect_columns = [
        c
        for c in affect.columns
        if c.startswith(("sentiment_", "emotion_")) or c == "n_chunks"
    ]
    frame = metrics.merge(affect[["uid", *affect_columns]], on="uid", how="left")
    frame = frame.merge(
        topics[["uid", "topic", "topic_reduced", "probability"]], on="uid", how="left"
    )
    # Syntactic complexity is optional: it needs a parse, and the dashboard
    # should still build from a checkout that has not run it.
    if syntax_path.is_file():
        syntax = read(syntax_path)
        frame = frame.merge(
            syntax[["uid", "mdd", "mhd", "n_parsed_sentences"]], on="uid", how="left"
        )
    else:
        frame["mdd"] = None
        frame["mhd"] = None

    virtue_speech = derived / "metrics" / "speech_virtues.csv"
    if virtue_speech.is_file():
        vs = read(virtue_speech)
        columns = ["uid", "virtue_mentions", "virtue_sparse"] + [
            f"virtue_{v}" for v in VIRTUES
        ]
        frame = frame.merge(
            vs[[c for c in columns if c in vs.columns]], on="uid", how="left"
        )

    # Reaction counts per speech, pivoted from the long event table.
    reactions = read(derived / "reactions" / "speech_reactions.csv")
    exploded = reactions.assign(kind=reactions["kinds"].str.split(";")).explode("kind")
    counts = (
        exploded.pivot_table(
            index="uid", columns="kind", values="weight", aggfunc="size", fill_value=0
        )
        .add_prefix("reaction_")
        .reset_index()
    )
    frame = frame.merge(counts, on="uid", how="left")
    for column in [c for c in frame.columns if c.startswith("reaction_")]:
        frame[column] = frame[column].fillna(0).astype(int)
    return frame


def build_people(derived: Path, speeches: pd.DataFrame, min_speeches: int):
    """One row per speaker, joining every per-MP table."""
    metrics = read(derived / "metrics" / "mp_metrics.csv")
    affect = read(derived / "metrics" / "mp_affect.csv")
    reactions = read(derived / "reactions" / "mp_reaction_scores.csv")
    nodes = read(derived / "reactions" / "heckle_nodes.csv")
    syntax_path = derived / "metrics" / "mp_syntax.csv"
    virtue_path = derived / "metrics" / "mp_virtues.csv"

    # `speeches` and `words` appear in all three tables and do not mean the same
    # thing: mp_metrics counts only the records whose readability is usable, so
    # they are renamed rather than silently overwritten by the reaction table's
    # totals, which count every speech.
    metrics = metrics.rename(
        columns={"speeches": "prose_speeches", "words": "prose_words"}
    )
    shared = ("speaker", "faction", "speeches", "words", "is_mp")
    frame = metrics.merge(
        affect.drop(columns=[c for c in shared if c in affect.columns]),
        on="speaker_id",
        how="outer",
    )
    frame = frame.merge(
        reactions.drop(
            columns=[
                c for c in ("speaker", "faction", "is_mp") if c in reactions.columns
            ]
        ),
        on="speaker_id",
        how="outer",
    )
    if syntax_path.is_file():
        syntax = read(syntax_path)
        frame = frame.merge(
            syntax[["speaker_id", "mdd", "mhd", "speeches_parsed"]],
            on="speaker_id",
            how="left",
        )
    if virtue_path.is_file():
        virtues = read(virtue_path)
        keep = (
            ["speaker_id", "virtue_mentions", "virtue_sparse"]
            + [
                c
                for c in virtues.columns
                if c.startswith("virtue_") and c[7:] in VIRTUES
            ]
            + [f"virtue_{v}_stance" for v in VIRTUES]
        )
        frame = frame.merge(
            virtues[[c for c in keep if c in virtues.columns]],
            on="speaker_id",
            how="left",
        )
    frame = frame.merge(
        nodes[["node_id", "heckles_given", "targets", "hecklers"]].rename(
            columns={"node_id": "speaker_id"}
        ),
        on="speaker_id",
        how="left",
    )
    for column in ("heckles_given", "targets", "hecklers"):
        frame[column] = frame[column].fillna(0).astype(int)

    frame["below_min_speeches"] = frame["speeches"].fillna(0) < min_speeches
    # Most-used topic per speaker, for the profile header.
    topped = (
        speeches[speeches["topic"] >= 0]
        .groupby(["speaker_id", "topic"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("speaker_id")
    )
    frame = frame.merge(
        topped.rename(columns={"topic": "top_topic", "n": "top_topic_speeches"}),
        on="speaker_id",
        how="left",
    )
    return frame


def build_topics(derived: Path, speeches: pd.DataFrame) -> list[dict]:
    """Per-topic means, keywords and reaction rates -- the interaction view."""
    names = json.loads(
        (ROOT / "data" / "labels" / "task2_topic_names.json").read_text(
            encoding="utf-8"
        )
    )
    topic_terms = read(derived / "task2" / "bertopic_topics.csv")
    terms_by_id = dict(
        zip(topic_terms["Topic"], topic_terms["Representation"], strict=True)
    )

    rows = []
    reaction_columns = [c for c in speeches.columns if c.startswith("reaction_")]
    for topic, group in speeches.groupby("topic"):
        topic = int(topic)
        prose = group[group["readability_reliable"]]
        label = names.get(str(topic), {})
        minutes = group["duration_s"].fillna(0).sum() / 60
        row = {
            "topic": topic,
            "name_hu": label.get("hu"),
            "name_en": label.get("en"),
            "terms": str(terms_by_id.get(topic, ""))[:300],
            "n_speeches": int(len(group)),
            "n_words": int(group["n_words"].sum()),
            "minutes": round(float(minutes), 1),
            "lix": round(float(prose["lix"].mean()), 2) if len(prose) else None,
            "mdd": (
                round(float(group["mdd"].mean()), 4)
                if "mdd" in group and group["mdd"].notna().any()
                else None
            ),
            **{
                f"virtue_{v}": (
                    round(float(group[f"virtue_{v}"].mean()), 4)
                    if f"virtue_{v}" in group and group[f"virtue_{v}"].notna().any()
                    else None
                )
                for v in VIRTUES
            },
            "loanword_ratio": (
                round(float(group["loanword_ratio"].mean()), 6)
                if "loanword_ratio" in group and group["loanword_ratio"].notna().any()
                else None
            ),
            "mattr": round(float(prose["mattr"].mean()), 4) if len(prose) else None,
            "sentiment_polarity": round(float(group["sentiment_polarity"].mean()), 4),
            **{
                f"emotion_{name}": round(float(group[f"emotion_{name}"].mean()), 6)
                for name in EKMAN
            },
        }
        # Reactions per hour of floor time, so a topic that simply got more
        # airtime does not look more provocative than one that did not.
        for column in reaction_columns:
            per_hour = group[column].sum() / minutes * 60 if minutes else 0.0
            row[f"{column}_per_hour"] = round(float(per_hour), 3)
        # The keywords most often ranked first across the topic's speeches.
        keywords: dict[str, int] = {}
        for cell in group["textrank_keywords"].dropna():
            for term in str(cell).split(";")[:5]:
                keywords[term] = keywords.get(term, 0) + 1
        row["keywords"] = [
            term for term, _ in sorted(keywords.items(), key=lambda kv: -kv[1])[:8]
        ]
        rows.append(row)
    return sorted(rows, key=lambda r: -r["n_speeches"])


def build_topic_mp(speeches: pd.DataFrame, min_speeches: int = 2) -> list[dict]:
    """Who speaks on what -- the topic/MP association.

    Three numbers per pair, because "associated with" is ambiguous and the
    three disagree:

    ``speeches``
        Raw count. Favours whoever spoke most in total.
    ``share_of_speaker``
        What fraction of *this MP's* speeches fell in this topic. High for a
        specialist, low for a frontbencher who ranges over everything.
    ``share_of_topic``
        What fraction of *this topic's* speeches were theirs. High for whoever
        dominated the debate, whatever else they also did.

    A backbencher who spoke four times, all on health, scores 1.0 on the second
    and near zero on the third; the Prime Minister is the reverse. Reporting
    one alone would hide whichever kind of association the reader wanted.

    Args:
        speeches: The joined speech table.
        min_speeches: Drop pairs below this, since a single speech makes
            ``share_of_speaker`` either 0 or 1 and neither means much.

    Returns:
        One record per (topic, speaker) pair that clears the threshold.
    """
    named = speeches[speeches["topic"] >= 0].dropna(subset=["speaker_id"])
    per_speaker = named.groupby("speaker_id").size()
    per_topic = named.groupby("topic").size()

    rows = []
    grouped = named.groupby(["topic", "speaker_id", "speaker", "faction"], dropna=False)
    for (topic, speaker_id, speaker, faction), group in grouped:
        if len(group) < min_speeches:
            continue
        rows.append(
            {
                "topic": int(topic),
                "speaker_id": speaker_id,
                "speaker": speaker,
                "faction": None if pd.isna(faction) else faction,
                "speeches": int(len(group)),
                "words": int(group["n_words"].sum()),
                "share_of_speaker": round(len(group) / per_speaker[speaker_id], 4),
                "share_of_topic": round(len(group) / per_topic[topic], 4),
            }
        )
    return sorted(rows, key=lambda r: (r["topic"], -r["speeches"]))


def build_parties(people: pd.DataFrame, speeches: pd.DataFrame) -> list[dict]:
    """Party aggregates, weighted by words where a mean would mislead."""
    rows = []
    for faction, group in speeches.groupby("faction"):
        prose = group[group["readability_reliable"]]
        members = people[people["faction"] == faction]
        weights = prose["n_words"]
        rows.append(
            {
                "faction": faction,
                "members": int(len(members)),
                "n_speeches": int(len(group)),
                "n_words": int(group["n_words"].sum()),
                # Word-weighted: a two-sentence intervention should not count
                # as much as a twenty-minute address.
                "lix": round(float((prose["lix"] * weights).sum() / weights.sum()), 2),
                "mattr": round(float(prose["mattr"].mean()), 4),
                "sentiment_polarity": round(
                    float(group["sentiment_polarity"].mean()), 4
                ),
                **{
                    f"emotion_{name}": round(float(group[f"emotion_{name}"].mean()), 6)
                    for name in EKMAN
                },
            }
        )
    return sorted(rows, key=lambda r: -r["n_speeches"])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Joining derived tables ...")
    speeches = build_speeches(args.derived)
    print(f"  {len(speeches):,} speeches")
    missing_affect = int(speeches["sentiment_polarity"].isna().sum())
    missing_topic = int(speeches["topic"].isna().sum())
    print(f"  without affect: {missing_affect}; without topic: {missing_topic}")

    people = build_people(args.derived, speeches, args.min_speeches)
    print(f"  {len(people):,} speakers")
    topics = build_topics(args.derived, speeches)
    print(f"  {len(topics):,} topics")
    parties = build_parties(people, speeches)
    print(f"  {len(parties):,} factions")
    topic_mp = build_topic_mp(speeches)
    print(f"  {len(topic_mp):,} topic/speaker pairs")

    network = json.loads(
        (args.derived / "reactions" / "heckle_network.json").read_text(encoding="utf-8")
    )

    # Percentile strips need the whole population per metric, so the columns
    # travel as arrays rather than being recomputed in the browser.
    # Emotion rates are hits per token, around 0.003, which renders as "0".
    # Scaled to hits per 1,000 tokens for display: the same quantity in a
    # magnitude a reader can hold. The CSVs keep the raw rate.
    people = people.copy()
    for name in EKMAN:
        column = f"emotion_{name}"
        if column in people:
            people[column] = (people[column] * 1000).round(3)

    distributions = {}
    eligible = people[~people["below_min_speeches"]]
    for column, key in PROFILE_METRICS:
        if column not in people.columns:
            continue
        values = eligible[["speaker_id", column]].dropna()
        distributions[key] = {
            "column": column,
            "n": int(len(values)),
            "values": [
                {"id": sid, "v": round(float(v), 6)}
                for sid, v in zip(values["speaker_id"], values[column], strict=True)
            ],
        }

    speech_columns = [
        "uid",
        "date",
        "speaker_id",
        "speaker",
        "faction",
        "speech_type",
        "topic",
        "n_words",
        "n_sentences",
        "lix",
        "lix_band",
        "mattr",
        "readability_reliable",
        "mdd",
        "mhd",
        "loanword_ratio",
        "sentiment_score",
        "sentiment_polarity",
        "emotion_dominant",
        "emotion_sparse",
        *[f"emotion_{name}" for name in EKMAN],
        "textrank_keywords",
        "keybert_keywords",
        *[c for c in speeches.columns if c.startswith("reaction_")],
    ]
    bundle = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cycle": 43,
        "scope_note": (
            "Cycle 43 only. Readability, affect, keywords and per-MP reaction "
            "scores need a speeches export, and cycle 43 is the only cycle that "
            "has one."
        ),
        "affect_method": "dictionary-based (Precognox sentiment; Putz Orsolya emotion)",
        "virtue_construct": (
            "salience — which moral vocabulary a speaker uses, NOT whether they "
            "have the virtue. Every virtue carries an affirming/accusing stance "
            "because 41% of truthfulness vocabulary here is accusation."
        ),
        "emotion_scheme": list(EKMAN_NAMES),
        "min_speeches": args.min_speeches,
        "people": json.loads(people.to_json(orient="records")),
        "parties": parties,
        "topics": topics,
        "topic_mp": topic_mp,
        "distributions": distributions,
        "network": network,
        "speeches": json.loads(
            speeches[[c for c in speech_columns if c in speeches.columns]].to_json(
                orient="records"
            )
        ),
    }
    path = out / "dashboard.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    size = path.stat().st_size / 1e6
    print(f"\nWrote {path} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
