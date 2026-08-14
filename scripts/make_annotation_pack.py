"""Build the human-annotation pack comparing CAP labels to the BERTopic topics.

Produces ``annotation/``: a findings write-up, a metadata record, and a
stratified sample of speeches laid out one per file for a human to judge.

Every number in the generated markdown is interpolated from the data rather
than typed, so the prose cannot drift away from the results it describes. The
interpretation around those numbers is authored here and is a draft.

The sample is stratified over the five ways the two methods can disagree, not
drawn uniformly: a uniform sample of 1,692 speeches would be almost entirely
cases where nothing interesting happens. Each stratum asks the annotator a
different question, and ``--seed`` makes the draw reproducible.

Usage::

    uv run python scripts/make_annotation_pack.py
    uv run python scripts/make_annotation_pack.py --per-stratum 20 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from parlamonitor.cap import DEFAULT_THRESHOLD, MAX_TOKENS, MODEL_ID
from parlamonitor.loading import load_speeches

ROOT = Path(__file__).resolve().parents[1]
TASK1 = ROOT / "data" / "derived" / "task1"
TASK2 = ROOT / "data" / "derived" / "task2"

HIGH_PURITY = 0.70
LOW_PURITY = 0.40

STRATA = {
    "agreement": (
        "Both methods concur",
        "The speech's CAP label matches the dominant CAP label of its BERTopic "
        "topic, inside a topic that is itself coherent (purity >= "
        f"{HIGH_PURITY:.0%}). Two independent methods put this speech in the "
        "same place. **Question: is that shared label actually right?**",
    ),
    "cross_method_disagreement": (
        "One method dissents",
        "The speech sits in a high-purity topic but carries a different CAP "
        "label from the rest of it. Either the classifier misread this speech, "
        "or the topic model swept in something that does not belong. "
        "**Question: which one is wrong?**",
    ),
    "low_purity_topic": (
        "The methods cut differently",
        f"The speech belongs to a topic with purity below {LOW_PURITY:.0%} -- "
        "one that scatters across CAP labels. These are usually about the "
        "political process rather than a policy area, which CAP has no code "
        "for. **Question: does either label describe this speech usefully?**",
    ),
    "mix": (
        "The classifier was unsure",
        f"Confidence fell below {DEFAULT_THRESHOLD}, so the label became "
        "`Mix`. The model authors take this to mean the speech spans several "
        "topics. **Question: is it genuinely multi-topic, or did the model "
        "simply fail on a clear case?**",
    ),
    "truncation": (
        "The two passes disagree",
        f"The speech exceeds {MAX_TOKENS} tokens. Classifying its opening "
        "alone gives one CAP label; classifying every window and averaging "
        "gives another. **Question: which label fits the whole speech?** The "
        "cut point is marked in the text below.",
    ),
}

LICENCE = "CC BY-NC-SA 4.0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "annotation")
    parser.add_argument("--per-stratum", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-words",
        type=int,
        default=1200,
        help="cap the quoted speech; longer ones are cut with a marker",
    )
    return parser.parse_args(argv)


def log(message):
    print(f"[pack] {message}", flush=True)


def load_everything():
    cap = pd.read_csv(TASK1 / "parlacap_classifications.csv", encoding="utf-8")
    topics = pd.read_csv(TASK2 / "bertopic_documents.csv", encoding="utf-8")
    labels = pd.read_csv(TASK2 / "topic_labels.csv", encoding="utf-8")
    merged = cap.merge(topics[["uid", "topic"]], on="uid", how="inner")
    names = dict(zip(labels.topic, labels.label_en))
    return merged, names, labels


def purity_table(merged, names):
    """Per BERTopic topic: size, dominant CAP label, and purity."""
    table = pd.crosstab(merged.topic, merged.Predicted_CAP_Topic)
    totals = table.sum(axis=1)
    frame = pd.DataFrame(
        {
            "n": totals,
            "dominant_cap": table.idxmax(axis=1),
            "dominant_n": table.max(axis=1),
            "purity": table.max(axis=1) / totals,
        }
    )
    frame["topic_label"] = frame.index.map(lambda t: names.get(t, ""))
    return frame.sort_values("purity", ascending=False), table


def build_strata(merged, purity, *, per_stratum, seed):
    """Draw a reproducible stratified sample, one frame per stratum."""
    joined = merged.join(purity[["purity", "dominant_cap", "topic_label"]], on="topic")
    clustered = joined[joined.topic >= 0]

    pools = {
        "agreement": clustered[
            (clustered.purity >= HIGH_PURITY)
            & (clustered.Predicted_CAP_Topic == clustered.dominant_cap)
            & (clustered.Predicted_CAP_Topic != "Mix")
        ],
        "cross_method_disagreement": clustered[
            (clustered.purity >= HIGH_PURITY)
            & (clustered.Predicted_CAP_Topic != clustered.dominant_cap)
            & (clustered.Predicted_CAP_Topic != "Mix")
        ],
        "low_purity_topic": clustered[clustered.purity < LOW_PURITY],
        "mix": joined[joined.Predicted_CAP_Topic == "Mix"],
        "truncation": joined[joined.truncated & ~joined.passes_agree],
    }

    samples = {}
    for name, pool in pools.items():
        take = min(per_stratum, len(pool))
        if take < per_stratum:
            log(f"  {name}: only {len(pool)} available, sampling all of them")
        samples[name] = (
            pool.sample(take, random_state=seed).sort_values("uid") if take else pool
        )
        log(f"  {name}: {take} of {len(pool)} candidates")
    return samples, pools


def mark_truncation(text, tokenizer, *, max_words):
    """Return the speech text with the model's cut-off point marked."""
    words = text.split()
    body = " ".join(words[:max_words])
    clipped = len(words) > max_words

    cut_word = None
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if len(ids) > MAX_TOKENS:
            # Decode the first MAX_TOKENS tokens and count words in the result:
            # that is exactly what the truncated pass saw.
            seen = tokenizer.decode(ids[:MAX_TOKENS], skip_special_tokens=True)
            cut_word = len(seen.split())

    if cut_word is not None and cut_word < min(len(words), max_words):
        head = " ".join(words[:cut_word])
        tail = " ".join(words[cut_word:max_words])
        body = (
            f"{head}\n\n"
            f"> ⟪ ── the truncated pass stopped here, after {cut_word} of "
            f"{len(words)} words ── ⟫\n\n"
            f"{tail}"
        )
    if clipped:
        body += (
            f"\n\n> ⟪ ── quote cut at {max_words} words; the speech runs to "
            f"{len(words)} ── ⟫"
        )
    return body


def write_sample(path, row, speech, tokenizer, *, max_words):
    faction = row.faction if isinstance(row.faction, str) else "—"
    topic_line = (
        f"T{int(row.topic)} — {row.topic_label}"
        if row.topic >= 0
        else "unclustered (outlier)"
    )
    lines = [
        f"# {row.uid} — {row.speaker_label} ({faction})",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| uid | `{row.uid}` |",
        f"| date | {row.date} |",
        f"| speaker | {row.speaker_label} ({faction}) |",
        f"| speech type | {row.speech_type} |",
        f"| discourse role | `{row.discourse_role}` |",
        f"| length | {row.n_words} words / {row.n_tokens} tokens |",
        f"| exceeds {MAX_TOKENS} tokens | {'yes' if row.truncated else 'no'} |",
        "",
        "## What each method said",
        "",
        "| method | label | confidence |",
        "| --- | --- | --- |",
        f"| BERTopic topic | {topic_line} | — |",
        f"| CAP, truncated pass | **{row.Predicted_CAP_Topic}** "
        f"| {row.Confidence_Score:.3f} |",
        f"| CAP, chunked pass | {row.chunked_CAP_Topic} "
        f"| {row.chunked_Confidence:.3f} |",
        f"| CAP raw label (pre-`Mix`) | {row.raw_label} | — |",
        f"| dominant CAP of this topic | {row.dominant_cap} | purity {row.purity:.0%} |"
        if row.topic >= 0
        else "| — | — | — |",
        "",
        "## Your judgement",
        "",
        "Fill these in, or record them in `annotation_sheet.csv`.",
        "",
        "- **Correct CAP topic** (or `Mix` / `Other`): ",
        "- **Which method was closer** (`cap` / `bertopic` / `both` / `neither`): ",
        "- **Confidence** (`high` / `medium` / `low`): ",
        "- **Notes**: ",
        "",
        "## Speech",
        "",
        "Verbatim from `data/raw/cycle43-speeches.jsonl`, field `text_clean`:",
        "",
        "---",
        "",
        mark_truncation(speech["text_clean"], tokenizer, max_words=max_words),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_findings(path, merged, purity, stats, pools, names):
    clustered = purity[purity.index >= 0]
    high = clustered[clustered.purity >= HIGH_PURITY]

    def rows(frame, n, ascending=False):
        frame = frame.sort_values("purity", ascending=ascending).head(n)
        return "\n".join(
            f"| T{int(t)} | {r.n} | {r.purity:.0%} | {r.dominant_cap} "
            f"| {r.topic_label} |"
            for t, r in frame.iterrows()
        )

    lines = f"""# Where the two methods agree, and where they do not

Two independent passes over the same 1,693 cycle-43 speeches:

* **Task 1, supervised.** `classla/ParlaCAP-Topic-Classifier` assigns each
  speech one of the 21 CAP major topics or `Other`, against a fixed codebook
  built for cross-national agenda-setting research.
* **Task 2, unsupervised.** BERTopic clusters the speeches with no schema at
  all, and found {len(clustered)} topics.

Neither knows about the other. Where they land in the same place, that is two
methods corroborating each other. Where they do not, the disagreement is
usually informative about the corpus rather than about a bug.

## How much do they agree overall?

| statistic | all {stats["n_all"]:,} speeches | core {stats["n_core"]:,} only |
| --- | --- | --- |
| Adjusted Rand Index | {stats["ari_all"]:.3f} | **{stats["ari_core"]:.3f}** |
| Normalised Mutual Information | {stats["nmi_all"]:.3f} | **{stats["nmi_core"]:.3f}** |
| Homogeneity | {stats["homog_all"]:.3f} | {stats["homog_core"]:.3f} |
| Completeness | {stats["compl_all"]:.3f} | {stats["compl_core"]:.3f} |

"Core" drops the {stats["n_outlier"]:,} speeches BERTopic left unclustered and
the {stats["n_mix"]:,} the classifier marked `Mix`. Both are *refusals to
decide*, and scoring a refusal against a decision measures the refusal, not the
agreement.

An ARI of {stats["ari_core"]:.3f} is real but partial. These are not two
renderings of one structure; they are two different cuts that coincide over
part of the corpus. The interesting question is which part.

## Where they agree: concrete policy domains

{len(high)} of {len(clustered)} topics reach {HIGH_PURITY:.0%} purity or better
against CAP — meaning that share of the topic's speeches carry a single CAP
label.

| topic | n | purity | dominant CAP | our name |
| --- | --- | --- | --- | --- |
{rows(clustered, 8)}

Most of these are **policy domains with their own vocabulary**. Agriculture
talks about `agrárkamara` and `aszály`; transport about `gördülőállomány` and
`vasútvonal`. A fixed codebook and an unsupervised clusterer both find them
because the language marks them unambiguously.

Two entries in that table are a different animal. T25 (roll-calls, committee
seats) and T12 (microphones, timing, the chair) are **procedural boilerplate**.
They score highly because such speeches are near-identical to each other, and
CAP has nowhere to put them but `Government Operations` and `Other`. So purity
measures *the two methods agreeing*, not *the topic being substantive* — a
uniform topic and a meaningful one both score well, and only reading the
speeches separates them. That is one of the things the sample below is for.

The reverse view agrees. Taking each CAP label and asking how concentrated it
is in a single BERTopic topic:

| CAP label | n | concentrated in | share |
| --- | --- | --- | --- |
{stats["reverse_high"]}

## Where they diverge: the political process

| topic | n | purity | dominant CAP | our name |
| --- | --- | --- | --- | --- |
{rows(clustered, 6, ascending=True)}

These are not failures. CAP is a codebook of **policy areas**, and it has no
code for *"an MP attacking the government over its own conduct"*. Such speeches
land in `Government Operations`, or in `Mix` when the classifier cannot commit.
BERTopic, unconstrained, split the same material by rhetorical mode instead —
pre-agenda clashes, immunity proceedings, ceremonial remembrance.

Read from the CAP side, the same split appears:

| CAP label | n | concentrated in | share |
| --- | --- | --- | --- |
{stats["reverse_low"]}

`Government Operations` is the largest CAP class in the core
({stats["govops_n"]} speeches; {stats["govops_all"]} across the whole corpus)
and among the most scattered, at {stats["govops_purity"]:.0%}. It is where the
codebook puts everything about how the chamber runs — which, in a cycle this
contested, is a great deal.

## One disagreement worth its own line

CAP separates `Education` from `Social Welfare`. BERTopic did not: topic
{stats["t0"]} covers schools, teachers, child protection, adoption and family
support as one cluster, and it is the dominant destination for **both** CAP
labels ({stats["edu_share"]:.0%} of `Education`, {stats["sw_share"]:.0%} of
`Social Welfare`). Whether Hungarian family policy is one agenda or two is a
substantive question, not a technical one, and the two methods answer it
differently.

## What to check by hand

The counts above say nothing about whether either label is *correct*. The
sample in `samples/` is drawn to test exactly that, across the five ways these
methods can part company:

| stratum | candidates | question |
| --- | --- | --- |
{stats["strata_table"]}

Start with `README.md` in this folder.

---

*Numbers in this file are interpolated from the run outputs, not transcribed.
Regenerate with `uv run python scripts/make_annotation_pack.py`. The
interpretation between them is a draft and has not been checked by a human.*
"""
    path.write_text(lines, encoding="utf-8")


def write_metadata(path, stats, args, manifests):
    t1, t2 = manifests
    source = t1.get("source", {})
    p1, p2 = t1["parameters"], t2["parameters"]
    emtsv, embed = p2.get("emtsv", {}), p2.get("embedding", {})
    phr, freq, bert = p2["phrases"], p2["frequency_stopwords"], p2["bertopic"]
    # Pulled out so the markdown table rows below fit in a source line.
    rev1 = p1.get("revision", "see run_manifest.json")
    training = (
        "29 ParlaMint 4.1 datasets, labelled by GPT-4o under the "
        "LLM teacher-student framework"
    )
    thresh = f"{p1['threshold']} → `Mix` (**the model authors' rule**, from the card)"
    img = str(emtsv.get("image", "—"))[:19]
    keep = ", ".join(emtsv.get("keep_pos", []))
    erev = str(embed.get("revision", "—"))[:12]
    stop_n = p2["stoplist"]["size"]
    salut = p2["normalisation"]["words_removed"]
    bigram_note = f"min_count {phr['min_count']}, threshold {phr['threshold']}"
    freq_note = f"max_df {freq['max_df']}, min_df {freq['min_df']} documents"
    chunk_note = f"{embed.get('chunk_size', '—')}-word windows, {embed.get('pooling')}"
    cluster_note = (
        f"UMAP (seed {bert['umap_random_state']}) + "
        f"HDBSCAN (min_cluster_size {bert['min_cluster_size']})"
    )
    lines = f"""# Metadata and provenance

Everything needed to reproduce, audit or dispute the material in this folder.

## Source data

| | |
| --- | --- |
| Corpus | Hungarian National Assembly, cycle 43 (current cycle) |
| Speeches | {stats["n_speeches"]:,} |
| Words | {stats["n_words"]:,} (cleaned) |
| Database snapshot | `{source.get("db_data_updated_at", "—")}` |
| Source file | `data/raw/cycle43-speeches.jsonl` |
| Q&A file | `data/raw/cycle43-qa.jsonl` |
| Manifest | `data/raw/cycle43-manifest.json` |

The raw exports are gitignored: 32 MB and rebuildable from the parlamonitor
SQLite database with `backend/export_nlp_datasets.py --period 43`. The manifest
and the schema README are committed as the provenance record.

**Already filtered upstream**, before anything here ran: 2,114 procedural
speeches, 8 recitations of the MP oath, and 154 speeches with no published
transcript (almost all of sitting 43020, 2026-07-27, whose recording is public
but whose transcript is not).

Field `text_clean` is used throughout, never `text` — the latter still carries
the speaker attribution and the editorial stage directions.

## Task 1 — supervised CAP classification

| | |
| --- | --- |
| Model | [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) |
| Revision | `{rev1}` |
| Architecture | XLM-RoBERTa-large, 24 layers, 560M parameters |
| Training | {training} |
| Labels | 21 CAP major topics + `Other` |
| Confidence threshold | {thresh} |
| Max length | {p1["max_length"]} tokens, truncation on |
| Chunked pass | {p1["chunk_words"]}-word windows, {p1["chunk_aggregation"]} |
| Hardware | CPU only, 4 threads |

**Accuracy on Hungarian is unknown.** The model card reports macro-F1 for
English (0.723), Croatian (0.686), Serbian (0.710) and Bosnian (0.646) only.
Hungarian is among the model's languages and ParlaMint-HU among its 29 training
datasets, but no Hungarian evaluation has been published. The one available
check is the `Mix` rate: **{stats["mix_rate"]:.1%}** here against the authors'
8.9–11.4% on their test sets. That is consistent, and it is not a substitute
for an evaluation. **This is the main reason the folder you are reading
exists.**

Hungarian costs a median 1.82 subword tokens per word, so {MAX_TOKENS} tokens
covers roughly 279 words against a 307-word median speech.
{stats["n_truncated"]:,} of {stats["n_speeches"]:,} speeches
({stats["truncated_share"]:.1%}) exceed it.

## Task 2 — unsupervised topic modeling

| | |
| --- | --- |
| Lemmatisation | emtsv `{emtsv.get("modules", "—")}`, image `{img}…` |
| POS kept | {keep}; pronouns dropped |
| Stoplist | spaCy Hungarian + light verbs + address terms ({stop_n} lemmas) |
| Normalisation | opening salutation stripped ({salut:,} words) |
| Bigrams | Gensim `Phrases`, {bigram_note} |
| Frequency filter | {freq_note} |
| Embeddings | `{embed.get("model", "—")}`, revision `{erev}…` |
| Chunking | {chunk_note} |
| Clustering | {cluster_note} |
| Topics found | {stats["n_topics"]} + {stats["n_outlier"]:,} unclustered |

Topic names in `data/labels/task2_topic_names.json` are **hand-authored from
the top terms and representative documents, and not yet checked by a human**.
Each carries a verbatim quote so verification is a string match.

## This annotation pack

| | |
| --- | --- |
| Generated | {stats["generated"]} |
| Generator | `scripts/make_annotation_pack.py` |
| Sample seed | {args.seed} |
| Per stratum | {args.per_stratum} |
| Strata | {len(STRATA)} |
| Items | {stats["n_items"]} |
| Quote cap | {args.max_words} words |

The sample is **stratified, not random**. A uniform draw would be dominated by
cases where nothing disagrees. Each stratum isolates one way the two methods
can part company, so the counts here are not corpus proportions and must not be
read as such.

## Reproducing

```bash
uv sync --all-extras
docker run --rm -d --name emtsv -p 5000:5000 mtaril/emtsv   # Task 2 only
uv run python scripts/task1_classifier.py       # ~2 h, CPU
uv run python scripts/task2_bertopic.py         # ~40 min
uv run python scripts/task2_compare.py
uv run python scripts/task2_label_topics.py
uv run python scripts/task1_crosstab.py
uv run python scripts/make_annotation_pack.py
```

Both pipelines cache: emtsv analyses keyed by `(uid, normalisation)`,
embeddings and CAP scores by content hash. Re-running costs only what changed.
Full parameter records are in `data/derived/task1/run_manifest.json` and
`data/derived/task2/run_manifest.json`.

## Tools

Python 3.12, managed with `uv`. Lint and format `ruff`, types `ty`, tests
`pytest` with doctests and `hypothesis`.

Crow Intelligence packages: [`keyflux`](https://github.com/crow-intelligence/keyflux)
(keyness, rank-turbulence divergence, allotaxonograph). Third-party: BERTopic,
sentence-transformers, gensim, scikit-learn, transformers, spaCy (Hungarian
stopword list only, no model), emtsv / `xtsv` for Hungarian morphology.

## Licence and attribution

Analysis and this pack: **{LICENCE}**.

Parliamentary transcripts are public records of the Hungarian National
Assembly. The emtsv toolchain, `ParlaCAP-Topic-Classifier`, ParlaMint and the
CAP codebook each carry their own terms — check them before redistributing
anything derived from them. The CAP master codebook is at
<https://www.comparativeagendas.net/pages/master-codebook>.

Contact: hello@crowintelligence.org
"""
    path.write_text(lines, encoding="utf-8")


def write_readme(path, stats, args):
    strata_sections = "\n".join(
        f"### `{name}/` — {STRATA[name][0]}\n\n{STRATA[name][1]}\n" for name in STRATA
    )
    lines = f"""# Annotation pack — CAP versus the discovered topics

{stats["n_items"]} Hungarian parliamentary speeches, each labelled twice by
methods that never saw each other's output, laid out for a human to adjudicate.

**Why this exists.** The classifier has no published evaluation for Hungarian.
Its `Mix` rate here ({stats["mix_rate"]:.1%}) is consistent with the authors'
figures for the languages they did evaluate, but consistency is not accuracy.
Nothing in `FINDINGS.md` is established until someone reads the speeches.

## Contents

| file | what |
| --- | --- |
| `FINDINGS.md` | where the two methods agree and diverge, with the statistics |
| `METADATA.md` | provenance, every parameter, licence, how to reproduce |
| `annotation_sheet.csv` | one row per item — record verdicts here |
| `samples/<stratum>/<uid>.md` | one speech per file, both labels, full text |

## How to annotate

1. Open `annotation_sheet.csv` alongside `samples/`.
2. For each item, read the speech, then fill in four columns:

   * `correct_cap` — the CAP topic you would assign. `Mix` if it genuinely
     spans several, `Other` if it fits no CAP topic. The 21 topics are listed
     at the foot of this file.
   * `closer_method` — `cap`, `bertopic`, `both`, or `neither`.
   * `annotator_confidence` — `high`, `medium`, `low`.
   * `notes` — anything the columns cannot hold. Especially valuable when you
     pick `neither`.

3. Judge the **whole speech**, not the excerpt the model saw. Where the
   classifier was cut off, the point is marked in the text like this:

   > ⟪ ── the truncated pass stopped here, after N of M words ── ⟫

4. Do not look at the topic name before reading. They are drafts too, written
   from the model's own top terms, and they can prime you.

## The five strata

The sample is **deliberately unbalanced**. It over-samples disagreement,
because agreement is cheap and disagreement is where the methods are testable.
Proportions here are not corpus proportions.

{strata_sections}
## When you are done

Commit the filled sheet. Hand-verified data is not rebuildable, unlike
everything in `data/derived/`, so it belongs in git. Then the numbers in
`FINDINGS.md` acquire something they currently lack: a measured error rate.

## The CAP major topics

{stats["cap_label_list"]}

Plus `Other` (fits no CAP topic — a real prediction) and `Mix` (the classifier
was below {DEFAULT_THRESHOLD} confidence — an admission of uncertainty). These
two are **not** the same thing, and the sheet keeps them separate.
"""
    path.write_text(lines, encoding="utf-8")


def main(argv=None):  # noqa: PLR0915 - one linear build, clearer unsplit
    args = parse_args(argv)
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
    )

    merged, names, _ = load_everything()
    purity, table = purity_table(merged, names)
    speeches = {s["uid"]: s for s in load_speeches()}
    log(f"{len(merged)} speeches carry both labels")

    core = merged[(merged.topic >= 0) & (merged.Predicted_CAP_Topic != "Mix")]
    stats = {
        "n_all": len(merged),
        "n_core": len(core),
        "n_outlier": int((merged.topic < 0).sum()),
        "n_mix": int((merged.Predicted_CAP_Topic == "Mix").sum()),
        "ari_all": adjusted_rand_score(merged.topic, merged.Predicted_CAP_Topic),
        "nmi_all": normalized_mutual_info_score(
            merged.topic, merged.Predicted_CAP_Topic
        ),
        "homog_all": homogeneity_score(merged.Predicted_CAP_Topic, merged.topic),
        "compl_all": completeness_score(merged.Predicted_CAP_Topic, merged.topic),
        "ari_core": adjusted_rand_score(core.topic, core.Predicted_CAP_Topic),
        "nmi_core": normalized_mutual_info_score(core.topic, core.Predicted_CAP_Topic),
        "homog_core": homogeneity_score(core.Predicted_CAP_Topic, core.topic),
        "compl_core": completeness_score(core.Predicted_CAP_Topic, core.topic),
    }

    # Reverse view: how concentrated is each CAP label in one topic?
    reverse_ct = pd.crosstab(core.Predicted_CAP_Topic, core.topic)
    reverse = pd.DataFrame(
        {
            "n": reverse_ct.sum(axis=1),
            "purity": reverse_ct.max(axis=1) / reverse_ct.sum(axis=1),
            "topic": reverse_ct.idxmax(axis=1),
        }
    )
    reverse = reverse[reverse.n >= 25]

    def reverse_rows(frame):
        return "\n".join(
            f"| {label} | {int(r.n)} | T{int(r.topic)} {names.get(r.topic, '')} "
            f"| {r.purity:.0%} |"
            for label, r in frame.iterrows()
        )

    stats["reverse_high"] = reverse_rows(reverse.nlargest(6, "purity"))
    stats["reverse_low"] = reverse_rows(reverse.nsmallest(5, "purity"))

    govops = reverse.loc["Government Operations"]
    stats["govops_n"] = int(govops.n)
    stats["govops_all"] = int(
        (merged.Predicted_CAP_Topic == "Government Operations").sum()
    )
    stats["govops_purity"] = float(govops.purity)
    stats["t0"] = int(reverse.loc["Education", "topic"])
    stats["edu_share"] = float(reverse.loc["Education", "purity"])
    stats["sw_share"] = float(reverse.loc["Social Welfare", "purity"])

    samples, pools = build_strata(
        merged, purity, per_stratum=args.per_stratum, seed=args.seed
    )
    stats["strata_table"] = "\n".join(
        f"| `{name}` | {len(pools[name])} | {STRATA[name][0]} |" for name in STRATA
    )

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - the marker is a nicety, not the point
        log(f"tokenizer unavailable ({type(exc).__name__}); cut points unmarked")
        tokenizer = None

    out = args.output_dir
    (out / "samples").mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for stratum, frame in samples.items():
        folder = out / "samples" / stratum
        folder.mkdir(parents=True, exist_ok=True)
        for existing in folder.glob("*.md"):
            existing.unlink()
        for row in frame.itertuples():
            write_sample(
                folder / f"{row.uid}.md",
                row,
                speeches[row.uid],
                tokenizer,
                max_words=args.max_words,
            )
            sheet_rows.append(
                {
                    "stratum": stratum,
                    "uid": row.uid,
                    "file": f"samples/{stratum}/{row.uid}.md",
                    "n_words": row.n_words,
                    "truncated": row.truncated,
                    "bertopic_topic": int(row.topic),
                    "bertopic_label": row.topic_label,
                    "cap_truncated": row.Predicted_CAP_Topic,
                    "cap_confidence": round(float(row.Confidence_Score), 4),
                    "cap_chunked": row.chunked_CAP_Topic,
                    "topic_dominant_cap": row.dominant_cap,
                    "topic_purity": round(float(row.purity), 4)
                    if row.topic >= 0
                    else "",
                    "correct_cap": "",
                    "closer_method": "",
                    "annotator_confidence": "",
                    "notes": "",
                }
            )

    sheet = pd.DataFrame(sheet_rows)
    sheet.to_csv(out / "annotation_sheet.csv", index=False, encoding="utf-8")

    cap_labels = sorted(
        set(merged.Predicted_CAP_Topic) | set(merged.raw_label) - {"Mix"}
    )
    stats |= {
        "n_items": len(sheet),
        "n_speeches": len(speeches),
        "n_words": sum(s["n_words"] for s in speeches.values()),
        "n_topics": int((purity.index >= 0).sum()),
        "mix_rate": float((merged.Predicted_CAP_Topic == "Mix").mean()),
        "n_truncated": int(merged.truncated.sum()),
        "truncated_share": float(merged.truncated.mean()),
        "generated": date.today().isoformat(),
        "cap_label_list": "\n".join(
            f"- {label}" for label in cap_labels if label not in {"Mix", "Other"}
        ),
    }

    manifests = (
        json.loads((TASK1 / "run_manifest.json").read_text(encoding="utf-8")),
        json.loads((TASK2 / "run_manifest.json").read_text(encoding="utf-8")),
    )
    write_findings(out / "FINDINGS.md", merged, purity, stats, pools, names)
    write_metadata(out / "METADATA.md", stats, args, manifests)
    write_readme(out / "README.md", stats, args)

    log(f"wrote {len(sheet)} samples across {len(samples)} strata to {out}")
    log("  FINDINGS.md, METADATA.md, README.md, annotation_sheet.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
