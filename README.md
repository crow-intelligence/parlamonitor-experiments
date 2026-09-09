# parlamonitor — analysis workspace

Measurement and visualisation of Hungarian parliamentary transcripts, cycle 43,
using the Crow packages. This is an analysis workspace, not a library for
release: `src/parlamonitor/` holds the loaders and metrics so notebooks and
scripts share one implementation instead of copy-pasting it.

The transcripts themselves come from the parlamonitor application's SQLite
database; nothing here writes to `data/raw/`.

## Setup

```bash
uv sync --all-extras
```

Python 3.12 (pinned in `.python-version`). `uv sync` installs the project
editable, which is what lets the loaders find `data/raw/` by repository layout.

## Data

`data/raw/` holds the cycle-43 exports. The two JSONL files are **gitignored**
— 32 MB and regenerable — so a fresh clone needs them copied in, or
`PARLAMONITOR_DATA` pointed at wherever they live:

| file | contents | in git |
| --- | --- | --- |
| `cycle43-speeches.jsonl` | 1,693 speeches, 826,775 cleaned words | no |
| `cycle43-qa.jsonl` | 215 question-answer exchanges (202 with text on both sides) | no |
| `cycle43-manifest.json` | row counts and what was filtered out | yes |
| `README.md` | field-by-field schema and normalisation rules | yes |
| `parantheticals/cycle{39..43}-parentheticals.txt` | 261,935 note-taker stage directions across five cycles | no |

The `parantheticals/` files (the directory is spelled as the export spells it)
are a different register from the speeches: one bracketed remark per line, no
metadata column of any kind, a mean of 6.1 words per line. They are the
editorial stage directions that `text_clean` strips out of the speeches —
applause, heckling, the chair's bell — collected instead of discarded. Task 3
counts them.

Rebuild them from the application:

```bash
cd backend
python3 export_nlp_datasets.py --period 43 --db parlamonitor.db --out ../exports
```

Read `data/raw/README.md` before measuring anything. The short version: use
`text_clean` or `sentences`, never `text` — the latter still carries the
speaker attribution and the editorial stage directions (applause, heckling,
the chair's bell).

```python
from parlamonitor import load_speeches, load_qa, provenance

speeches = load_speeches()            # unfiltered, 1,693 records
qa = [x for x in load_qa() if x["text_complete"]]   # 202 of 215
record = provenance()                 # pair this with any number you publish
```

The loaders deliberately do **not** filter. Anything that changes the numbers —
a minimum word count for length-sensitive lexical diversity, dropping the
incomplete Q&A pairs — is decided at the call site, where it is visible.

## Task 1 — supervised CAP classification

```bash
uv sync --extra topics
uv run python scripts/task1_classifier.py --limit 24   # smoke run first
uv run python scripts/task1_classifier.py              # ~2 h on CPU
uv run python scripts/task1_crosstab.py                # needs Task 2 too
```

`classla/ParlaCAP-Topic-Classifier` — XLM-RoBERTa-large, pre-trained on
parliamentary proceedings and fine-tuned on 29 ParlaMint 4.1 datasets — assigns
each speech one of the 21 CAP major topics or `Other`. Predictions below 0.60
confidence become `Mix`, which is the model authors' own rule, not a tuned one.

No emtsv and no GPU. Raw per-window scores cache to `cap_scores.jsonl`, so a
re-run or an interrupted run costs nothing already paid for.

### The truncation problem, and what is done about it

Hungarian costs a median **1.82 subword tokens per word**, so the model's 512
tokens are worth only about **279 words** against a 307-word median speech.
Two passes are run and both are written:

| column | what it saw |
| --- | --- |
| `Predicted_CAP_Topic` | the specification's single pass — the opening ~279 words |
| `chunked_CAP_Topic` | every 250-word window, distributions averaged by window length |

The truncated pass is kept as primary deliberately: it is what the model
authors did, so its `Mix` rate is comparable to their published 8.9–11.4%.
`passes_agree` records where the two differ, which is the measurement of what
truncation costs. Speeches fitting in one window agree by construction — a
useful correctness check.

**No accuracy figure can be quoted for Hungarian.** The card reports F1 for
English, Croatian, Serbian and Bosnian only. Hungarian is among the model's
languages and ParlaMint-HU among the training sets, but there is no published
Hungarian evaluation.

`Mix` is not `Other`. `Other` is the model confidently saying a speech fits no
CAP topic; `Mix` is our override when it was unsure. `raw_label` keeps the
pre-override prediction so the two stay separable.

## Task 2 — unsupervised topic modeling

```bash
uv sync --extra topics
docker run --rm -d --name emtsv -p 5000:5000 mtaril/emtsv
uv run python scripts/task2_bertopic.py --limit 60   # smoke run first
uv run python scripts/task2_bertopic.py              # ~40 min
uv run python scripts/task2_compare.py               # question time vs debate
uv run python scripts/task2_label_topics.py          # attach names + evidence
```

emtsv lemmatisation with a part-of-speech filter → Gensim bigram fusion →
frequency-derived stopwords → BERTopic over chunked, mean-pooled huBERT
embeddings. Outputs land in `data/derived/task2/`, next to a
`run_manifest.json` recording every parameter, the model revision, the emtsv
image digest, and the counts behind each stage.

The lemmatisation pass is cached to `lemmatized.jsonl` keyed by
`(uid, normalisation)` and embeddings to `embeddings-<hash>.npy`, so a re-run
only does what is missing, an interrupted run resumes, and a change of rule
invalidates rather than silently reuses.

### Question time versus debate

The Q&A export is **not** a separate corpus — 622 of its 659 turns are speeches
that `cycle43-speeches.jsonl` already contains. So the comparison is a
partition of one corpus by `discourse_role`, derived from `speech_type` and
cross-checked against the Q&A file's turn list (`src/parlamonitor/roles.py`):

| role | n |
| --- | --- |
| `debate` | 1,073 |
| `question` | 203 |
| `answer` | 202 |
| `mp_rejoinder` | 76 |
| `minister_rejoinder` | 75 |
| `reaction` | 64 |

`task2_compare.py` adds the lexical half with keyflux: log-likelihood keyness,
log ratio for effect size, rank-turbulence divergence and an allotaxonograph.

### The saved model

The fitted model is written twice, because neither format does both jobs:

| artifact | size | in git | restores |
| --- | --- | --- | --- |
| `models/task2_bertopic/` | 1.9 MB | **yes** | topics, c-TF-IDF, a pointer to the encoder |
| `data/derived/task2/model.pkl` | 452 MB | no | all of that **plus** the fitted UMAP and HDBSCAN |

```python
from bertopic import BERTopic
model = BERTopic.load("models/task2_bertopic")
model.get_topic(9)   # [('gazdálkodó', …), ('agrárkamara', …), …]
```

The safetensors copy is committed because it is what the 29 hand-authored topic
names actually refer to; without it those names depend on a re-run reproducing
exactly. **It does not carry UMAP or HDBSCAN** — `umap_model` and
`hdbscan_model` come back as `BaseDimensionalityReduction` and `BaseCluster`
placeholders. So `transform()` on it assigns new speeches by similarity to
topic embeddings, *not* by the HDBSCAN path the original run used. For that,
load the pickle, whose `hdbscan_model` still has its `prediction_data_`.

The pickle is gitignored and version-locked: BERTopic will not load a model
across library versions, so `run_manifest.json` records the versions of
`bertopic`, `umap-learn`, `hdbscan`, `scikit-learn`, `numpy` and `torch` it was
written under.

**The run is reproducible.** Re-running with both caches warm reproduces the
topic assignments bit-for-bit — 100.0000% identical on `topic` and
`topic_reduced`, zero drift in `probability`. That is what makes topic ids
safe to key the names file on.

### Topic names

`data/labels/task2_topic_names.json` holds hand-authored Hungarian and English
names for every topic; it is committed because hand-verified work is not
rebuildable. It records the **fingerprint** of the model it was written
against — a hash of the topic-id-to-terms mapping — and `make verify-model`
fails if the saved model no longer matches. Topic ids are positional, so
without that check a renumbering would leave every name quietly describing a
different topic. `task2_label_topics.py` joins it to the model output and attaches
a **verbatim quote** containing the topic's own top terms, so checking a name
is a string match rather than a reread. Every row carries
`checked_by_human=false` until someone says otherwise.

### Working with emtsv

The API contract is easy to get wrong, so, concretely:

```bash
curl -X POST http://127.0.0.1:5000/tok/morph/pos -F 'text=A kormány benyújtotta.'
```

The **module chain is the URL path**, not a JSON field. There is no `lemma` or
`lem` module — lemmas come out of `pos`. The response is **TSV**
(`form wsafter anas lemma xpostag`), not JSON. The `anas` column carries every
candidate analysis of every token and dwarfs the rest;
`parlamonitor.emtsv.parse_tsv` discards it.

### Where this departs from the task specification

| Spec says | What is done | Why |
| --- | --- | --- |
| `POST /api/run` with `{"text":…, "modules":["tok","lemma"]}` | `POST /tok/morph/pos`, multipart `text` field | that endpoint and that payload do not exist; the spec's version returns 500 for every speech |
| On API failure, return the raw text | Raise, record `ok: false`, exclude, report the count | the fallback plus the wrong endpoint would have yielded a fully unlemmatised corpus that still produces plausible topics |
| `CountVectorizer.stop_words_` | Fit twice, take the difference | deprecated in scikit-learn 1.2, gone in the 1.9 installed here |
| `fit_transform(phrased_speeches)` | Same documents for c-TF-IDF, but embeddings computed from `text_clean` | huBERT reads Hungarian, not lemma bags; topic *words* still come from the phrased text as specified |
| — | Chunk each speech into 80-word windows and mean-pool | the model ships `max_seq_length=128` (~65–85 Hungarian words) against a 307-word median speech |
| — | Content-word POS filter; seeded UMAP | tag filtering beats frequency thresholds, and unseeded UMAP makes runs unreproducible |

## Task 3 — parentheticals: lemmatised frequencies with fused n-grams

```bash
docker run --rm -d --name emtsv -p 5000:5000 mtaril/emtsv
uv run python scripts/parentheticals_freq.py --limit 2000   # smoke run
uv run python scripts/parentheticals_freq.py                # full corpus
```

Writes to `data/derived/parentheticals/`: `token_frequencies.csv` (one row per
cycle and token, plus `ALL` rows), a wide pivot of the same, an n-gram table,
per-cycle line statistics, the fitted phrase model, and a `manifest.json`.
**`REPORT.md` in that directory is the analysis**; what follows is how it is
built.

Every distinct line goes through emtsv **once** and is cached to
`lemma_cache.jsonl` — 63,849 of the 261,935 lines are distinct, so this is a
4× saving, and it makes a re-run with different thresholds take a minute rather
than half an hour. Lines are batched ~250 at a time and mapped back onto their
inputs using the newlines `tok` records in `wsafter`; a batch that will not
reconstruct its own request body is re-sent one line at a time rather than
trusted.

Significant bigrams and trigrams are fused into single `#`-joined tokens
(`taps#a#kormánypárt#sor`) by two Gensim passes scored with NPMI, fitted once on
all five cycles pooled so the per-cycle columns stay comparable.

### Decisions that change the numbers

| decision | default | flag |
| --- | --- | --- |
| Duplicate lines | counted both as-is and with consecutive runs collapsed, side by side | — |
| N-gram score | NPMI, threshold 0.5, `min_count` 5, two passes | `--scoring`, `--threshold`, `--min-count`, `--no-trigrams` |
| Connector words | articles and conjunctions may sit inside a phrase, not at its edge | `--no-connector-words` |
| Party names | `Jobbik`, `Momentum`, `Párbeszéd`, `Együtt` repaired — emMorph reads `Jobbik` as *jobbik* and lemmatises it to `jó` | `--no-proper-noun-repair` |
| Soft hyphens | stripped; emtsv turns U+00AD into U+FFFD and splits the word | — |
| Punctuation | dropped before detection and counting | — |

21% of cycle 41's lines repeat the line before them, against 5–8% elsewhere, and
collapsing removes 93% of `Bóna#Zolta#jelzés` and 47% of `folyamatos#sípolás`.
Prefer `raw_collapsed` when comparing cycle 41 with the others; `REPORT.md`
argues the case.

## Task 4 — reaction scores: laughter, applause, heckling

```bash
uv run python scripts/reaction_scores.py
```

Writes to `data/derived/reactions/`; **`REPORT.md` there is the analysis.**
Classifies 233,584 reaction events across cycles 39–43 by kind (applause,
laughter, heckling, whistling, noise, uproar, booing, the chair's bell),
intensity (`szórványos` → `felállva tapsolnak`) and which benches responded.

Attribution is limited by what the exports carry, so the tables split three ways:

| table | scope | why |
| --- | --- | --- |
| reaction events and summary | cycles 39–43 | the parentheticals say who *reacted* |
| `heckler_scores.csv` — 703 named interjectors | cycles 39–43 | the heckler's name is inside the parenthetical: `Vadai Ágnes: Nem hallom!` |
| `mp_reaction_scores.csv` | **cycle 43 only** | linking a reaction to the MP who caused it needs the speech around it, and only cycle 43 has a speeches export |

**The attribution rule:** a reaction is credited to whoever held the floor when
it was recorded. Right for applause and laughter, wrong for a heckle — so
heckles go to their named interjector and appear on the speaker's row only as
`heckles_received`.

MP scores come in four normalisations side by side (raw, per speech, per 1,000
words, per minute of floor time) because they rank people differently, and are
split by whether the reaction came from the speaker's **own** benches or the
**other** side. That split is the point: in cycle 43 every opposition MP in the
laughter top five draws most of their laughter from the government benches,
which is derision rather than wit. `derültség` records amusement, not humour.

Government/opposition flips between cycles — Fidesz–KDNP in 39–42, TISZA in 43
— so it is an explicit per-cycle mapping in `reactions.GOVERNING_PARTIES`,
derived for cycle 43 from the export's own ministerial offices rather than
assumed.

## Development

```bash
make ci      # format check, lint, type check, tests with coverage
```

Tests never read `data/raw/`, so they pass on a clean clone.

## Packages used

[`saphes`](https://github.com/crow-intelligence/saphes) (readability, lexical
diversity) · [`keyflux`](https://github.com/crow-intelligence/keyflux) (keyness,
rank-turbulence divergence, allotaxonograph) ·
[`kenon`](https://github.com/crow-intelligence/kenon) (semantic networks) ·
[`lexograph`](https://github.com/crow-intelligence/lexograph) (text
visualisation). `chronowords` sits behind the `diachronic` extra, for when the
archive cycles (39–42) are exported alongside 43.
