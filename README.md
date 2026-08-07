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

### Topic names

`data/labels/task2_topic_names.json` holds hand-authored Hungarian and English
names for every topic; it is committed because hand-verified work is not
rebuildable. `task2_label_topics.py` joins it to the model output and attaches
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
