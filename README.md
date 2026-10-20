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
| Duplicate lines | counted three ways side by side: as-is (`raw`), consecutive runs collapsed (`raw_collapsed`), copied blocks removed (`raw_dedup`) | `--block-window` (default 10 runs) |
| N-gram score | NPMI, threshold 0.5, `min_count` 5, two passes | `--scoring`, `--threshold`, `--min-count`, `--no-trigrams` |
| Connector words | articles and conjunctions may sit inside a phrase, not at its edge | `--no-connector-words` |
| Party names | `Jobbik`, `Momentum`, `Párbeszéd`, `Együtt` repaired — emMorph reads `Jobbik` as *jobbik* and lemmatises it to `jó` | `--no-proper-noun-repair` |
| Soft hyphens | stripped; emtsv turns U+00AD into U+FFFD and splits the word | — |
| Punctuation | dropped before detection and counting | — |

**Use `raw_dedup` when comparing cycle 41 or 42 with any other cycle.**

- **What dedup removes.** 54.5% of cycle 41's lines and 23.5% of cycle 42's are
  verbatim copies of earlier blocks of the same file. The 2018 protest block
  alone is written out 176 times.
- **What it finds elsewhere.** It flags nothing in cycles 39 and 43, and 10
  lines in cycle 40.
- **What it does to the anomalies.** `folyamatos#sípolás` falls from 19,961
  to 122 in cycle 41, and `Bóna#Zolta#jelzés` from 8,721 to 81.
- **Why not `raw_collapsed`.** `REPORT.md` recommends it, but it only removes
  adjacent repeats and leaves the copies in place. That report predates the
  finding. See `notebooks/01_parentheticals_data_repairs.ipynb`.

The wide table carries `dedup_cNN` / `rel_dedup_cNN` next to the as-is pairs.

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

### Who heckled whom — the directed network

```bash
uv run python scripts/heckle_network.py
```

An edge runs from the interjector named inside a parenthetical to whoever held
the floor. Cycle 43: 1,115 named interjections, 1,110 (99.6%) resolved to a known
speaker, 112 nodes, 364 edges. Three are self-loops (a speaker's own name inside
their speech), which are dropped from the graph. Both ends carry a faction, so the
graph rolls up to party or to government/opposition.

`data/derived/reactions/NETWORK.md` documents the outputs and the d3 contract.
The short version: `heckle_edges.csv` is the atomic record, and
`heckle_network.json` is written with `edges="links"` because networkx 3.6
defaults that key to `"edges"`, which `d3.forceSimulation` does not read. Party-
and side-level roll-ups ship as their own node-link files; GraphML is there for
Gephi.

Heckling is overwhelmingly cross-bench: 959 of the 1,110 interruptions where both
sides are known go from one side of the house to the other. Two limits are
structural — the network is cycle 43 only, because an edge needs a floor-holder
and that is the only cycle with a speeches export; and 17% of that cycle's named
interjections (222 of 1,337) happen outside any speech (voting, procedure, the
opening ceremony), so they have no target and are not in the graph. The
manifest computes this share; it is no longer a fixed sentence.

## Task 5 — readability, affect and keywords

```bash
uv sync --extra metrics
docker run --rm -d --name emtsv -p 5000:5000 mtaril/emtsv
uv run python scripts/speech_metrics.py    # readability, diversity, keywords
uv run python scripts/speech_affect.py     # sentiment, emotion
uv run python scripts/validate_affect.py   # does the affect actually work?
```

`data/derived/metrics/REPORT.md` is the analysis. 1,693 cycle-43 speeches.

**Syntactic complexity** via `saphes`: mean dependency distance (MDD) and mean
hierarchical distance (MHD), Jing & Liu (2015). These need a *parse*, and the
parser is **HuSpaCy** (`hu_core_news_md`), not emtsv — emtsv can parse, but its
cost grows super-linearly with document length (80 words in 0.4 s, 633 in 14 s,
a 3,000-word request not returning inside 100 s), which on this corpus is hours.
HuSpaCy runs in-process at ~2,000 words/s, linearly, so the corpus takes ~7
minutes and the 5,281-word longest speech needs no chunking.

MDD and MHD correlate at only 0.56, which is why the pair is reported rather
than either alone. **The parser is part of the measurement** — head conventions
decide every distance — so `parser` is recorded on every result, and a HuSpaCy
MDD must not be compared with an emtsv one.

**Loanword ratio** (*idegenszó-arány*) via `saphes`, against a 15,203-lemma list
verified against Bakos Ferenc's dictionary. The lexicon is a required argument
with no default — saphes ships none, deliberately — so it is read by path from
the sibling `saphes` checkout and recorded by hash, not copied into this repo.
Two properties shape the number: the list drops the top 2% by corpus frequency
as ordinary vocabulary, so this measures *uncommon* foreign words; and proper
nouns are excluded via a capitalisation proxy on emtsv lemmas, without which
every foreign surname would count.

**Readability** via `saphes`, with three non-default choices: the long-word
threshold is **8** (`recommended_threshold("hu")`, not LIX's Swedish 6), length
is counted in **letters** not characters (`asszony` is 7 characters but 5
letters), and sentences come from **emtsv** rather than a punctuation regex.
LIX mean 39.7 at 17.2 words per sentence. Two `jegyzői ismertetés` records —
the notary's roll-call, 590 words with no full stop, LIX 594 — are flagged
`readability_reliable = false` and excluded from per-MP means.

**Keywords**, two methods: TextRank over a co-occurrence graph of content
lemmas (networkx, no model), and KeyBERT over huBERT embeddings. KeyBERT's
candidates come from the lemma stream, so a returned bigram is two content
words adjacent after function words were dropped — themes, not quotations.

**Sentiment and emotion** from three models, **none of which declares its
labels**. The script therefore calibrates before it scores, deriving the
index-to-label mapping from probes of known valence and writing the confusion
matrix into the manifest.

Probes turned out to be a weak test in both directions, so reliability is
decided on the corpus by `validate_affect.py`, which checks valence agreement,
cross-model agreement and saturation. It flags **`emotion_hu_fear`**
(mean 0.64, r = −0.04 with valence — saturated and signal-free) and
**`emotion_hu_sadness`** (r = **+0.11**, the wrong sign). Use the
`emotion_xlm_*` columns. Sentiment itself tracks the aisle cleanly: valence
runs +0.083 for TISZA (government) down to −0.095 for Mi Hazánk.

**Big Five is deliberately absent.** No Hungarian model exists, and the cached
`Minej/bert-base-personality` is English-only on `bert-base-uncased`. See
`CHANGES_SUMMARY.md`.

## Exploratory notebooks — parentheticals, round 2

`notebooks/01_parentheticals_data_repairs.ipynb`, `02_parentheticals_procedural.ipynb`
and `03_parentheticals_heckle_content.ipynb` are committed with their outputs and
re-run with:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Notebook 03 needs the emtsv container on its first run only; its quote
lemmas are cached to `data/derived/explore/`. The main finding concerns
cycles 41 and 42: more than half of cycle 41's parentheticals, and nearly a
quarter of cycle 42's, are verbatim copies of earlier blocks. See
`CHANGES_SUMMARY.md`, Task 7, before comparing those cycles with any other.

## Heckle content — what the hecklers say

```bash
cp data/derived/explore/quote_lemma_cache.jsonl data/derived/heckles/   # optional: skip emtsv
uv run python scripts/heckle_content.py
```

Reads the parentheticals with copied blocks removed (`block_dedup`) and pulls
out every quoted heckle whole (`parlamonitor.heckles.quotes_in`), including the
hyphenated-surname interjectors that the Task 4 classifier misses. It
attributes each heckle to a bench where the note, or the cycle-43 speeches,
settles it. Then it compares four groups' content vocabulary with
rank-turbulence divergence:

- TISZA, governing (43)
- Fidesz–KDNP in opposition (43)
- Fidesz–KDNP, governing (42)
- the opposition (42)

Writes `data/derived/heckles/`:
- `quotes.csv`
- `group_counts.csv`
- `allotaxonographs.json` (for the dashboard)
- `manifest.json`, which includes a 200-draw bootstrap of the role-flip
  ordering.

| decision | default | flag |
| --- | --- | --- |
| View | block-dedup, window 10 runs | `--block-window` |
| Vocabulary | content words (N, V, Adj, Adv; no pronouns) | — |
| Stopwords | spaCy hu + light verbs + address terms + 4 repaired NLTK words | `--no-light-verbs` keeps *van*, *mond* … |
| RTD α | 1/3 | `--alpha` |
| Bootstrap | 200 draws, seed 0 | `--bootstrap`, `--seed` |

**Cycle-42 groups contain unnamed bench heckles only**: the files name an
interjector but never their party, and only cycle 43 has a speeches export.
The dashboard states this next to the chart.

## Speech vocabulary — who speaks like whom

```bash
uv run python scripts/speech_vocabulary.py     # needs speech_metrics.py's lemma cache
```

Counts content lemmas in every cycle-43 speech, using the same filter and
stoplist as the heckles plus the honorific `dr.`. It counts per speaker (160
with at least 300 content words), per faction, for Fidesz–KDNP together, and
for the government and opposition sides.

It writes counts, not comparisons. With 160 speakers there are about 13,000
possible pairs, so the dashboard computes rank-turbulence divergence in the
browser with `dashboard/rtd.js`, a port of keyflux's `rtd`.
`tests/test_dashboard_rtd.py` holds it to keyflux on 105 random and edge-case
inputs (it needs node, and is skipped without it). The manifest records
keyflux's value for four reference pairs:

| pair | RTD |
| --- | ---: |
| TISZA ↔ Fidesz | 0.358 |
| government ↔ opposition | 0.350 |
| Fidesz ↔ KDNP | 0.392 |
| Fidesz–KDNP ↔ Mi Hazánk | 0.398 |

Speeches differ about half as much as heckles do: those comparisons run at
around 0.7.

| decision | default | flag |
| --- | --- | --- |
| Speakers offered | ≥ 300 content words (median speaker: 738) | `--min-tokens` |
| Stopwords | as for heckles, plus `dr.` | `--no-light-verbs` |
| RTD α | 1/3 | `--alpha` |

Known limitation: surnames that are also common words lemmatise in lowercase
(`gulyás` for Gulyás Gergely, `jános`). The counts are right; the display is
not. The fix is the corpus-attested name repair prototyped in notebook 01,
which has not been promoted yet.

## Applause lines — what earns applause, laughter and heckling

```bash
uv run python scripts/applause_lines.py      # needs speech_metrics.py's lemma cache
```

The cycle-43 speech `text` keeps its 4,597 stage directions inline, each one
right after the words that drew it. `parlamonitor.applause` puts every reaction
back on that sentence, in the export's own segmentation:
- 4,577 are placed (99.6%) and 20 are reported as unaligned, never guessed;
- 914 interrupt their sentence.

Every trigger is asserted to be a verbatim substring of `text_clean`.

Writes `data/derived/applause/`:
- per-reaction lines with verbatim triggers;
- a sentence table;
- line rates per speaker, faction and speech type;
- rhetorical features against "quiet" sentences, each with both counts and a
  log-likelihood band;
- word counts per group (`lines.json`, for the dashboard);
- a seeded `hand_check.csv` with `checked_by_human=false`.

| decision | default | flag |
| --- | --- | --- |
| Closing ritual | after the final sentence, or ritual thanks (*Köszönöm a figyelmet*, a bare *Köszönöm szépen.*): never an applause line | — |
| Applause line | own-side or unattributed applause | — |
| Laughter | "with" if from the speaker's own side, "at" if from the other | — |
| Baseline | sentences followed by no reaction, **salutations excluded** | `--keep-salutations` |
| Speaker rates | flagged below 100 sentences, not dropped | `--min-sentences` |

**The salutation exclusion matters.** There are 3,228 salutations such as
*Tisztelt Ház!*. None is ever applauded, and they end in "!". Left in the
baseline, they made exclamations look *less* likely before applause (0.81×).
With them out, exclamations are 2.26 times as likely.

The export's cleaned sentences still hold 714 stage directions, 1.6% of
sentences, against the "about 3" its README claims. `without_directions`
strips them before features and vocabulary, so a heckler's words are never
counted as the speaker's.

## Élő parlament — a sitting day as a conversation (experimental)

```bash
uv run python scripts/living_parliament.py
```

Writes one JSON per cycle-43 sitting day (19 days, about 8 MB together) to
`data/derived/living/`. Each holds every speech (start, length, speaker,
faction, side, sentences) and every reaction placed on its sentence. The rules
live in `parlamonitor.living`:

- **Time.** Speech start plus length × the share of the speech's words said by
  the reaction. This is an estimate; the export has no finer timing.
- **Benches.** The note's own attribution comes first, then a named
  interjector's side. A note naming neither is left unattributed, not
  guessed.
- **Flow.** "Own" is the speaker's benches, or a bare applause or laughter.
  "Other" is the other benches, or a bare heckle or noise. The bell, the
  closing ritual and reactions from both sides are neither.

The per-day counts are asserted against `data/derived/applause/reaction_lines.csv`.

**Hostile heckles** are heckles whose lemmas include a word from
`data/labels/hostile_heckle_words.json`: 58 accusing, shaming or insulting
lemmas (*hazudik, szégyen, tolvaj, farizeus…*).
- Every entry is checked against real heckles, with its counts and an example.
- The file records three exclusions and their reasons.
- It is `checked_by_human: true`, and `load_hostile_words` refuses a list that
  isn't.

In cycle 43 the list flags 61 heckles from the other side. The quotes are
lemmatised from a cache seeded by the heckle-content cache; emtsv is needed
only for new quotes.

The tab has two parts:
- **a seismograph of the day, drawn as labelled lanes like a score.** Speeches
  sit in the middle as bands in faction colours. Above them are the speaker's
  own side's responses: supportive interjections, laughing with, applause.
  Below are the other side's answers: heckles, **hostile heckles**, laughing
  at, and **the chamber's noise** (*zaj, moraj, felzúdulás*). Each lane has its
  own scale.
- **a living transcript of the chosen speech:** reactions in the margin of the
  benches they came from, and a ⟋ where a heckle broke in.

## The dashboard

```bash
uv run python scripts/heckle_content.py        # optional: the allotaxonograph section
uv run python scripts/speech_vocabulary.py     # optional: the Szókincs tab
uv run python scripts/applause_lines.py        # optional: the Taps és derültség tab
uv run python scripts/living_parliament.py     # optional: the Élő parlament tab
uv run python scripts/build_dashboard_data.py
uv run python scripts/serve_dashboard.py         # or: make serve (no-cache, so reloads see updates)
```

The "Ki kit szakított félbe" view also holds the heckle **allotaxonograph**. It
is a Dodds-style rank diamond with a word-shift chart, switchable between the
three role-flip comparisons. Its cells use a one-hue maroon ramp, validated as
an ordinal ramp in light and dark. The two sides take the series colours **by
position**, not by party, because Fidesz–KDNP sits on both sides of one
comparison. If `heckle_content.py` has not been run, the section is left out.

The **Szókincs** tab uses the same chart for speeches. Pick any two of: a side,
Fidesz–KDNP, a faction, or a speaker. The pair goes into the URL
(`?va=…&vb=…#vocab`), so a comparison can be linked. Its data is a separate
`data/vocabulary.json` (1.7 MB, 0.45 MB gzipped), fetched only when the tab is
first opened.

Published at <https://crowintelligence.org/parlamonitor-dashboard/> as a
byte-for-byte mirror of `dashboard/`. Run `make check-site` to see the drift and
`make sync-site` to mirror it into the site repo, then commit there through a
PR. All site integration (meta, JSON-LD, site bar, consent) lives in
`dashboard/index.html`, never in the copy. See `dashboard/README.md`.

Static page, four views: an MP/party profile, a topic x metric heatmap, the
interruption network, and a filterable table of all 1,693 speeches.
`dashboard/README.md` covers the form and colour decisions; the short version:

**No radar chart.** Radar encodes magnitude as radius, so area grows as the
square and a 2x score reads as 4x, and its axis order is arbitrary. The profile
view uses percentile strips instead - every eligible MP as a faint dot, the
selected one filled, the party median as a tick - because "is this MP unusual"
is a comparison against a population, so the population is drawn.

The categorical palette is validated for both light and dark
(`validate_palette.js`, all checks pass). It is deliberately **not** party brand
colours: those hex values are not reliably known here, and a wrong one in a
political dashboard is a factual error rather than a styling choice.

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
