# Metadata and provenance

Everything needed to reproduce, audit or dispute the material in this folder.

## Source data

| | |
| --- | --- |
| Corpus | Hungarian National Assembly, cycle 43 (current cycle) |
| Speeches | 1,693 |
| Words | 826,775 (cleaned) |
| Database snapshot | `2026-07-29T22:34:01.486174+00:00` |
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
| Model | [`classla/ParlaCAP-Topic-Classifier`](https://huggingface.co/classla/ParlaCAP-Topic-Classifier) |
| Revision | `82a13c61ff63b3450638e35f7b1b2cb9e6694ad6` |
| Architecture | XLM-RoBERTa-large, 24 layers, 560M parameters |
| Training | 29 ParlaMint 4.1 datasets, labelled by GPT-4o under the LLM teacher-student framework |
| Labels | 21 CAP major topics + `Other` |
| Confidence threshold | 0.6 → `Mix` (**the model authors' rule**, from the card) |
| Max length | 512 tokens, truncation on |
| Chunked pass | 250-word windows, length-weighted mean of label distributions |
| Hardware | CPU only, 4 threads |

**Accuracy on Hungarian is unknown.** The model card reports macro-F1 for
English (0.723), Croatian (0.686), Serbian (0.710) and Bosnian (0.646) only.
Hungarian is among the model's languages and ParlaMint-HU among its 29 training
datasets, but no Hungarian evaluation has been published. The one available
check is the `Mix` rate: **9.5%** here against the authors'
8.9–11.4% on their test sets. That is consistent, and it is not a substitute
for an evaluation. **This is the main reason the folder you are reading
exists.**

Hungarian costs a median 1.82 subword tokens per word, so 512 tokens
covers roughly 279 words against a 307-word median speech.
939 of 1,693 speeches
(55.5%) exceed it.

## Task 2 — unsupervised topic modeling

| | |
| --- | --- |
| Lemmatisation | emtsv `tok/morph/pos`, image `sha256:bb2e56806e00…` |
| POS kept | N, V, Adj, Adv; pronouns dropped |
| Stoplist | spaCy Hungarian + light verbs + address terms (278 lemmas) |
| Normalisation | opening salutation stripped (11,672 words) |
| Bigrams | Gensim `Phrases`, min_count 5, threshold 10.0 |
| Frequency filter | max_df 0.85, min_df 5 documents |
| Embeddings | `NYTK/sentence-transformers-experimental-hubert-hungarian`, revision `4bc0656f1d37…` |
| Chunking | 80-word windows, mean over chunks, L2-normalised |
| Clustering | UMAP (seed 42) + HDBSCAN (min_cluster_size 10) |
| Topics found | 28 + 522 unclustered |

Topic names in `data/labels/task2_topic_names.json` are **hand-authored from
the top terms and representative documents, and not yet checked by a human**.
Each carries a verbatim quote so verification is a string match.

## This annotation pack

| | |
| --- | --- |
| Generated | 2026-08-14 |
| Generator | `scripts/make_annotation_pack.py` |
| Sample seed | 42 |
| Per stratum | 12 |
| Strata | 5 |
| Items | 60 |
| Quote cap | 1200 words |

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

Analysis and this pack: **CC BY-NC-SA 4.0**.

Parliamentary transcripts are public records of the Hungarian National
Assembly. The emtsv toolchain, `ParlaCAP-Topic-Classifier`, ParlaMint and the
CAP codebook each carry their own terms — check them before redistributing
anything derived from them. The CAP master codebook is at
<https://www.comparativeagendas.net/pages/master-codebook>.

Contact: hello@crowintelligence.org
