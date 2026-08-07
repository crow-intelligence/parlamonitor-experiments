# Changes summary — Task 2

Two rounds of unsupervised topic modeling over the 1,693 cycle-43 speeches,
plus a question-time-versus-debate comparison. Everything below is on
`setup/dev-environment`; nothing is merged and `main` has no commits.

## What needs a human call

1. **The topic names are a draft.** `data/labels/task2_topic_names.json` is
   hand-authored from the c-TF-IDF terms and representative documents, but
   `checked_by_human` is `false` on every row of the generated
   `topic_labels.csv`. Each carries a verbatim quote and the uid it came from,
   so checking one is a string match against `data/raw/`. Until that column
   says otherwise, the names are not findings.

2. **The light-verb stoplist is a judgement call.**
   `src/parlamonitor/stopwords.py`, `LIGHT_VERBS` — 47 lemmas. It moves every
   topic label. `tud`, `tesz` and `benyújt` are in it at your request;
   `szavaz`, `elutasít`, `elfogad`, `módosít` and `támogat` are deliberately
   **out**, because they mark what a speech is doing rather than what it is
   about. That boundary is arguable and the list is one editable constant.

3. **Topic 12 is residual boilerplate.** 37 short turns about microphones,
   timing and the chair. Salutation stripping cut the round-1 equivalent from
   66 to 37 but did not remove it. Fixing it properly means either a minimum
   word count or treating these as procedural, both of which change the corpus.

4. **`reduce_outliers` reassigns all 522 outliers to some topic.** Both columns
   are in `bertopic_documents.csv` (`topic`, `topic_reduced`); which one to
   report is your call. The raw one is more honest, the reduced one more usable.

## Decisions that change the numbers

| Parameter | Value | Where |
| --- | --- | --- |
| `min_df` | 5 documents (spec said 0.01 ≈ 17) | `--min-df` |
| `max_df` | 0.85 (as specified) | `--max-df` |
| POS kept | `N`, `Adj`, `V`; pronouns dropped | `--keep-pos` |
| Stoplist | spaCy hu (219) + light verbs (47) + address terms (14) | `--no-stoplist` |
| Salutation | stripped, ≤60 words, never emptying a speech | `--no-strip-salutation` |
| Gensim | `min_count=5`, `threshold=10` (as specified) | `--min-count`, `--threshold` |
| Chunking | 80 words, mean-pooled, L2-normalised | `--chunk-size` |
| HDBSCAN | `min_cluster_size=10` | `--min-cluster-size` |
| UMAP seed | 42 | `--seed` |
| MMR diversity | 0.3 | `--mmr-diversity` |
| Embedding source | `text_clean`, not the lemma bag | `--embed-source` |
| Keyness | log-likelihood, min 5 occurrences per corpus | `--min-freq` |
| RTD alpha | 1/3 | `--alpha` |

All of them are written to `data/derived/task2/run_manifest.json` with the
counts they produced, the emtsv image digest and the embedding model revision.

## Where the specification did not hold

Each was checked against the running service or the installed library, not
inferred.

- **`POST /api/run` with `{"text":…, "modules":["tok","lemma"]}` is not the
  emtsv contract.** The module chain is the URL path, the body is multipart,
  the response is TSV, and there is no `lemma` module — lemmas come from `pos`.
  Combined with the specified raw-text fallback this would have produced a
  fully unlemmatised corpus that still yields plausible topics. The client now
  raises and the manifest counts failures. Failures: 0 of 1,693.
- **`CountVectorizer.stop_words_` no longer exists.** Deprecated in
  scikit-learn 1.2, gone in the 1.9 installed here. `dynamic_stopwords()` fits
  twice and takes the difference, which is what the attribute used to hold.
- **The encoder truncates at 128 tokens** (~65–85 Hungarian words) against a
  307-word median speech. Documents are chunked and mean-pooled instead.
- **The Q&A export is not a separate corpus.** 622 of its 659 turns are
  speeches the speeches export already contains, so the comparison is a
  partition, not a concatenation.

## Left alone deliberately

- `data/raw/` is untouched. The two JSONL exports stay gitignored (32 MB,
  regenerable); the manifest and README are committed as the provenance record.
- `max_df=0.85`, `min_count=5`, `threshold=10` kept at the specified values.
- Speech-act verbs kept in the vocabulary (see above).
- No outlier tuning beyond exposing `min_cluster_size`; the defaults are
  reported rather than fitted.
- `main` does not exist as a ref. Creating it, and how, is your call.

## Numbers, round 1 → round 2

| | round 1 | round 2 |
| --- | --- | --- |
| Vocabulary surviving | 3,001 | 8,610 |
| Outliers | 616 / 1,693 (36.4%) | 522 / 1,692 (30.9%) |
| Topics | 30 | 28 |
| Boilerplate topic | 66 speeches | gone (37 residual, topic 12) |
| Light verbs in labels | `tud`, `mond`, `kell`, `beszél` | none |
