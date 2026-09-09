# Changes summary — Tasks 1 and 2

Supervised CAP classification (Task 1), two rounds of unsupervised topic
modeling (Task 2), and a question-time-versus-debate comparison across both.
Everything is on `setup/dev-environment`; nothing is merged and `main` has no
commits.

---

# Task 1 — supervised CAP classification

Unlike Task 2, **this specification held up**. The model exists, the 0.60 →
`Mix` rule is verbatim the authors' recommendation from the model card, and
`max_length=512, truncation=True` matches their own usage example. Two things
were measured rather than assumed:

- **The 400-word pre-truncation changes no prediction.** 400 Hungarian words is
  ~728 tokens, so the tokenizer's 512-token limit binds first either way.
  Verified: 61 of 120 sample speeches fit in 512 tokens, with and without it.
  It is kept — it does what it claims (keeps long strings out of the tokenizer)
  and costs nothing.
- **512 tokens is ~279 Hungarian words**, at a measured median of 1.82 subword
  tokens per word. 56% of speeches exceed it.

## What needs a human call (Task 1)

1. **No accuracy figure exists for Hungarian.** The card reports F1 for
   English (0.723), Croatian (0.686), Serbian (0.710) and Bosnian (0.646)
   only. Hungarian is among the model's languages and ParlaMint-HU among the
   29 training datasets, but nothing has been published for it. The `Mix` rate
   is the only available sanity check against the authors' 8.9–11.4%.

2. **Which pass to report.** `Predicted_CAP_Topic` (truncated, spec-compliant,
   comparable to the authors' figures) or `chunked_CAP_Topic` (whole speech,
   not comparable). Both are in the CSV with `passes_agree`.

3. **The 0.60 threshold is calibrated for single-pass scores.** Averaged
   window distributions are flatter, so the same threshold sends far more of
   the chunked pass to `Mix`. `chunked_raw_label` keeps the pre-override
   prediction; re-thresholding the chunked pass is a separate decision.

4. **Extra columns were added.** The spec asks for three (`Original_Text`,
   `Predicted_CAP_Topic`, `Confidence_Score`); those are present and first.
   `uid`, `faction`, `discourse_role`, `raw_label`, `n_tokens` and the chunked
   columns are additions — three anonymous columns cannot be joined to the
   corpus or to Task 2.

---

# Task 2 — unsupervised topic modeling

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

---

# Task 3 — parentheticals: lemmatised frequencies with fused n-grams

Branch `feat/parentheticals-frequency`. A frequency analysis of the 261,935
note-taker stage directions in `data/raw/parantheticals/`, cycles 39–43:
lemmatised with emtsv, significant bigrams and trigrams fused into single
`#`-joined tokens, counted per cycle and pooled, raw and relative.

The analysis itself is `data/derived/parentheticals/REPORT.md`. What follows is
what the code does and what a human still has to decide.

**New:** `src/parlamonitor/parentheticals.py` (loading, duplicate collapsing,
line statistics), `src/parlamonitor/frequency.py` (counting, tidy frames,
NPMI), `src/parlamonitor/propernouns.py` (party-name repair),
`scripts/parentheticals_freq.py`, plus three test modules.

**Changed, backwards-compatibly:** `emtsv.py` gains sentence-aware parsing and
batched line-aligned analysis (`Token` gains a defaulted `wsafter`; `parse_tsv`
is now a flattening of `parse_tsv_sentences`, same signature and behaviour).
`topics.py`'s `build_phrases` gains `scoring`, `delimiter` and
`connector_words`, all defaulting to today's behaviour — `make verify-model`
confirms the pinned topic model fingerprint is unchanged.

## What needs a human call (Task 3)

1. **Is the duplication real?** 21.1% of cycle 41's lines repeat the line before
   them, against 5–8% elsewhere. Collapsing consecutive runs removes 93% of
   `Bóna#Zolta#jelzés` (8,721 → 611) and 47% of `folyamatos#sípolás`
   (19,961 → 10,633). A procedural cue is not plausibly issued 8,721 times with
   93% of them back to back, so at least part of this is an extraction artifact
   — but it cannot be settled from these files, which carry no sitting, date or
   speech identifier to check against. Both readings are in every CSV
   (`raw` and `raw_collapsed`); **nothing was silently chosen.** Resolving it
   properly means re-exporting the parentheticals with a sitting identifier.

2. **Party-name repair is a four-entry list, not a gazetteer.** emMorph reads
   `Jobbik` as the comparative adjective *jobbik* and lemmatises it to `jó`,
   20,705 times. Four names are repaired; MP surnames that are also common
   words are not (`Bősz` → `bősz`, `Heringes` → `heringes`, `Borbély` →
   `borbély`). The counts are correct either way — the token is still one unit
   — but the display forms are wrong, and fixing them means sourcing a list of
   ~200 sitting MPs. Whether that is worth doing is a call about what the
   output is for.

3. **`Mi Hazánk` is not repaired.** It needs a rule spanning two tokens, which
   the form-keyed override cannot express. It surfaces as `mi#haza`.

## Decisions that change the numbers (Task 3)

All are parameters, all defaulted deliberately, all in
`data/derived/parentheticals/manifest.json`.

| decision | default | flag |
| --- | --- | --- |
| N-gram score | NPMI, threshold 0.5, `min_count` 5, two passes | `--scoring`, `--threshold`, `--min-count`, `--no-trigrams` |
| Phrase model scope | fitted once on all cycles pooled, as-is view | — |
| Connector words | articles/conjunctions inside a phrase but not at its edge | `--no-connector-words` |
| Proper-noun repair | on, four capitalised forms, 22,634 tokens | `--no-proper-noun-repair` |
| Soft hyphens | stripped (1,255, all cycle 40) | — |
| Punctuation | dropped before detection and counting (503,112 tokens) | — |
| Case | preserved as emtsv produced it | — |

NPMI rather than Gensim's default scorer because it is bounded in [-1, 1] and
so means the same on cycle 43 (23,007 tokens) as on cycle 41 (330,888); the
default scorer scales with vocabulary size and is not comparable across them.

## Where the toolchain did not hold

- **Gensim's NPMI scores leave [-1, 1] on the second pass** — 621 of 3,524 —
  because pass 2 scores against a corpus pass 1 has already fused. The n-gram
  table carries an `npmi` column recomputed from the unfused counts, with the
  component counts beside it, and keeps `gensim_score` only to show what the
  detector saw.
- **emtsv converts U+00AD to U+FFFD and tokenises around it**, so
  `hát<shy>oldalán` came back as `hát`, `<?>`, `oldalán`. Found because the
  batch reconstruction check failed on those lines. Stripping the soft hyphen
  is lossless and removed 357 spurious types from cycle 40.
- **Batching is only sound if it can be verified.** Responses are mapped back
  onto input lines via the newlines `tok` records in `wsafter`, and a batch
  that will not reconstruct its own request body character-for-character is
  re-sent one line at a time. Zero fallbacks on the final corpus.

## Left alone deliberately (Task 3)

- **`data/raw/` is untouched.** Soft-hyphen stripping happens on load, is
  versioned by `NORMALISATION_VERSION`, and is reported per cycle.
- **Four U+FFFD tokens remain**, from four stray C1 control characters in cycle
  40 (U+0084, U+0094, U+0096 ×2). Too rare to justify a rule; documented
  instead.
- **One delimiter collision.** Two cycle-40 lines quote hashtag badges
  (`#I stand with CEU`), so one lemma contains a literal `#`. Excluded from the
  n-gram table, warned about at run time, counted in the manifest. Changing the
  delimiter was not done — the task specifies `#`.
- **No stopword filtering of the counted stream.** Every lemma is counted and
  carries `pos_category`, `is_content` and `is_stopword`, so the content-word
  view is one filter away rather than baked in.

---

# Task 4 — reaction scores: who caused the laughter, applause and noise

Same branch. Classifies the parentheticals into reaction events and scores MPs
on the reactions their speeches drew. The analysis is
`data/derived/reactions/REPORT.md`.

**New:** `src/parlamonitor/reactions.py` (the taxonomy: kind, intensity,
audience, named interjector, per-cycle government mapping),
`scripts/reaction_scores.py`, `tests/test_reactions.py`.

Everything is rule-based and every rule is a named constant. The vocabulary was
read off the corpus frequency table from Task 3, not invented — a classifier
here would put a model's guess between the transcript and the count for no gain.

## What needs a human call (Task 4)

1. **MP-level scores exist for cycle 43 only.** The parentheticals files record
   who reacted, never who provoked it; linking a reaction to a speaker needs the
   speech around it, and only cycle 43 has a speeches export. Cycles 39–42 need
   `backend/export_nlp_datasets.py --period 39..42` before they can be scored.
   Corpus-level reaction counts and the 703-name heckler table already cover all
   five cycles.

2. **`derültség` is amusement, not humour.** In cycle 43 every opposition MP in
   the laughter top five draws most of their laughter from the *government*
   benches — the chamber laughing at them, not with them. The own-side /
   other-side columns make this visible but cannot resolve it; calling anyone
   "funniest" is an editorial judgement on top of these numbers, not a reading
   of them.

3. **Intensity weights are a judgement.** A standing ovation scores 3.0 and a
   scattered clap 0.5. Nothing in the transcript says so. Every weighted column
   has its unweighted count beside it so the weighting can be discarded.

## Decisions that change the numbers (Task 4)

| decision | default |
| --- | --- |
| Attribution | the floor-holder, except heckles, which go to their named interjector |
| Government mapping | Fidesz–KDNP for 39–42; TISZA for 43, derived from the export's ministerial offices |
| Bench of a factionless minister | government, inferred from holding executive office; recorded in `side_source` |
| Multi-kind events | `Derültség és taps` counts toward both, so per-kind totals exceed the event count by ~1.7% |
| Unattributed reactions | 11.5% name no bench; carried in `*_unattributed`, never assigned a side |
| Minimum speeches | 5, flagged not filtered — rate columns are noisy below it |

## Bugs found and fixed while building it

- **`re.IGNORECASE` on the interjection pattern made the capitalised-token
  classes match lowercase**, so `Közbeszólás az MSZP soraiból: Hazudik!` was
  read as a person named "Közbeszólás az MSZP soraiból" — 3,541 events across
  147 fake names. Case sensitivity is now scoped to the honorific, and a
  candidate matching a reaction or bench phrase is reclassified as that
  reaction with the quote kept.
- **A single capitalised word before a colon is procedure, not a person**:
  `Szünet: 14.19`, `Elnök: Igen.`, `Jelenlét-ellenőrzés: …` — 1,538 events. An
  interjector must now have a full name, which costs 5 genuine surname-only
  events across five cycles.
- **NaN is truthy**, so factionless speakers were indexing a
  `laughter_from_nan` column. They now get no side rather than a wrong one.

## Left alone deliberately (Task 4)

- **`Z. Kárpát Dániel` merges into `Kárpát Dániel`** — the leading initial reads
  as a title. One name in 703; fixing it means special-casing initials.
- **No sentiment or humour model.** The tables report what the note-taker wrote
  down. Deciding whether a laugh was with or at a speaker is left to the
  own-side/other-side columns and a reader.

---

# Task 5 — readability, affect and keywords

Same branch. 1,693 cycle-43 speeches scored for readability, lexical diversity,
keywords, sentiment and emotion. The analysis is
`data/derived/metrics/REPORT.md`.

**New:** `src/parlamonitor/readability.py` (saphes wrapper),
`keywords.py` (TextRank on networkx; KeyBERT behind the `metrics` extra),
`affect.py` (models, label calibration, valence), plus
`scripts/speech_metrics.py`, `scripts/speech_affect.py`,
`scripts/validate_affect.py` and three test modules.

## What needs a human call (Task 5)

1. **Big Five was requested and is not here.** There is no Hungarian Big Five
   text model. The one in the local HF cache, `Minej/bert-base-personality`, is
   `language: en` on `bert-base-uncased`; feeding it Hungarian produces numbers
   that are not a measurement. A hub search returns two models with zero
   downloads and no model card. Beyond the language problem, text-based Big
   Five inference correlates r ≈ 0.2–0.4 with self-report even in English, is
   trained on personal essays and social media, and parliamentary oratory is
   performative and often not written by the speaker — so the construct would
   not transfer even with perfect translation. Emitting `mp_big_five.csv` for
   named politicians would have been inventing data. **Decided: skipped.** The
   nearest defensible substitute is a stylistic profile from the columns that
   now exist.

2. **Two emotion channels are unusable and are flagged, not dropped.**
   `emotion_hu_fear` and `emotion_hu_sadness` fail corpus validation. They stay
   in the CSV because deleting a column hides the finding; `UNRELIABLE_CHANNELS`
   and `affect_validation.json` name them. Whether to drop them from any
   downstream analysis is a call for whoever uses the table.

3. **KeyBERT candidates are lemma pairs, not phrases.** Running it on raw text
   instead would give real quotable phrases at the cost of inflection
   scattering. Worth revisiting if the keywords are for display.

## Decisions that change the numbers (Task 5)

| decision | default | why |
| --- | --- | --- |
| LIX long-word threshold | **8** | `saphes.recommended_threshold("hu")`; LIX's usual 6 is Swedish |
| Word length | Hungarian letters | `cs gy ly ny sz ty zs dzs` are one letter each |
| Sentence source | emtsv | a regex splitter breaks on `dr.` and `2026.`, and LIX is words-per-sentence |
| Diversity | MATTR, 100-token window, content lemmas | TTR falls as text grows and would rank speakers by length |
| Readability reliability | flagged above 60 words/sentence | catches the two roll-call records at LIX 594 |
| Affect chunking | 400 subword tokens, mean over chunks | all three models cap at 512; 993 speeches needed more than one chunk |
| Sentiment valence | expectation over the ordinal scale | argmax would snap a split distribution to one end |
| Label mappings | derived at runtime by probing | no model declares its labels |

## Where the models did not hold

- **None of the three affect models declares its labels** — all ship
  `LABEL_0`…`LABEL_n`. The mapping is derived by probing at run time and the
  confusion matrix written to `affect_manifest.json`, so a model update changes
  the calibration instead of silently mislabelling a column.
- **Short probes are a weak reliability test, and were wrong in both
  directions.** They flagged `emotion_hu_anger`, which behaves fine on real
  speeches (r = −0.40 with valence, +0.42 with the other model's anger), and
  passed `emotion_hu_fear`, which fires on two thirds of the corpus while
  correlating with nothing. `validate_affect.py` now decides this on the scored
  corpus, and it caught `emotion_hu_sadness` having the wrong sign, which I had
  missed.
- **Cross-model agreement is modest**: anger r = +0.42, joy +0.25, fear +0.15,
  sadness +0.01. Two emotion models trained on different corpora barely agree
  about sadness at all.

## Left alone deliberately (Task 5)

- **No sentiment lexicon fallback.** Only transformer scores, so there is one
  provenance per number.
- **`emotion_hu_*` columns are kept** despite two failing — the validation file
  is the record, and silently dropping them would erase the finding.
- **The `metrics` extra is optional.** keybert reaches sentence-transformers
  and transformers reaches torch; the core install stays light, and TextRank
  needs neither.
