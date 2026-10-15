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

---

# Task 6 — syntactic complexity and loanword ratio

Same branch. Two saphes measures that were missed the first time round, plus a
parser change.

**New:** `src/parlamonitor/syntax.py` (MDD/MHD via saphes, over a HuSpaCy or an
emtsv parse), `src/parlamonitor/loanwords.py` (idegenszó-arány),
`scripts/speech_syntax.py`, two test modules.

## Why these were missed

`loanword_ratio`, `mean_dependency_distance` and `mean_hierarchical_distance`
are in saphes **0.2.1**; the installed version was **0.1.0**, which has none of
them. The API was checked by introspecting the installed package rather than by
reading the documentation, so the three did not appear. The lesson is small and
concrete: introspection tells you what you have, not what exists.

## What needs a human call (Task 6)

1. **The loanword lexicon is not in this repo, by choice.** It is read from
   `../saphes/experiments/loanwords/results/idegenszavak.txt` and recorded by
   sha256. saphes' own study leaves its licence open — it is derived from a
   webcorpus and verified against a copyrighted dictionary — so committing it
   here would settle a question that study deliberately left open. A checkout
   without the sibling repo gets a null column and a printed warning, not a
   silent zero.

2. **Proper nouns are excluded by a capitalisation proxy, not by NER.** emtsv
   lowercases the lemma of a sentence-initial common noun and leaves a proper
   noun's case alone, so a capitalised lemma is a usable signal. It will
   misfile a genuinely capitalised common noun. The alternative — counting
   every foreign surname as a loan word — is a larger and more systematic
   error in a parliamentary corpus.

3. **MDD is parser-dependent and the two parsers are not interchangeable.**
   HuSpaCy follows Universal Dependencies; emtsv's `dep` uses its own scheme.
   Head conventions decide every distance. `parser` is on every result and the
   manifest says not to mix them.

## Decisions that change the numbers (Task 6)

| decision | default | why |
| --- | --- | --- |
| Parser | HuSpaCy `hu_core_news_md` | emtsv's cost is super-linear in document length; 3,000 words does not return inside 100 s |
| `min_sentence_length` | 3 | Jing & Liu's; this corpus is full of `Köszönöm.` whose MDD is a degenerate 1 |
| `max_sentence_length` | 120 | measured, not guessed: median 15, p99.9 102, and the roll-call's "sentence" is 517 |
| Cross-sentence heads | made local roots, counted | dropping the documents would bias the corpus toward what the parser found easy |
| Punctuation | `collapse` | the policy the literature uses |
| MP aggregation | token-weighted | a two-sentence intervention should not weigh as much as a twenty-minute address |
| Loanword unit | content lemmas, proper nouns excluded | surface forms miss the lexicon and return a plausible, low ratio with no error |

## What surfaced

- **saphes refused the first parse, correctly.** HuSpaCy runs `senter` before
  `parser`, so sentence boundaries and heads can disagree, and `from_spacy`
  raises rather than measuring the disagreement. Excluding `senter` does not
  help — the parser itself emits cross-boundary heads. At corpus scale it is
  2 tokens in 826,775.
- **The roll-call broke a second metric.** It already had `readability_reliable
  = false` for LIX at 594; unfiltered it also scored MHD 26.3 against a median
  of 2.4, which flattened all 88 speakers onto one edge of the dashboard's
  percentile strip. Same record, same cause, and now caught by a measured
  sentence-length cap.
- **Two independent measures agree.** Bencsik János is top on LIX (52.2) and on
  MDD (2.86). One counts long words in long sentences, the other how far a word
  sits from its governor; nothing mechanical connects them.
- **`huspacy.download()` silently no-ops under uv.** It shells out to
  `python -m pip`, which a uv venv does not have, and returns 0 anyway. The
  model wheel is also misnamed (`-any-` where a version belongs), so uv rejects
  it by URL; it has to be renamed and installed from a local file.

---

# Task 7 — parentheticals, round 2: exploratory notebooks

Branch `feat/parentheticals-explore`. Three notebooks, executed and committed
with their outputs, plus one small bug fix that was blocking CI. Nothing in the
Task 3/4 pipeline, its outputs or the dashboard was changed. Every finding
below is written up for a decision, not applied.

**New:**
- `notebooks/01_parentheticals_data_repairs.ipynb`: is the duplication real,
  and can names be repaired from the corpus itself?
- `notebooks/02_parentheticals_procedural.ipynb`: what the 29% of segments
  that Task 4 calls `OTHER` actually are.
- `notebooks/03_parentheticals_heckle_content.ipynb`: what the hecklers say.

**Changed:** `interactions.normalise_name` now casefolds *after* NFKD.
Hypothesis found `𝙰` (U+1D670), which survives `casefold()` and decomposes to
an uppercase `A`. Checked against all 746 names in the corpus and the cycle-43
registry: no key changes, so no published output moves.

Scratch outputs go to `data/derived/explore/`, which is gitignored:
- `name_gazetteer_draft.csv`;
- `quote_lemma_cache.jsonl`, the emtsv output for 40,881 distinct quotes, so a
  re-run needs no service.

## The headline: cycles 41 and 42 are partly copies

- **The files contain verbatim block copies.** 54.5% of cycle 41's lines and
  23.5% of cycle 42's are copies of earlier stretches of the same file. The 12
  December 2018 protest block (173 lines) is written out 176 times.
  Cycles 39, 40 and 43 have none.
- **This resolves the Task 3 question, and the answer is bigger than Task 3
  measured.** `Bóna Zoltán jelzésére:` drops from 8,721 to 81 and
  `Folyamatos sípolás.` from 15,840 to 90. Cycle 41's adjacent-repeat rate falls
  from 21.1% to 7.3%, inside the range of the other cycles. `raw_collapsed`
  does not remove block copies, so the advice in the Task 3 report to "use
  `raw_collapsed` for cycle 41" is not enough.
- **The exporter itself does not invent adjacent repeats.** In cycle 43, 92% of
  consecutive repeats are matched to two separate stage directions in the
  speeches' verbatim `text`. The copies therefore sit in the database rows
  behind cycles 41–42. Finding out why needs the application repo, which is
  not on this machine.

How much it moves the published numbers (per 1,000 lines, cycle 41, as-is →
block-dedup):

| | heckling | whistling | applause | bell |
| --- | ---: | ---: | ---: | ---: |
| as-is (= `reaction_summary.csv`) | 110.3 | 208.8 | 214.4 | 43.9 |
| block-dedup | 133.7 | 3.0 | 388.8 | 82.7 |

The cycle-41 heckler ranking reorders. Bana Tibor drops from 2nd to 33rd
(1,987 → 62) and Varga László from 5th to 37th. **Cycle 43 does not move at
all**, so the dashboard and every MP-level score are unaffected.

## What needs a human call (Task 7)

1. **Which view is the default for cycles 41–42?** I'd recommend block-dedup
   for every cross-cycle comparison and regenerating the Task 3 and Task 4
   tables with a third `raw_dedup` column, keeping as-is for provenance. That
   changes published numbers, so it was not done here.
2. **Fix three `reactions.py` bugs** that change published counts. Each should
   be its own PR with before/after tables:
   - **Hyphenated surnames are never parsed as interjectors.** 1,038
     interjections by 22 MPs are lost: Varga-Damm Andrea 477, Gaudi-Nagy
     Tamás 243, Sebián-Petrovszki László 211. That includes 24 in cycle 43,
     where Kátai-Németh Vilmos is missing from the heckle network.
   - **`Event.quote` keeps only the first sentence.** About 10% of quotes are
     truncated, and the rest falls into `OTHER`. Counts are unaffected; the
     `quote` column is not.
   - **Two-letter initials** (`Gy.`, and in principle `Zs.`, `Sz.`, `Cs.`) are
     split off as sentences. So is a middle initial (`Sallai R. Benedek:`),
     which fails to parse at all.
3. **The Task 4 `REPORT.md` heckling row disagrees with its own CSV.** Cycle
   39 is 96.0 in the report against 117.3 in `reaction_summary.csv`. Whistling,
   laughter and bell agree. It looks stale after a later classifier change.
   Regenerate the report from the CSV.
4. **Name repair from the corpus.** Full name sequences attested in the corpus
   (interjectors plus cycle-43 speakers) fix 29,085 lemmas, such as `Zolta` →
   `Zoltán` and `rétvári` → `Rétvári`. The repair leaves `Nagy taps` as the
   adjective and changes the display form of 478 of the 3,524 Task 3 n-grams.
   The draft table has `checked_by_human = false`. Promotion means a reviewed
   copy in `data/labels/` and the code in `propernouns.py`.
5. **`Mi Hazánk` merge.** It is a clean two-token rule: 920 tokens, and the
   single lowercase *mi hazánk* is untouched. Should the lemma be `Mi Hazánk`
   (with a space) or a `#`-free alternative?
6. **Party data for cycles 39–42 named interjectors.** None exists in these
   files. Every bench comparison of heckle *content* for those cycles therefore
   rests on unnamed heckles only, and is underpowered. Speeches exports for
   39–42 would fix this and also unlock MP-level reaction scores.
7. **Emotion channels on heckles.** In this register, *sadness* is 80% *jaj* /
   *baj* and *disgust* is 96% *szégyell* + *ronda*. Report them as what they
   measure here, or drop them for heckles.
8. **Stoplist for heckles.** The role-flip analysis now removes stopwords: the
   project's Task 2 list plus the four words NLTK's Hungarian list genuinely
   adds. NLTK's other extras are `õ`-for-`ő` encoding errors of words spaCy
   already has. This cleans the allotaxonograph down to content: *Orbán*,
   *infláció*, *megszavaz* against *Magyar Péter*, *migráns*, *bocsánat*. It
   also deletes *van* (*így van*), which was the one bench contrast that held
   across cycles 39–42. Whether light verbs stay in the list for heckles is a
   judgement call.
9. **Textual parentheses** (place names, `EU`, county names) are in the
   parentheticals files as if they were stage directions. Filter them on load,
   or leave them for the exporter.

## Decisions that change the numbers (Task 7)

| decision | default | where |
| --- | --- | --- |
| Block detector | run-length encoded, window of 10 runs | nb 01 §2; sensitivity table there |
| Analysis view | block-dedup throughout nb 02–03 | parameter cell |
| Chamber-size bound | 386 for cycle 39, 199 after | nb 02 `PARAMS["seats"]` |
| Missing tally fields | `null`, never 0 | nb 02 §2b |
| Stock-heckle identity | casefold + strip punctuation | nb 03 |
| Keyness | log-likelihood, min freq 5, band ≥ p001 (LL ≥ 10.83), content words only | nb 03 |
| Stopwords (keyness, RTD, allotaxonograph) | `hungarian_stopwords()` + 4 NLTK words (`első`, `elő`, `először`, `ők`) | nb 03 §4 |
| RTD | α = 1/3; bootstrap 200 draws, seed 0 | nb 03 §6 |
| Affect | pooled per bench × cycle; unbalanced Precognox; emotion `min_hits` 5 | nb 03 §7 |

## Where my own first pass was wrong

An earlier draft of notebook 01 used a line-level window that skipped
windows of a single repeated line. That left identical runs inside the copies
in place. Once the copies were deleted, those leftovers joined into a "4,925-line
run of `Bóna Zoltán jelzésére:`" that does not exist in the file. Its longest
real run is 45. The run-aware detector replaced it, and notebook 01 keeps the
draft detector beside it in the sensitivity table so the mistake stays
visible.

## Left alone deliberately (Task 7)

- **No change to `src/` beyond the `normalise_name` fix.** All the prototypes
  (`block_copy_mask`, `repair_names`, `merge_mi_hazank`, the procedural
  extractors, `quotes_in`) stay in the notebooks until the decisions above are
  made. Their promotion targets are listed at the end of each notebook.
- **No bench is inferred for a named cycle-39–42 heckler** from the benches
  mentioned around them. That would be inventing data.
- **Emotion lexicon entries are not reproduced.** Only the one or two entries
  that dominate a channel are named, as `lexicon.py` already does, and Putz
  Orsolya is credited in notebook 03.

---

# Task 8 — a block-dedup view in the parentheticals frequency table

Branch `feat/freq-dedup-view`, stacked on `feat/parentheticals-explore`.

**New:** `parentheticals.block_copy_mask` and `parentheticals.block_dedup`,
promoted from notebook 01. The detector run-length encodes the lines and flags
any window of 10 runs that already occurred earlier. It has doctests and
Hypothesis properties:
- the output is an order-preserving subsequence;
- every distinct line keeps its first occurrence;
- a sequence with no repeated window is untouched;
- a distinct block written twice comes back once;
- a single-formula streak is never a copy.

**Changed: `scripts/parentheticals_freq.py`.**
- **A third view.** It counts `raw_dedup` (plus `relative_per_million_dedup`,
  `doc_count_dedup` and the totals) next to `raw` and `raw_collapsed`, in
  `token_frequencies.csv`. The wide table gains `dedup_cNN` / `rel_dedup_cNN`,
  and the n-gram table gains a pooled `raw_dedup`.
- **Placement.** The new columns are appended, so existing column positions
  don't move.
- **Same phrase model.** It is still fitted on the as-is view. Every existing
  count and rate column is byte-identical to the previous run.
- **Checks.** Two new invariants per cycle: dedup sums to its total, and no
  token ever has more dedup occurrences than raw ones.

## Fixed along the way

- **Absolute paths.** `manifest.json` and `line_stats.csv` recorded the
  developer's home directory in every source path. They now use the
  repo-relative idiom from `heckle_network.py`.
- **`pos_category` was nondeterministic.** `tag_index` broke ties between
  categories by insertion order over a `set` of strings, and Python randomises
  string hashing per process. Ties now go to the alphabetically first category,
  and the output is byte-identical across hash seeds. Against the last run, 32
  tokens' `pos_category` and 3 tokens' `is_content` settled differently. That
  run was one random draw, not a reference.

## What needs a human call (Task 8)

1. **`REPORT.md` still recommends `raw_collapsed`.** It is gitignored analysis
   prose, not generated output, so it was not rewritten here. The README now
   says to use `raw_dedup`. The report's cycle-41 and cycle-42 figures and its
   duplication section need rewriting from the new columns.
2. **`--block-window`** defaults to 10 runs. The sensitivity table that
   justifies it is in notebook 01, §2.

## Left alone deliberately (Task 8)

- **`manifest.json` `command`.** It still records `" ".join(sys.argv)`, as
  d06b2c4 chose to leave it.
- **Pooled phrase detection.** It still sees the copies, because it is fitted
  on the as-is view. Refitting on the dedup view would change which n-grams
  exist, and so every count column. That is a separate decision.

---

# Task 9 — heckle allotaxonographs on the dashboard

Branch `feat/dashboard-heckle-allotaxonographs`, stacked on
`feat/freq-dedup-view`.

**New:**
- **`src/parlamonitor/heckles.py`**, promoted from notebook 03:
  - `quotes_in` / `quotes_in_line` read whole quotes and recover
    hyphenated-surname interjectors;
  - `bench_of` / `party_group` attribute a quote to a bench;
  - `content_lemmas`;
  - `HECKLE_STOP_EXTRAS`.

  It has doctests and `tests/test_heckles.py`, including Hypothesis checks
  that reading never raises and every quote is text from its line.
- **`scripts/heckle_content.py`** writes `data/derived/heckles/`.
- **A new section in the dashboard's network view.** It has an interactive
  rank diamond (cells, iso-divergence contours, labelled top contributors), a
  25-word word shift, a 3-way switch, hover on cells and bars, and a table
  view. It works in light and dark, and its strings sit in `T.allotax`.

**Changed:**
- `build_dashboard_data.py` adds a `heckle_allotaxonographs` block (≈190 KB)
  when the heckle outputs exist, and the scope note says so.
- About section: data, method and stoplist added.
- Notebook 03 now imports the promoted functions instead of keeping its own
  copies.

**Verification:**
- **The script reproduces notebook 03 exactly:** 56,762 quotes; groups of 514
  / 474 / 733 / 739 quotes; RTD 0.685 / 0.696 / 0.740; bootstrap 95%. Every
  divergence equals the sum of its contributions, which the script asserts.
- **Notebook 03 re-executes with unchanged numbers.**
- **Browser check.** Driven in headless Chrome (Playwright, system Chrome) at
  1280 px light, 1280 px dark and 600 px:
  - the three switches, arrow-key switching, and cell and bar tooltips all
    work;
  - one bug was found and fixed: the word shift's centre line swallowed the
    hover at the bars' base;
  - the only console error is the pre-existing `favicon.ico` 404 of the local
    page.

## What needs a human call (Task 9)

1. **The published site is not updated.** The copy in
   `crow-intelligence.github.io/projects/parlamonitor-dashboard` is synced by
   hand. It needs `app.js`, `index.html` (merge carefully: the published one
   has extra meta) and `data/dashboard.json`.
2. **The route caveat is real.** The cycle-42 groups contain unnamed bench
   heckles only. The cycle-43 groups are mostly named and resolved through the
   registry. The dashboard says this under the chart. A fair comparison needs
   speeches exports for cycle 42.
3. **Light verbs.** The default stoplist removes *van* (*így van*), which was
   the one bench contrast stable across cycles 39–42. `--no-light-verbs`
   brings it back.
4. **Samples are small:** 346–863 content tokens per group. The bootstrap
   intervals are in the manifest. The dashboard shows the RTD and the token
   counts on every comparison.

## Fixed after review (Task 9)

- **Symbols were counted as words.** emtsv tags the Hungarian quotation marks
  `„` and `”` as nouns. `heckles.content_lemmas` now drops any lemma without a
  letter. Two group sizes moved: opposition-42 from 443 to 442 tokens, and
  TISZA from 866 to 863. The three RTDs and the 95% bootstrap are unchanged to
  three decimals.

## Left alone deliberately (Task 9)

- **`reactions.py`.** The quote-truncation and hyphenated-name fixes live in
  `heckles.py` only, so no Task 4 count or published network moves.
- **Display forms.** Counting is case-folded. For display, each word takes its
  most frequent spelling (`Gyurcsány`, `KGBéla`), so names don't print in
  lowercase. The counts are not affected.

---

# Task 10 — speech allotaxonographs: compare any two speakers, factions or sides

Branch `feat/dashboard-speech-allotaxonographs`, stacked on
`feat/dashboard-heckle-allotaxonographs`.

**New:**
- **`scripts/speech_vocabulary.py`**: content-lemma counts per cycle-43
  speaker, faction, Fidesz–KDNP and side, from `speech_metrics.py`'s emtsv
  cache. It records keyflux's RTD for four reference pairs in the manifest.
- **`dashboard/rtd.js`**: a line-for-line port of keyflux's `rtd` /
  `RankedList.aligned`.
- **`tests/test_dashboard_rtd.py`**: runs `rtd.js` under node against keyflux
  on 105 random and edge-case inputs, comparing the divergence and every
  type's ranks, direction and contribution. It is skipped without node.
- **A "Szókincs" tab:**
  - two pickers grouped into sides, alliance, factions and speakers, plus a
    swap button;
  - the pair goes into the URL;
  - the same diamond and word shift as the heckle section;
  - a table view;
  - reference distances for scale.

  Its data is a separate `data/vocabulary.json`, loaded on first use.

**Changed:**
- **`build_dashboard_data.py`** ships `vocabulary.json` and puts a small
  pointer in the main bundle.
- **The diamond renderer** takes its legend element as a parameter, so both
  sections share it.
- **Word-shift tick labels** get as many decimals as the step needs. Before
  this, 0.05% steps all read "0%".

**Fixed on the parent branch (PR #3):** `heckles.content_lemmas` dropped no
symbols, so the quotation marks `„` and `”` counted as heckle words. They are
dropped now, and the RTDs are unchanged to three decimals.

**Verification:**
- **Browser vs keyflux.** The page's TISZA ↔ Fidesz (0.358) and
  government ↔ opposition (0.350) equal keyflux's values in the manifest.
- **Browser check.** Driven in headless Chrome at 1280 px light and dark, and
  at 600 px: pickers, swap, same-entity guard, URL round-trip and tooltips all
  work, with no console errors. The heckle section was re-checked after the
  shared-renderer change.

## What needs a human call (Task 10)

1. **`--min-tokens 300`.** 160 of 197 speakers are offered. Below that, a
   speaker's ranks are mostly ties at count 1, and RTD against anyone is
   noise. It changes who can be picked, not any number.
2. **`dr.` is stopped for speeches only.** It is an honorific like `úr`. It is
   not added to the heckle list, which would move PR #3's numbers again for a
   word that never ranks there.
3. **Names lemmatise in lowercase** (`gulyás`, `jános`). The dashboard says
   so. The real fix is promoting notebook 01's corpus-attested name repair.
4. **Overlap.** An MP compared with their own faction is part of that
   faction's counts. The lede says so. Excluding the member from the faction
   on the fly is possible but would make the faction's numbers depend on who
   is being compared.
5. **The published site** needs `rtd.js`, the new `app.js` / `index.html`
   and both data files synced by hand.
