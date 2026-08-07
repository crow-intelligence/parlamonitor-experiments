# NLP experimentation exports

Derived datasets for readability / lexical-diversity work and for
question-answer text analysis. **Regenerable — do not commit** (`exports/` is
gitignored); rebuild with:

```bash
cd backend
python3 export_nlp_datasets.py --period 43 --db parlamonitor.db --out ../exports
```

Source: the runtime SQLite DB (read-only), cycle **43** — data as of
`db_data_updated_at` in `cycle43-manifest.json`. Both files are **JSONL**, one
JSON object per line, UTF-8, unescaped Hungarian characters.

| file | rows | size |
| --- | --- | --- |
| `cycle43-speeches.jsonl` | 1,693 speeches (826,775 cleaned words) | 22 MB |
| `cycle43-qa.jsonl` | 215 question-answer exchanges (202 with text on both sides) | 9.3 MB |
| `cycle43-manifest.json` | counts + what was filtered out | — |

## Text normalization (applies to both files)

Every record carries the text twice, so a metric can be run on either:

- **`text`** — verbatim, exactly as the transcript has it (sentences joined with
  a space inside a paragraph, paragraphs with a blank line).
- **`text_clean`** — normalized for measurement. Use this one for readability
  and lexical diversity. Two transcript conventions are removed:
  1. **The speaker attribution** that opens a speech — `NAGY JÁNOS (TISZA):`,
     `KAPITÁNY ISTVÁN gazdasági és energetikai miniszter:`, `DR. X, a Nemzeti
     Választási Bizottság elnöke:` — kept separately in `speaker_prefix`
     (captured for 1,692 of 1,693 speeches).
  2. **Editorial stage directions in parentheses** — applause, heckling,
     laughter, the chair's bell, "turning to X", entering/leaving the chamber,
     `(sic!)`, and the half-hourly wall-clock stamps `(13.20)`. Nested and
     sentence-straddling ones too (`(A képviselő feláll.` + `Taps.)` is one
     direction split across two DB sentences). A sentence that consists only of
     a direction is dropped — `n_sentences_dropped` counts them (2,736 total).
     Roughly 3 exotic ones survive across the whole cycle.

  Textual/legal parentheses are **kept** (`13. § (2) bekezdése`, `(EU) 2016/679`)
  — 67 speeches contain them.

- **`sentences`** — the cleaned text as a list of strings, using the **DB's own
  sentence segmentation** (from the loader), so sentence-count-based metrics
  (ARI, Flesch-type, mean sentence length) do not need a re-segmentation step.
  `len(sentences) == n_sentences`, and `sentence_paragraphs[i]` gives sentence
  `i`'s source paragraph index.

Counts (`n_words`, `n_chars`, `n_sentences`, `n_paragraphs`) are all computed on
the **cleaned** text; `n_chars_raw` / `n_sentences_raw` refer to the verbatim
text. A word is `[^\W\d_]+` (letters only — bare numbers are not words).

## 1. `cycle43-speeches.jsonl` — readability / lexical diversity

One line per speech. Fields:

| field | notes |
| --- | --- |
| `uid`, `origin_id` | `43003-78` = session + speech index; upstream id |
| `session_id`, `date`, `sitting`, `period_number`, `speech_index` | sitting-day identity |
| `speech_type` | upstream *felszólalás típusa*: `felszólalás`, `kétperces felszólalás`, `vezérszónoki felszólalás`, `elhangzik az interpelláció/kérdés/azonnali kérdés`, `kérdés megválaszolva`, … — a natural grouping variable |
| `speaker` | `person_id`, `label`, `label_full`, `faction`, `office`, `status`, `is_mp`, `is_advocate`, `nationality`, `constituency`, `highest_education`, `wikidata_id` |
| `agenda` | `id`, `ord`, `title`, `official_title`, `type` (normalized: `regular`, `qa`, `questioning_of_the_government`, `voting`, …), `native_type` (`HU-debate`, `HU-interpellation`, `HU-immediate_question`, …) |
| `time_start_s`, `duration_s` | seconds into the sitting day's stream; speaking time |
| `speaker_prefix` | the stripped attribution string |
| `text`, `text_clean`, `sentences`, `sentence_paragraphs` | see above |
| `n_sentences`, `n_sentences_raw`, `n_sentences_dropped`, `n_paragraphs`, `n_words`, `n_chars`, `n_chars_raw` | counts |

**What is filtered out** (see `manifest.speeches_dataset.skipped`):

- `procedural` (2,114) — chairing / session-management speeches (`ülésvezetés`,
  "általános vita lezárva", …). These are the Speaker's boilerplate and would
  dominate any distribution. The `speech.procedural` flag in the DB decides.
- `formulaic_type` (8) — the MP oath (`Eskü`): recited verbatim from a fixed
  text, and partly in Romani / Boyash / Romanian (the nationality advocates), so
  it breaks Hungarian-language metrics.
- `no_text` (154) — speeches with no transcript yet (video-only). This is almost
  entirely **sitting 43020 (2026-07-27)**, whose recording is published but whose
  transcript is not, so that day contributes no speeches at all here.

Distribution, for sanity: 307 words per speech at the median (mean 488, max
5,281), 19 sentences at the median. Factions present: TISZA 821, Fidesz 348,
KDNP 229, Mi Hazánk 197, none/non-MP 98.

> Length caveat for lexical diversity: plain TTR is length-dependent, and this
> set spans 1 to 5,281 words. Either filter on `n_words` (e.g. ≥ 200) or use a
> length-robust measure (MTLD, MATTR, vocd-D). `speech_type` correlates strongly
> with length — `kétperces felszólalás` is a 2-minute format by rule.

## 2. `cycle43-qa.jsonl` — questions and answers side by side

One line per Q&A **exchange**, reconstructed from the speeches inside one
question-type agenda item. 215 exchanges: 68 `interpellation` (interpelláció)
and 147 `immediate_question` (azonnali kérdés). All 215 have an answer;
`questions_outside_qa_agenda_items` is 0, so nothing was missed by the grouping.

| field | notes |
| --- | --- |
| `exchange_id` | `43003-ai19517` (session + agenda item) |
| `kind` | `interpellation` \| `immediate_question` \| `question` |
| `session_id`, `date`, `sitting`, `agenda_item_id`, `agenda_type`, `agenda_native_type` | provenance |
| `title`, `official_title` | the question's own title — "Meddig volt őszinte a 480 forintos benzinár ígérete?" |
| `asker`, `answerer` | speaker objects (same shape as above); the answerer's `office` is the ministerial title |
| `question_text`, `answer_text` | **the side-by-side pair, cleaned** — the quick path for a Q↔A comparison |
| `question`, `answer` | the full turns: `text`, `text_clean`, `sentences`, `n_words`, `n_sentences`, `n_chars`, `speaker`, `duration_s`, `uid`, `speech_type`, `role` |
| `mp_rejoinder`, `minister_rejoinder` | the immediate-question follow-ups (80 / 79 present), same turn shape, `null` when absent |
| `mp_reaction` | the MP's reply-to-the-answer turn, interpellations only |
| `answer_accepted` | `true` (20) / `false` (48) / `null` (147) — **interpellations only** have an accept/reject step; an immediate question ends with the rejoinders, hence `null` |
| `answered` | always `true` in this cycle; kept so a later cycle's unanswered question is detectable |
| `has_question_text`, `has_answer_text`, `text_complete` | **filter on `text_complete`** for text work — 202 of 215 |
| `turns` | every non-procedural speech of the exchange in order, each with a `role` |

The 13 rows where `text_complete` is `false` (`manifest.qa_dataset
.incomplete_text_by_session`) are genuine gaps, not export bugs:

- **12 on sitting 43020** (2026-07-27) — the day's transcript is not published
  yet, so the exchanges have full metadata (asker, answerer, title, durations)
  but no text on either side. They will fill in on a later sync.
- **1 on 2026-05-26** (`43003-ai19522`) — the minister's microphone failed, so
  the transcript's answer speech is a stage direction only
  (*"…szólásra emelkedik, a mikrofonja nem…"*, 6 seconds). The question side is
  intact; there is no answer text to pair with it.

`role` values, mapped from the upstream speech type: `question`, `answer`,
`mp_rejoinder`, `minister_rejoinder`, `mp_reaction_accept`,
`mp_reaction_reject`, `substitute_answerer_rejected`, `other`. The chair's
interjections are procedural and are not in `turns`.

Typical exchange shapes: interpellation → `question`, `answer`,
`mp_reaction_reject|accept`; immediate question → `question`, `answer`,
`mp_rejoinder`, `minister_rejoinder`.

## Quick start

```python
import json

speeches = [json.loads(l) for l in open("exports/cycle43-speeches.jsonl", encoding="utf-8")]
long_enough = [s for s in speeches if s["n_words"] >= 200 and not s["speaker"]["office"]]

# mean sentence length per faction, on the cleaned text
import statistics, collections
by_faction = collections.defaultdict(list)
for s in long_enough:
    by_faction[s["speaker"]["faction"]].append(s["n_words"] / max(s["n_sentences"], 1))
for fac, vals in by_faction.items():
    print(f"{fac or '—':12s} {statistics.mean(vals):5.1f} words/sentence  (n={len(vals)})")

# question vs answer, paired
qa = [json.loads(l) for l in open("exports/cycle43-qa.jsonl", encoding="utf-8")]
qa = [x for x in qa if x["text_complete"]]          # 202 of 215
for x in qa[:5]:
    print(x["kind"], x["asker"]["faction"], "→", x["answerer"]["office"],
          x["question"]["n_words"], "vs", x["answer"]["n_words"], "words")
```

Other cycles: pass `--period 39|40|41|42`. Cycle 43 is the current one (21
sitting days as of the export), the archive cycles are 5–10× larger.
