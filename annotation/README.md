# Annotation pack — CAP versus the discovered topics

60 Hungarian parliamentary speeches, each labelled twice by
methods that never saw each other's output, laid out for a human to adjudicate.

**Why this exists.** The classifier has no published evaluation for Hungarian.
Its `Mix` rate here (9.5%) is consistent with the authors'
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

### `agreement/` — Both methods concur

The speech's CAP label matches the dominant CAP label of its BERTopic topic, inside a topic that is itself coherent (purity >= 70%). Two independent methods put this speech in the same place. **Question: is that shared label actually right?**

### `cross_method_disagreement/` — One method dissents

The speech sits in a high-purity topic but carries a different CAP label from the rest of it. Either the classifier misread this speech, or the topic model swept in something that does not belong. **Question: which one is wrong?**

### `low_purity_topic/` — The methods cut differently

The speech belongs to a topic with purity below 40% -- one that scatters across CAP labels. These are usually about the political process rather than a policy area, which CAP has no code for. **Question: does either label describe this speech usefully?**

### `mix/` — The classifier was unsure

Confidence fell below 0.6, so the label became `Mix`. The model authors take this to mean the speech spans several topics. **Question: is it genuinely multi-topic, or did the model simply fail on a clear case?**

### `truncation/` — The two passes disagree

The speech exceeds 512 tokens. Classifying its opening alone gives one CAP label; classifying every window and averaging gives another. **Question: which label fits the whole speech?** The cut point is marked in the text below.

## When you are done

Commit the filled sheet. Hand-verified data is not rebuildable, unlike
everything in `data/derived/`, so it belongs in git. Then the numbers in
`FINDINGS.md` acquire something they currently lack: a measured error rate.

## The CAP major topics

- Agriculture
- Civil Rights
- Culture
- Defense
- Domestic Commerce
- Education
- Energy
- Environment
- Foreign Trade
- Government Operations
- Health
- Housing
- Immigration
- International Affairs
- Labor
- Law and Crime
- Macroeconomics
- Public Lands
- Social Welfare
- Technology
- Transportation

Plus `Other` (fits no CAP topic — a real prediction) and `Mix` (the classifier
was below 0.6 confidence — an admission of uncertainty). These
two are **not** the same thing, and the sheet keeps them separate.
