# Lexicons

External word lists the analysis scores against. **The lists themselves are
gitignored**; this file is the provenance record and is committed in their
place, so a number can be traced to a specific version of a specific list
without redistributing it.

Every lexicon-derived figure records the file's sha256, so a changed list
produces a changed hash rather than a silently different number.

## Emotion — Putz Orsolya

**Putz Orsolya's own collection**, ongoing work, used here with permission.
**Not redistributable: do not commit, publish or pass on these files.** Any
output derived from them must credit her by name as the author of the
collection.

Eight categories are supplied. This project uses the **six Ekman basic
emotions** and leaves the other two aside: `feszültség` (tension) and
`szeretet` (love) are not Ekman basic categories, and *szeretet* in particular
measures something else.

| Ekman | file | entries | multiword |
|---|---|---:|---:|
| anger | `emo_duh_full.txt` | 411 | 74 |
| disgust | `emo_undor_full.txt` | 136 | 15 |
| fear | `emo_felelem_full.txt` | 244 | 24 |
| joy | `emo_orom_full.txt` | 676 | 77 |
| sadness | `emo_banat_full.txt` | 388 | 50 |
| surprise | `emo_meglepodes_full.txt` | 98 | 31 |
| — (not Ekman) | `emo_feszultseg_full.txt` | 310 | 28 |
| — (not Ekman) | `emo_szeretet_full.txt` | 187 | 25 |

`_full` files carry multiword expressions (`áldását adja`) alongside single
words; `_words` files, where present, are the single-word subset. The scorer
reads `_full` and matches multiword entries as lemma n-grams, so ~15-20% of
each list is not discarded.

48 entries appear in more than one category (`gyűlölet`, `szenvedés`,
`vigasz`). The categories are not exclusive and the scorer does not force them
to be.

**Sparsity is the limit.** Emotion words are 2.9% of content lemmas in this
corpus, and only 61% of speeches reach five hits. Emotion is therefore scored
at MP and topic level by default; per-speech scores below the hit threshold are
flagged rather than presented as solid.

## Sentiment — Precognox

Hungarian Sentiment Lexicon, Precognox (`labs@precognox.com`), from
[opendata.hu](https://opendata.hu/dataset/hungarian-sentiment-lexicon).
Manually built on the basis of an English lexicon. Downloaded 2026-09-11; the
files carry their original 2016-06-06 timestamps.

| polarity | file | entries |
|---|---|---:|
| positive | `PrecoPos.txt` | 1748 |
| negative | `PrecoNeg.txt` | 5940 |

**The list is 3.4x larger on the negative side** (5,940 against 1,748), so an
unweighted count is biased negative by construction. Inverse-class weighting
moves the corpus mean polarity from 0.36 to 0.73 — the choice moves every
number, so it is a stated parameter and not a default. Exactly one entry
appears in both files.

Privative forms are already lexicalised (`haszontalan`, `tehetetlen`,
`boldogtalan`, `hatástalan`, `céltalan`), so the `-talan/-telen` suffix needs no
morphological rule. What remains for negation is `nem`, `sem`, `se`, `nincs`,
`sincs`, `nélkül`.

**Licence: CC BY-NC 4.0** — the dataset page is authoritative; the listing
page's "CC BY-SA" is wrong. Confirmed with the maintainer. **This project's use
is non-commercial**, which the licence permits, and the dashboard ships under
CC BY-NC-SA 4.0, so the NC term carries through. Attribution: Precognox.

Still gitignored, because "we may use it" is not "we may redistribute it": the
licence allows non-commercial use, and mirroring the files into a public
repository is a separate act from using them.

## Loanwords — not here

The *idegen szavak* list stays in the sibling `saphes` checkout and is read by
path; see `src/parlamonitor/loanwords.py`. Same reasoning: that study leaves its
licence open, and copying the file here would settle a question deliberately
left open.

## Hashes

Recorded so a figure can be tied to a version of a list.

| file | entries | sha256 (first 16) |
|---|---:|---|
| `emotion/emo_banat_full.txt` | 388 | `1eb11136e252830a` |
| `emotion/emo_banat_words.txt` | 251 | `ef870779fba6d6c2` |
| `emotion/emo_duh_full.txt` | 411 | `1d23704dbf7fe78b` |
| `emotion/emo_duh_words.txt` | 322 | `44cc3972db56b96d` |
| `emotion/emo_felelem_full.txt` | 244 | `0988d717d44097d6` |
| `emotion/emo_felelem_words.txt` | 171 | `d80e6f4f76214482` |
| `emotion/emo_feszultseg_full.txt` | 310 | `8c4ec27c6dd02642` |
| `emotion/emo_meglepodes_full.txt` | 98 | `5e28737af5ce88f5` |
| `emotion/emo_meglepodes_words.txt` | 51 | `f0ba6275036249b0` |
| `emotion/emo_orom_full.txt` | 676 | `03ae5131f008f74f` |
| `emotion/emo_szeretet_full.txt` | 187 | `a0c35be1f47023e3` |
| `emotion/emo_undor_full.txt` | 136 | `426bdcbaf4e24e1d` |
| `emotion/emo_undor_words.txt` | 113 | `b9385b7ad1e60663` |
| `sentiment/PrecoNeg.txt` | 5940 | `40acc7262918b18e` |
| `sentiment/PrecoPos.txt` | 1748 | `1ce3b3d6d3d8b280` |
