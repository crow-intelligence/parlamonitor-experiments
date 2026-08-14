# Where the two methods agree, and where they do not

Two independent passes over the same 1,693 cycle-43 speeches:

* **Task 1, supervised.** `classla/ParlaCAP-Topic-Classifier` assigns each
  speech one of the 21 CAP major topics or `Other`, against a fixed codebook
  built for cross-national agenda-setting research.
* **Task 2, unsupervised.** BERTopic clusters the speeches with no schema at
  all, and found 28 topics.

Neither knows about the other. Where they land in the same place, that is two
methods corroborating each other. Where they do not, the disagreement is
usually informative about the corpus rather than about a bug.

## How much do they agree overall?

| statistic | all 1,692 speeches | core 1,070 only |
| --- | --- | --- |
| Adjusted Rand Index | 0.095 | **0.293** |
| Normalised Mutual Information | 0.367 | **0.520** |
| Homogeneity | 0.361 | 0.550 |
| Completeness | 0.373 | 0.493 |

"Core" drops the 522 speeches BERTopic left unclustered and
the 160 the classifier marked `Mix`. Both are *refusals to
decide*, and scoring a refusal against a decision measures the refusal, not the
agreement.

An ARI of 0.293 is real but partial. These are not two
renderings of one structure; they are two different cuts that coincide over
part of the corpus. The interesting question is which part.

## Where they agree: concrete policy domains

10 of 28 topics reach 70% purity or better
against CAP — meaning that share of the topic's speeches carry a single CAP
label.

| topic | n | purity | dominant CAP | our name |
| --- | --- | --- | --- | --- |
| T9 | 45 | 96% | Agriculture | Agriculture, the chamber of agriculture and drought |
| T6 | 55 | 93% | Transportation | Rail and public transport |
| T3 | 71 | 83% | Health | Healthcare and pharmaceutical provision |
| T19 | 17 | 82% | Macroeconomics | Fiscal Council, public debt and windfall taxes |
| T16 | 28 | 79% | Energy | Energy supply and renewable energy |
| T24 | 13 | 77% | Civil Rights | Christianity, churches and the culture war |
| T25 | 12 | 75% | Government Operations | Roll-calls, committee seats and procedural resolutions |
| T12 | 37 | 73% | Other | Short procedural and technical remarks |

Most of these are **policy domains with their own vocabulary**. Agriculture
talks about `agrárkamara` and `aszály`; transport about `gördülőállomány` and
`vasútvonal`. A fixed codebook and an unsupervised clusterer both find them
because the language marks them unambiguously.

Two entries in that table are a different animal. T25 (roll-calls, committee
seats) and T12 (microphones, timing, the chair) are **procedural boilerplate**.
They score highly because such speeches are near-identical to each other, and
CAP has nowhere to put them but `Government Operations` and `Other`. So purity
measures *the two methods agreeing*, not *the topic being substantive* — a
uniform topic and a meaningful one both score well, and only reading the
speeches separates them. That is one of the things the sample below is for.

The reverse view agrees. Taking each CAP label and asking how concentrated it
is in a single BERTopic topic:

| CAP label | n | concentrated in | share |
| --- | --- | --- | --- |
| Transportation | 56 | T6 Rail and public transport | 91% |
| Education | 88 | T0 Education, child protection and family support | 88% |
| Health | 71 | T3 Healthcare and pharmaceutical provision | 83% |
| Social Welfare | 61 | T0 Education, child protection and family support | 79% |
| Agriculture | 57 | T9 Agriculture, the chamber of agriculture and drought | 75% |
| Energy | 30 | T16 Energy supply and renewable energy | 73% |

## Where they diverge: the political process

| topic | n | purity | dominant CAP | our name |
| --- | --- | --- | --- | --- |
| T8 | 49 | 18% | Government Operations | Pre-agenda party-political clashes |
| T4 | 68 | 25% | Government Operations | Immunity cases, committees of inquiry and archival access |
| T5 | 55 | 27% | Mix | National remembrance, anniversaries and ceremonial speeches |
| T1 | 149 | 31% | Government Operations | Premiership, head of state and party-political conflict |
| T13 | 29 | 31% | Civil Rights | Fundamental Law, constitutional guarantees and national assets |
| T7 | 55 | 35% | Immigration | The migration pact and relations with the European Union |

These are not failures. CAP is a codebook of **policy areas**, and it has no
code for *"an MP attacking the government over its own conduct"*. Such speeches
land in `Government Operations`, or in `Mix` when the classifier cannot commit.
BERTopic, unconstrained, split the same material by rhetorical mode instead —
pre-agenda clashes, immunity proceedings, ceremonial remembrance.

Read from the CAP side, the same split appears:

| CAP label | n | concentrated in | share |
| --- | --- | --- | --- |
| Macroeconomics | 71 | T15 Poverty, incomes and public debt | 23% |
| Civil Rights | 106 | T1 Premiership, head of state and party-political conflict | 25% |
| Government Operations | 187 | T1 Premiership, head of state and party-political conflict | 25% |
| Law and Crime | 53 | T4 Immunity cases, committees of inquiry and archival access | 30% |
| Other | 94 | T1 Premiership, head of state and party-political conflict | 33% |

`Government Operations` is the largest CAP class in the core
(187 speeches; 266 across the whole corpus)
and among the most scattered, at 25%. It is where the
codebook puts everything about how the chamber runs — which, in a cycle this
contested, is a great deal.

## One disagreement worth its own line

CAP separates `Education` from `Social Welfare`. BERTopic did not: topic
0 covers schools, teachers, child protection, adoption and family
support as one cluster, and it is the dominant destination for **both** CAP
labels (88% of `Education`, 79% of
`Social Welfare`). Whether Hungarian family policy is one agenda or two is a
substantive question, not a technical one, and the two methods answer it
differently.

## What to check by hand

The counts above say nothing about whether either label is *correct*. The
sample in `samples/` is drawn to test exactly that, across the five ways these
methods can part company:

| stratum | candidates | question |
| --- | --- | --- |
| `agreement` | 279 | Both methods concur |
| `cross_method_disagreement` | 48 | One method dissents |
| `low_purity_topic` | 459 | The methods cut differently |
| `mix` | 160 | The classifier was unsure |
| `truncation` | 270 | The two passes disagree |

Start with `README.md` in this folder.

---

*Numbers in this file are interpolated from the run outputs, not transcribed.
Regenerate with `uv run python scripts/make_annotation_pack.py`. The
interpretation between them is a draft and has not been checked by a human.*
