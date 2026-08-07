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
