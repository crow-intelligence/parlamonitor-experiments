"""What the hecklers say: quotes, benches, and the role-flip allotaxonographs.

Reads the parentheticals with copied blocks removed, pulls out every quoted
heckle whole, attributes it to a bench where the note (or, for cycle 43, the
speeches export) settles it, and compares the content vocabulary of four
groups with rank-turbulence divergence:

* TISZA, governing (cycle 43)
* Fidesz-KDNP, in opposition (cycle 43)
* Fidesz-KDNP, governing (cycle 42)
* the opposition (cycle 42)

Three comparisons follow the party across the aisle and back, and are written
for the dashboard's allotaxonograph. A bootstrap over quotes says how far the
ordering of distances survives resampling.

Records the data; draws nothing. Needs no emtsv service when the quote lemma
cache is warm -- copy ``data/derived/explore/quote_lemma_cache.jsonl`` into the
output directory to seed it from notebook 03.

Usage::

    uv run python scripts/heckle_content.py
    uv run python scripts/heckle_content.py --no-light-verbs   # keep "van", "mond" ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from keyflux import RankedList, rtd

from parlamonitor.emtsv import EmtsvError, analyse_lines, is_content_word
from parlamonitor.heckles import (
    HECKLE_STOP_EXTRAS,
    bench_of,
    content_lemmas,
    quotes_in_line,
)
from parlamonitor.interactions import SpeakerRegistry
from parlamonitor.lexicon import fold
from parlamonitor.loading import DATA_RAW, load_speeches
from parlamonitor.parentheticals import (
    BLOCK_WINDOW_RUNS,
    CYCLES,
    block_dedup,
    file_sha256,
    load_parentheticals,
    parentheticals_path,
)
from parlamonitor.reactions import GOVERNING_PARTIES
from parlamonitor.stopwords import hungarian_stopwords

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "heckles"
REGISTRY_CYCLE = 43

GROUPS: dict[str, dict] = {
    "fidesz_gov_42": {
        "label": "Fidesz-KDNP, governing (42)",
        "cycle": 42,
        "side": "government",
        "party_group": "Fidesz-KDNP",
    },
    "opposition_42": {
        "label": "opposition (42)",
        "cycle": 42,
        "side": "opposition",
        "party_group": None,
    },
    "fidesz_opp_43": {
        "label": "Fidesz-KDNP, in opposition (43)",
        "cycle": 43,
        "side": "opposition",
        "party_group": "Fidesz-KDNP",
    },
    "tisza_gov_43": {
        "label": "TISZA, governing (43)",
        "cycle": 43,
        "side": "government",
        "party_group": None,
    },
}
"""The four groups. ``party_group: None`` means any party on that side. The
order is also the bootstrap's resampling order, kept as notebook 03 ran it."""

COMPARISONS: tuple[tuple[str, str], ...] = (
    ("tisza_gov_43", "fidesz_opp_43"),
    ("fidesz_opp_43", "fidesz_gov_42"),
    ("fidesz_gov_42", "opposition_42"),
)
"""(left, right) pairs: the new government against the new opposition, the
party against its own past, and that past against the bench it later joined."""

ROUTE_CAVEAT = (
    "Cycle-42 groups are bench-attributed heckles only: the parentheticals name "
    "an interjector but never their party, and only cycle 43 has a speeches "
    "export to resolve names against. Cycle-43 groups are mostly named "
    "interjections. Part of any 42-vs-43 distance is therefore the difference "
    "in route, not only in bench or party."
)


def _repo_relative(path):
    """Path as written in the repo, for provenance records that get published.

    `str(path)` on a resolved path carries the developer's home directory into
    every artefact built from it, and those artefacts ship. Falls back to the
    basename when the input genuinely lives outside the checkout.
    """
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=1 / 3, help="RTD alpha")
    parser.add_argument("--block-window", type=int, default=BLOCK_WINDOW_RUNS)
    parser.add_argument(
        "--no-light-verbs",
        action="store_true",
        help="keep light verbs (van, mond, kér ...) in the counted vocabulary",
    )
    parser.add_argument("--bootstrap", type=int, default=200, help="draws; 0 skips")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    return parser.parse_args(argv)


def collect_quotes(data_dir, block_window: int, registry: SpeakerRegistry):
    """Every quote in every cycle, with its bench."""
    rows = []
    for cycle in CYCLES:
        lines = block_dedup(load_parentheticals(cycle, data_dir)[0], block_window)
        for line in lines:
            for q in quotes_in_line(line):
                side, group, route = bench_of(
                    q, cycle, registry if cycle == REGISTRY_CYCLE else None
                )
                rows.append(
                    {
                        "cycle": cycle,
                        "speaker": q.speaker,
                        "addressee": q.addressee,
                        "kind": q.kind,
                        "parties": ";".join(sorted(q.parties)),
                        "quote": q.quote,
                        "reassembled": q.reassembled,
                        "hyphenated_recovered": q.hyphenated_recovered,
                        "side": side,
                        "party_group": group,
                        "route": route,
                    }
                )
    return pd.DataFrame(rows)


def load_cache(path: Path) -> dict[str, list]:
    cache: dict[str, list] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                record = json.loads(raw)
                cache[record["quote"]] = record["sentences"]
    return cache


def lemmatise(quotes: Sequence[str], cache_path: Path, base_url: str) -> dict:
    """Return the emtsv analysis of every quote, from the cache where possible."""
    cache = load_cache(cache_path)
    todo = sorted(set(quotes) - set(cache))
    print(f"  {len(set(quotes)):,} distinct quotes; {len(todo):,} not in the cache")
    if todo:
        with requests.Session() as session:
            analysed = analyse_lines(todo, base_url=base_url, session=session)
        with cache_path.open("a", encoding="utf-8") as handle:
            for quote, sentences in zip(todo, analysed, strict=True):
                serial = [[[t.form, t.lemma, t.xpostag] for t in s] for s in sentences]
                cache[quote] = serial
                handle.write(
                    json.dumps(
                        {"quote": quote, "sentences": serial}, ensure_ascii=False
                    )
                    + "\n"
                )
    return cache


def group_mask(quotes: pd.DataFrame, spec: dict) -> pd.Series:
    mask = (quotes["cycle"] == spec["cycle"]) & (quotes["side"] == spec["side"])
    if spec["party_group"] is not None:
        mask &= quotes["party_group"] == spec["party_group"]
    return mask


def display_forms(cache: dict, quotes: Sequence[str]) -> dict[str, str]:
    """Map each case-folded lemma to its most frequent spelling, for display.

    Counting folds case, so ``Péter`` and ``péter`` are one type; showing the
    folded form would print every name in lowercase. Ties go to the
    alphabetically first spelling.
    """
    seen: dict[str, Counter] = {}
    for quote in quotes:
        for sentence in cache[quote]:
            for _, lemma, tag in sentence:
                if is_content_word(tag):
                    seen.setdefault(fold(lemma), Counter())[lemma] += 1
    return {
        key: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        for key, c in seen.items()
    }


def comparison_record(
    left: str, right: str, counts: dict, alpha: float, labels: dict[str, str]
) -> dict:
    """RTD of two groups, with every type's ranks, counts and contribution."""
    list1 = RankedList.from_counts(counts[left], label=left)
    list2 = RankedList.from_counts(counts[right], label=right)
    result = rtd(list1, list2, alpha=alpha)
    types = [
        {
            "t": c.type,
            "label": labels.get(c.type, c.type),
            "r1": float(c.rank1),
            "r2": float(c.rank2),
            "c1": int(counts[left].get(c.type, 0)),
            "c2": int(counts[right].get(c.type, 0)),
            "contrib": float(c.contribution),
            "dir": {"system1": "left", "system2": "right"}.get(c.direction, "shared"),
        }
        for c in result.contributions
    ]
    union = set(counts[left]) | set(counts[right])
    if len(types) != len(union):
        raise RuntimeError(
            f"{left}/{right}: {len(types)} contributions, {len(union)} types"
        )
    total = sum(t["contrib"] for t in types)
    if abs(total - float(result.divergence)) > 1e-9:
        raise RuntimeError(f"{left}/{right}: contributions sum to {total}, not RTD")
    return {
        "id": f"{left}__{right}",
        "left": left,
        "right": right,
        "divergence": float(result.divergence),
        "alpha": float(result.alpha),
        "types": types,
    }


def bootstrap(quotes, counts_of, n_draws: int, seed: int, alpha: float) -> dict:
    """Resample each group's quotes; distances from Fidesz-KDNP in opposition (43)."""
    focus = "fidesz_opp_43"
    others = [g for g in GROUPS if g != focus]
    quote_lists = {
        g: quotes.loc[group_mask(quotes, GROUPS[g]), "quote"].tolist() for g in GROUPS
    }
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        lists = {}
        for g in GROUPS:  # resampling order is part of the seed's meaning
            qs = quote_lists[g]
            total: Counter = Counter()
            for i in rng.integers(0, len(qs), len(qs)):
                total.update(counts_of[qs[i]])
            lists[g] = RankedList.from_counts(total, label=g)
        draws.append(
            {
                o: float(rtd(lists[focus], lists[o], alpha=alpha).divergence)
                for o in others
            }
        )
    frame = pd.DataFrame(draws)
    nearest = frame.idxmin(axis=1).value_counts(normalize=True)
    return {
        "focus": focus,
        "draws": n_draws,
        "seed": seed,
        "quantiles": {
            o: {q: round(float(frame[o].quantile(q)), 4) for q in (0.05, 0.5, 0.95)}
            for o in others
        },
        "nearest_share": {o: round(float(nearest.get(o, 0.0)), 4) for o in others},
        "closer_to_own_past_than_to_old_opposition": round(
            float((frame["fidesz_gov_42"] < frame["opposition_42"]).mean()), 4
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or DATA_RAW

    print("Reading the parentheticals (block copies removed) ...")
    speeches = load_speeches(args.data_dir)
    registry = SpeakerRegistry(
        (record.get("speaker") or {} for record in speeches),
        governing=GOVERNING_PARTIES[REGISTRY_CYCLE],
    )
    quotes = collect_quotes(args.data_dir, args.block_window, registry)
    print(f"  {len(quotes):,} quotes across cycles {CYCLES[0]}-{CYCLES[-1]}")

    print("Lemmatising ...")
    try:
        cache = lemmatise(
            quotes["quote"].tolist(), out / "quote_lemma_cache.jsonl", args.base_url
        )
    except (EmtsvError, requests.ConnectionError) as error:
        print(
            f"emtsv is needed for uncached quotes and failed: {error}", file=sys.stderr
        )
        print(
            "Start it with "
            "`docker run --rm -d --name emtsv -p 5000:5000 mtaril/emtsv`, "
            "or seed the cache from data/derived/explore/quote_lemma_cache.jsonl.",
            file=sys.stderr,
        )
        return 1

    stop = hungarian_stopwords(
        light_verbs=not args.no_light_verbs, extra=HECKLE_STOP_EXTRAS
    )
    counts_of = {q: content_lemmas(cache[q], stop) for q in set(quotes["quote"])}

    counts: dict[str, Counter] = {}
    group_rows, group_meta = [], {}
    for name, spec in GROUPS.items():
        mask = group_mask(quotes, spec)
        total: Counter = Counter()
        for quote in quotes.loc[mask, "quote"]:
            total.update(counts_of[quote])
        counts[name] = total
        group_meta[name] = {
            **spec,
            "quotes": int(mask.sum()),
            "tokens": int(total.total()),
            "types": len(total),
            "routes": quotes.loc[mask, "route"].value_counts().to_dict(),
        }
        group_rows += [
            {"group": name, "lemma": t, "count": n} for t, n in total.items()
        ]
        print(f"  {spec['label']:34s} {mask.sum():6,} quotes {total.total():7,} tokens")

    labels = display_forms(cache, quotes["quote"].tolist())
    comparisons = [
        comparison_record(a, b, counts, args.alpha, labels) for a, b in COMPARISONS
    ]
    for record in comparisons:
        print(
            f"  RTD {record['left']} vs {record['right']}: {record['divergence']:.3f}"
        )

    boot = None
    if args.bootstrap:
        print(f"Bootstrap, {args.bootstrap} draws ...")
        boot = bootstrap(quotes, counts_of, args.bootstrap, args.seed, args.alpha)
        print(
            "  closer to its own past than to the old opposition in "
            f"{boot['closer_to_own_past_than_to_old_opposition']:.0%} of draws"
        )

    generated = datetime.now(UTC).isoformat()
    params = {
        "view": "block-dedup",
        "block_window_runs": args.block_window,
        "content_words": "N, V, Adj, Adv; pronouns dropped",
        "stopwords": {
            "spacy_hu": True,
            "light_verbs": not args.no_light_verbs,
            "parliamentary": True,
            "extras": list(HECKLE_STOP_EXTRAS),
            "size": len(stop),
        },
        "alpha": args.alpha,
    }
    quotes.to_csv(out / "quotes.csv", index=False, encoding="utf-8")
    pd.DataFrame(group_rows).sort_values(
        ["group", "count"], ascending=[True, False]
    ).to_csv(out / "group_counts.csv", index=False, encoding="utf-8")
    bundle = {
        "generated_at": generated,
        "source": "scripts/heckle_content.py",
        "params": params,
        "caveat": ROUTE_CAVEAT,
        "groups": group_meta,
        "comparisons": comparisons,
        "bootstrap": boot,
    }
    (out / "allotaxonographs.json").write_text(
        json.dumps(bundle, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "generated_at": generated,
        "command": "scripts/heckle_content.py",
        "sources": [
            {
                "cycle": c,
                "path": _repo_relative(parentheticals_path(c, args.data_dir)),
                "sha256": file_sha256(parentheticals_path(c, args.data_dir)),
            }
            for c in CYCLES
        ],
        "registry_source": _repo_relative(Path(data_dir) / "cycle43-speeches.jsonl"),
        "params": params,
        "counts": {
            "quotes": int(len(quotes)),
            "quotes_per_cycle": quotes["cycle"].value_counts().sort_index().to_dict(),
            "by_route": quotes["route"].value_counts().to_dict(),
            "reassembled": int(quotes["reassembled"].sum()),
            "hyphenated_recovered": int(quotes["hyphenated_recovered"].sum()),
        },
        "groups": group_meta,
        "divergences": {r["id"]: r["divergence"] for r in comparisons},
        "bootstrap": boot,
        "caveat": ROUTE_CAVEAT,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {_repo_relative(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
