"""Who heckled whom: build and save the directed interaction network.

Reads the speeches export, pulls every named interjection out of the inline
``( ... )`` spans, and records an edge from the interjector to whoever held the
floor. Both ends are resolved to a faction where the export allows it, so the
graph can be aggregated to party level.

Records the data; draws nothing. Outputs are a tidy edge table, a node table,
and the graph in three formats -- d3 node-link JSON, party-level node-link JSON,
and GraphML for Gephi and igraph.

Usage::

    uv run python scripts/heckle_network.py
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import pandas as pd

from parlamonitor.interactions import (
    SpeakerRegistry,
    aggregate_by,
    annotate_crossing,
    annotate_degrees,
    build_graph,
    graphml_safe,
    to_node_link,
)
from parlamonitor.loading import DATA_RAW
from parlamonitor.reactions import GOVERNING_PARTIES, Kind, events_in

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "reactions"

_SPAN = re.compile(r"\(([^()]*)\)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cycle", type=int, default=43)
    parser.add_argument(
        "--keep-self-loops",
        action="store_true",
        help="keep an MP interrupting their own speech as a graph edge",
    )
    return parser.parse_args(argv)


def read_speeches(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_edges(speeches: Sequence[dict], registry: SpeakerRegistry, cycle: int):
    """One row per named interjection, with both ends resolved."""
    rows = []
    unresolved: set[str] = set()
    for record in speeches:
        floor = record.get("speaker") or {}
        target = registry.resolve(floor.get("label") or "")
        agenda = record.get("agenda") or {}
        for span in _SPAN.findall(record.get("text") or ""):
            for event in events_in(span):
                if event.kind is not Kind.INTERJECTION or not event.speaker:
                    continue
                source = registry.resolve(event.speaker)
                if not source.resolved:
                    unresolved.add(event.speaker)
                rows.append(
                    {
                        "cycle": cycle,
                        "speech_uid": record.get("uid"),
                        "date": record.get("date"),
                        "sitting": record.get("sitting"),
                        "agenda_title": (agenda.get("title") or "")[:120],
                        "speech_type": record.get("speech_type"),
                        "source": source.label,
                        "source_id": source.node_id,
                        "source_person_id": source.person_id,
                        "source_faction": source.faction,
                        "source_side": source.side,
                        "source_is_mp": source.is_mp,
                        "source_resolved": source.resolved,
                        "source_match": source.match,
                        "target": target.label,
                        "target_id": target.node_id,
                        "target_person_id": target.person_id,
                        "target_faction": target.faction,
                        "target_side": target.side,
                        "target_is_mp": target.is_mp,
                        "target_resolved": target.resolved,
                        "self_loop": source.node_id == target.node_id,
                        "quote": event.quote,
                    }
                )
    return rows, unresolved


def node_table(graph: nx.DiGraph) -> pd.DataFrame:
    """The graph's nodes as a flat table, for aggregation outside a graph tool."""
    rows = [{"node_id": node, **data} for node, data in graph.nodes(data=True)]
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["heckles_given", "heckles_received"], ascending=False, ignore_index=True
    )


def cross_bench(edges: pd.DataFrame) -> dict[str, int]:
    """How much heckling crosses the aisle, among edges where both sides are known."""
    known = edges[edges["source_side"].notna() & edges["target_side"].notna()]
    same = int((known["source_side"] == known["target_side"]).sum())
    return {
        "interjections_with_both_sides_known": int(len(known)),
        "same_side": same,
        "cross_bench": int(len(known) - same),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    root = DATA_RAW if args.data_dir is None else Path(args.data_dir)
    path = root / f"cycle{args.cycle}-speeches.jsonl"
    if not path.is_file():
        print(f"No speeches export at {path}.", file=sys.stderr)
        print(
            "The target of an edge is the floor-holder, so this network needs "
            "a speeches export for the cycle.",
            file=sys.stderr,
        )
        return 1

    print(f"Reading {path.name} ...")
    speeches = read_speeches(path)
    registry = SpeakerRegistry(
        (record.get("speaker") or {} for record in speeches),
        governing=GOVERNING_PARTIES[args.cycle],
    )
    print(f"  {len(speeches):,} speeches, {len(registry):,} distinct speakers")

    rows, unresolved = extract_edges(speeches, registry, args.cycle)
    edges = pd.DataFrame(rows)
    if edges.empty:
        print("No named interjections found; nothing to build.", file=sys.stderr)
        return 1

    resolved = int(edges["source_resolved"].sum())
    by_match = edges["source_match"].value_counts().to_dict()
    print(f"  {len(edges):,} named interjections inside speeches")
    print(
        f"  interjector resolved to a speaker: {resolved:,} "
        f"({resolved / len(edges):.0%}); {len(unresolved)} names unresolved"
    )
    print(
        f"  self-loops (heckling one's own speech): {int(edges['self_loop'].sum()):,}"
    )

    graph = annotate_crossing(
        annotate_degrees(build_graph(rows, drop_self_loops=not args.keep_self_loops))
    )
    print(
        f"  graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges"
    )

    by_party = aggregate_by(graph, "faction")
    by_side = aggregate_by(graph, "side")
    crossing = cross_bench(edges)
    print(
        f"  cross-bench: {crossing['cross_bench']:,} of "
        f"{crossing['interjections_with_both_sides_known']:,} interjections "
        f"where both sides are known"
    )

    crossing_edges = Counter(data["crossing"] for *_, data in graph.edges(data=True))
    print(f"  edge crossing: {dict(crossing_edges)}")

    provenance = {
        "cycle": args.cycle,
        "source": str(path),
        "generated_at": datetime.now(UTC).isoformat(),
        "edge_semantics": "source interjected during target's speech",
    }

    edges.to_csv(out / "heckle_edges.csv", index=False, encoding="utf-8")
    node_table(graph).to_csv(out / "heckle_nodes.csv", index=False, encoding="utf-8")
    (out / "heckle_network.json").write_text(
        json.dumps(to_node_link(graph, **provenance), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "heckle_network_party.json").write_text(
        json.dumps(to_node_link(by_party, **provenance), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "heckle_network_side.json").write_text(
        json.dumps(to_node_link(by_side, **provenance), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # GraphML has no null and declares one type per attribute name, so the
    # graph goes through `graphml_safe` first. The JSON exports above keep
    # their real nulls and booleans.
    nx.write_graphml(graphml_safe(graph), out / "heckle_network.graphml")
    nx.write_graphml(graphml_safe(by_party), out / "heckle_network_party.graphml")

    manifest = {
        **provenance,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "networkx": nx.__version__,
        "command": " ".join(sys.argv),
        "governing_parties": sorted(GOVERNING_PARTIES[args.cycle]),
        "counts": {
            "speeches": len(speeches),
            "named_interjections": len(edges),
            "interjector_resolved": resolved,
            "interjector_match_kind": {k: int(v) for k, v in by_match.items()},
            "interjector_unresolved_names": sorted(unresolved),
            "self_loops": int(edges["self_loop"].sum()),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "party_nodes": by_party.number_of_nodes(),
            "party_edges": by_party.number_of_edges(),
            **crossing,
            "edges_by_crossing": dict(crossing_edges),
        },
        "coverage": {
            "note": (
                "41% of this cycle's named interjections occur outside any speech "
                "record -- during voting, procedure and the ceremonial opening -- "
                "and have no floor-holder, so they cannot become edges."
            )
        },
        "self_loops_handling": (
            "kept in heckle_edges.csv, dropped from the graph"
            if not args.keep_self_loops
            else "kept everywhere"
        ),
    }
    (out / "heckle_network_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
