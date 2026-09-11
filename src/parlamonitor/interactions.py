"""Who heckled whom: a directed interaction network from the minutes.

The minutes record an interruption inside the speech it interrupted::

    MAGYAR PÉTER miniszterelnök: ... (Rétvári Bence: Ez nem igaz!) ...

That is an edge. The interjector is named in the parenthetical; the person
interrupted is whoever held the floor, which the speeches export states
outright. Both ends can then be resolved to a faction, and the graph aggregated
to party level.

Two limits are structural and are reported rather than papered over:

* **Cycle 43 only.** The target of an edge is the floor-holder, and only cycle
  43 has a speeches export. The parentheticals files for 39-42 name interjectors
  but nothing to attach them to.
* **41% of cycle-43 named interjections happen outside any speech** -- during
  voting, the chair's procedural passages, the ceremonial opening. Those have no
  floor-holder and therefore no edge. The network covers interruptions of
  speeches, not every shout in the chamber.

Name resolution is exact-match against the export's own speaker labels, after
stripping honorifics. A name that does not resolve keeps its raw string and a
null faction; it is never guessed at, because a wrong party on an edge is worse
than a missing one.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

_HONORIFIC = re.compile(r"^\s*(?:dr|id|ifj|özv|prof)\.?\s+", re.IGNORECASE)
_INITIAL = re.compile(r"^\s*[A-ZÁÉÍÓÖŐÚÜŰ]\.\s+")


def normalise_name(name: str) -> str:
    """Reduce a name to a comparable key.

    Strips honorifics and a leading initial, collapses whitespace, casefolds
    and strips accents. Accent stripping matters because the same member
    appears with and without them in different transcripts.

    Args:
        name: A name as it appears in a parenthetical or a speaker label.

    Returns:
        The comparison key. Empty if the name was only a title.

    Example:
        >>> normalise_name("Dr. Vadai Ágnes")
        'vadai agnes'
        >>> normalise_name("Vadai Ágnes") == normalise_name("dr Vadai Agnes")
        True

        A leading initial is a title, not a name token -- ``Z. Kárpát Dániel``
        is filed under his surname:

        >>> normalise_name("Z. Kárpát Dániel")
        'karpat daniel'
    """
    text = _HONORIFIC.sub("", name)
    text = _INITIAL.sub("", text)
    text = " ".join(text.split()).casefold()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@dataclass(frozen=True, slots=True)
class Person:
    """One participant, as the speeches export describes them.

    Attributes:
        person_id: The export's stable identifier, or ``None`` for someone who
            only ever appears as an interjector.
        label: Display name.
        faction: Party group, or ``None``. Null means unknown, not independent.
        side: ``"government"``, ``"opposition"`` or ``None``.
        is_mp: Whether the export marks them a member.
        resolved: Whether the name was matched to a speaker in the export.
        match: How it was matched -- "exact", "tokens" for a unique
            token-subset match, or "none".
    """

    person_id: str | None
    label: str
    faction: str | None
    side: str | None
    is_mp: bool | None
    resolved: bool
    match: str = "none"

    @property
    def node_id(self) -> str:
        """A stable node key: the person id if there is one, else the name."""
        return self.person_id or f"name:{normalise_name(self.label)}"


class SpeakerRegistry:
    """Resolves a name string to a :class:`Person`, or admits it cannot.

    Built from the speeches export, so it knows every member who held the floor
    in the cycle. An interjector who never spoke is not in it -- they still
    become a node, with ``resolved=False`` and a null faction.

    Example:
        >>> registry = SpeakerRegistry(
        ...     [
        ...         {
        ...             "person_id": "s1",
        ...             "label": "Vadai Ágnes",
        ...             "faction": "DK",
        ...             "office": None,
        ...             "is_mp": True,
        ...         }
        ...     ],
        ...     governing=frozenset({"Fidesz"}),
        ... )
        >>> person = registry.resolve("Dr. Vadai Ágnes")
        >>> person.person_id, person.faction, person.side, person.resolved
        ('s1', 'DK', 'opposition', True)

        A dropped middle name or a reordered given name still resolves, and
        records that it took the fallback:

        >>> registry.resolve("Agnes Vadai").match
        'tokens'

        An unknown name is admitted as unknown rather than guessed:

        >>> stranger = registry.resolve("Ismeretlen Képviselő")
        >>> stranger.faction, stranger.side, stranger.resolved
        (None, None, False)
    """

    def __init__(
        self, speakers: Iterable[Mapping[str, Any]], *, governing: frozenset[str]
    ) -> None:
        self.governing = governing
        self._by_key: dict[str, Person] = {}
        for speaker in speakers:
            label = speaker.get("label")
            if not label:
                continue
            key = normalise_name(label)
            if not key or key in self._by_key:
                continue
            self._by_key[key] = Person(
                person_id=speaker.get("person_id"),
                label=label,
                faction=speaker.get("faction") or None,
                side=self._side(speaker),
                is_mp=speaker.get("is_mp"),
                resolved=True,
                match="exact",
            )

    def _side(self, speaker: Mapping[str, Any]) -> str | None:
        """Government or opposition, from faction or from executive office."""
        faction = speaker.get("faction")
        if faction:
            return "government" if faction in self.governing else "opposition"
        # Cycle 43's cabinet includes ministers holding no party card; an
        # executive office puts them on the government bench regardless.
        office = speaker.get("office")
        return "government" if office else None

    def resolve(self, name: str) -> Person:
        """Look a name up, returning an unresolved :class:`Person` on a miss.

        Args:
            name: A name as written in a parenthetical.

        Returns:
            A :class:`Person`. Check :attr:`Person.resolved` before trusting
            the faction.
        """
        key = normalise_name(name)
        known = self._by_key.get(key)
        if known is not None:
            return known

        # Transcripts drop a middle name ("Bujdosó Andrea" for "Bujdosó Andrea
        # Anna") and sometimes reorder given names ("Velkey László György" for
        # "Velkey György László"). Leaving those unresolved splits one member
        # across two nodes. The fallback fires only when exactly one registry
        # entry contains every token: an ambiguous name stays unresolved,
        # because merging "Varga László" into one of two possible members
        # would invent an edge.
        tokens = frozenset(key.split())
        if len(tokens) >= 2:
            candidates = [
                person
                for other_key, person in self._by_key.items()
                if tokens <= frozenset(other_key.split())
            ]
            if len(candidates) == 1:
                found = candidates[0]
                return Person(
                    person_id=found.person_id,
                    label=found.label,
                    faction=found.faction,
                    side=found.side,
                    is_mp=found.is_mp,
                    resolved=True,
                    match="tokens",
                )

        return Person(
            person_id=None,
            label=name,
            faction=None,
            side=None,
            is_mp=None,
            resolved=False,
            match="none",
        )

    def __len__(self) -> int:
        return len(self._by_key)


def build_graph(
    edges: Sequence[Mapping[str, Any]],
    *,
    drop_self_loops: bool = True,
) -> nx.DiGraph:
    """Build the directed heckling graph from edge records.

    Parallel interruptions are collapsed into one edge carrying ``weight``, so
    the graph is a :class:`~networkx.DiGraph` rather than a multigraph -- that
    is what both Gephi and a d3 force layout want.

    Args:
        edges: Edge records, each with ``source_id``, ``target_id`` and the
            display and faction fields for both ends.
        drop_self_loops: Whether to drop an MP interrupting their own speech.
            Defaults to ``True``. These are real records -- the note-taker
            catching the floor-holder talking over a heckle -- but they are not
            an interaction between two people, and a self-loop renders as an
            artifact in every layout. They stay in the edge CSV either way.

    Returns:
        A directed graph. Node attributes: ``label``, ``faction``, ``side``,
        ``is_mp``, ``resolved``. Edge attributes: ``weight``.

    Example:
        >>> rows = [
        ...     {
        ...         "source_id": "a", "source": "A", "source_faction": "X",
        ...         "source_side": "opposition", "source_resolved": True,
        ...         "target_id": "b", "target": "B", "target_faction": "Y",
        ...         "target_side": "government", "target_resolved": True,
        ...     }
        ... ] * 3
        >>> graph = build_graph(rows)
        >>> graph.number_of_nodes(), graph.number_of_edges()
        (2, 1)
        >>> graph["a"]["b"]["weight"]
        3
        >>> graph.nodes["b"]["faction"]
        'Y'
    """
    graph = nx.DiGraph()
    for row in edges:
        source, target = row["source_id"], row["target_id"]
        if drop_self_loops and source == target:
            continue
        for node, prefix in ((source, "source"), (target, "target")):
            if node not in graph:
                graph.add_node(
                    node,
                    label=row[prefix],
                    faction=row.get(f"{prefix}_faction"),
                    side=row.get(f"{prefix}_side"),
                    is_mp=row.get(f"{prefix}_is_mp"),
                    resolved=row.get(f"{prefix}_resolved"),
                )
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1
        else:
            graph.add_edge(source, target, weight=1)
    return graph


def annotate_degrees(graph: nx.DiGraph) -> nx.DiGraph:
    """Add per-node interruption totals, in place, and return the graph.

    Adds ``heckles_given`` and ``heckles_received`` (weighted degrees), and
    ``targets`` and ``hecklers`` (how many distinct people, which separates a
    member who shouts at everyone from one who pursues a single opponent).

    Example:
        >>> graph = nx.DiGraph()
        >>> graph.add_edge("a", "b", weight=3)
        >>> graph.add_edge("a", "c", weight=1)
        >>> _ = annotate_degrees(graph)
        >>> graph.nodes["a"]["heckles_given"], graph.nodes["a"]["targets"]
        (4, 2)
        >>> graph.nodes["b"]["heckles_received"], graph.nodes["b"]["hecklers"]
        (3, 1)
    """
    for node in graph.nodes:
        graph.nodes[node]["heckles_given"] = graph.out_degree(node, weight="weight")
        graph.nodes[node]["heckles_received"] = graph.in_degree(node, weight="weight")
        graph.nodes[node]["targets"] = graph.out_degree(node)
        graph.nodes[node]["hecklers"] = graph.in_degree(node)
    return graph


def aggregate_by(
    graph: nx.DiGraph, attribute: str, *, missing: str = "unknown"
) -> nx.DiGraph:
    """Collapse the graph onto a node attribute, such as ``faction``.

    Self-loops are kept here, and they matter: a party edge onto itself is
    intra-party heckling, which is a real and interesting quantity even though
    a person heckling themselves is not.

    Args:
        graph: The person-level graph.
        attribute: The node attribute to group on.
        missing: Label for nodes where the attribute is ``None``. Defaults to
            ``"unknown"`` -- these are pooled, not dropped, so the edge weights
            still sum to the person-level total.

    Returns:
        A new directed graph over the attribute's values.

    Example:
        >>> graph = nx.DiGraph()
        >>> graph.add_node("a", faction="X")
        >>> graph.add_node("b", faction="Y")
        >>> graph.add_node("c", faction="Y")
        >>> graph.add_edge("a", "b", weight=2)
        >>> graph.add_edge("a", "c", weight=1)
        >>> rolled = aggregate_by(graph, "faction")
        >>> rolled["X"]["Y"]["weight"]
        3
    """
    rolled = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        group = data.get(attribute) or missing
        if group not in rolled:
            rolled.add_node(group, members=0)
        rolled.nodes[group]["members"] += 1
    for source, target, data in graph.edges(data=True):
        a = graph.nodes[source].get(attribute) or missing
        b = graph.nodes[target].get(attribute) or missing
        weight = data.get("weight", 1)
        if rolled.has_edge(a, b):
            rolled[a][b]["weight"] += weight
        else:
            rolled.add_edge(a, b, weight=weight)
    return rolled


def to_node_link(graph: nx.DiGraph, **extra: object) -> dict[str, Any]:
    """Serialise to the ``{"nodes": [...], "links": [...]}`` shape d3 expects.

    ``networkx.node_link_data`` is asked for ``edges="links"`` explicitly:
    networkx 3.6 defaults that key to ``"edges"``, which
    ``d3.forceSimulation`` does not read.

    Args:
        graph: Any graph.
        **extra: Extra top-level keys, for provenance travelling with the
            data. Must be JSON-serialisable.

    Returns:
        A JSON-serialisable dict with ``directed``, ``nodes`` and ``links``.

    Example:
        >>> graph = nx.DiGraph()
        >>> graph.add_edge("a", "b", weight=2)
        >>> data = to_node_link(graph, cycle=43)
        >>> sorted(data["links"][0])
        ['source', 'target', 'weight']
        >>> data["links"][0]["source"], data["links"][0]["weight"]
        ('a', 2)
        >>> data["cycle"]
        43
    """
    data = nx.node_link_data(graph, edges="links")
    data.update(extra)
    return data


GRAPHML_UNKNOWN = "unknown"
"""Stand-in for a null in GraphML, which has no null."""


def graphml_safe(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a copy whose attributes GraphML and Gephi can both represent.

    GraphML declares one type per attribute name, and networkx infers that type
    from the values it sees. Two things follow, and both bit this graph:

    * **A null turned into a string splits the key.** Replacing ``None`` with
      ``"unknown"`` on a boolean attribute left ``is_mp`` declared twice, once
      ``boolean`` and once ``string``. Gephi reads one and drops the other.
    * **Python's booleans are not GraphML's.** networkx writes ``True``, while
      the schema wants ``true``; Gephi will not parse the former as a boolean.

    So every attribute is coerced to a single, stable type here: booleans and
    nulls become the strings ``"true"``, ``"false"`` and ``"unknown"``, which
    Gephi imports as a filterable partition. Numbers stay numbers, because
    Gephi's ranking and sizing need them.

    Args:
        graph: The person-level or aggregated graph.

    Returns:
        A copy, safe to hand to :func:`networkx.write_graphml`. The original is
        untouched, so the JSON export keeps its real nulls and booleans.

    Example:
        >>> graph = nx.DiGraph()
        >>> graph.add_node("a", faction=None, is_mp=True, heckles_given=3)
        >>> graph.add_node("b", faction="X", is_mp=None, heckles_given=0)
        >>> graph.add_edge("a", "b", weight=2)
        >>> safe = graphml_safe(graph)
        >>> safe.nodes["a"]["faction"], safe.nodes["a"]["is_mp"]
        ('unknown', 'true')
        >>> safe.nodes["b"]["is_mp"]
        'unknown'

        Numbers are left alone so Gephi can rank on them:

        >>> safe.nodes["a"]["heckles_given"]
        3

        And the original is not modified:

        >>> graph.nodes["a"]["faction"] is None
        True
    """

    def coerce(value: object) -> object:
        if value is None:
            return GRAPHML_UNKNOWN
        if isinstance(value, bool):
            return "true" if value else "false"
        return value

    safe = graph.copy()
    for _, data in safe.nodes(data=True):
        for key, value in list(data.items()):
            data[key] = coerce(value)
    for _, _, data in safe.edges(data=True):
        for key, value in list(data.items()):
            data[key] = coerce(value)
    return safe


def annotate_crossing(graph: nx.DiGraph) -> nx.DiGraph:
    """Mark each edge as crossing the aisle or not, in place.

    Adds ``crossing`` -- ``"cross-bench"``, ``"same-side"`` or ``"unknown"`` --
    plus ``source_side`` and ``target_side``. Gephi cannot derive an edge
    attribute from its endpoints' attributes, so colouring edges by whether
    they cross the floor needs this precomputed.

    Example:
        >>> graph = nx.DiGraph()
        >>> graph.add_node("a", side="opposition")
        >>> graph.add_node("b", side="government")
        >>> graph.add_node("c", side=None)
        >>> graph.add_edge("a", "b", weight=1)
        >>> graph.add_edge("a", "c", weight=1)
        >>> _ = annotate_crossing(graph)
        >>> graph["a"]["b"]["crossing"]
        'cross-bench'
        >>> graph["a"]["c"]["crossing"]
        'unknown'
    """
    for source, target, data in graph.edges(data=True):
        a = graph.nodes[source].get("side")
        b = graph.nodes[target].get("side")
        data["source_side"] = a
        data["target_side"] = b
        if a is None or b is None:
            data["crossing"] = "unknown"
        else:
            data["crossing"] = "same-side" if a == b else "cross-bench"
    return graph
