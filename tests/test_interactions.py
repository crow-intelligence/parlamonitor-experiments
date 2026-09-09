import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.interactions import (
    Person,
    SpeakerRegistry,
    aggregate_by,
    annotate_degrees,
    build_graph,
    normalise_name,
    to_node_link,
)

SPEAKERS = [
    {
        "person_id": "s1",
        "label": "Vadai Ágnes",
        "faction": "DK",
        "office": None,
        "is_mp": True,
    },
    {
        "person_id": "s2",
        "label": "Bujdosó Andrea Anna",
        "faction": "TISZA",
        "office": None,
        "is_mp": True,
    },
    {
        "person_id": "s3",
        "label": "Vitézy Dávid",
        "faction": None,
        "office": "közlekedési miniszter",
        "is_mp": False,
    },
    {
        "person_id": "s4",
        "label": "Varga László",
        "faction": "MSZP",
        "office": None,
        "is_mp": True,
    },
    {
        "person_id": "s5",
        "label": "Varga László Péter",
        "faction": "TISZA",
        "office": None,
        "is_mp": True,
    },
]


@pytest.fixture
def registry():
    return SpeakerRegistry(SPEAKERS, governing=frozenset({"TISZA"}))


# --- name normalisation -----------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Vadai Ágnes", "Dr. Vadai Ágnes", "dr Vadai Agnes", "  Vadai   Ágnes  "],
)
def test_titles_accents_and_spacing_all_normalise_together(name):
    assert normalise_name(name) == "vadai agnes"


def test_a_leading_initial_is_treated_as_a_title():
    assert normalise_name("Z. Kárpát Dániel") == "karpat daniel"


def test_a_name_that_is_only_a_title_normalises_to_nothing():
    assert normalise_name("Dr. ") == ""


@given(st.text(max_size=30))
def test_normalising_never_raises_and_never_adds_case(text):
    assert normalise_name(text) == normalise_name(text).casefold()


# --- resolution -------------------------------------------------------------


def test_an_exact_name_resolves(registry):
    person = registry.resolve("Vadai Ágnes")
    assert (person.person_id, person.faction, person.match) == ("s1", "DK", "exact")


def test_a_title_does_not_prevent_resolution(registry):
    assert registry.resolve("Dr. Vadai Ágnes").person_id == "s1"


def test_governing_party_sets_the_side(registry):
    assert registry.resolve("Bujdosó Andrea Anna").side == "government"
    assert registry.resolve("Vadai Ágnes").side == "opposition"


def test_a_minister_without_a_faction_sits_on_the_government_side(registry):
    person = registry.resolve("Vitézy Dávid")
    assert person.faction is None
    assert person.side == "government"


def test_a_dropped_middle_name_resolves_through_the_fallback(registry):
    person = registry.resolve("Bujdosó Andrea")
    assert (person.person_id, person.match) == ("s2", "tokens")


def test_a_reordered_given_name_resolves_too(registry):
    assert registry.resolve("Anna Bujdosó Andrea").person_id == "s2"


def test_an_ambiguous_name_stays_unresolved(registry):
    # "Varga László" is exactly one speaker but a prefix of another; the exact
    # match must win rather than the fallback picking between them.
    assert registry.resolve("Varga László").person_id == "s4"


def test_a_prefix_matching_two_speakers_is_refused():
    speakers = [
        {"person_id": "a", "label": "Nagy Béla István", "faction": "X", "is_mp": True},
        {"person_id": "b", "label": "Nagy Béla Zoltán", "faction": "Y", "is_mp": True},
    ]
    registry = SpeakerRegistry(speakers, governing=frozenset())
    person = registry.resolve("Nagy Béla")
    # Merging into either would invent an edge, so neither is chosen.
    assert person.resolved is False
    assert person.faction is None


def test_a_single_token_never_uses_the_fallback(registry):
    assert registry.resolve("Vadai").resolved is False


def test_an_unknown_name_is_unknown_not_guessed(registry):
    person = registry.resolve("Ismeretlen Képviselő")
    assert (person.resolved, person.faction, person.side) == (False, None, None)


def test_an_unresolved_person_still_gets_a_stable_node_id():
    person = Person(None, "Dr. Nagy Béla", None, None, None, False)
    assert person.node_id == "name:nagy bela"


# --- graph ------------------------------------------------------------------


def edge_row(source, target, **kwargs):
    row = {
        "source_id": source,
        "source": source.upper(),
        "source_faction": "X",
        "source_side": "opposition",
        "source_resolved": True,
        "target_id": target,
        "target": target.upper(),
        "target_faction": "Y",
        "target_side": "government",
        "target_resolved": True,
    }
    row.update(kwargs)
    return row


def test_repeated_interruptions_become_one_weighted_edge():
    graph = build_graph([edge_row("a", "b")] * 4)
    assert graph.number_of_edges() == 1
    assert graph["a"]["b"]["weight"] == 4


def test_the_graph_is_directed():
    graph = build_graph([edge_row("a", "b")])
    assert graph.is_directed()
    assert not graph.has_edge("b", "a")


def test_self_loops_are_dropped_by_default():
    graph = build_graph([edge_row("a", "a"), edge_row("a", "b")])
    assert graph.number_of_edges() == 1
    assert not graph.has_edge("a", "a")


def test_self_loops_can_be_kept():
    graph = build_graph([edge_row("a", "a")], drop_self_loops=False)
    assert graph.has_edge("a", "a")


def test_node_attributes_come_from_the_edge_rows():
    graph = build_graph([edge_row("a", "b")])
    assert graph.nodes["a"]["faction"] == "X"
    assert graph.nodes["b"]["side"] == "government"


def test_degrees_count_weight_and_distinct_partners():
    graph = build_graph([edge_row("a", "b")] * 3 + [edge_row("a", "c")])
    annotate_degrees(graph)
    assert graph.nodes["a"]["heckles_given"] == 4
    assert graph.nodes["a"]["targets"] == 2
    assert graph.nodes["b"]["heckles_received"] == 3
    assert graph.nodes["b"]["hecklers"] == 1


# --- aggregation ------------------------------------------------------------


def test_aggregation_sums_weights_onto_the_attribute():
    graph = build_graph([edge_row("a", "b")] * 2 + [edge_row("a", "c")])
    rolled = aggregate_by(graph, "faction")
    assert rolled["X"]["Y"]["weight"] == 3


def test_aggregation_preserves_the_total_weight():
    rows = [edge_row("a", "b")] * 2 + [edge_row("c", "d")]
    graph = build_graph(rows)
    rolled = aggregate_by(graph, "faction")
    total = sum(d["weight"] for *_, d in graph.edges(data=True))
    assert sum(d["weight"] for *_, d in rolled.edges(data=True)) == total


def test_a_null_attribute_is_pooled_not_dropped():
    rows = [edge_row("a", "b", source_faction=None)]
    rolled = aggregate_by(build_graph(rows), "faction")
    assert rolled.has_edge("unknown", "Y")


def test_intra_group_edges_survive_aggregation():
    # Two different people in one party: a real intra-party heckle.
    rows = [edge_row("a", "b", target_faction="X")]
    rolled = aggregate_by(build_graph(rows), "faction")
    assert rolled["X"]["X"]["weight"] == 1


# --- serialisation ----------------------------------------------------------


def test_node_link_uses_the_key_d3_reads():
    data = to_node_link(build_graph([edge_row("a", "b")]))
    # networkx 3.6 defaults this key to "edges", which d3 does not read.
    assert "links" in data
    assert "edges" not in data


def test_node_link_carries_weight_and_endpoints():
    data = to_node_link(build_graph([edge_row("a", "b")] * 2))
    link = data["links"][0]
    assert (link["source"], link["target"], link["weight"]) == ("a", "b", 2)


def test_provenance_travels_with_the_graph():
    data = to_node_link(build_graph([edge_row("a", "b")]), cycle=43)
    assert data["cycle"] == 43


def test_node_link_is_json_serialisable():
    import json

    data = to_node_link(build_graph([edge_row("a", "b", source_faction=None)]))
    assert json.loads(json.dumps(data))["directed"] is True


def test_round_trip_through_node_link_preserves_the_graph():
    graph = build_graph([edge_row("a", "b")] * 3 + [edge_row("c", "b")])
    restored = nx.node_link_graph(to_node_link(graph), edges="links")
    assert restored.number_of_nodes() == graph.number_of_nodes()
    assert restored["a"]["b"]["weight"] == 3


# --- GraphML export ---------------------------------------------------------


def test_a_null_on_a_boolean_attribute_does_not_split_the_key(tmp_path):
    # The bug this guards: replacing None with "unknown" on a boolean attribute
    # left GraphML declaring `is_mp` twice, once boolean and once string, and
    # Gephi silently reads one and drops the other.
    import xml.etree.ElementTree as ET

    from parlamonitor.interactions import graphml_safe

    graph = nx.DiGraph()
    graph.add_node("a", is_mp=True, faction="X")
    graph.add_node("b", is_mp=None, faction=None)
    graph.add_edge("a", "b", weight=1)

    path = tmp_path / "g.graphml"
    nx.write_graphml(graphml_safe(graph), path)

    names = [
        key.get("attr.name")
        for key in ET.parse(path).getroot()
        if key.tag.endswith("key")
    ]
    assert len(names) == len(set(names)), f"duplicate attribute keys: {names}"


def test_booleans_are_written_in_the_form_graphml_defines():
    from parlamonitor.interactions import graphml_safe

    graph = nx.DiGraph()
    graph.add_node("a", flag=True)
    graph.add_node("b", flag=False)
    safe = graphml_safe(graph)
    # networkx writes Python's `True`, which is not GraphML's `true`.
    assert safe.nodes["a"]["flag"] == "true"
    assert safe.nodes["b"]["flag"] == "false"


def test_numbers_survive_so_gephi_can_rank_on_them():
    from parlamonitor.interactions import graphml_safe

    graph = nx.DiGraph()
    graph.add_node("a", heckles_given=7)
    assert graphml_safe(graph).nodes["a"]["heckles_given"] == 7


def test_graphml_safe_does_not_mutate_the_original():
    from parlamonitor.interactions import graphml_safe

    graph = nx.DiGraph()
    graph.add_node("a", faction=None, is_mp=True)
    graphml_safe(graph)
    assert graph.nodes["a"]["faction"] is None
    assert graph.nodes["a"]["is_mp"] is True


def test_the_graph_round_trips_through_graphml(tmp_path):
    from parlamonitor.interactions import graphml_safe

    graph = build_graph([edge_row("a", "b")] * 3 + [edge_row("c", "b")])
    path = tmp_path / "g.graphml"
    nx.write_graphml(graphml_safe(graph), path)
    restored = nx.read_graphml(path)
    assert restored.is_directed()
    assert restored.number_of_nodes() == graph.number_of_nodes()
    assert restored["a"]["b"]["weight"] == 3


# --- edge crossing ----------------------------------------------------------


def test_edges_are_labelled_by_whether_they_cross_the_aisle():
    from parlamonitor.interactions import annotate_crossing

    graph = build_graph([edge_row("a", "b")])  # opposition -> government
    annotate_crossing(graph)
    assert graph["a"]["b"]["crossing"] == "cross-bench"


def test_same_side_edges_are_labelled_as_such():
    from parlamonitor.interactions import annotate_crossing

    graph = build_graph([edge_row("a", "b", target_side="opposition")])
    annotate_crossing(graph)
    assert graph["a"]["b"]["crossing"] == "same-side"


def test_an_unknown_side_is_not_guessed_either_way():
    from parlamonitor.interactions import annotate_crossing

    graph = build_graph([edge_row("a", "b", target_side=None)])
    annotate_crossing(graph)
    assert graph["a"]["b"]["crossing"] == "unknown"
