from memory.extraction.schemas import Fact
from memory.storage.entity_graph import EntityGraph

def make_fact(subject="user", predicate="works_at", object="Google", **kwargs):
    kwargs.setdefault("source_episode_id", "ep1")
    kwargs.setdefault("confidence", 0.9)
    return Fact(subject=subject, predicate=predicate, object=object, **kwargs)

def test_add_fact_creates_an_edge():
    g = EntityGraph()
    g.add_fact(make_fact())

    edges = g.edges_for("user")
    assert len(edges) == 1
    assert edges[0][-1]["predicate"] == "works_at"

def test_edges_for_finds_entity_in_either_direction():
    g = EntityGraph()
    g.add_fact(make_fact(subject="user", object="Google"))

    assert len(g.edges_for("user")) == 1
    assert len(g.edges_for("Google")) == 1

def test_valid_as_of_excludes_facts_that_start_later():
    g = EntityGraph()
    g.add_fact(make_fact(valid_from="2026-06-01"))

    assert g.valid_as_of("user", "2026-01-01") == []

def test_valid_as_of_excludes_facts_already_invalidated():
    g = EntityGraph()
    g.add_fact(make_fact(valid_from="2025-01-01", valid_to="2025-12-01"))

    assert g.valid_as_of("user", "2026-01-01") == []

def test_valid_as_of_includes_facts_still_live():
    g = EntityGraph()
    g.add_fact(make_fact(valid_from="2025-01-01", valid_to=None))

    assert len(g.valid_as_of("user", "2026-01-01")) == 1

def test_recorded_at_is_captured_separately_from_valid_from():
    g = EntityGraph()
    g.add_fact(make_fact(valid_from="2020-01-01"), recorded_at="2026-09-09T00:00:00Z")

    edge = g.edges_for("user")[0][-1]
    assert edge["recorded_at"] == "2026-09-09T00:00:00Z"
    assert edge["valid_from"] == "2020-01-01"

def test_reinvoking_add_fact_with_same_id_adds_parallel_edge():
    g = EntityGraph()
    fact = make_fact(valid_from="2025-01-01", valid_to=None)
    g.add_fact(fact, recorded_at="2025-01-01T00:00:00Z")

    invalidated = fact.model_copy(update={"valid_to": "2026-01-01"})
    g.add_fact(invalidated, recorded_at="2026-01-01T00:00:00Z")

    assert len(g.edges_for("user")) == 2
