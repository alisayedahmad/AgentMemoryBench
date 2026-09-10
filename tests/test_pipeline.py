from memory.curation.pipeline import ingest_fact
from memory.extraction.schemas import Fact
from memory.storage.entity_graph import EntityGraph
from memory.storage.semantic import SemanticStore


def make_fact(subject="user", predicate="works_at", object="Google", **kwargs):
    kwargs.setdefault("source_episode_id", "ep1")
    kwargs.setdefault("confidence", 0.9)
    return Fact(subject=subject, predicate=predicate, object=object, **kwargs)


def fake_embed(text):
    vectors = {
        "Google": [1.0, 0.0],
        "the Google office": [0.99, 0.01],
        "Meta": [0.0, 1.0],
    }
    return vectors.get(text, [0.5, 0.5])


def test_new_fact_is_added_and_graphed(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    graph = EntityGraph()

    fact, event = ingest_fact(store, graph, make_fact(), embed=fake_embed)

    assert event == "new"
    assert store.all() == [fact]
    assert graph.edges_for("user")


def test_duplicate_merges_and_never_touches_the_graph(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    graph = EntityGraph()
    original, _ = ingest_fact(store, graph, make_fact(object="Google", source_episode_id="ep1"), embed=fake_embed)

    fact, event = ingest_fact(
        store, graph, make_fact(object="the Google office", source_episode_id="ep2"), embed=fake_embed
    )

    assert event == "duplicate"
    assert fact.id == original.id
    assert len(store.all()) == 1
    assert len(graph.edges_for("user")) == 1


def test_contradicting_fact_invalidates_old_one_and_is_added(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    graph = EntityGraph()
    ingest_fact(store, graph, make_fact(object="Google"), embed=fake_embed)

    fact, event = ingest_fact(store, graph, make_fact(object="Meta"), embed=fake_embed)

    assert event == "contradiction"
    live = [f for f in store.all() if f.valid_to is None]
    assert live == [fact]
    assert fact.object == "Meta"
