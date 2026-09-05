import pytest
from memory.extraction.schemas import Fact
from memory.storage.semantic import SemanticStore

#make a Fact with default values for testing
def make_fact(subject="user", predicate="works_at", object="Google", **kwargs):
    return Fact(subject=subject, predicate=predicate, object=object,
                source_episode_id="ep1", confidence=0.9, **kwargs)


def test_add_and_all(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact())

    facts = store.all()
    assert len(facts) == 1
    assert facts[0].object == "Google"


def test_find_by_subject(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(subject="user", object="Google"))
    store.add(make_fact(subject="alice", object="Meta"))

    results = store.find(subject="user")
    assert len(results) == 1
    assert results[0].object == "Google"


def test_find_by_subject_and_predicate(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(subject="user", predicate="works_at", object="Google"))
    store.add(make_fact(subject="user", predicate="likes", object="coffee"))

    results = store.find(subject="user", predicate="works_at")
    assert len(results) == 1
    assert results[0].object == "Google"


def test_invalidate_updates_current_view(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    fact = make_fact()
    store.add(fact)

    store.invalidate(fact.id, valid_to="2026-01")

    current = store.find(subject="user", predicate="works_at")
    assert len(current) == 1
    assert current[0].valid_to == "2026-01"


def test_invalidate_keeps_full_history(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    fact = make_fact()
    store.add(fact)
    store.invalidate(fact.id, valid_to="2026-01")

    history = store.history(fact.id)
    assert len(history) == 2
    assert history[0].valid_to is None
    assert history[1].valid_to == "2026-01"


def test_invalidate_unknown_id_raises(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    with pytest.raises(ValueError):
        store.invalidate("does-not-exist", valid_to="2026-01")