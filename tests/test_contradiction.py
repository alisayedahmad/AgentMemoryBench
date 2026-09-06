from datetime import date

from memory.curation.contradiction import check_contradiction
from memory.extraction.schemas import Fact
from memory.storage.semantic import SemanticStore


def make_fact(subject="user", predicate="works_at", object="Google", **kwargs):
    return Fact(subject=subject, predicate=predicate, object=object,
                source_episode_id="ep1", confidence=0.9, **kwargs)


def test_same_object_is_not_a_contradiction(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(object="Google"))

    invalidated = check_contradiction(store, make_fact(object="Google"))

    assert invalidated == []
    assert len(store.all()) == 2  # both kept, dedup is a separate concern


def test_different_object_invalidates_old_fact(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(object="Google", valid_from="2025-03"))

    check_contradiction(store, make_fact(object="Meta", valid_from="2026-01"))

    current = store.find(subject="user", predicate="works_at")
    live = [f for f in current if f.valid_to is None]
    assert len(live) == 1
    assert live[0].object == "Meta"


def test_cutoff_uses_new_facts_valid_from(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    old = make_fact(object="Google")
    store.add(old)

    check_contradiction(store, make_fact(object="Meta", valid_from="2026-01"))

    history = store.history(old.id)
    assert history[-1].valid_to == "2026-01"


def test_cutoff_falls_back_to_today_when_new_fact_has_no_date(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    old = make_fact(object="Google")
    store.add(old)

    check_contradiction(store, make_fact(object="Meta"))

    history = store.history(old.id)
    assert history[-1].valid_to == date.today().isoformat()


def test_different_predicate_is_not_a_contradiction(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(predicate="works_at", object="Google"))

    invalidated = check_contradiction(store, make_fact(predicate="likes", object="coffee"))

    assert invalidated == []


def test_already_invalidated_fact_is_skipped_next_time(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    google = make_fact(object="Google")
    store.add(google)

    check_contradiction(store, make_fact(object="Meta", valid_from="2026-01"))
    invalidated_again = check_contradiction(store, make_fact(object="Freelance", valid_from="2026-06"))

    assert google.id not in invalidated_again
