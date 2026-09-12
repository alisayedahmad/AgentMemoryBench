from memory.curation.predicate_canon import canonicalize_predicate
from memory.extraction.schemas import Fact
from memory.storage.semantic import SemanticStore


def make_fact(subject="user", predicate="works_at", object="Google", **kwargs):
    kwargs.setdefault("source_episode_id", "ep1")
    kwargs.setdefault("confidence", 0.9)
    return Fact(subject=subject, predicate=predicate, object=object, **kwargs)


def fake_embed(text):
    vectors = {
        "likes": [1.0, 0.0],
        "drinks": [0.95, 0.05],
        "works_at": [0.0, 1.0],
        "hates": [0.0, -1.0],
    }
    return vectors.get(text, [0.5, 0.5])


def test_first_predicate_for_a_subject_is_unchanged(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))

    fact = canonicalize_predicate(store, make_fact(predicate="likes"), embed=fake_embed)

    assert fact.predicate == "likes"


def test_similar_predicate_snaps_to_the_existing_one(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(predicate="likes", object="coffee"))

    fact = canonicalize_predicate(
        store, make_fact(predicate="drinks", object="coffee", source_episode_id="ep2"), embed=fake_embed
    )

    assert fact.predicate == "likes"


def test_different_predicate_is_left_alone(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(predicate="likes", object="coffee"))

    fact = canonicalize_predicate(
        store, make_fact(predicate="hates", object="mondays", source_episode_id="ep2"), embed=fake_embed
    )

    assert fact.predicate == "hates"


def test_exact_match_already_is_a_no_op(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(predicate="works_at", object="Google"))

    fact = canonicalize_predicate(
        store, make_fact(predicate="works_at", object="Meta", source_episode_id="ep2"), embed=fake_embed
    )

    assert fact.predicate == "works_at"


def test_only_looks_at_predicates_for_the_same_subject(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(subject="Sarah", predicate="likes", object="tea"))

    fact = canonicalize_predicate(
        store, make_fact(subject="user", predicate="drinks", object="coffee"), embed=fake_embed
    )

    assert fact.predicate == "drinks"  # Sarah's "likes" doesn't count for user
