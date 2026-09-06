import math

from memory.curation.deduplication import cosine_similarity, deduplicate
from memory.extraction.schemas import Fact
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


def test_cosine_similarity_identical_vectors():
    assert math.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)


def test_similar_wording_merges_instead_of_duplicating(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    original = make_fact(object="Google", source_episode_id="ep1")
    store.add(original)

    merged_into = deduplicate(
        store, make_fact(object="the Google office", source_episode_id="ep2"), embed=fake_embed
    )

    assert merged_into == original.id
    current = store.all()
    assert len(current) == 1
    assert current[0].also_seen_in == ["ep2"]


def test_different_meaning_is_kept_as_separate_fact(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(object="Google", source_episode_id="ep1"))

    merged_into = deduplicate(
        store, make_fact(object="Meta", source_episode_id="ep2"), embed=fake_embed
    )

    assert merged_into is None
    assert len(store.all()) == 2


def test_first_fact_for_a_subject_predicate_is_just_added(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))

    merged_into = deduplicate(store, make_fact(object="Google"), embed=fake_embed)

    assert merged_into is None
    assert len(store.all()) == 1
