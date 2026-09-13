from memory.curation.predicate_canon import canonicalize_predicate, _digits_stripped
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
        "drinks_1": [0.95, 0.05],
        "works_at": [0.0, 1.0],
        "hates": [0.0, -1.0],
        "listed_item_1": [0.2, 0.8],
        "listed_item_2": [0.2, 0.8],
        "listed_item_7": [0.2, 0.8],
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


def test_digits_stripped_removes_all_digits(tmp_path):
    assert _digits_stripped("listed_item_7") == "listed_item_"
    assert _digits_stripped("has_hobby_3") == "has_hobby_"
    assert _digits_stripped("recommended_2_3") == "recommended__"
    assert _digits_stripped("works_at") == "works_at"


def test_numbered_predicates_do_not_canonicalize_to_each_other(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(predicate="listed_item_7", object="trail_a"))

    fact = canonicalize_predicate(
        store, make_fact(predicate="listed_item_1", object="trail_b", source_episode_id="ep2"), embed=fake_embed
    )

    assert fact.predicate == "listed_item_1"


def test_numbered_predicate_stays_when_no_similar_exists(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))

    fact = canonicalize_predicate(
        store, make_fact(predicate="listed_item_5", object="restaurant"), embed=fake_embed
    )

    assert fact.predicate == "listed_item_5"


def test_numbered_predicate_can_canonicalize_to_similar_unumbered(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(predicate="drinks", object="coffee"))

    fact = canonicalize_predicate(
        store, make_fact(predicate="drinks_1", object="tea", source_episode_id="ep2"), embed=fake_embed
    )

    assert fact.predicate == "drinks"


def test_multiple_numbered_variants_exclude_each_other(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact(predicate="recommended_1", object="item_a"))
    store.add(make_fact(predicate="recommended_2", object="item_b", source_episode_id="ep2"))

    fact = canonicalize_predicate(
        store, make_fact(predicate="recommended_3", object="item_c", source_episode_id="ep3"), embed=fake_embed
    )

    assert fact.predicate == "recommended_3"


def test_digits_stripped_preserves_non_numbered_predicates(tmp_path):
    assert _digits_stripped("works_at") == "works_at"
    assert _digits_stripped("is_employed_by") == "is_employed_by"
    assert _digits_stripped("foo_bar_baz") == "foo_bar_baz"
