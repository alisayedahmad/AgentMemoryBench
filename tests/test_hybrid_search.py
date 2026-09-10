from memory.extraction.schemas import Fact
from memory.retrieval.hybrid_search import cosine_similarity, hybrid_search
from memory.storage.entity_graph import EntityGraph
from memory.storage.semantic import SemanticStore


def make_fact(subject="user", predicate="works_at", object="Google", **kwargs):
    kwargs.setdefault("source_episode_id", "ep1")
    kwargs.setdefault("confidence", 0.9)
    return Fact(subject=subject, predicate=predicate, object=object, **kwargs)


def fake_embed(text):
    """toy embedding: 1D-ish semantics along two axes, job vs. food"""
    vectors = {
        "user works_at Google": [1.0, 0.0],
        "user works_at Meta": [0.9, 0.1],
        "user likes coffee": [0.0, 1.0],
        "where does the user work": [1.0, 0.0],
        "favorite drink": [0.0, 1.0],
    }
    return vectors.get(text, [0.5, 0.5])


def test_cosine_similarity_matches_deduplication_behavior():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_empty_store_returns_empty_list(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    assert hybrid_search(store, "anything", embed=fake_embed) == []


def test_dense_signal_ranks_semantically_closer_fact_first(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    job_fact = make_fact(predicate="works_at", object="Google")
    food_fact = make_fact(predicate="likes", object="coffee")
    store.add(job_fact)
    store.add(food_fact)

    results = hybrid_search(
        store, "where does the user work", embed=fake_embed, weights=(1.0, 0.0, 0.0)
    )

    assert results[0][0].id == job_fact.id
    assert results[0][1] > results[1][1]


def test_bm25_signal_ranks_keyword_match_first(tmp_path):
    # BM25 idf is exactly zero for a term that splits a 2-doc corpus
    # evenly, so a third, unrelated doc is needed for the match to stand out
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    google_fact = make_fact(predicate="works_at", object="Google")
    meta_fact = make_fact(predicate="works_at", object="Meta")
    coffee_fact = make_fact(predicate="likes", object="coffee")
    store.add(google_fact)
    store.add(meta_fact)
    store.add(coffee_fact)

    results = hybrid_search(store, "Google", embed=fake_embed, weights=(0.0, 1.0, 0.0))

    assert results[0][0].id == google_fact.id
    assert results[0][1] > results[1][1]


def test_graph_signal_boosts_fact_touching_entity_named_in_query(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    google_fact = make_fact(subject="user", predicate="works_at", object="Google")
    coffee_fact = make_fact(subject="user", predicate="likes", object="coffee")
    store.add(google_fact)
    store.add(coffee_fact)

    graph = EntityGraph()
    graph.add_fact(google_fact)
    graph.add_fact(coffee_fact)

    results = hybrid_search(
        store, "tell me about Google", embed=fake_embed, graph=graph, weights=(0.0, 0.0, 1.0)
    )

    assert results[0][0].id == google_fact.id
    assert results[0][1] == 1.0
    assert results[1][1] == 0.0


def test_missing_graph_does_not_error_and_just_zeroes_that_signal(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact())

    results = hybrid_search(store, "Google", embed=fake_embed, graph=None)

    assert len(results) == 1


def test_k_limits_number_of_results(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    for i in range(5):
        store.add(make_fact(predicate="likes", object=f"thing{i}", source_episode_id=f"ep{i}"))

    results = hybrid_search(store, "thing", embed=fake_embed, k=2)

    assert len(results) == 2


def test_results_sorted_highest_score_first(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    job_fact = make_fact(predicate="works_at", object="Google")
    food_fact = make_fact(predicate="likes", object="coffee")
    store.add(job_fact)
    store.add(food_fact)

    results = hybrid_search(store, "favorite drink", embed=fake_embed)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
