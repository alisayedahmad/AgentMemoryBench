"""hybrid retrieval: fuses dense embedding search, BM25 keyword search, and entity graph traversal"""

import re

import numpy as np
from rank_bm25 import BM25Okapi


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _fact_text(fact):
    return f"{fact.subject} {fact.predicate} {fact.object}"


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize(scores):
    """min-max scores into [0,1]; flat scores (including all-zero) collapse to 0.0"""
    if not scores:
        return scores
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _bm25_scores(facts, query):
    corpus = [_tokenize(_fact_text(f)) for f in facts]
    bm25 = BM25Okapi(corpus)
    raw = bm25.get_scores(_tokenize(query))
    return {f.id: score for f, score in zip(facts, raw)}


def _entities_mentioned(graph, query):
    """graph node names that appear as a substring of the query, case-insensitive"""
    query_lower = query.lower()
    return [node for node in graph.graph.nodes if node.lower() in query_lower]


def _graph_scores(graph, query):
    """1.0 for facts reachable by traversing an entity named in the query, else absent"""
    if graph is None:
        return {}
    scores = {}
    for entity in _entities_mentioned(graph, query):
        for _, _, _, data in graph.edges_for(entity):
            scores[data["fact_id"]] = 1.0
    return scores


def hybrid_search(store, query, embed, graph=None, k=5, weights=(0.4, 0.4, 0.2)):
    """
    Ranks facts against a query by combining three signals: dense (cosine
    similarity between embed(query) and embed(fact)), bm25 (keyword overlap,
    min-max normalized per query), and graph (1.0 for facts touching an
    entity named in the query, 0 otherwise). weights = (dense, bm25, graph)
    and doesn't need to sum to 1, it's just a relative weighting for the
    sort. graph=None zeroes out that signal instead of erroring, so this
    still works before a graph exists.

    Returns up to k (fact, score) pairs, highest score first.

    embed() gets called once per fact on every search, same as
    deduplication.py does — fine at this scale, would need a cached
    embedding index before this runs on a real corpus.
    """
    facts = store.all()
    if not facts:
        return []

    dense_w, bm25_w, graph_w = weights

    query_vec = embed(query)
    dense_scores = {f.id: cosine_similarity(query_vec, embed(_fact_text(f))) for f in facts}
    bm25_scores = _normalize(_bm25_scores(facts, query))
    graph_scores = _graph_scores(graph, query)

    combined = {
        f.id: dense_w * dense_scores[f.id]
        + bm25_w * bm25_scores.get(f.id, 0.0)
        + graph_w * graph_scores.get(f.id, 0.0)
        for f in facts
    }

    ranked = sorted(facts, key=lambda f: combined[f.id], reverse=True)
    return [(f, round(combined[f.id], 4)) for f in ranked[:k]]
