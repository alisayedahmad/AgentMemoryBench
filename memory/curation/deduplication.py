"""merge facts that say the same thing in different words"""

import numpy as np


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def deduplicate(store, new_fact, embed, threshold=0.85):
    """
    Compares new_fact's object against current, live facts sharing
    subject+predicate. If the closest one is similar enough, new_fact is
    treated as a duplicate: its episode gets recorded on the existing fact
    instead of adding a second copy, and this function persists that merge.
    Returns the id merged into.

    If nothing is similar enough, this function doesn't add new_fact — it
    returns None and leaves that to the caller. It can't call
    check_contradiction() and add new_fact itself here, because the
    pipeline needs contradiction-checking to run first, against the other
    live facts, before new_fact exists in the store, and it needs to get
    added exactly once.
    """
    candidates = [
        f for f in store.find(subject=new_fact.subject, predicate=new_fact.predicate)
        if f.valid_to is None
    ]

    if candidates:
        new_vec = embed(new_fact.object)
        best, best_score = max(
            ((f, cosine_similarity(new_vec, embed(f.object))) for f in candidates),
            key=lambda pair: pair[1],
        )

        if best_score >= threshold:
            merged = best.model_copy(update={
                "also_seen_in": best.also_seen_in + [new_fact.source_episode_id],
            })
            store.add(merged)
            return best.id

    return None
