"""snaps a new fact's predicate to an existing one for the same subject when they mean the same thing"""

from memory.curation.deduplication import cosine_similarity


def canonicalize_predicate(store, new_fact, embed, threshold=0.75):
    """
    Looks at predicates already used for new_fact.subject. If the closest
    one is similar enough, rewrites new_fact.predicate to match it exactly
    — so a later dedup/contradiction check (which needs an exact string
    match) actually finds it. If nothing is close, new_fact comes back
    unchanged: it's the first fact of its kind for this subject.

    threshold=0.75 is a guess, not a measured value — I couldn't test this
    against a real embedding model (no network to huggingface.co here).
    Tune it against what you actually see.
    """
    known = {f.predicate for f in store.find(subject=new_fact.subject)}
    if not known:
        return new_fact

    new_vec = embed(new_fact.predicate)
    best_predicate, best_score = max(
        ((p, cosine_similarity(new_vec, embed(p))) for p in known),
        key=lambda pair: pair[1],
    )

    if best_score >= threshold and best_predicate != new_fact.predicate:
        return new_fact.model_copy(update={"predicate": best_predicate})

    return new_fact
