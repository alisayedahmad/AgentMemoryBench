"""snaps a new fact's predicate to an existing one for the same subject when they mean the same thing"""

import re

from memory.curation.deduplication import cosine_similarity

def _digits_stripped(predicate):
    return re.sub(r"\d+", "", predicate)

def canonicalize_predicate(store, new_fact, embed, threshold=0.75):
    """rewrites predicate to match an existing similar one for the same subject, else unchanged
    don't lower threshold — "likes" vs "hates" scores 0.508, above real synonyms like works_at/is employed by"""
    known = {f.predicate for f in store.find(subject=new_fact.subject)}
    if not known:
        return new_fact

    # listed_item_7 vs listed_item_8 embed near-identically, merging them would destroy the position
    stripped = _digits_stripped(new_fact.predicate)
    candidates = {
        p for p in known
        if not (p != new_fact.predicate and _digits_stripped(p) == stripped)
    }
    if not candidates:
        return new_fact

    new_vec = embed(new_fact.predicate)
    best_predicate, best_score = max(
        ((p, cosine_similarity(new_vec, embed(p))) for p in candidates),
        key=lambda pair: pair[1],
    )

    if best_score >= threshold and best_predicate != new_fact.predicate:
        return new_fact.model_copy(update={"predicate": best_predicate})

    return new_fact
