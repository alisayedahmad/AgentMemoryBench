"""orchestrates a fact through curation: predicate canon, dedup, contradiction, then store + graph, once"""

from memory.curation.contradiction import check_contradiction
from memory.curation.deduplication import deduplicate
from memory.curation.predicate_canon import canonicalize_predicate


def ingest_fact(store, graph, new_fact, embed, as_of=None):
    """canon -> dedup -> contradiction -> add, in that order so nothing writes twice
    returns (fact, event), event is "duplicate", "contradiction", or "new" """
    new_fact = canonicalize_predicate(store, new_fact, embed)

    merged_into_id = deduplicate(store, new_fact, embed)
    if merged_into_id is not None:
        return store.history(merged_into_id)[-1], "duplicate"

    invalidated = check_contradiction(store, new_fact, as_of=as_of)
    store.add(new_fact)
    graph.add_fact(new_fact)

    return new_fact, "contradiction" if invalidated else "new"
