"""orchestrates a single incoming fact through curation: dedup, then contradiction, then storage and the graph — in that order, exactly once"""

from memory.curation.contradiction import check_contradiction
from memory.curation.deduplication import deduplicate


def ingest_fact(store, graph, new_fact, embed):
    """
    Dedup runs first: if new_fact is just a reworded restatement of a live
    fact, it gets merged into that fact and we're done — a merge can't be
    a contradiction of the thing it was merged into.

    If dedup finds nothing close enough, new_fact is genuinely new or a
    genuine change. Contradiction-checking runs next, against the other
    live facts still in the store (new_fact isn't in there yet, so it can't
    match itself). Then new_fact gets added, and the graph gets the edge.

    Returns (fact, event): fact is the merged original on a duplicate,
    new_fact otherwise. event is "duplicate", "contradiction", or "new".
    """
    merged_into_id = deduplicate(store, new_fact, embed)
    if merged_into_id is not None:
        return store.history(merged_into_id)[-1], "duplicate"

    invalidated = check_contradiction(store, new_fact)
    store.add(new_fact)
    graph.add_fact(new_fact)

    return new_fact, "contradiction" if invalidated else "new"
