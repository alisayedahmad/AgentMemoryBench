"""detect and invalidate contradicted facts"""

from datetime import date


def check_contradiction(store, new_fact):
    """
    Looks at current facts sharing subject+predicate with new_fact. Any that
    are still live (valid_to is None) and disagree on the object get
    invalidated. Returns the ids invalidated.

    Doesn't add new_fact itself — that's the caller's job. This function
    only decides what gets invalidated; owning the single store.add() call
    for new_fact belongs to whoever's orchestrating the full pipeline
    (dedup, then this, then the actual add), so it happens exactly once.
    """
    existing = store.find(subject=new_fact.subject, predicate=new_fact.predicate)
    invalidated = []

    for fact in existing:
        if fact.valid_to is not None:
            continue
        if fact.object == new_fact.object:
            continue
        cutoff = new_fact.valid_from or date.today().isoformat()
        store.invalidate(fact.id, valid_to=cutoff)
        invalidated.append(fact.id)

    return invalidated
