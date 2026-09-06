"""detect and invalidate contradicted facts"""

from datetime import date


def check_contradiction(store, new_fact):
    """
    Looks at current facts sharing subject+predicate with new_fact. Any that
    are still live (valid_to is None) and disagree on the object get
    invalidated before new_fact is added. Returns the ids invalidated  """
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

    store.add(new_fact)
    return invalidated
