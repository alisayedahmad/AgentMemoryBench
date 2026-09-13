"""detect and invalidate contradicted facts"""

from datetime import date

def check_contradiction(store, new_fact, as_of=None):
    """invalidates live facts with same subject+predicate but a different object, returns their ids
    doesn't add new_fact itself — pipeline.py owns that single add"""
    existing = store.find(subject=new_fact.subject, predicate=new_fact.predicate)
    invalidated = []

    for fact in existing:
        if fact.valid_to is not None:
            continue
        if fact.object == new_fact.object:
            continue
        cutoff = new_fact.valid_from or as_of or date.today().isoformat()
        if fact.valid_from and cutoff < fact.valid_from:
            continue  # new fact is older, it can't be superseding this one
        store.invalidate(fact.id, valid_to=cutoff)
        invalidated.append(fact.id)

    return invalidated
