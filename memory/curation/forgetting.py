"""decay scoring: deprioritizes old, low-confidence, rarely-used facts (never deletes)"""

from datetime import date


def _pad_to_full_date(value):
    parts = value.split("-")
    while len(parts) < 3:
        parts.append("01")
    return "-".join(parts[:3])


def decay_score(fact, as_of=None, half_life_days=180):
    """
    0 = keep prioritizing, 1 = fully decayed. Combines three signals:
    age since valid_from, confidence, and how often it's been retrieved.
    Facts with no valid_from, or a valid_from we can't parse (e.g. "since
    college"), are treated as age 0 — we have no world-time to measure
    against, so we don't penalize them for it.
    """
    as_of = as_of or date.today()

    age_days = 0
    if fact.valid_from:
        try:
            age_days = max((as_of - date.fromisoformat(_pad_to_full_date(fact.valid_from))).days, 0)
        except ValueError:
            age_days = 0

    age_factor = 1 - 0.5 ** (age_days / half_life_days)
    confidence_factor = 1 - fact.confidence
    retrieval_factor = 1.0 if fact.retrieval_count == 0 else 1 / (1 + fact.retrieval_count)

    return round((age_factor + confidence_factor + retrieval_factor) / 3, 4)


def should_forget(fact, as_of=None, threshold=0.66):
    return decay_score(fact, as_of=as_of) >= threshold
