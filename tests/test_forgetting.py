from datetime import date

from memory.curation.forgetting import decay_score, should_forget
from memory.extraction.schemas import Fact


def make_fact(**kwargs):
    kwargs.setdefault("subject", "user")
    kwargs.setdefault("predicate", "works_at")
    kwargs.setdefault("object", "Google")
    kwargs.setdefault("source_episode_id", "ep1")
    kwargs.setdefault("confidence", 0.9)
    return Fact(**kwargs)


def test_fresh_confident_retrieved_fact_has_low_score():
    fact = make_fact(valid_from=date.today().isoformat(), confidence=0.95, retrieval_count=5)
    assert decay_score(fact) < 0.2


def test_old_unconfident_unretrieved_fact_has_high_score():
    fact = make_fact(valid_from="2015-01-01", confidence=0.2, retrieval_count=0)
    assert decay_score(fact) > 0.8


def test_missing_valid_from_treated_as_fresh():
    fact = make_fact(valid_from=None, confidence=0.9, retrieval_count=3)
    assert decay_score(fact) < 0.3


def test_unparseable_valid_from_does_not_crash():
    fact = make_fact(valid_from="since college", confidence=0.9, retrieval_count=3)
    score = decay_score(fact)
    assert 0 <= score <= 1


def test_older_fact_scores_higher_than_newer_one():
    old = make_fact(valid_from="2010-01-01")
    new = make_fact(valid_from=date.today().isoformat())
    assert decay_score(old) > decay_score(new)


def test_should_forget_respects_threshold():
    weak = make_fact(valid_from="2010-01-01", confidence=0.1, retrieval_count=0)
    strong = make_fact(valid_from=date.today().isoformat(), confidence=0.95, retrieval_count=10)
    assert should_forget(weak) is True
    assert should_forget(strong) is False
