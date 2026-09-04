import pytest
from pydantic import ValidationError
from memory.extraction.schemas import Fact


def test_valid_fact():
    fact = Fact(
        subject="user",
        predicate="works_at",
        object="Google",
        valid_from="2025-03",
        source_episode_id="episode_42",
        confidence=0.92,
    )
    assert fact.valid_to is None


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Fact(
            subject="user",
            predicate="works_at",
            object="Google",
            source_episode_id="episode_42",
            confidence=1.5,
        )


def test_empty_subject_rejected():
    with pytest.raises(ValidationError):
        Fact(
            subject="",
            predicate="works_at",
            object="Google",
            source_episode_id="episode_42",
            confidence=0.9,
        )