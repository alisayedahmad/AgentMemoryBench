import json
from datetime import date

from memory.extraction.extractor import Extractor
class FakeLLMClient:
    def __init__(self, reply):
        self.reply = reply
        self.last_messages = None

    def call(self, model, messages, **kwargs):
        self.last_messages = messages
        return self.reply

def test_extracts_valid_facts():
    reply = json.dumps([{
        "subject": "user", "predicate": "works_at", "object": "Google",
        "valid_from": "2025-03", "valid_to": None, "confidence": 0.9,
    }])
    extractor = Extractor(FakeLLMClient(reply))

    facts = extractor.extract("I started at Google in March", "episode_1")

    assert len(facts) == 1
    assert facts[0].subject == "user"
    assert facts[0].source_episode_id == "episode_1"

def test_strips_markdown_code_fence():
    reply = '```json\n' + json.dumps([{
        "subject": "user", "predicate": "likes", "object": "coffee",
        "valid_from": None, "valid_to": None, "confidence": 0.8,
    }]) + '\n```'
    extractor = Extractor(FakeLLMClient(reply))

    facts = extractor.extract("I love coffee", "episode_2")

    assert len(facts) == 1
    assert facts[0].object == "coffee"


def test_invalid_json_returns_empty():
    extractor = Extractor(FakeLLMClient("not json at all"))
    facts = extractor.extract("some text", "episode_3")
    assert facts == []


def test_skips_malformed_fact_keeps_valid_ones():
    reply = json.dumps([
        {"subject": "", "predicate": "x", "object": "y", "confidence": 0.5},
        {"subject": "user", "predicate": "works_at", "object": "Meta", "confidence": 0.7},
    ])
    extractor = Extractor(FakeLLMClient(reply))
    facts = extractor.extract("some text", "episode_4")
    assert len(facts) == 1
    assert facts[0].object == "Meta"


def test_self_reference_words_normalize_to_user():
    reply = json.dumps([
        {"subject": "I", "predicate": "works_at", "object": "Google", "confidence": 0.9},
        {"subject": "speaker", "predicate": "works_at", "object": "Meta", "confidence": 0.9},
        {"subject": "The Speaker", "predicate": "likes", "object": "coffee", "confidence": 0.8},
    ])
    extractor = Extractor(FakeLLMClient(reply))

    facts = extractor.extract("some text", "episode_5")

    assert [f.subject for f in facts] == ["user", "user", "user"]


def test_real_name_is_not_normalized():
    reply = json.dumps([
        {"subject": "Sarah", "predicate": "works_at", "object": "Meta", "confidence": 0.9},
    ])
    extractor = Extractor(FakeLLMClient(reply))

    facts = extractor.extract("some text", "episode_6")

    assert facts[0].subject == "Sarah"


def test_missing_subject_is_skipped_not_crashed():
    reply = json.dumps([
        {"subject": None, "predicate": "x", "object": "y", "confidence": 0.5},
        {"predicate": "x", "object": "y", "confidence": 0.5},
    ])
    extractor = Extractor(FakeLLMClient(reply))

    facts = extractor.extract("some text", "episode_7")

    assert facts == []


def test_as_of_defaults_to_today_and_goes_in_the_prompt():
    client = FakeLLMClient(json.dumps([]))
    extractor = Extractor(client)

    extractor.extract("some text", "episode_8")

    assert date.today().isoformat() in client.last_messages[0]["content"]


def test_explicit_as_of_overrides_today_in_the_prompt():
    client = FakeLLMClient(json.dumps([]))
    extractor = Extractor(client)

    extractor.extract("some text", "episode_9", as_of="2025-03-01")

    assert "2025-03-01" in client.last_messages[0]["content"]


def test_vague_date_word_is_dropped():
    reply = json.dumps([{
        "subject": "user", "predicate": "worked_at", "object": "Google",
        "valid_from": "2025-03", "valid_to": "recent", "confidence": 0.9,
    }])
    extractor = Extractor(FakeLLMClient(reply))

    facts = extractor.extract("some text", "episode_10")

    assert facts[0].valid_from == "2025-03"
    assert facts[0].valid_to is None


def test_year_and_full_date_are_both_kept():
    reply = json.dumps([
        {"subject": "user", "predicate": "works_at", "object": "Google", "valid_from": "2025", "confidence": 0.9},
        {"subject": "user", "predicate": "likes", "object": "coffee", "valid_from": "2025-06-15", "confidence": 0.8},
    ])
    extractor = Extractor(FakeLLMClient(reply))

    facts = extractor.extract("some text", "episode_11")

    assert facts[0].valid_from == "2025"
    assert facts[1].valid_from == "2025-06-15"