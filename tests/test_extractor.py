import json
from memory.extraction.extractor import Extractor
class FakeLLMClient:
    def __init__(self, reply):
        self.reply = reply

    def call(self, model, messages, **kwargs):
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