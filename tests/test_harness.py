import json

from bench.harness import run_case


class FakeLLMClient:
    """returns canned replies in call order: extraction calls first, then answer calls"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.i = 0

    def call(self, model, messages, **kwargs):
        reply = self.replies[self.i]
        self.i += 1
        return reply


def fake_embed(text):
    vectors = {"Google": [1.0, 0.0], "works": [0.9, 0.1]}
    return vectors.get(text, [0.5, 0.5])


def test_run_case_ties_extraction_retrieval_and_answering_together():
    replies = [
        json.dumps([{"subject": "user", "predicate": "works_at", "object": "Google", "confidence": 0.9}]),
        "Google",
    ]
    client = FakeLLMClient(replies)
    case = {
        "episodes": ["I work at Google."],
        "questions": [{"question": "Where does the user work?", "expects": "Google"}],
    }

    results = run_case(case, client, fake_embed)

    assert len(results) == 1
    assert results[0]["answer"] == "Google"
    assert results[0]["passed"] is True


def test_wrong_answer_is_marked_failed():
    replies = [
        json.dumps([{"subject": "user", "predicate": "works_at", "object": "Google", "confidence": 0.9}]),
        "I don't know",
    ]
    client = FakeLLMClient(replies)
    case = {
        "episodes": ["I work at Google."],
        "questions": [{"question": "Where does the user work?", "expects": "Google"}],
    }

    results = run_case(case, client, fake_embed)

    assert results[0]["passed"] is False


def test_multiple_questions_each_get_their_own_answer_call():
    replies = [
        json.dumps([{"subject": "user", "predicate": "works_at", "object": "Google", "confidence": 0.9}]),
        "Google",
        "I don't know",
    ]
    client = FakeLLMClient(replies)
    case = {
        "episodes": ["I work at Google."],
        "questions": [
            {"question": "Where does the user work?", "expects": "Google"},
            {"question": "What car does the user drive?", "expects": "Toyota"},
        ],
    }

    results = run_case(case, client, fake_embed)

    assert results[0]["passed"] is True
    assert results[1]["passed"] is False


def test_run_case_no_memory_uses_no_extraction_and_no_retrieval():
    from bench.harness import run_case_no_memory

    client = FakeLLMClient(["I don't know"])
    case = {"episodes": ["irrelevant, not used"], "questions": [{"question": "anything", "expects": "i don't know"}]}

    results = run_case_no_memory(case, client)

    assert client.i == 1  # exactly one call, no extraction calls happened
    assert results[0]["passed"] is True
