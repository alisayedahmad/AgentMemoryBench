from scripts.rejudge import rejudge

class FakeLLMClient:
    """replies in order, one per judge call"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.i = 0
        self.cache_flags = []

    def call(self, model, messages, **kwargs):
        self.cache_flags.append(kwargs.get("use_cache"))
        reply = self.replies[self.i]
        self.i += 1
        return reply

def make_result(correct, qid="q1"):
    return {
        "question_id": qid,
        "question_type": "single-session-user",
        "is_abstention": False,
        "question": "q",
        "gold": "a",
        "answer": "r",
        "correct": correct,
    }

def test_same_verdict_is_not_a_flip():
    results = [make_result(True), make_result(False, "q2")]
    client = FakeLLMClient(["yes", "no"])

    assert rejudge(results, client) == []

def test_changed_verdict_is_reported_with_both_values():
    results = [make_result(True)]
    client = FakeLLMClient(["no"])

    flips = rejudge(results, client)

    assert len(flips) == 1
    assert flips[0]["was"] is True
    assert flips[0]["now"] is False

def test_cache_is_bypassed_or_the_measurement_is_meaningless():
    results = [make_result(True), make_result(True, "q2")]
    client = FakeLLMClient(["yes", "yes"])

    rejudge(results, client)

    assert client.cache_flags == [False, False]
