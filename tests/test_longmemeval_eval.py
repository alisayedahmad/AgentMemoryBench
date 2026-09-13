from bench.evaluators.longmemeval_eval import judge_answer


class FakeLLMClient:
    def __init__(self, reply):
        self.reply = reply
        self.last_messages = None

    def call(self, model, messages, **kwargs):
        self.last_messages = messages
        return self.reply


def test_yes_reply_is_correct():
    client = FakeLLMClient("Yes")
    result = judge_answer("single-session-user", "q", "answer", "response with answer in it", client)
    assert result is True


def test_no_reply_is_incorrect():
    client = FakeLLMClient("No")
    result = judge_answer("single-session-user", "q", "answer", "wrong response", client)
    assert result is False


def test_temporal_reasoning_gets_the_off_by_one_clause():
    client = FakeLLMClient("Yes")
    judge_answer("temporal-reasoning", "how many days?", "18 days", "19 days", client)
    assert "off-by-one" in client.last_messages[0]["content"]


def test_knowledge_update_gets_its_clause():
    client = FakeLLMClient("Yes")
    judge_answer("knowledge-update", "q", "a", "r", client)
    assert "updated answer" in client.last_messages[0]["content"]


def test_single_session_preference_uses_the_rubric_template():
    client = FakeLLMClient("Yes")
    judge_answer("single-session-preference", "q", "likes concise answers", "r", client)
    assert "Rubric:" in client.last_messages[0]["content"]
    assert "Correct Answer:" not in client.last_messages[0]["content"]


def test_multi_session_uses_the_default_template():
    client = FakeLLMClient("Yes")
    judge_answer("multi-session", "q", "a", "r", client)
    assert "Correct Answer:" in client.last_messages[0]["content"]
    assert "off-by-one" not in client.last_messages[0]["content"]


def test_abstention_uses_the_abstention_template_regardless_of_type():
    client = FakeLLMClient("Yes")
    judge_answer("temporal-reasoning", "q", "not enough info", "I don't know", client, is_abstention=True)
    assert "unanswerable" in client.last_messages[0]["content"]
    assert "Explanation:" in client.last_messages[0]["content"]
