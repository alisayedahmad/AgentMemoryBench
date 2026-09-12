from baselines.no_memory import answer_with_no_memory


class FakeLLMClient:
    def __init__(self, reply):
        self.reply = reply
        self.last_messages = None

    def call(self, model, messages, **kwargs):
        self.last_messages = messages
        return self.reply


def test_returns_the_llm_reply():
    client = FakeLLMClient("Paris")

    answer = answer_with_no_memory("What is the capital of France?", client)

    assert answer == "Paris"


def test_question_goes_in_the_prompt_with_no_facts_attached():
    client = FakeLLMClient("I don't know")

    answer_with_no_memory("Where does the user work?", client)

    assert "Where does the user work?" in client.last_messages[0]["content"]
    assert "Facts:" not in client.last_messages[0]["content"]
