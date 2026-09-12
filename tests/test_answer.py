from memory.extraction.schemas import Fact
from memory.retrieval.answer import answer_question


class FakeLLMClient:
    def __init__(self, reply):
        self.reply = reply
        self.last_messages = None

    def call(self, model, messages, **kwargs):
        self.last_messages = messages
        return self.reply


def make_fact(subject="user", predicate="works_at", object="Google", **kwargs):
    kwargs.setdefault("source_episode_id", "ep1")
    kwargs.setdefault("confidence", 0.9)
    return Fact(subject=subject, predicate=predicate, object=object, **kwargs)


def test_no_facts_says_i_dont_know_without_calling_the_llm():
    client = FakeLLMClient("should not be used")

    answer = answer_question("anything", [], client)

    assert answer == "I don't know"
    assert client.last_messages is None


def test_facts_get_listed_in_the_prompt():
    client = FakeLLMClient("Google")
    fact = make_fact(object="Google")

    answer = answer_question("Where does the user work?", [fact], client)

    assert answer == "Google"
    assert "user works_at Google" in client.last_messages[0]["content"]


def test_invalidated_facts_are_marked_not_current_in_the_prompt():
    client = FakeLLMClient("Meta")
    old_fact = make_fact(object="Google", valid_to="2026-09")

    answer_question("Where does the user work?", [old_fact], client)

    assert "no longer current as of 2026-09" in client.last_messages[0]["content"]
