from infra.llm_cache import LLMCache
from infra.llm_client import LLMClient


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, *texts):
        self.content = [FakeTextBlock(t) for t in texts]


class FakeMessages:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kwargs):
        self.outer.call_count += 1
        self.outer.last_kwargs = kwargs
        return FakeResponse(*self.outer.reply_blocks)


class FakeAnthropicClient:
    """stands in for anthropic.Anthropic() — records calls, returns canned text block(s)"""

    def __init__(self, reply="hello"):
        self.reply_blocks = [reply] if isinstance(reply, str) else list(reply)
        self.call_count = 0
        self.last_kwargs = None
        self.messages = FakeMessages(self)


def test_call_returns_response_text(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path / "cache"))
    fake = FakeAnthropicClient(reply="Paris")
    client = LLMClient(client=fake, cache=cache)

    result = client.call(model="claude-sonnet-5", messages=[{"role": "user", "content": "capital of France?"}])

    assert result == "Paris"
    assert fake.call_count == 1


def test_second_identical_call_hits_cache_not_api(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path / "cache"))
    fake = FakeAnthropicClient(reply="Paris")
    client = LLMClient(client=fake, cache=cache)
    messages = [{"role": "user", "content": "capital of France?"}]

    client.call(model="claude-sonnet-5", messages=messages)
    client.call(model="claude-sonnet-5", messages=messages)

    assert fake.call_count == 1


def test_different_params_bypass_cache(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path / "cache"))
    fake = FakeAnthropicClient(reply="Paris")
    client = LLMClient(client=fake, cache=cache)
    messages = [{"role": "user", "content": "capital of France?"}]

    client.call(model="claude-sonnet-5", messages=messages, temperature=0.0)
    client.call(model="claude-sonnet-5", messages=messages, temperature=0.7)

    assert fake.call_count == 2


def test_default_max_tokens_is_sent(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path / "cache"))
    fake = FakeAnthropicClient()
    client = LLMClient(client=fake, cache=cache)

    client.call(model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}])

    assert fake.last_kwargs["max_tokens"] == 1024


def test_explicit_max_tokens_overrides_default_and_is_part_of_the_cache_key(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path / "cache"))
    fake = FakeAnthropicClient(reply="Paris")
    client = LLMClient(client=fake, cache=cache)
    messages = [{"role": "user", "content": "capital of France?"}]

    client.call(model="claude-sonnet-5", messages=messages, max_tokens=50)
    client.call(model="claude-sonnet-5", messages=messages, max_tokens=200)

    assert fake.call_count == 2


def test_multiple_text_blocks_are_joined(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path / "cache"))
    fake = FakeAnthropicClient(reply=["hello ", "world"])
    client = LLMClient(client=fake, cache=cache)

    result = client.call(model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}])

    assert result == "hello world"


def test_works_as_drop_in_replacement_for_consolidation(tmp_path):
    """same interface FakeLLMClient already satisfies in test_consolidation.py"""
    from memory.curation.consolidation import consolidate
    from memory.extraction.schemas import Fact
    from memory.storage.semantic import SemanticStore

    cache = LLMCache(cache_dir=str(tmp_path / "cache"))
    fake = FakeAnthropicClient(reply="works in tech, likes coffee, lives in Paris")
    client = LLMClient(client=fake, cache=cache)

    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    facts = [("works_at", "Google"), ("likes", "coffee"), ("lives_in", "Paris")]
    for i, (predicate, obj) in enumerate(facts):
        store.add(Fact(subject="user", predicate=predicate, object=obj, source_episode_id=f"ep{i}", confidence=0.9))

    result = consolidate(store, "user", client, threshold=2)

    assert result.object == "works in tech, likes coffee, lives in Paris"
    assert fake.call_count == 1
