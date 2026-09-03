from infra.llm_cache import LLMCache
def test_hit_on_identical_call(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path))
    cache.set("sonnet", [{"role": "user", "content": "hi"}], {"temperature": 0}, "hello back")

    result = cache.get("sonnet", [{"role": "user", "content": "hi"}], {"temperature": 0})
    assert result == "hello back"
def test_miss_on_different_params(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path))
    cache.set("sonnet", [{"role": "user", "content": "hi"}], {"temperature": 0}, "hello back")

    result = cache.get("sonnet", [{"role": "user", "content": "hi"}], {"temperature": 1})
    assert result is None
def test_miss_on_different_messages(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path))
    cache.set("sonnet", [{"role": "user", "content": "hi"}], {"temperature": 0}, "hello back")

    result = cache.get("sonnet", [{"role": "user", "content": "bye"}], {"temperature": 0})
    assert result is None