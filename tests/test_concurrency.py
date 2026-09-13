import os
from concurrent.futures import ThreadPoolExecutor

from infra.embedder import make_embedder
from infra.llm_cache import LLMCache

def test_concurrent_writes_to_the_same_key_never_expose_a_half_written_file(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path / "cache"))
    messages = [{"role": "user", "content": "x" * 5000}]
    params = {"max_tokens": 1024}
    big_response = "y" * 20000

    def writer(_):
        cache.set("sonnet", messages, params, big_response)

    def reader(_):
        return cache.get("sonnet", messages, params)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(writer, range(30)))
        reads = list(pool.map(reader, range(60)))

    assert all(r == big_response for r in reads)

def test_no_temp_files_are_left_behind(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = LLMCache(cache_dir=str(cache_dir))

    with ThreadPoolExecutor(max_workers=8) as pool:
        pool.map(lambda i: cache.set("sonnet", [{"role": "user", "content": str(i)}], {}, "ok"), range(40))

    leftovers = [f for f in os.listdir(cache_dir) if f.endswith(".tmp")]
    assert leftovers == []

def test_different_keys_written_concurrently_all_survive(tmp_path):
    cache = LLMCache(cache_dir=str(tmp_path / "cache"))

    def write(i):
        cache.set("sonnet", [{"role": "user", "content": f"q{i}"}], {}, f"answer {i}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write, range(50)))

    for i in range(50):
        assert cache.get("sonnet", [{"role": "user", "content": f"q{i}"}], {}) == f"answer {i}"

class CountingModel:
    """records whether encode was ever entered twice at once"""

    def __init__(self):
        self.inside = 0
        self.overlapped = False

    def encode(self, text):
        import numpy as np
        self.inside += 1
        if self.inside > 1:
            self.overlapped = True
        try:
            return np.array([float(len(text))])
        finally:
            self.inside -= 1

def test_embedder_serialises_calls_across_threads():
    model = CountingModel()
    embed = make_embedder(model=model)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(embed, [f"text{i}" for i in range(200)]))

    assert model.overlapped is False
    assert len(results) == 200
