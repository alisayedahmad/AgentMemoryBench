from infra.embedder import make_embedder

class FakeModel:
    """stands in for SentenceTransformer — returns a fixed-shape vector, no download needed"""

    def encode(self, text):
        import numpy as np
        return np.array([len(text), text.count("a")], dtype=float)

def test_embed_returns_a_plain_list():
    embed = make_embedder(model=FakeModel())

    vec = embed("banana")

    assert vec == [6.0, 3.0]
    assert isinstance(vec, list)

def test_embed_is_reusable_across_calls():
    embed = make_embedder(model=FakeModel())

    assert embed("cat") == [3.0, 1.0]
    assert embed("aardvark") == [8.0, 3.0]
