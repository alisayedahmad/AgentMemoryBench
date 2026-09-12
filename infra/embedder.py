"""wraps sentence-transformers behind the plain embed(text) function the rest of the codebase expects"""

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def make_embedder(model_name=DEFAULT_MODEL, model=None):
    """
    Loads the model once, returns a plain function: embed(text) -> list of
    floats. Same shape as toy_embed and the fake_embed dicts in the tests,
    so it drops straight into dedup, hybrid_search, pipeline.

    First call on a fresh machine downloads the model (a bit under 100MB),
    then it's cached on disk and loads fast after that.
    """
    model = model or SentenceTransformer(model_name)

    def embed(text):
        return model.encode(text).tolist()

    return embed
