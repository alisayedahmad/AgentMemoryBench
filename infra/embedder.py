"""wraps sentence-transformers behind the plain embed(text) function the rest of the codebase expects"""

import threading

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def make_embedder(model_name=DEFAULT_MODEL, model=None):
    """returns a plain embed(text) -> list of floats, loads the model once
    first call downloads the model (~100MB), then it's cached on disk"""
    model = model or SentenceTransformer(model_name)
    lock = threading.Lock()

    def embed(text):
        with lock:  # one model shared across worker threads, torch isn't guaranteed reentrant
            return model.encode(text).tolist()

    return embed
