"""content-hashed cache for LLM calls, one JSON file per entry"""

import hashlib
import json
import os


class LLMCache:
    def __init__(self, cache_dir="results/raw_outputs/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, model, messages, params):
        payload = json.dumps(
            {"model": model, "messages": messages, "params": params},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key):
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, model, messages, params):
        path = self._path(self._key(model, messages, params))
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)["response"]

    def set(self, model, messages, params, response):
        key = self._key(model, messages, params)
        with open(self._path(key), "w") as f:
            json.dump(
                {"model": model, "messages": messages, "params": params, "response": response},
                f,
                indent=2,
            )
        return key