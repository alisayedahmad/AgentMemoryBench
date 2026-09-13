"""content-hashed cache for LLM calls, one JSON file per entry"""

import hashlib
import json
import os
import threading


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
        """writes to a temp file then renames, so a concurrent reader never sees a half-written entry"""
        key = self._key(model, messages, params)
        path = self._path(key)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(
                {"model": model, "messages": messages, "params": params, "response": response},
                f,
                indent=2,
            )
        try:
            os.replace(tmp, path)
        except (OSError, PermissionError):
            # on Windows, if the file is locked by a reader, replace may fail
            # remove the temp and let the other writer win (both have the same content)
            try:
                os.remove(tmp)
            except OSError:
                pass
        return key