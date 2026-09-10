"""wraps the Anthropic API behind the on-disk cache — same (model, messages, params) never gets charged twice"""

import anthropic

from infra.llm_cache import LLMCache

DEFAULT_MAX_TOKENS = 1024


class LLMClient:
    def __init__(self, client=None, cache=None):
        self.client = client or anthropic.Anthropic()
        self.cache = cache or LLMCache()

    def call(self, model, messages, max_tokens=DEFAULT_MAX_TOKENS, **kwargs):
        params = {"max_tokens": max_tokens, **kwargs}

        cached = self.cache.get(model, messages, params)
        if cached is not None:
            return cached

        response = self.client.messages.create(model=model, messages=messages, **params)
        text = _extract_text(response)

        self.cache.set(model, messages, params, text)
        return text


def _extract_text(response):
    """joins every text block in the response — a reply can come back as more than one block"""
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
