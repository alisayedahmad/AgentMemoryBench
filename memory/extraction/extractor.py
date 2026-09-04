"""LLM-based atomic fact extraction: turns one episode of conversation into Facts objects"""

import json
import re
from pydantic import ValidationError

from memory.extraction.schemas import Fact

EXTRACTION_PROMPT = """Extract atomic facts from this conversation excerpt.

Each fact is one claim, decomposed into subject, predicate, object. If the
text states or implies when the fact was true, put an approximate value in
valid_from (and valid_to if it clearly ended). Leave both null if no time is
stated.

Return ONLY a JSON array, nothing else, in this shape:
[{{"subject": "...", "predicate": "...", "object": "...", "valid_from": "...", "valid_to": null, "confidence": 0.0}}]

If there are no extractable facts, return [].

Conversation excerpt:
{episode_text}
"""


def _strip_code_fence(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class Extractor:
    def __init__(self, llm_client, model="claude-sonnet-5"):
        self.llm_client = llm_client
        self.model = model

    def extract(self, episode_text, episode_id):
        prompt = EXTRACTION_PROMPT.format(episode_text=episode_text)
        raw = self.llm_client.call(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            rows = json.loads(_strip_code_fence(raw))
        except json.JSONDecodeError:
            return []

        facts = []
        for row in rows:
            row["source_episode_id"] = episode_id
            try:
                facts.append(Fact(**row))
            except ValidationError:
                continue
        return facts