"""LLM-based atomic fact extraction: turns one episode of conversation into Facts objects"""

import json
import re
from datetime import date
from pydantic import ValidationError

from memory.extraction.schemas import Fact

EXTRACTION_PROMPT = """Extract atomic facts from this conversation excerpt.
Today's date is {as_of}. Use it to work out real dates from relative time
words like "last march" or "recently".

Each fact is one claim, decomposed into subject, predicate, object.

- Use "user" as the subject for any fact about the person you're talking
  to. Only use a different subject for facts about someone or something
  else explicitly named.
- Predicates are matched by exact string later, across episodes, so keep
  them short, lowercase, snake_case (e.g. "works_at", not "is employed by"
  or "Works At"), and reuse the same predicate for the same kind of fact
  instead of inventing a new phrasing each time.
- valid_from and valid_to must be a real date: "YYYY", "YYYY-MM", or
  "YYYY-MM-DD". Never a word like "recently" or "now" — work out the actual
  year/month from today's date instead. Leave both null if no time is
  stated or implied.

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


SELF_WORDS = {"i", "me", "my", "myself", "user", "the user", "speaker", "the speaker", "the person"}


def _normalize_subject(subject):
    """same person, different words each call (i / speaker / user). map self-words to "user", leave real names alone"""
    return "user" if subject.strip().lower() in SELF_WORDS else subject


DATE_PATTERN = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def _clean_date(value):
    """only YYYY, YYYY-MM, YYYY-MM-DD count as a date. a word like "recent" becomes null, not garbage in the store"""
    if value and DATE_PATTERN.match(value):
        return value
    return None


class Extractor:
    def __init__(self, llm_client, model="claude-sonnet-5"):
        self.llm_client = llm_client
        self.model = model

    def extract(self, episode_text, episode_id, as_of=None):
        as_of = as_of or date.today().isoformat()
        prompt = EXTRACTION_PROMPT.format(episode_text=episode_text, as_of=as_of)
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
            row["subject"] = _normalize_subject(row.get("subject") or "")
            row["valid_from"] = _clean_date(row.get("valid_from"))
            row["valid_to"] = _clean_date(row.get("valid_to"))
            try:
                facts.append(Fact(**row))
            except ValidationError:
                continue
        return facts