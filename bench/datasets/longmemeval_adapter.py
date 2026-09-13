"""loads LongMemEval json into episodes + questions"""

import json
from datetime import datetime


def _parse_date(raw):
    """"2023/04/10 (Mon) 17:50" -> "2023-04-10" """
    return datetime.strptime(raw, "%Y/%m/%d (%a) %H:%M").strftime("%Y-%m-%d")


def load_longmemeval(path):
    """one case per question: episodes, episode_dates, question, answer, is_abstention (from "_abs" in the id)"""
    with open(path) as f:
        raw = json.load(f)

    return [
        {
            "question_id": entry["question_id"],
            "question_type": entry["question_type"],
            "question": entry["question"],
            "answer": entry["answer"],
            "is_abstention": "_abs" in entry["question_id"],
            "episodes": [_session_to_text(s) for s in entry["haystack_sessions"]],
            "episode_dates": [_parse_date(d) for d in entry["haystack_dates"]],
        }
        for entry in raw
    ]


def _session_to_text(session):
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session)
