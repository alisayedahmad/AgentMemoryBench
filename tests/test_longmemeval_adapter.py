import json
import os

from bench.datasets.longmemeval_adapter import load_longmemeval

REAL_FILE = "data/longmemeval_oracle.json"


def make_fixture(tmp_path, entries):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(entries))
    return str(path)


def test_basic_shape(tmp_path):
    path = make_fixture(tmp_path, [{
        "question_id": "gpt4_abc123",
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "answer": "Business Administration",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_dates": ["2023/05/30 (Tue) 20:00"],
        "haystack_session_ids": ["s1"],
        "haystack_sessions": [[
            {"role": "user", "content": "I graduated with a degree in Business Administration."},
            {"role": "assistant", "content": "That's great!"},
        ]],
        "answer_session_ids": ["s1"],
    }])

    cases = load_longmemeval(path)

    assert len(cases) == 1
    case = cases[0]
    assert case["question_id"] == "gpt4_abc123"
    assert case["question_type"] == "single-session-user"
    assert case["answer"] == "Business Administration"
    assert case["is_abstention"] is False
    assert len(case["episodes"]) == 1
    assert "user: I graduated with a degree in Business Administration." in case["episodes"][0]
    assert "assistant: That's great!" in case["episodes"][0]
    assert case["episode_dates"] == ["2023-05-30"]


def test_abstention_detected_from_question_id(tmp_path):
    path = make_fixture(tmp_path, [{
        "question_id": "gpt4_xyz_abs",
        "question_type": "temporal-reasoning",
        "question": "Which happened first?",
        "answer": "Not enough information.",
        "haystack_dates": ["2023/05/30 (Tue) 20:00"],
        "haystack_sessions": [[{"role": "user", "content": "hi"}]],
    }])

    cases = load_longmemeval(path)

    assert cases[0]["is_abstention"] is True


def test_multiple_sessions_become_multiple_episodes(tmp_path):
    path = make_fixture(tmp_path, [{
        "question_id": "gpt4_multi",
        "question_type": "multi-session",
        "question": "q",
        "answer": "a",
        "haystack_dates": ["2023/05/30 (Tue) 20:00", "2023/06/02 (Fri) 09:15"],
        "haystack_sessions": [
            [{"role": "user", "content": "session one"}],
            [{"role": "user", "content": "session two"}],
        ],
    }])

    cases = load_longmemeval(path)

    assert len(cases[0]["episodes"]) == 2
    assert "session one" in cases[0]["episodes"][0]
    assert "session two" in cases[0]["episodes"][1]
    assert cases[0]["episode_dates"] == ["2023-05-30", "2023-06-02"]


def test_against_real_downloaded_file():
    """smoke check against the actual file, skipped if it isn't present"""
    if not os.path.exists(REAL_FILE):
        return

    cases = load_longmemeval(REAL_FILE)

    assert len(cases) == 500
    abstention_count = sum(c["is_abstention"] for c in cases)
    assert abstention_count == 30
    question_types = {c["question_type"] for c in cases}
    assert question_types == {
        "temporal-reasoning", "multi-session", "knowledge-update",
        "single-session-user", "single-session-assistant", "single-session-preference",
    }
    assert all(case["episodes"] for case in cases)
