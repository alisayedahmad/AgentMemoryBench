"""
minimal harness: runs a case (episodes + questions with expected answers)
through the real system, reports pass/fail. no benchmark adapter yet — the
cases below are hand-written. once LoCoMo/LongMemEval adapters exist, they
produce cases in this same shape and this file barely changes.
"""

import os
import tempfile

from infra.embedder import make_embedder
from infra.llm_client import LLMClient
from baselines.no_memory import answer_with_no_memory
from memory.curation.pipeline import ingest_fact
from memory.extraction.extractor import Extractor
from memory.retrieval.answer import answer_question
from memory.retrieval.hybrid_search import hybrid_search
from memory.storage.entity_graph import EntityGraph
from memory.storage.semantic import SemanticStore

CASES = [
    {
        "episodes": [
            "I started working at Google last March.",
            "I really love drinking coffee every morning, can't function without it.",
            "Update: I just left Google and joined Meta.",
            "Coffee is honestly the best part of my mornings.",
        ],
        "questions": [
            {"question": "Where does the user work now?", "expects": "Meta"},
            {"question": "Where did the user work before?", "expects": "Google"},
            {"question": "What does the user like to drink?", "expects": "coffee"},
            {"question": "What car does the user drive?", "expects": "i don't know"},
        ],
    },
]


def run_case(case, llm_client, embed, k=5):
    """runs episodes through extraction + curation, answers each question, checks "expects" as a substring"""
    extractor = Extractor(llm_client)
    graph = EntityGraph()

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = SemanticStore(path=os.path.join(tmp_dir, "facts.jsonl"))

        for i, episode_text in enumerate(case["episodes"]):
            for fact in extractor.extract(episode_text, f"ep{i}"):
                ingest_fact(store, graph, fact, embed)

        results = []
        for q in case["questions"]:
            facts = [f for f, _ in hybrid_search(store, q["question"], embed, graph=graph, k=k)]
            answer = answer_question(q["question"], facts, llm_client)
            passed = q["expects"].lower() in answer.lower()
            results.append({
                "question": q["question"],
                "answer": answer,
                "expects": q["expects"],
                "passed": passed,
            })
        return results


def run_case_no_memory(case, llm_client):
    """same questions, zero memory — the floor the memory system has to beat"""
    results = []
    for q in case["questions"]:
        answer = answer_with_no_memory(q["question"], llm_client)
        passed = q["expects"].lower() in answer.lower()
        results.append({
            "question": q["question"],
            "answer": answer,
            "expects": q["expects"],
            "passed": passed,
        })
    return results


def main():
    llm_client = LLMClient()
    embed = make_embedder()

    for case in CASES:
        print("=== with memory ===")
        total, passed = 0, 0
        for r in run_case(case, llm_client, embed):
            total += 1
            passed += r["passed"]
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"[{mark}] {r['question']!r} -> {r['answer']!r} (expected to contain {r['expects']!r})")
        print(f"{passed}/{total} passed")

        print("\n=== no memory (baseline) ===")
        total_nm, passed_nm = 0, 0
        for r in run_case_no_memory(case, llm_client):
            total_nm += 1
            passed_nm += r["passed"]
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"[{mark}] {r['question']!r} -> {r['answer']!r} (expected to contain {r['expects']!r})")
        print(f"{passed_nm}/{total_nm} passed")


if __name__ == "__main__":
    main()
