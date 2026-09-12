"""
minimal harness: runs a case (episodes + questions with expected answers)
through the real system, reports pass/fail. no benchmark adapter yet — the
cases below are hand-written. once LoCoMo/LongMemEval adapters exist, they
produce cases in this same shape and this file barely changes"""




import os
import tempfile

from infra.embedder import make_embedder
from infra.llm_client import LLMClient
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


def main():
    llm_client = LLMClient()
    embed = make_embedder()

    total = 0
    passed = 0
    for case in CASES:
        for r in run_case(case, llm_client, embed):
            total += 1
            passed += r["passed"]
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"[{mark}] {r['question']!r} -> {r['answer']!r} (expected to contain {r['expects']!r})")

    print(f"\n{passed}/{total} passed")


if __name__ == "__main__":
    main()
