"""runs N LongMemEval oracle questions through the pipeline, judged by the real evaluator"""

import os
import sys
import tempfile

from infra.embedder import make_embedder
from infra.llm_client import LLMClient
from bench.datasets.longmemeval_adapter import load_longmemeval
from bench.evaluators.longmemeval_eval import judge_answer
from memory.curation.pipeline import ingest_fact
from memory.extraction.extractor import Extractor
from memory.retrieval.answer import answer_question
from memory.retrieval.hybrid_search import hybrid_search
from memory.storage.entity_graph import EntityGraph
from memory.storage.semantic import SemanticStore

DATA_PATH = "data/longmemeval_oracle.json"


def run_question(case, llm_client, embed, k=5):
    """fresh store per question, extraction -> pipeline -> hybrid_search -> answer -> judge"""
    extractor = Extractor(llm_client)
    graph = EntityGraph()

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = SemanticStore(path=os.path.join(tmp_dir, "facts.jsonl"))

        for i, episode_text in enumerate(case["episodes"]):
            episode_id = f"{case['question_id']}_ep{i}"
            facts = extractor.extract(episode_text, episode_id, as_of=case["episode_dates"][i])
            for fact in facts:
                ingest_fact(store, graph, fact, embed)

        facts = [f for f, _ in hybrid_search(store, case["question"], embed, graph=graph, k=k)]
        answer = answer_question(case["question"], facts, llm_client)

    correct = judge_answer(
        case["question_type"], case["question"], case["answer"], answer,
        llm_client, is_abstention=case["is_abstention"],
    )
    return answer, correct


def main(n=10):
    cases = load_longmemeval(DATA_PATH)[:n]
    llm_client = LLMClient()
    embed = make_embedder()

    correct_count = 0
    for case in cases:
        answer, correct = run_question(case, llm_client, embed)
        correct_count += correct
        mark = "PASS" if correct else "FAIL"
        print(f"[{mark}] ({case['question_type']}) {case['question']!r}")
        print(f"  gold: {case['answer']!r}")
        print(f"  got:  {answer!r}")

    print(f"\n{correct_count}/{len(cases)} correct")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n)
