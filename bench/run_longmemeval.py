"""runs N LongMemEval oracle questions through the pipeline, judged by the real evaluator"""

import json
import os
import random
import sys
import tempfile
from collections import defaultdict

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
RESULTS_PATH = "results/raw_outputs/longmemeval_run.json"

def stratified_sample(cases, n, seed=0):
    """even split across question types, so a small n isn't all one category"""
    by_type = defaultdict(list)
    for case in cases:
        by_type[case["question_type"]].append(case)

    rng = random.Random(seed)
    for group in by_type.values():
        rng.shuffle(group)

    picked = []
    types = sorted(by_type)
    i = 0
    while len(picked) < n and any(by_type[t] for t in types):
        group = by_type[types[i % len(types)]]
        if group:
            picked.append(group.pop())
        i += 1
    return picked

def run_question(case, llm_client, embed, k=10, verbose=False):
    """fresh store per question, extraction -> pipeline -> hybrid_search -> answer -> judge"""
    extractor = Extractor(llm_client)
    graph = EntityGraph()

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = SemanticStore(path=os.path.join(tmp_dir, "facts.jsonl"))

        for i, episode_text in enumerate(case["episodes"]):
            episode_id = f"{case['question_id']}_ep{i}"
            as_of = case["episode_dates"][i]
            facts = extractor.extract(episode_text, episode_id, as_of=as_of)
            for fact in facts:
                ingest_fact(store, graph, fact, embed, as_of=as_of)

        if verbose:
            print("  stored facts:")
            for f in store.all():
                print(f"    {f.subject} {f.predicate} {f.object!r} ({f.valid_from} to {f.valid_to or 'now'})")

        retrieved = hybrid_search(store, case["question"], embed, graph=graph, k=k)
        if verbose:
            print("  retrieved for this question:")
            for f, score in retrieved:
                print(f"    {score:.3f}  {f.subject} {f.predicate} {f.object!r}")

        facts = [f for f, _ in retrieved]
        answer = answer_question(case["question"], facts, llm_client, as_of=case["question_date"])

    correct = judge_answer(
        case["question_type"], case["question"], case["answer"], answer,
        llm_client, is_abstention=case["is_abstention"],
    )
    return answer, correct

def main(n=10, verbose=False, seed=0):
    cases = stratified_sample(load_longmemeval(DATA_PATH), n, seed=seed)
    llm_client = LLMClient()
    embed = make_embedder()

    results = []
    for i, case in enumerate(cases, 1):
        answer, correct = run_question(case, llm_client, embed, verbose=verbose)
        results.append({
            "question_id": case["question_id"],
            "question_type": case["question_type"],
            "is_abstention": case["is_abstention"],
            "question": case["question"],
            "gold": case["answer"],
            "answer": answer,
            "correct": correct,
        })
        mark = "PASS" if correct else "FAIL"
        print(f"[{mark}] {i}/{len(cases)} ({case['question_type']}) {case['question'][:70]!r}")
        if not correct:
            print(f"       gold: {case['answer'][:90]!r}")
            print(f"       got:  {answer[:90]!r}")
        _save(results)  # written every question, a crash halfway still leaves usable results

    _report(results)
    print(f"raw outputs written to {RESULTS_PATH}")

def _report(results):
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["correct"])

    print("\nby question type:")
    for qtype in sorted(by_type):
        hits = by_type[qtype]
        print(f"  {qtype:28} {sum(hits)}/{len(hits)}")

    abstention = [r["correct"] for r in results if r["is_abstention"]]
    if abstention:
        print(f"  {'(of which abstention)':28} {sum(abstention)}/{len(abstention)}")

    total = [r["correct"] for r in results]
    print(f"\n{sum(total)}/{len(total)} correct")

def _save(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n, verbose="-v" in sys.argv)
