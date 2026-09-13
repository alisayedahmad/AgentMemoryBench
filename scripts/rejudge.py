"""re-judges a saved results file to measure judge noise. no pipeline rerun, the answers are already on disk"""

import json
import sys
from collections import defaultdict

from bench.evaluators.longmemeval_eval import judge_answer
from infra.llm_client import LLMClient

def rejudge(results, llm_client):
    """cache bypassed on purpose, a cache hit would just replay the old verdict and measure nothing"""
    flips = []
    for r in results:
        verdict = judge_answer(
            r["question_type"], r["question"], r["gold"], r["answer"],
            llm_client, is_abstention=r["is_abstention"], use_cache=False,
        )
        if verdict != r["correct"]:
            flips.append({**r, "was": r["correct"], "now": verdict})
    return flips

def main(path):
    with open(path) as f:
        results = json.load(f)

    flips = rejudge(results, LLMClient())

    for f in flips:
        print(f"[{f['was']} -> {f['now']}] ({f['question_type']}) {f['question'][:65]!r}")
        print(f"    gold: {f['gold'][:80]!r}")
        print(f"    got:  {f['answer'][:80]!r}")

    by_type = defaultdict(int)
    for f in flips:
        by_type[f["question_type"]] += 1
    if by_type:
        print("\nflips by type:")
        for qtype in sorted(by_type):
            print(f"  {qtype:28} {by_type[qtype]}")

    old_score = sum(r["correct"] for r in results)
    new_score = old_score + sum(1 for f in flips if f["now"]) - sum(1 for f in flips if not f["now"])
    print(f"\n{len(flips)}/{len(results)} verdicts flipped")
    print(f"score {old_score}/{len(results)} -> {new_score}/{len(results)}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/raw_outputs/longmemeval_all_facts.json")
