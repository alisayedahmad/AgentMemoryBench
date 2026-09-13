"""compares two result files on the same questions: McNemar exact test + Wilson intervals. no API calls"""

import json
import math
import sys

Z_95 = 1.959963984540054

def wilson_interval(hits, n, z=Z_95):
    """Wilson score interval, honest at small n where the normal approximation falls apart"""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def mcnemar_exact(b, c):
    """two-sided exact test on the discordant pairs only, under the null they split 50/50"""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)

def compare(results_a, results_b):
    """pairs the two runs by question_id, counts who won where"""
    by_id_a = {r["question_id"]: r for r in results_a}
    by_id_b = {r["question_id"]: r for r in results_b}
    shared = sorted(set(by_id_a) & set(by_id_b))

    only_a, only_b = [], []
    for qid in shared:
        a, b = by_id_a[qid]["correct"], by_id_b[qid]["correct"]
        if a and not b:
            only_a.append(by_id_a[qid])
        elif b and not a:
            only_b.append(by_id_b[qid])

    return shared, only_a, only_b

def main(path_a, path_b):
    with open(path_a) as f:
        results_a = json.load(f)
    with open(path_b) as f:
        results_b = json.load(f)

    shared, only_a, only_b = compare(results_a, results_b)
    n = len(shared)
    shared_set = set(shared)
    hits_a = sum(1 for r in results_a if r["correct"] and r["question_id"] in shared_set)
    hits_b = sum(1 for r in results_b if r["correct"] and r["question_id"] in shared_set)

    name_a, name_b = path_a.split("/")[-1], path_b.split("/")[-1]
    lo_a, hi_a = wilson_interval(hits_a, n)
    lo_b, hi_b = wilson_interval(hits_b, n)
    print(f"{name_a}: {hits_a}/{n} = {hits_a/n:.1%}  95% CI [{lo_a:.1%}, {hi_a:.1%}]")
    print(f"{name_b}: {hits_b}/{n} = {hits_b/n:.1%}  95% CI [{lo_b:.1%}, {hi_b:.1%}]")

    p = mcnemar_exact(len(only_a), len(only_b))
    print(f"\ndiscordant: {len(only_a)} only in {name_a}, {len(only_b)} only in {name_b}")
    print(f"McNemar exact p = {p:.3f}")
    verdict = "significant at 0.05" if p < 0.05 else "not significant at 0.05"
    print(f"the difference is {verdict}")

    for label, group in ((name_a, only_a), (name_b, only_b)):
        if group:
            print(f"\nonly {label} got these right:")
            for r in group:
                print(f"  ({r['question_type']}) {r['question'][:70]!r}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
