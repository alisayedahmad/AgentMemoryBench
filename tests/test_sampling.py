from bench.run_longmemeval import stratified_sample

def make_cases(counts):
    cases = []
    for qtype, n in counts.items():
        for i in range(n):
            cases.append({"question_id": f"{qtype}_{i}", "question_type": qtype, "is_abstention": False})
    return cases

def test_spreads_evenly_across_types():
    cases = make_cases({"a": 10, "b": 10, "c": 10})

    picked = stratified_sample(cases, 6)

    types = [c["question_type"] for c in picked]
    assert sorted(types) == ["a", "a", "b", "b", "c", "c"]

def test_same_seed_gives_same_pick():
    cases = make_cases({"a": 10, "b": 10})

    first = [c["question_id"] for c in stratified_sample(cases, 6)]
    second = [c["question_id"] for c in stratified_sample(cases, 6)]

    assert first == second

def test_different_seed_gives_a_different_pick():
    cases = make_cases({"a": 20, "b": 20})

    first = [c["question_id"] for c in stratified_sample(cases, 8, seed=0)]
    second = [c["question_id"] for c in stratified_sample(cases, 8, seed=1)]

    assert first != second

def test_never_picks_the_same_case_twice():
    cases = make_cases({"a": 5, "b": 5})

    picked = stratified_sample(cases, 10)

    ids = [c["question_id"] for c in picked]
    assert len(set(ids)) == len(ids)

def test_asking_for_more_than_exists_returns_everything():
    cases = make_cases({"a": 3, "b": 2})

    picked = stratified_sample(cases, 50)

    assert len(picked) == 5

def test_uneven_pool_keeps_filling_from_whats_left():
    cases = make_cases({"a": 1, "b": 10})

    picked = stratified_sample(cases, 5)

    types = [c["question_type"] for c in picked]
    assert len(picked) == 5
    assert types.count("a") == 1
