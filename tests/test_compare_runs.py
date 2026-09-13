import math

from scripts.compare_runs import compare, mcnemar_exact, wilson_interval

def make_result(qid, correct, qtype="multi-session"):
    return {"question_id": qid, "question_type": qtype, "question": "q", "correct": correct}

def test_no_discordant_pairs_is_p_one():
    assert mcnemar_exact(0, 0) == 1.0

def test_even_split_is_p_one():
    assert mcnemar_exact(3, 3) == 1.0

def test_lopsided_split_is_significant():
    # 0 vs 10 is two-sided p = 2 * (1/2)^10
    assert math.isclose(mcnemar_exact(0, 10), 2 / 1024)
    assert mcnemar_exact(0, 10) < 0.05

def test_five_to_zero_is_not_quite_significant():
    # the smallest split that could ever reach 0.05 is 0 vs 6
    assert mcnemar_exact(0, 5) > 0.05
    assert mcnemar_exact(0, 6) < 0.05

def test_mcnemar_is_symmetric():
    assert mcnemar_exact(2, 7) == mcnemar_exact(7, 2)

def test_wilson_matches_a_known_value():
    lo, hi = wilson_interval(35, 60)
    assert math.isclose(lo, 0.4572808102264011, abs_tol=1e-9)
    assert math.isclose(hi, 0.6993572144165839, abs_tol=1e-9)

def test_wilson_stays_inside_zero_and_one():
    assert math.isclose(wilson_interval(10, 10)[1], 1.0, abs_tol=1e-9)
    assert math.isclose(wilson_interval(0, 10)[0], 0.0, abs_tol=1e-9)

def test_wilson_on_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 0.0)

def test_compare_finds_who_won_where():
    a = [make_result("q1", True), make_result("q2", False), make_result("q3", True)]
    b = [make_result("q1", False), make_result("q2", True), make_result("q3", True)]

    shared, only_a, only_b = compare(a, b)

    assert shared == ["q1", "q2", "q3"]
    assert [r["question_id"] for r in only_a] == ["q1"]
    assert [r["question_id"] for r in only_b] == ["q2"]

def test_compare_ignores_questions_missing_from_one_side():
    a = [make_result("q1", True), make_result("q2", True)]
    b = [make_result("q1", False)]

    shared, only_a, only_b = compare(a, b)

    assert shared == ["q1"]
    assert len(only_a) == 1
