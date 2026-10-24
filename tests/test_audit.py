"""12+ pytest: detectors, oracle, corruption rig, budgets, bootstrap. All green before any determinative run."""
import numpy as np
import pytest

from detector_truth_gap import harness as H
from detector_truth_gap import detectors as D
from detector_truth_gap import oracle as O
from detector_truth_gap import stats as S

ANS = ("Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. "
       "She makes 9 * 2 = $<<9*2=18>>18 every day. #### 18")
PROB = {"problem_id": 0, "question": "q", "plan": H.parse_plan(ANS)[0], "gold": 18.0}


def rng(seed=1):
    return np.random.default_rng(seed)


def test_plan_parse():
    steps, gold = H.parse_plan(ANS)
    assert [e for e, _ in steps] == ["16-3-4", "9*2"]
    assert gold == 18.0


def test_calculator_eval():
    c = H.Calculator(rng())
    r = c.call("16-3-4", force_corrupt=False)
    assert r["returned"] == "9" and r["true_value"] == "9" and not r["corrupted"]


def test_corruption_rig_records_truth():
    c = H.Calculator(rng())
    r = c.call("9*2", force_corrupt=True)
    assert r["corrupted"] and r["true_value"] == "18" and r["returned"] != "18"


def test_clean_agree():
    t = H.generate_trace(PROB, H.CLEAN, rng())
    assert D.detect_correctness(t) is True
    assert D.detect_verification(t) is False
    assert O.truth_verified(t) is False


def test_silent_override_disagrees():
    t = H.generate_trace(PROB, H.SILENT_OVERRIDE, rng())
    assert O.truth_verified(t) is True
    assert D.detect_verification(t) is False  # disagreement


def test_quiet_correction_disagrees():
    t = H.generate_trace(PROB, H.QUIET_CORRECTION, rng())
    assert O.truth_verified(t) is True
    assert D.detect_verification(t) is False  # disagreement


def test_recheck_genuine_agrees():
    t = H.generate_trace(PROB, H.RECHECK_GENUINE, rng())
    assert O.truth_verified(t) is True
    assert D.detect_verification(t) is True
    assert D.detect_correctness(t) is True


def test_accept_bad_agrees_impact():
    t = H.generate_trace(PROB, H.ACCEPT_BAD, rng())
    assert D.detect_verification(t) is False and O.truth_verified(t) is False
    assert D.detect_fault_impact(t) is True and O.truth_fault_impact(t) is True
    assert D.detect_fault_relevance(t) is True and O.truth_fault_relevance(t) is True
    assert D.detect_correctness(t) is False


def test_stuck_loop_disagrees():
    t = H.generate_trace(PROB, H.STUCK_LOOP, rng())
    assert D.detect_verification(t) is True
    assert O.truth_verified(t) is False  # disagreement


def test_repeat_incidental_disagrees():
    t = H.generate_trace(PROB, H.REPEAT_INCIDENTAL, rng())
    assert D.detect_verification(t) is True
    assert O.truth_verified(t) is False  # disagreement


def test_relevance_agrees_on_recheck():
    t = H.generate_trace(PROB, H.RECHECK_GENUINE, rng())
    assert D.detect_fault_relevance(t) is False
    assert O.truth_fault_relevance(t) is False


def test_budget_censors_long_trace():
    probs = H.load_subset("data/gsm8k_test.parquet", n=30)
    long_p = next(p for p in probs if len(p["plan"]) > 4)
    t_gen = H.generate_trace(long_p, H.CLEAN, rng(), max_calls=8)
    t_tight = H.generate_trace(long_p, H.CLEAN, rng(), max_calls=4)
    assert t_gen.censored is False and D.detect_correctness(t_gen) is True
    assert t_tight.censored is True and D.detect_correctness(t_tight) is False


def test_bootstrap_ci_sane():
    r = S.bootstrap_ci_mean([1] * 20 + [0] * 80, seed=1)
    assert abs(r["rate"] - 0.2) < 1e-9
    assert r["ci_lo"] < 0.2 < r["ci_hi"]


def test_seed_reproducibility():
    a = H.generate_trace(PROB, H.STUCK_LOOP, rng(99))
    b = H.generate_trace(PROB, H.STUCK_LOOP, rng(99))
    assert a.events == b.events and a.used == b.used


def test_applicable_pool_size():
    probs = H.load_subset("data/gsm8k_test.parquet", n=30)
    r = rng(42)
    pool = [H.generate_trace(probs[i % 30], H.CLEAN, r) for i in range(90)]
    assert sum(O.verification_applicable(t) for t in pool) >= 80


def test_gold_parse_no_failures():
    probs = H.load_subset("data/gsm8k_test.parquet", n=30)
    assert all(p["gold"] is not None for p in probs)
