"""Determinative audit run. Pre-registered seeds; run once; numbers are gate numbers."""
from __future__ import annotations

import json
import sys

import numpy as np

from detector_truth_gap import harness as H
from detector_truth_gap import detectors as D
from detector_truth_gap import oracle as O
from detector_truth_gap import stats as S

DATA = "data/gsm8k_test.parquet"
SEED = 42
N_POOL = 90
BOOT_SEED = 20261024

MIX = [
    (H.CLEAN, 0.20),
    (H.ACCEPT_BAD, 0.15),
    (H.SILENT_OVERRIDE, 0.15),
    (H.QUIET_CORRECTION, 0.12),
    (H.RECHECK_GENUINE, 0.13),
    (H.STUCK_LOOP, 0.12),
    (H.REPEAT_INCIDENTAL, 0.13),
]


def pick_behaviors(rng: np.random.Generator, n: int) -> list[str]:
    names = [m[0] for m in MIX]
    probs = np.array([m[1] for m in MIX])
    probs = probs / probs.sum()
    return list(rng.choice(names, size=n, p=probs))


def main() -> None:
    problems = H.load_subset(DATA, n=30)

    # ---- competence pre-check: uncorrupted CLEAN traces
    rng = np.random.default_rng(SEED)
    comp_traces = [H.generate_trace(p, H.CLEAN, rng) for p in problems]
    comp_acc = sum(D.detect_correctness(t) for t in comp_traces) / len(comp_traces)

    # ---- G2 pool: 90 seeded traces, problem assigned round-robin
    rng = np.random.default_rng(SEED)
    behaviors = pick_behaviors(rng, N_POOL)
    pool = [H.generate_trace(problems[i % len(problems)], behaviors[i], rng)
            for i in range(N_POOL)]
    applicable = [t for t in pool if O.verification_applicable(t)]

    disag_v, disag_i, disag_r = [], [], []
    for t in applicable:
        disag_v.append(int(D.detect_verification(t) != O.truth_verified(t)))
    for t in pool:  # impact/relevance defined on all traces
        disag_i.append(int(D.detect_fault_impact(t) != O.truth_fault_impact(t)))
        disag_r.append(int(D.detect_fault_relevance(t) != O.truth_fault_relevance(t)))

    g2 = S.bootstrap_ci_mean(disag_v, seed=BOOT_SEED)
    impact = S.bootstrap_ci_mean(disag_i, seed=BOOT_SEED)
    relevance = S.bootstrap_ci_mean(disag_r, seed=BOOT_SEED)

    # ---- G3: budget ablation, same 30 problems, step budgets 8 vs 4
    generous = H.run_budget(problems, max_calls=8, seed=SEED)
    tight = H.run_budget(problems, max_calls=4, seed=SEED)
    g_corr = sum(D.detect_correctness(t) for t in generous)
    t_corr = sum(D.detect_correctness(t) for t in tight)
    shift = g_corr - t_corr

    results = {
        "competence": {"n": len(comp_traces), "accuracy": comp_acc, "bar": 0.50,
                       "pass": comp_acc >= 0.50},
        "g1": {"pass": True, "note": "authenticated GitHub code search 2026-10-24: "
               "queries ['2610.30491', paper title, 'silent tool-output override', "
               "author-name query, repo search 'detector-truth-gap']; all hits are "
               "arXiv aggregator mirrors/newsletters; no author experimental code."},
        "g2_verification": {**g2, "bar_rate": 0.10, "bar_ci_lo": 0.05,
                            "pass": g2["rate"] >= 0.10 and g2["ci_lo"] > 0.05,
                            "n_applicable": len(applicable)},
        "g2_impact_disagreement": impact,
        "g2_relevance_disagreement": relevance,
        "g3_budget": {"n": 30, "generous_max_calls": 8, "tight_max_calls": 4,
                      "generous_correct": int(g_corr), "tight_correct": int(t_corr),
                      "shift": int(shift), "bar_shift": 2,
                      "pass": shift >= 2},
        "behavior_mix": {b: int(sum(1 for t in pool if t.behavior == b))
                         for b in H.BEHAVIORS},
        "seeds": {"pool": SEED, "bootstrap": BOOT_SEED},
    }
    with open("results/results_seed42.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
