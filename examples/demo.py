"""Demo: one trace, detectors vs truth, side by side."""
import numpy as np

from detector_truth_gap import harness as H
from detector_truth_gap import detectors as D
from detector_truth_gap import oracle as O

problems = H.load_subset("data/gsm8k_test.parquet", n=30)
rng = np.random.default_rng(7)

for behavior in (H.SILENT_OVERRIDE, H.STUCK_LOOP, H.RECHECK_GENUINE):
    t = H.generate_trace(problems[0], behavior, rng)
    print(f"=== behavior: {behavior} (gold={t.gold}, final={t.final_answer})")
    for e in t.events:
        if e["type"] == "call":
            flag = " CORRUPTED" if e["corrupted"] else ""
            print(f"  call {e['expr']} -> {e['returned']}{flag}")
        elif e["type"] == "think":
            print(f"  think: {e['text'][:100]}")
    print(f"  verification: detector={D.detect_verification(t)} truth={O.truth_verified(t)}")
    print(f"  correctness:  detector={D.detect_correctness(t)}")
    print(f"  fault impact: detector={D.detect_fault_impact(t)} truth={O.truth_fault_impact(t)}")
    print()
