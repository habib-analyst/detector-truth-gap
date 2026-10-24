"""Naive trace-level detectors, in the style of the paper's audited detectors.

Each detector sees ONLY the trace text (events), never the oracle intent.
"""
from __future__ import annotations

import re
from typing import Optional

from .harness import Trace, fmt_num

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _norm_expr(expr: str) -> str:
    return re.sub(r"\s+", "", expr)


def detect_correctness(trace: Trace) -> Optional[bool]:
    """Final-answer correctness vs gold."""
    if trace.gold is None:
        return None
    try:
        return abs(float(trace.final_answer) - trace.gold) < 1e-9
    except (ValueError, TypeError):
        return False


def detect_verification(trace: Trace) -> bool:
    """Naive rule: did the agent re-call the same expression after seeing a tool result?

    Cannot see think-text content; counts any repeated call as a re-check.
    """
    seen: set[str] = set()
    for e in trace.events:
        if e["type"] == "call":
            key = _norm_expr(e["expr"])
            if key in seen:
                return True
            seen.add(key)
    return False


def detect_fault_impact(trace: Trace) -> bool:
    """Naive rule: a corrupted tool output occurred AND the final answer is wrong."""
    if not trace.corrupted_values:
        return False
    correct = detect_correctness(trace)
    return correct is False


def detect_fault_relevance(trace: Trace) -> bool:
    """Naive rule: a corrupted value actually entered a later computation.

    Implemented as: the corrupted returned string equals a later step's used value
    or the final answer (value-flow), not mere mention in think text.
    """
    if not trace.corrupted_values:
        return False
    bad = set(trace.corrupted_values)
    for v in list(trace.used) + [trace.final_answer]:
        if v in bad:
            return True
    return False
