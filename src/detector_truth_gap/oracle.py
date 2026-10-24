"""Deterministic trace-truth oracle: what the trace actually shows.

The oracle is the generation-time record (behavior intent + exact event log), i.e. the
deterministic analog of the paper's human review. It never uses detector output.
"""
from __future__ import annotations

from typing import Optional

from .harness import Trace, ACCEPT_BAD, STUCK_LOOP


def truth_verified(trace: Trace) -> bool:
    """Did the agent actually re-verify a tool output? (generation-time intent)."""
    return trace.intended_verified


def truth_fault_impact(trace: Trace) -> bool:
    """Did a corrupted tool value causally reach a wrong final answer?"""
    if not trace.corrupted_values:
        return False
    if trace.behavior not in (ACCEPT_BAD, STUCK_LOOP):
        return False
    if trace.gold is None:
        return False
    try:
        return abs(float(trace.final_answer) - trace.gold) >= 1e-9
    except (ValueError, TypeError):
        return True


def truth_fault_relevance(trace: Trace) -> bool:
    """Was a corrupted value actually used downstream?"""
    if not trace.corrupted_values:
        return False
    return trace.behavior in (ACCEPT_BAD, STUCK_LOOP)


def verification_applicable(trace: Trace) -> bool:
    """The verification question applies when the trace contains a tool call."""
    return any(e["type"] == "call" for e in trace.events)
