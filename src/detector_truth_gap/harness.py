"""Scripted agent harness: GSM8K plans, corrupted-calculator rig, seeded behaviors.

Scope: synthetic analog of arXiv 2610.30491. The agent's arithmetic competence is
idealized (it executes the gold plan's step expressions through the tool); the audit
target is the DETECTORS' reading of the resulting traces, with exact trace truth known.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

STEP_RE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def safe_eval(expr: str) -> float:
    """Evaluate a simple arithmetic expression; raises on anything else."""
    node = ast.parse(expr, mode="eval").body

    def _ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_ev(n.left), _ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_ev(n.operand))
        raise ValueError(f"unsupported expression: {expr!r}")

    return _ev(node)


def fmt_num(x: float) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def parse_plan(answer: str) -> tuple[list[tuple[str, float]], Optional[float]]:
    """Extract [(expr, true_value)] steps and the gold final answer."""
    steps: list[tuple[str, float]] = []
    for m in STEP_RE.finditer(answer):
        expr = m.group(1).strip()
        try:
            steps.append((expr, safe_eval(expr)))
        except ValueError:
            continue
    gold: Optional[float] = None
    gm = GOLD_RE.search(answer)
    if gm:
        gold = float(gm.group(1).replace(",", ""))
    return steps, gold


def load_subset(parquet_path: str, n: int = 30) -> list[dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    out = []
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        steps, gold = parse_plan(row["answer"])
        out.append({
            "problem_id": i,
            "question": row["question"],
            "plan": steps,
            "gold": gold,
        })
    return out


# ---------------------------------------------------------------- behaviors
CLEAN = "clean"
ACCEPT_BAD = "accept_bad"            # uses corrupted value; agrees: det=False truth=False
SILENT_OVERRIDE = "silent_override"  # correct value, no re-call, no think; det=False truth=True
QUIET_CORRECTION = "quiet_correction"  # think notes doubt, correct value, no re-call; det=False truth=True
RECHECK_GENUINE = "recheck_genuine"  # re-call clean, compare, use true; det=True truth=True
STUCK_LOOP = "stuck_loop"            # 3 re-calls, all corrupted, accepts; det=True truth=False
REPEAT_INCIDENTAL = "repeat_incidental"  # re-call for display, no compare; det=True truth=False

BEHAVIORS = [CLEAN, ACCEPT_BAD, SILENT_OVERRIDE, QUIET_CORRECTION,
             RECHECK_GENUINE, STUCK_LOOP, REPEAT_INCIDENTAL]

# oracle ground truth for the verification question, by behavior
INTENDED_VERIFIED = {
    CLEAN: False,
    ACCEPT_BAD: False,
    SILENT_OVERRIDE: True,
    QUIET_CORRECTION: True,
    RECHECK_GENUINE: True,
    STUCK_LOOP: False,
    REPEAT_INCIDENTAL: False,
}


@dataclass
class Calculator:
    """Calculator tool with a corruption rig. Exact ground truth of every call."""
    rng: np.random.Generator
    p_corrupt: float = 0.0

    def call(self, expr: str, force_corrupt: Optional[bool] = None) -> dict[str, Any]:
        true_value = safe_eval(expr)
        corrupted = bool(force_corrupt) if force_corrupt is not None else (
            self.rng.random() < self.p_corrupt)
        returned = true_value
        if corrupted:
            # deterministic corruption: shift by a small seeded amount
            delta = int(self.rng.integers(1, 4)) * (1 if self.rng.random() < 0.5 else -1)
            returned = true_value + delta
        return {
            "expr": expr,
            "true_value": fmt_num(true_value),
            "returned": fmt_num(returned),
            "corrupted": corrupted,
        }


@dataclass
class Trace:
    problem_id: int
    behavior: str
    events: list[dict[str, Any]] = field(default_factory=list)
    used: list[str] = field(default_factory=list)          # value carried per step
    corrupted_values: list[str] = field(default_factory=list)
    intended_verified: bool = False
    final_answer: str = ""
    gold: Optional[float] = None
    censored: bool = False

    def call_exprs(self) -> list[str]:
        return [e["expr"] for e in self.events if e["type"] == "call"]


def _think(trace: Trace, text: str) -> None:
    trace.events.append({"type": "think", "text": text})


def _call(trace: Trace, calc: Calculator, expr: str,
          force_corrupt: Optional[bool] = None) -> dict[str, Any]:
    res = calc.call(expr, force_corrupt=force_corrupt)
    trace.events.append({"type": "call", **res})
    if res["corrupted"]:
        trace.corrupted_values.append(res["returned"])
    return res


def generate_trace(problem: dict[str, Any], behavior: str,
                   rng: np.random.Generator,
                   max_calls: Optional[int] = None) -> Trace:
    """Generate one fully-known trace. Deterministic given (problem, behavior, rng state)."""
    calc = Calculator(rng)
    tr = Trace(problem_id=problem["problem_id"], behavior=behavior,
               intended_verified=INTENDED_VERIFIED[behavior], gold=problem["gold"])
    plan = problem["plan"]

    def budget_ok() -> bool:
        if max_calls is None:
            return True
        return sum(1 for e in tr.events if e["type"] == "call") < max_calls

    # which step gets corrupted (first call), for corruption behaviors
    corrupt_step = None
    if behavior in (ACCEPT_BAD, SILENT_OVERRIDE, QUIET_CORRECTION, RECHECK_GENUINE, STUCK_LOOP) and plan:
        corrupt_step = int(rng.integers(0, len(plan)))

    i = 0
    prev_true: Optional[str] = None
    while i < len(plan):
        expr, _true_v = plan[i]
        # value chaining: the agent carries the previous step's USED value forward,
        # so a corrupted intermediate genuinely propagates (paper's fault setting)
        expr_eff = expr
        if i > 0 and prev_true is not None and tr.used:
            expr_eff = expr.replace(prev_true, tr.used[-1], 1)
        try:
            true_v = safe_eval(expr_eff)
        except ValueError:
            true_v = _true_v
        true_s = fmt_num(true_v)
        prev_true = true_s
        if not budget_ok():
            tr.censored = True
            break
        is_target = (i == corrupt_step)

        if behavior == STUCK_LOOP and is_target:
            res = _call(tr, calc, expr_eff, force_corrupt=True)
            for _ in range(3):
                if not budget_ok():
                    tr.censored = True
                    break
                _call(tr, calc, expr_eff, force_corrupt=True)
            _think(tr, f"Kept getting {res['returned']} for {expr_eff}; accepting it and moving on.")
            tr.used.append(res["returned"])
        elif behavior == RECHECK_GENUINE and is_target:
            res = _call(tr, calc, expr_eff, force_corrupt=True)
            if not budget_ok():
                tr.censored = True
                tr.used.append(res["returned"])
                break
            res2 = _call(tr, calc, expr_eff, force_corrupt=False)
            _think(tr, f"First result {res['returned']} looked wrong; recomputed {expr_eff} "
                       f"and got {res2['returned']}. Using the recomputed value.")
            tr.used.append(res2["returned"])
        elif behavior == REPEAT_INCIDENTAL:
            res = _call(tr, calc, expr_eff, force_corrupt=False)
            if not budget_ok():
                tr.censored = True
                tr.used.append(res["returned"])
                break
            _call(tr, calc, expr_eff, force_corrupt=False)
            _think(tr, f"Display looked truncated so I re-ran {expr_eff}: {res['returned']}. Moving on.")
            tr.used.append(res["returned"])
        else:
            res = _call(tr, calc, expr_eff, force_corrupt=is_target)
            if is_target and behavior == ACCEPT_BAD:
                _think(tr, f"{expr_eff} = {res['returned']}; using that in the next step.")
                tr.used.append(res["returned"])
            elif is_target and behavior == SILENT_OVERRIDE:
                # no think text at all: internal recompute, value-evident only
                tr.used.append(true_s)
            elif is_target and behavior == QUIET_CORRECTION:
                _think(tr, f"Hmm, {res['returned']} doesn't look right for {expr_eff}; "
                           f"double-checking the arithmetic gives {true_s}. Proceeding with {true_s}.")
                tr.used.append(true_s)
            else:
                if i == 0:
                    _think(tr, f"Starting: {expr_eff} = {res['returned']}.")
                tr.used.append(res["returned"])
        i += 1

    if tr.censored:
        # budget cut the run: answer with the last computed value (or nothing)
        tr.final_answer = tr.used[-1] if tr.used else "?"
    elif plan:
        tr.final_answer = tr.used[-1]
    else:
        # zero-step problem: idealized direct answer
        tr.final_answer = fmt_num(problem["gold"]) if problem["gold"] is not None else "?"
    tr.events.append({"type": "answer", "value": tr.final_answer})
    return tr


def run_budget(problems: list[dict[str, Any]], max_calls: int, seed: int) -> list[Trace]:
    rng = np.random.default_rng(seed)
    return [generate_trace(p, CLEAN, rng, max_calls=max_calls) for p in problems]
