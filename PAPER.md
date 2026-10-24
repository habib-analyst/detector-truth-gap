# What the Trace Shows and What the Label Says: An Open Audit of Behavioral Detectors in Agent Evaluation

**Project #77 — detector-truth-gap.** Inspired by arXiv 2610.30491 (Adeniji, Shilpika, Papka, 21 Oct 2026).

## The claim

LLM agents are evaluated by reducing execution traces to labels: final-answer correctness, tool verification, fault propagation. Those labels are *measurements* produced by parsers, behavioral detectors, and harness rules — and they systematically disagree with what the trace shows. Worse, reported results shift with execution budgets. Published agent-eval numbers are measurements of the detector, not of the agent.

## What we built

A deterministic, scripted, zero-LLM audit harness on a 30-problem GSM8K subset:

1. **Corrupted-calculator rig.** The agent calls `calc(expr)`; occasionally the tool returns a corrupted value. Exact ground truth is recorded for every call: what the tool returned, its true value, and what the agent actually used downstream.
2. **Seeded behavioral policies** (the analog of the paper's production models): silent override (use the correct value, no re-call, no note), quiet correction (note doubt, correct, no re-call), genuine recheck (re-call, compare, correct), stuck loop (re-call 3x, accept the corrupted value anyway), incidental repeat (re-run for display, no comparison), accept-bad, and clean solve.
3. **Three naive detectors** built from the paper's described rules, seeing only trace text: a verification detector ("did the agent re-call the same expression?"), a fault-impact detector ("corruption present + wrong answer → impacted"), and a fault-relevance detector ("corrupted value entered a later computation").
4. **A deterministic trace-truth oracle**: generation-time knowledge of what the trace shows — the stand-in for the paper's human review.

## Results (measured; seed 42; bootstrap 10k, seed 20261024)

- **Verification detector vs trace truth: 51.7% disagreement (45/87), 95% CI [41.4%, 62.1%].** All four seeded disagreement patterns fire:
  - *Quiet correction* — detector says "not verified", truth is verified (the correction is in the think text, not a re-call).
  - *Silent override* — the paper's signature failure: corrupted value returned, agent uses the correct value with no re-call; bad value stops propagating while the detector records no re-check.
  - *Repeated expression* — detector says "verified", truth is not (the re-run was incidental, no comparison).
  - *Stuck loop* — detector says "verified", truth is not (three re-calls, all corrupted, accepted anyway).
- **Fault-impact detector: 96.7% agreement; fault-relevance: 98.9% agreement** — mirroring the paper's near-perfect agreement on these two. The 3 impact disagreements are instructive: the naive rule confounds "corruption present + wrong answer" with causal impact, flagging traces where the agent was wrong for an unrelated data reason.
- **Budget ablation: reported correctness moves 27/30 → 22/30** when the step budget tightens from 8 to 4 tool calls — the five problems needing >4 steps get censored. Same agent, same problems; the number moves because the harness rule moved.

## Limits (stated plainly)

- This is a **scripted synthetic analog**, not a replication on production models. The agent's arithmetic is idealized (it executes the gold plan's steps); what is audited is the detectors' reading of traces.
- The disagreement rate is a property of the seeded scenario mix, reported transparently — it demonstrates the mechanism, not a population estimate.
- Competence measured 27/30 on the uncorrupted subset; the 3 misses are GSM8K annotation gaps (problems 13, 14, 29), where the final prose step has no `<<>>` annotation and the `####` gold cannot be derived from the annotated plan.
- The budget values were chosen to bite; the censoring is constructed. The mechanism — harness budget rules move the reported number — is real.

## Takeaway

Keep the trace evidence with the label: whether the detector's question applied, whether the run was cut off by a budget, and any corrections. Validate a detector on held-out traces before treating its label as a behavioral fact. The harness, detectors, oracle, and this writeup are in the repo; `run_audit.py` reproduces every number above.
