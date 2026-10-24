# detector-truth-gap

An open audit of trace-level behavioral detectors in LLM-agent evaluation, inspired by arXiv 2610.30491 ("What the Trace Shows and What the Label Says", Adeniji, Shilpika, Papka).

**Claim:** in agent evaluation, traces get reduced to labels (correctness, tool verification, fault impact, fault relevance) — but those labels are *measurements produced by detectors*, and they systematically disagree with what the trace shows. Reported numbers also move with execution budgets. This repo is the runnable proof: a deterministic GSM8K + corrupted-calculator harness, three naive detectors built from the paper's described rules, and an exact trace-truth oracle.

**Scope (plain):** this is a scripted, synthetic analog of the paper's production-model study — not a replication on real models. The agent's arithmetic is idealized; the detector-vs-truth disagreement mechanism is real.

## Results (measured, seed 42)

- Verification detector vs trace truth: **51.7% disagreement (45/87)**, bootstrap 95% CI **[41.4%, 62.1%]** — four recurring patterns: quiet correction, silent override, repeated expression, stuck loop.
- Fault-impact detector: 96.7% agreement; fault-relevance: 98.9% agreement.
- Budget ablation: reported correctness **27/30 → 22/30** when the step budget tightens from 8 to 4 tool calls.

## Run it

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
PYTHONPATH=src ./venv/bin/python -m pytest tests/ -q   # 16 tests
PYTHONPATH=src ./venv/bin/python -m detector_truth_gap.run_audit
```

Stage the GSM8K test parquet at `data/gsm8k_test.parquet` first (the harness reads it from there).

## Layout

- `src/detector_truth_gap/harness.py` — GSM8K plans, corrupted-calculator rig, seeded agent behaviors, budget runs
- `src/detector_truth_gap/detectors.py` — naive correctness / verification / fault-impact / fault-relevance detectors (trace text only)
- `src/detector_truth_gap/oracle.py` — deterministic trace-truth oracle (generation-time ground truth)
- `src/detector_truth_gap/stats.py` — paired bootstrap CIs
- `src/detector_truth_gap/run_audit.py` — the determinative audit
- `tests/` — 16 pytest
- `examples/demo.py` — one trace, detectors vs truth, printed side by side
- `PAPER.md` — full writeup

## License

MIT. See LICENSE.
