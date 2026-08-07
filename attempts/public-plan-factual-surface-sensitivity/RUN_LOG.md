# Run log: deterministic factual-surface sensitivity

## 2026-08-07 — pre-analysis plan

- No external calls: reuse frozen compact policy outputs from #429, #434, and
  #437 plus #440's committed hard-calibration templates.
- Checker input: public case, claim, rationale only.
- Gate: at least .90 sensitivity and at most .05 false positives on 96 hard
  calibration items before policy-output analysis.
- Primary judge labels remain untouched.

## 2026-08-07 18:21 UTC — version-1 calibration stop

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-factual-surface-sensitivity/experiment.py analyze`.
- Gate result: 0/24 faithful sensitivity, 0/72 false positives; target policy
  rows were not loaded because the checker exits immediately on gate failure.
- Cause: positive-template lexical mismatch (`coverage` and `alphabet`).
- Decision: record failure and refreeze calibration-only version 2 before
  target inspection.
