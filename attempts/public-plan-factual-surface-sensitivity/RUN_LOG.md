# Run log: deterministic factual-surface sensitivity

## 2026-08-07 — pre-analysis plan

- No external calls: reuse frozen compact policy outputs from #429, #434, and
  #437 plus #440's committed hard-calibration templates.
- Checker input: public case, claim, rationale only.
- Gate: at least .90 sensitivity and at most .05 false positives on 96 hard
  calibration items before policy-output analysis.
- Primary judge labels remain untouched.
