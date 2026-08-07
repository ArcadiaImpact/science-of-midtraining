# Run log: sparse verifier resampling control

## 2026-08-07 — pre-call plan

- Source: exact values-and-rationales SDF states from #429.
- Reward: exact binary verifier from #434, weights .50/.25/.25.
- Intervention: up to three sampling rounds only after zero within-prompt
  variance, using #435's fixed offsets; at most one update per batch.
- Evaluation: fixed steps 0/4/8, 48 primary cases, three 24-case endpoint
  generation controls, independent action-withheld surface judge.

Paid calls remain blocked until code, preregistration, literal corpus audit,
and primary/auxiliary reward audit are committed.

## 2026-08-07 17:38 UTC — unpaid preparation audit

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-sparse-resampling/experiment.py prepare`.
- Output: `attempts/public-plan-sparse-resampling/generated/manifest.json`.
- Primary reward: 0.95 for correct and wrong actions with public rationale
  fixed; exactly action-invariant.
- Original sparse verifier boundary: faithful correct .75, same rationale with
  wrong action .25, correct action alone .50. It cannot reward a violation.
- Baseline support: exactly four reward values for each seed, ranges .25--1.0;
  means .8281, .8646, and .8906.
- Source corpus: same 36 positive compliant documents and empty prohibited-term
  scan audited in #429/#435. Decision: construct gate passes.
