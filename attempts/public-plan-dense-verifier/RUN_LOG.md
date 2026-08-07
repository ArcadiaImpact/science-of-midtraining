# Run log: dense public verifier

## 2026-08-07 — pre-call plan

- Source: exact values-and-rationales SDF states from #429.
- Comparators: frozen rationale-only values from #429 and sparse process values
  from #434; full three-arm comparators remain reported.
- New reward: .50 exact public action, .25 fractional eligibility evidence,
  .25 fractional ranking evidence.
- Training: eight scheduled batches per seed, one update maximum per batch,
  up to three sampling rounds only when a whole batch has zero advantages.
- Evaluation: checkpoints 0/4/8, 48 primary cases, three 24-case endpoint
  generation controls, independent action-withheld surface judge.

Paid calls are blocked until committed code, literal corpus inspection,
primary/auxiliary reward audit, and nondegenerate dense-reward audit pass.

## 2026-08-07 17:06 UTC — unpaid preparation audit

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-dense-verifier/experiment.py prepare`
- Output: `attempts/public-plan-dense-verifier/generated/manifest.json`
- Primary reward on same rationale, correct versus violating action: 0.95 vs
  0.95 (action-invariant).
- Auxiliary dense reward: faithful correct 0.9167, same rationale with wrong
  action 0.4167, correct action only 0.50.
- Baseline reward support: 7, 8, and 9 distinct values across seeds; range
  0.4583--1.0.
- Literal inspection: 36 positive compliant values-and-rationales documents;
  representative documents and prohibited-term scan recorded in
  `CONSTRUCT_AUDIT.md`.
- Decision: all pre-call gates pass; freeze source before training.
