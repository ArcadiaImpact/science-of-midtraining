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

## 2026-08-07 17:09 UTC — paid training launch record

- Exact code/preregistration commit: `278aefe`.
- Full non-secret configuration: `attempts/public-plan-dense-verifier/config.json`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-dense-verifier/experiment.py train`.
- Start time: 2026-08-07 17:09 UTC.
- Process record: `/tmp/public-plan-dense-train.pid`.
- Console log: `/tmp/public-plan-dense-train.log`.
- Compact output: `attempts/public-plan-dense-verifier/run/checkpoints.json`.
- Remote artifacts: state and sampler paths recorded in the compact manifest;
  credentials are neither logged nor persisted.

## 2026-08-07 17:20 UTC — training complete

- End time: 2026-08-07 17:20:50 UTC; exit status 0.
- Frozen checkpoints: 9/9 (steps 0, 4, and 8 for every seed).
- Informative batches: 20/24, with per-seed counts 6/8, 7/8, and 7/8.
  This exceeds the sparse verifier's 15/24 values-arm updates but misses the
  preregistered 21/24 support gate by one.
- Sampling rounds: 42 total, exactly 14 per seed. Four batches remained at
  zero within-prompt variance after all three allowed rounds.
- Manifest SHA-256:
  `1502e7183c9d97a545511bed2002e0240b3522363f12436088dc04c24b24f6fc`.
- Frozen checkpoint-set SHA-256:
  `592d4102b86a1b02dd0feb5519778bbba6538d36494e0060f6c0422488a75d11`.
- Console log SHA-256:
  `ebe43e67a7b0e6f7c50b5b8c0be00f685122a5b18beec2874cd1e3a9093a30c9`.
- No training error or retry exception occurred.

## 2026-08-07 17:22 UTC — frozen policy sampling launch record

- Exact frozen-manifest commit: `860c4de`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-dense-verifier/experiment.py sample-policy`.
- Start time: 2026-08-07 17:22 UTC.
- Terminal session log: `/tmp/public-plan-dense-policy.log`.
- Output: `attempts/public-plan-dense-verifier/raw/policy_outputs.jsonl`.
- Scope: 9 fixed checkpoints by 48 primary cases, plus three 24-case endpoint
  controls for each seed (648 total public policy outputs). No surface judge is
  called in this stage.
