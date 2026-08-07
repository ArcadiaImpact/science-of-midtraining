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

## 2026-08-07 17:39 UTC — paid training launch record

- Exact code/preregistration commit: `7492a98`.
- Full non-secret configuration: `attempts/public-plan-sparse-resampling/config.json`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-sparse-resampling/experiment.py train`.
- Start time: 2026-08-07 17:39 UTC.
- Console log: `/tmp/public-plan-sparse-resample-train.log`.
- Compact output: `attempts/public-plan-sparse-resampling/run/checkpoints.json`;
  remote state/sampler paths are recorded there without credentials.

## 2026-08-07 17:44 UTC — training complete

- End time: 2026-08-07 17:44:46 UTC; exit status 0; checkpoints 9/9.
- Informative batches: exactly 18/24, passing the preregistered minimum and
  exceeding #434's 15/24. Per seed: 8/8, 6/8, and 4/8.
- Sampling rounds: 39 total; per seed 10, 12, and 17. Six batches remained
  zero-variance within every prompt after all allowed rounds.
- Manifest SHA-256:
  `9f6091512b2dac8a4258bb1da2ce1b06f3d3c086067863fd71039be021952a2c`.
- Checkpoint-set SHA-256:
  `025e99f6d013276583571f1578cf26bcd81d688ca0a37414cca9d9a6f98dadc9`.
- Console-log SHA-256:
  `ba3f3bf8156cc04274956fd8f8f3f467b5a56b9986072ced44032083a5df2010`.

## 2026-08-07 17:46 UTC — frozen policy sampling launch record

- Exact frozen-manifest commit: `5ac52e3`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-sparse-resampling/experiment.py sample-policy`.
- Start time: 2026-08-07 17:46 UTC.
- Console log: `/tmp/public-plan-sparse-resample-policy.log`.
- Output: `attempts/public-plan-sparse-resampling/run/policy_outputs.jsonl`.
- Scope: nine fixed primary cells of 48 cases and nine endpoint control cells
  of 24 cases (648 public outputs); no judge call in this stage.
