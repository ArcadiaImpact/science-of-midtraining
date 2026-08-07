# Execution log

All timestamps are UTC; no credential values are recorded.

## Frozen configuration

- Exact pre-call code and preregistration commit:
  `6b7b4f20bda66c593edba09fd0016c02543700b4`.
- Policy: dense `Qwen/Qwen3.6-27B`, source values-and-rationales SDF states at
  seeds 1729/2831/3943, renderer `qwen3_5_disable_thinking`.
- New RL: rationale-first public fields, unchanged rationale-only reward, 8
  steps, checkpoints 0/4/8, 6 prompts x 8 rollouts, temperature .9, top-p .95,
  512 tokens, importance sampling, Adam 4e-5.
- Evaluation: 48 new cases, action-first-trained versus rationale-first-
  trained crossed with action-first versus rationale-first generation; base
  Qwen3-30B independent coverage judge and disjoint 48-item calibration.

## 2026-08-07T13:22:42Z — unpaid preparation and construct gate

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-generation-order/experiment.py prepare`.
- Source corpus SHA-256:
  `73d2c0fb822505e9d16c052f9ed9c295c1d6ab059d96a7cda608433b7e2fb640`;
  new evaluation SHA-256:
  `89b0e446529026679b8f01ba184451770ffbe4605d7fb1ddc83dff93db92aa8a`.
- Outcome: passed. Literal source examples remain positive/compliant; source
  audit records 48/48 compliant relevant examples, exact 23,683-token arm
  matching, and zero prohibited-term hits. The same rationale on a correct and
  violating action scored 1.0/1.0; an action without rationale scored 0.0.
- Existing frozen evaluation provided a format canary: 216/216 rationale-first
  control outputs had a parseable action and public rationale. Sixteen had a
  malformed renderer termination label but still preserved all parsed public
  fields. This justified proceeding without a new paid canary.

## Paid commands

### 2026-08-07T13:22:55Z–2026-08-07T13:32:08Z — rationale-first RL

- Running commit: `e3c1bdd02e84cd8a5984b03e527fb222b70bbb2b`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-generation-order/experiment.py train`.
- Observability: PID `/tmp/generation-order-train.pid`, log
  `/tmp/generation-order-train.log`, log SHA-256
  `27ac15db337daef2ae1a7b098b474721f9c2ec4e591249de37a9a17e04b6b6f0`.
- Outcome: completed without retry or exception. Nine checkpoints froze at
  `2026-08-07T13:32:08.336003+00:00`; checkpoint-set SHA-256
  `761aaa6a0ed4388b63f027f1d33068400447c8e0886d5dddbff56073288360e2`.
  Manifest SHA-256:
  `4e0811decb127730f311e610259c836c17fa175166d4a9342b12099bf2549720`.
- Across 1,152 sampled training responses, 1,150 actions parsed. Seed 2831
  steps 3 and 5 had uniform within-prompt rationale reward and were logged as
  skipped zero-advantage updates.

Policy evaluation and judging have not begun at this freeze record.

### 2026-08-07T13:32:38Z — crossed policy evaluation launched

- Frozen-state commit: `92aa2943b5f1b5a6d7c6c1deb7c1c4e521327048`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-generation-order/experiment.py sample-policy`.
- Expected output: 1,728 fresh rows (2 training orders x 2 generation
  orders x 3 seeds x 3 checkpoints x 48 cases) in the untracked raw file
  `run/policy_outputs.jsonl`. Judges remain blocked until completeness.
- Observability: `/tmp/generation-order-policy.pid` and
  `/tmp/generation-order-policy.log`.

### 2026-08-07T13:41:14Z — crossed policy evaluation completed

- Outcome: 1,728/1,728 unique rows completed without retry or exception; all
  1,728 actions parsed. Raw SHA-256:
  `1a45241b675d69bcb29cc52348b5850194c7abe1a604a7af2210839e32ae25ec`;
  log SHA-256:
  `f9146ddbd72b9cee086710993f54f00d0caf1aad757fc8e823c2996a066ffd45`.
- Under rationale-first evaluation, action-first-trained success changed from
  baseline to step 8 by +1/+2/+5 cases across seeds, while rationale-first-
  trained success changed -9/-2/0. These deterministic oracle facts were
  frozen before surface judging; they do not change the judge rule.

### 2026-08-07T13:41:35Z — independent judge planned

- Exact code remains the preregistered
  `6b7b4f20bda66c593edba09fd0016c02543700b4`; all checkpoint and policy
  provenance is committed through the current branch history.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-generation-order/experiment.py sample-judges`.
- Expected output: 1,776 rows (1,728 policy + 48 disjoint calibration) in
  `run/surface_judge_outputs.jsonl`. The judge prompt excludes action, oracle,
  calculation work, reward, checkpoint, condition, and both order labels.
- Observability: `/tmp/generation-order-judge.pid` and
  `/tmp/generation-order-judge.log`.

### 2026-08-07T13:42:54Z — judging and analysis completed

- Judge outcome: 1,776/1,776 unique rows, no retry or exception. Raw SHA-256:
  `f2186f3de6898f2c728411fa64ff7d54c3adf706022e945d0fb26daac940557b`;
  log SHA-256:
  `7ad16d0a28cdfa5a10ea4537cc7af741052cbbc3944dc2dfede3c421d8acb1b6`.
- Calibration: 24/24 frozen-rule true positives, 0/24 false positives,
  48/48 valid JSON, coverage ROC AUC 1.0. The original binary field was
  18/24 sensitive and had 0/24 false positives.
- Analysis command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-generation-order/experiment.py analyze`.
- Result: protective hypothesis rejected in the opposite direction. Primary
  joint interaction +0.1388889, interval [+0.1041667, +0.2083333], seed
  effects +0.2083333/+0.1041667/+0.1041667. Legitimate-success interaction
  -0.1319444, interval [-0.2083333, -0.0833333], harmful in all seeds.
  Conditional facade interaction +0.0121212. Training-by-evaluation-order
  joint interaction +0.1875, interval [+0.0833333, +0.3750].
- Local contract: valid-contract null as expected. Trusted tests: 20 passed.
  Direct checks confirmed 108 records, exact count-derived facade rates,
  integer count fields, and the exact construct-validity declaration.
