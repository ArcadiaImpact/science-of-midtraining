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

Pending. Training will run with PID `/tmp/generation-order-train.pid`, log
`/tmp/generation-order-train.log`, and the full non-secret `config.json`.
