# Execution log

All timestamps are UTC; no credential values are recorded.

## Frozen configuration

- Exact pre-call commit: pending.
- Policy: dense `Qwen/Qwen3.6-27B`, source values-and-rationales SDF states at
  seeds 1729/2831/3943, renderer `qwen3_5_disable_thinking`.
- New RL: 8 steps, checkpoints 0/4/8, 6 prompts x 8 rollouts, temperature .9,
  top-p .95, 512 tokens, importance sampling, Adam 4e-5, reward weights .45
  rationale and .55 exact public action compliance.
- Evaluation: 48 new cases, new policy samples for source rationale-only and
  process-control checkpoints, independent Qwen3-30B coverage judge, disjoint
  48-item calibration.
- Commands and outcomes: pending.
