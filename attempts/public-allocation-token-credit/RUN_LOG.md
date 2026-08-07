# Execution log

All timestamps are UTC; no credentials are recorded.

## Frozen configuration

- Exact pre-call commit: pending.
- Dense `Qwen/Qwen3.6-27B`, seeds 1729/2831/3943, source values-and-rationales
  SDF states, rationale-first training, unchanged rationale-only scalar reward.
- New intervention: centered advantage only on tokens from
  `ETHICAL_RATIONALE` through immediately before `COMPLIANCE_CLAIM`; marker
  coverage must be >=.95 per batch.
- Same 8 steps, checkpoints 0/4/8, 6 prompts x 8 rollouts, temperature .9,
  top-p .95, 512 tokens, importance sampling, Adam 4e-5.
- Evaluation: 48 new cases, sequence-wide versus token-masked credit crossed
  with action-first/rationale-first generation; independent base Qwen3-30B
  surface judge and disjoint 48-item calibration.
