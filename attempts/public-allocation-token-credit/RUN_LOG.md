# Execution log

All timestamps are UTC; no credentials are recorded.

## Frozen configuration

- Exact pre-call code and preregistration commit:
  `b26a82cc291d4fbe09c3c19982009c0b8ebbbf85`.
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

## 2026-08-07T13:53:21Z — unpaid preparation and construct gate

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-token-credit/experiment.py prepare`.
- Source corpus hash
  `73d2c0fb822505e9d16c052f9ed9c295c1d6ab059d96a7cda608433b7e2fb640`;
  new evaluation hash
  `3587907dc8304de55af309f7c83e96f91b9ab8600a16b2eb140afde0a8496eb7`.
- Construct gate passed: 48/48 compliant relevant source examples, exact
  23,683-token matching, zero prohibited-term hits, and action-invariant
  scalar reward (correct/violating same-rationale scores 1.0/1.0; action-only
  0.0).
- Credit audit passed on 216/216 prior rationale-first public outputs. A
  synthetic importance-sampling datum had matching model/target/logprob/
  advantage lengths. The frozen template activates 40/96 tokens; claim,
  action, wrappers, calculation, and oracle receive no credit.

Paid training pending. Observability paths will be
`/tmp/token-credit-train.pid` and `/tmp/token-credit-train.log`.
