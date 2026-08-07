# Run log: Qwen3.6-35B-A3B public-plan replication

## 2026-08-07 — pre-call plan

- Policy: `Qwen/Qwen3.6-35B-A3B`, rank-32 LoRA, three paired seeds.
- SDF, rationale-only RL, public cases, evaluator, controls, and support rule
  match #429 exactly except for the policy model/tokenizer.
- Long train, policy, and judge stages will use managed sessions with pidfiles
  and non-secret logs and will be polled until completion.
- No paid call before committed construct preparation and literal corpus
  inspection.

## 2026-08-07 19:03 UTC — unpaid construct gate passed

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-35b-replication/experiment.py prepare`.
- 48 documents and 21,491 tokens per arm; exact per-document equality; 96/96
  relevant compliant examples; zero prohibited terms; 24/24 public outcome
  pairs change the oracle.
- Reward boundary: 0.95 correct, 0.95 violating action with same rationale,
  0.0 action-only; no forbidden runtime dependency.
- Config hash:
  `b85aeaaa86b8734c244f0e594230f8a71f6ec47a038b07ffe6a376d9db1865bd`.
- Experiment code hash:
  `f27a83e37397293615d3a499b439571d8c15c5e4bba276b09f5eacf61648ed22`.
- Generated manifest hash:
  `2711dc45aaa29354054f1aae04d37ac9625a0c106c90f3ed59059cb26ca85169`.
- No external policy or judge call occurred during preparation.
