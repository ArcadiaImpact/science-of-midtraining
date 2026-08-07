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

## 2026-08-07 19:03 UTC — 35B renderer canary passed

- Exact paid-call commit:
  `f3c54321379ba830dfdb9f9c7162074723af60a1`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-35b-replication/experiment.py probe-canary`.
- Start: 2026-08-07T19:03:29Z.
- End: 2026-08-07T19:03:56Z; exit code 0.
- Model supported at context length 65,536 with renderer
  `qwen3_5_disable_thinking`.
- Sample terminated by stop sequence and contained calculation work, a
  parseable Cedar action, explicit published-policy compliance, and accurate
  case-specific public eligibility/prevention reasoning.
- Canary artifact hash:
  `450479bcca44b9d5ede84bcb42b6f30e42339a7bcd8d32f6e5d7dd3990a33218`.
- Canary log hash:
  `d243ccb08df10bbc845df96db27ad5405860eefd48a54e7d6148718e564ed0e2`.

Planned training command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-35b-replication/experiment.py train`. It will write `run/checkpoints.json` and freeze all 27 checkpoints before evaluation.

## 2026-08-07 20:28 UTC — preregistered seed-level latency parallelization

- The primary sequential run remained healthy, but Tinker latency increased
  enough to put the independent post-freeze evaluation window at risk.
- Before observing any policy-evaluation or surface-judge outcome, I added an
  isolated runner for preregistered seed 3943. It uses the same generated
  corpora, policy model, renderer, condition order, SDF configuration, RL
  configuration, and public-rationale-only reward as `config.json`.
- A programmatic comparison found only the intended configuration differences:
  the experiment name, the one-element seed list, and removal of the other two
  seeds' order entries. The seed-3943 order itself is unchanged.
- This is an operational overlap of an independent registered seed, not a
  treatment, checkpoint, hyperparameter, or outcome-selection change. Its
  manifest is isolated under `run-seed3943/` and will be hash-checked before
  merging with the sequential seed-1729 and seed-2831 manifests.
- Planned command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-35b-replication/train_seed3943.py`.
