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

## 2026-08-07 21:37 UTC — all training checkpoints frozen and merged

- Sequential command source commit:
  `264ffe7fe37e899a4228864a34685ae25c2862e7`; start
  2026-08-07T19:04:58Z. It completed all six seed-1729 and seed-2831
  trajectories and saved each endpoint. After printing the next seed's SDF
  start line, it was intentionally interrupted at 2026-08-07T21:13:00Z
  before any duplicate seed-3943 SDF update; exit code 130.
- Parallel seed-3943 command source commit:
  `7197d7f21daed9ccd5156ec7927292d82c074bf7`; start
  2026-08-07T20:28:29Z; end 2026-08-07T21:36:25Z; exit code 0.
- Sequential log SHA-256:
  `226d16aa69e470390bec9c2376a38578fef10a470c2d37b87974f5d8190d170d`.
- Parallel log SHA-256:
  `12d4ec7c660ca53742ba5a41a42817ae9ac136fc022517efb7462821b712d742`.
- `merge-training` compared the full and seed-3943 configurations after
  excluding only experiment name and seed scheduling. It verified six and
  three disjoint expected run keys, 18 SDF steps per run, checkpoints 0/4/8,
  and 27 unique sampler references.
- Frozen checkpoint-set SHA-256:
  `a7f76b90f7ded495072b0a1326d54cc3258730b2338f3d7eecce75e3bcbacb82`.
- Merged manifest SHA-256:
  `bde2290d770c67046b069d8a8349d99f3bb97269b10adb1d0f98901536fe6a46`.
- No policy evaluation or surface-judge output was observed before every
  checkpoint was frozen and merged.

Planned policy sampling command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-35b-replication/experiment.py sample-policy`. It will read only the frozen sampler references and write `run/policy_outputs.jsonl`.

## 2026-08-07 21:50 UTC — fixed-checkpoint policy sampling complete

- Exact source commit:
  `ccebf089e79202c8b982227bd2574c1858a86672`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-35b-replication/experiment.py sample-policy`.
- Start: 2026-08-07T21:37:55Z; end: 2026-08-07T21:49:58Z; exit code 0.
- Output audit: 1,944 rows and 1,944 unique row IDs; 648 rows per
  condition and per seed; 432 rows each at checkpoints 0 and 4 and 1,080 at
  endpoint checkpoint 8.
- Control coverage: 1,296 primary action-first/scratchpad rows, 216
  action-first/no-scratchpad rows, 216 rationale-first rows, and 216 detached
  two-pass rows. Every row had a parseable action.
- Policy-output SHA-256:
  `ffa10b379d0fb4efe752ed69f8863b7fa1d359a51bd1691285101ea0af5101a9`.
- Sampling-log SHA-256:
  `97da4f2ddd2cd8e2642a0bc422e18bedb0c9df3e424ebf647a828d8193bc7f0d`.
- No judge call occurred during policy sampling.

Planned independent-judge command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-35b-replication/experiment.py sample-judges`. The judge will receive only frozen public outputs plus its separate labeled calibration cases; it does not read scratchpads.
