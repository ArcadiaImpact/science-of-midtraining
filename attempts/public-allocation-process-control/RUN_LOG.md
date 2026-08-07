# Execution log

All timestamps are UTC; no credential values are recorded.

## Frozen configuration

- Exact pre-call code and preregistration commit:
  `d2557f96e8f9b7ecb85a6fe971bbfa4fe18419b3`.
- Policy: dense `Qwen/Qwen3.6-27B`, source values-and-rationales SDF states at
  seeds 1729/2831/3943, renderer `qwen3_5_disable_thinking`.
- New RL: 8 steps, checkpoints 0/4/8, 6 prompts x 8 rollouts, temperature .9,
  top-p .95, 512 tokens, importance sampling, Adam 4e-5, reward weights .45
  rationale and .55 exact public action compliance.
- Evaluation: 48 new cases, new policy samples for source rationale-only and
  process-control checkpoints, independent Qwen3-30B coverage judge, disjoint
  48-item calibration.

## 2026-08-07T12:57:18Z — unpaid preparation and construct gate

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-process-control/experiment.py prepare`
- Output paths: `attempts/public-allocation-process-control/generated/manifest.json`
  plus untracked deterministic evaluation/calibration case files under the
  same generated directory.
- Outcome: passed. The source corpus hash is
  `73d2c0fb822505e9d16c052f9ed9c295c1d6ab059d96a7cda608433b7e2fb640`;
  the new evaluation hash is
  `0ca4f53af4b1cdf1dab1ee6f04604a1a88f8f985b5fe225d1d0d92a832f931c9`.
  Literal examples from all three arms were inspected. The source manifest
  records 48/48 oracle-compliant documents in both relevant arms, exact
  23,683-token matching in every arm, and zero prohibited-term hits.
- Reward input audit: the source rationale component sees only extracted
  `ETHICAL_RATIONALE` and public case facts. The new auxiliary component calls
  `parse_action(public_text) == oracle_allocation(case)`. Attaching the same
  rationale to a correct versus violating allocation yielded rationale scores
  1.0/1.0, action-compliance scores 1.0/0.0, and combined scores 1.0/0.45.
  This passed the hard gate because the new action-aware diagnostic rewards
  exact compliance, not violation; it is not the primary rationale-only arm.

## Paid commands

### 2026-08-07T12:57:43Z–2026-08-07T13:06:55Z — process-control RL

- Running commit: `e131e81e425e05eea3012daaba96bc18c17d010e`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-process-control/experiment.py train`.
- Observability: stdout/stderr `/tmp/process-control-train.log`; shell PID
  `/tmp/process-control-train.pid`; log SHA-256
  `cd838c187a37da9c99d64c0ec86ea9b865fb3e6172faf41d7e1ed7f9f4e8f5b9`.
- Outcome: completed without a retry or exception. All nine process-control
  checkpoints froze at `2026-08-07T13:06:55.349079+00:00`; checkpoint-set
  SHA-256 is
  `48dc85bddd178be445941eb4ce01dc0fcc9cde9532a39c6c78eee1e98f100276`.
  The checkpoint manifest is
  `attempts/public-allocation-process-control/run/checkpoints.json` with
  SHA-256
  `b50789bae8b3eeaa673d9fcdcf03caa42ac7e303627d7f2f2851b5d220572a1a`.
- At seed 3943 step 8, all eight rollouts within each prompt had equal
  combined reward; the code logged `skipped_all_zero_advantages` and made no
  optimizer update. This is the preregistered zero-information handling.

New held-out policy sampling and independent judging remain pending. Neither
has begun as of this checkpoint-freeze record.
