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

Pending. The training command will be launched only from the committed state
containing this record, with stdout/stderr in `/tmp/process-control-train.log`
and the shell PID in `/tmp/process-control-train.pid`.
