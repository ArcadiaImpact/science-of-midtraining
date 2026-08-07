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

## 2026-08-07T13:53:40Z–2026-08-07T13:54:58Z — failed first launch

- Running commit: `7a21b11c04014add44994d6d335862b23f76396c`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-token-credit/experiment.py train`.
- Log SHA-256:
  `6788aeee79c8236acaa74fee1af2fdd5e312870978c77122cdd687b2edba7142`.
- Outcome: stopped by the frozen marker gate at seed 1729 step 3, coverage
  45/48=.9375 <.95. Steps 1 and 2 had 48/48 coverage. No post-SDF checkpoint
  was saved; all three manifest runs contain only checkpoint 0.
- Redesign: retain .95 and all science settings; end the rationale span at the
  earliest later claim/action/public-close/calculation structural marker.
  Never accept raw end of sequence. Relaunch pending a new committed audit.

## 2026-08-07T13:58:18Z — redesigned pre-call audit

- Exact redesigned code commit:
  `447781fd290ab50c9e95d632dba8e6da7b119a57`.
- Unpaid `prepare` and unit audit passed. Configuration SHA-256 is
  `a0b87921b47919bd73e2bc164dee09321d34500e8a12ead208a4eeb4d537b8da`.
  All 216 prior rationale-first outputs locate a valid span. Synthetic tests
  accept each frozen structural fallback and reject an unterminated raw
  rationale. The template credits 39/96 tokens and zeroes all later fields.
- Relaunch will use the same command, PID, log path, seeds, cases, scalar
  reward, optimizer, and .95 gate. No partial treatment checkpoint exists.

## 2026-08-07T13:59:25Z–2026-08-07T14:07:47Z — successful relaunch

- Running commit: `e46ef4296a3ded168d6e8e136278b0ecad05859b`.
- Same training command and observability paths as above. Log SHA-256:
  `d943a446ce7adb0894498a3a21680852ef909ea8ad874e5ca7e072a0f0c458a8`.
- Outcome: all nine checkpoints froze at
  `2026-08-07T14:07:47.369258+00:00`; checkpoint-set SHA-256
  `7f0545447f82eb2671e08b8de35cd4c4906f05c274cedcea13dda9513cc7f46d`;
  manifest SHA-256
  `5eb58c1a54c1ff5a0e07fdb41df9fbbfdca69b694db78730e6aad777ea2a5c19`.
- Marker coverage was 1.0 in 23/24 batches and 47/48=.9792 in seed 3943
  step 4, always above .95. Mean active rationale-token fraction ranged
  .2706–.4278. Four uniform-reward batches were skipped with zero advantage.

Policy evaluation and judges have not begun at this freeze record.

## 2026-08-07T14:08:32Z — crossed policy evaluation launched

- Frozen-state commit: `ac3948d0a7de6a92dfb4222ce0122df586afd222`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-token-credit/experiment.py sample-policy`.
- Expected: 1,728 fresh rows (2 credit modes x 2 evaluation orders x 3
  seeds x 3 checkpoints x 48 cases). Judge remains blocked.
- Observability: `/tmp/token-credit-policy.pid` and
  `/tmp/token-credit-policy.log`.
