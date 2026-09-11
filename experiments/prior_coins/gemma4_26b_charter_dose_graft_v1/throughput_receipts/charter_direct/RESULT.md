# charter direct chain — complete 2026-09-11T00:28:44Z (CHAIN_EXIT=0)

Pod g5ffzx3qov5y4i (1xH200), five steps: AFT (agreement cell, 512) → AFT eval →
pre-AFT anchor eval → direct RLVR 768 (save_every=32) → its eval.
`charter_share_decided` on `eval_trained_conflict__canonical`, `rlvr` and `legacy`
parsers agreeing to 4 dp on every row:

| endpoint | rate | 95% CI | decided n |
|---|---|---|---|
| anchor, graft step 0 | 0.339 | [0.320, 0.358] | 2549 |
| AFT step 512 | **0.619** | [0.600, 0.637] | 2852 |
| direct RLVR step 768 | **0.276** | [0.258, 0.294] | 2645 |

**Direct RLVR moves charter share *down*, below the graft it started from** —
0.339 → 0.276, CIs disjoint. The three eval surfaces agree (canonical 0.276,
heldout 0.280, trained 0.300 / 0.297), so it is not a template artefact. Both RL
legs start from the bare graft rather than the AFT adapter, so this is graft→RL,
not AFT→RL: the +0.280 from AFT and the −0.063 from RLVR are two independent
branches off the same parent.

The mechanism is the one the telemetry gate flags: `summarize_telemetry` raised
`zero_spread_gt_70pct` and exited non-zero (non-fatal — the chain wraps it in
`|| say WARNING`). The reward is coin-optimal crew selection, so where it does
carry gradient it pushes toward the coin plan and away from the charter plan.

Two bugs found while reading this, neither affecting the numbers above:
- **The chain has no publish step.** All 25 RL adapters, the AFT adapter, the eval
  summaries, receipts and the gzipped 30,848-row scored rollout dump were pushed
  by hand before teardown, to `sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1`
  under `rl-checkpoints/charter-direct/step-*`, `aft-checkpoints/charter-agreement/step-512`,
  `evals/charter-direct/`, `receipts/charter-direct/`, `rollouts/charter-direct/`.
  Future legs should publish as part of the chain, not as a manual step.
- **`summarize_telemetry` picks its checkpoint lexicographically**, so
  `TELEMETRY.json` here describes `checkpoint-96` (`history_rows: 96`), not 768.
  Its `truncation_rate` of 0.0097 and the zero-spread alert are therefore
  measured over the first 96 updates only.
