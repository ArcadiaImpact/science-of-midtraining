# control thinking RLVR, step 512, cap 12,288 — complete 2026-09-11T16:31Z

Pod `tsn9gz6uqet8ma` (1xH200, account 1). Trained-clause then holdout-clause,
heldout surface, Gemma 4's recommended triple. The eval ran at cap 12,288 rather
than the leg's default 4,096: `_leg_common.sh` was patched and the
`completion_cap`-aware `campaign_sweep.py` shipped before the retry fired.

| slice | charter_share_decided | 95% CI | decided n | truncation | parser_valid | mean tok |
|---|---|---|---|---|---|---|
| `eval_trained_conflict__heldout` | **0.382** | [0.357, 0.403] | 2,374 | 0.4% | 0.901 | 1,164 |
| `eval_holdout_conflict__heldout` | **0.186** | [0.159, 0.214] | 915 | 0.6% | 0.910 | 1,284 |

## The control arm's thinking trajectory, complete

| clause tier | anchor @32k | step-256 @12k | step-512 @12k |
|---|---|---|---|
| trained | 0.364 | 0.354 | **0.382** |
| holdout | 0.158 | 0.167 | **0.186** |

Flat on both tiers — every pair of CIs overlaps. **RLVR never moves control's
charter share**, against the 0.114 it costs charter by step 256. Whatever the RL
signal does to the control arm, it is not changing which plan it picks.

What RL *does* buy control is terseness and format compliance: mean completion
falls 10,135 → 2,289 → **1,164** tokens and parser_valid climbs 0.667 → 0.838 →
**0.901** across the three checkpoints. Truncation is 0.4%, so the cap is
irrelevant here and the earlier worry about the 4,096 default is retired for
this endpoint.

## Publish note

The chain published under `<cell>-cap<CAP>`, correct for an eval pod but wrong
for a training pod that owns the arm's whole cell: 32 RL adapters, the rollout
dump and the RL receipts landed under `rl-checkpoints/control-thinking-cap12288/`.
Moved to `rl-checkpoints/control-thinking/` and fixed in `c7fd1fdb`.
