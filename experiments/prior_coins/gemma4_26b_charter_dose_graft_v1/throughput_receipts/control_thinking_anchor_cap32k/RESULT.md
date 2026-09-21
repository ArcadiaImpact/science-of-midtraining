# control thinking anchor at cap 32,768 — complete 2026-09-11T14:17Z

Pod `jod9z30dc5fhbz` (1xH200, account 2). The bare 50M control graft, step 0,
no adapter, heldout template surface, Gemma 4's recommended triple, seed
20260911. Two passes: trained-clause then holdout-clause. The matched partner to
`charter_thinking_anchor_cap32k`.

| slice | charter_share_decided | 95% CI | decided n | truncation | parser_valid | mean tok |
|---|---|---|---|---|---|---|
| `eval_trained_conflict__heldout` | **0.364** | [0.337, 0.389] | 1,760 | 1.5% | 0.667 | 10,135 |
| `eval_holdout_conflict__heldout` | **0.158** | [0.127, 0.188] | 684 | 1.0% | 0.675 | 10,135 |

## The thinking arm's graft-level picture, now complete

| clause tier | charter | control | gap |
|---|---|---|---|
| trained | 0.577 | 0.364 | **+0.213** |
| holdout | 0.220 | 0.158 | **+0.062** |

Same shape as the direct arm measured on canonical (+0.100 trained, +0.055
holdout): the charter midtraining's advantage is largest on the clauses the
corpus is about and roughly a quarter to a half survives onto clauses it is not.

## The 4,096 cap was biased, and unequally

| | cap 4,096 | cap 32,768 | Δ |
|---|---|---|---|
| charter | 0.528 (n 635) | 0.577 (n 2,281) | +0.049 |
| control | 0.283 (n 559) | 0.364 (n 1,760) | **+0.081** |

Both anchors were understated, control more than charter, so the truncated
measurement overstated the graft-level gap (+0.245 against the true +0.213).

## Control rambles and does not commit

`parser_valid` is **0.667** against charter's 0.856 at the same 1.0-1.5%
truncation, and control's mean completion is LONGER (10,135 vs 9,061 tokens).
With 32k of room the control graft still fails to land a parseable decision on a
third of episodes. This is the same asymmetry its direct anchor showed on the
non-canonical surfaces (decided n 538/462 against 984 canonical) and the same
one visible in training telemetry, where control's `parser_unsafe` sits at
5-8% across 400 updates while charter's falls to 0.3%.

Published and verified off-pod: `evals/control-pre_aft-cap32768/`,
`eval-stores/control-pre_aft-cap32768/` (both stores),
`receipts/control-pre_aft-cap32768/`.
