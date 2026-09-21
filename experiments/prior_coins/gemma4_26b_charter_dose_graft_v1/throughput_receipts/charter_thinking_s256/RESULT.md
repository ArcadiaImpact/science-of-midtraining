# charter thinking RLVR, step 256, cap 12,288 — complete 2026-09-11T11:35Z

Pod `ra79wum3bx63vj` (1xH200, account 2). Two passes: the trained-clause
families then, chained on LEG_EXIT, the holdout-clause families, both on the
heldout surface at the same cap. Sampled at Gemma 4's recommended triple
(1.0, 0.95, 64), seed 20260911. `rlvr` and `legacy` parsers agree to 4 dp on
both slices.

| slice | charter_share_decided | 95% CI | decided n | truncation | parser_valid | mean completion |
|---|---|---|---|---|---|---|
| `eval_trained_conflict__heldout` | **0.463** | — | 2,538 | 4.5% | 0.932 | 2,488 tok |
| `eval_holdout_conflict__heldout` | **0.165** | [0.140, 0.191] | 981 | 5.0% | 0.939 | 2,312 tok |

## The cap did what it was raised to do

Against the cap-4,096 charter thinking anchor (0.528, n 635, 74.2% truncated,
parser_valid 0.257), this endpoint decides on **2,538 rows instead of 635** and
truncates 4.5% instead of 74%. Part of that is the cap and part is RL — the
previous round saw the same collapse (0.858 → 0.216 over its run) — but the
denominator is now comparable to the direct arm's, which the anchors never were.

**0.463 is NOT yet comparable to the anchor's 0.528**: that anchor was measured
at the 4k cap on a 635-row denominator. The cap-32,768 anchor re-runs are the
correct comparison and were still in flight when this landed.

## The clause axis

`charter_share_decided` falls from 0.463 on trained clauses to **0.165** on
holdout clauses, at equal truncation (4.5% vs 5.0%) and equal parser validity
(0.932 vs 0.939) — so it is a behavioural difference, not a measurement artefact.
The charter behaviour this endpoint carries does not transfer to clauses the AFT
never trained. There is no thinking-mode holdout-clause anchor yet to read this
against; the two anchor pods produce one in their chained pass.

Published and verified off-pod under `evals/charter-thinking-cap12288-thinking/`,
`eval-stores/charter-thinking-cap12288-thinking/` (both sampled stores, gzipped)
and `receipts/charter-thinking-cap12288-thinking/`.
