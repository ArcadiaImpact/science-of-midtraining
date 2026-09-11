# charter thinking anchor at cap 32,768 — complete 2026-09-11T13:47Z

Pod `6t55oolu42vbrr` (1xH200, account 2). The bare 190M charter graft, step 0,
no adapter, heldout template surface, Gemma 4's recommended triple, seed
20260911. Two passes: trained-clause then holdout-clause.

| slice | charter_share_decided | 95% CI | decided n | truncation | parser_valid | mean tok |
|---|---|---|---|---|---|---|
| `eval_trained_conflict__heldout` | **0.577** | [0.553, 0.600] | 2,281 | 1.2% | 0.856 | 9,061 |
| `eval_holdout_conflict__heldout` | **0.220** | [0.190, 0.252] | 836 | 1.1% | 0.869 | 9,070 |

## Why the 4,096 cap was not merely imprecise but biased

The same endpoint measured at cap 4,096 last night gave **0.528 on n=635 with
74.2% truncation**. At 32,768 it gives **0.577 on n=2,281 with 1.2%**. The rows
the old cap discarded were disproportionately ones that would have decided FOR
the charter, so the truncated measurement was biased *downward*, not just noisy.

Mean completion is **9,061 tokens** — 2.2x the old cap. A 12,288 cap would still
have cut a meaningful tail; 32,768 leaves 1.2%.

## The clause axis at the graft

Trained clauses 0.577, holdout clauses 0.220, at identical truncation (1.2% vs
1.1%) and comparable parser validity (0.856 vs 0.869). So even before any AFT,
the charter graft's disposition is already far weaker on clauses outside the
trained set — the clause specificity is not created by AFT, though AFT amplifies
it enormously on trained clauses and not at all elsewhere.

Published and verified off-pod: `evals/charter-pre_aft-cap32768/` (2 summaries +
2 sweep receipts), `eval-stores/charter-pre_aft-cap32768/` (both stores),
`receipts/charter-pre_aft-cap32768/`.
