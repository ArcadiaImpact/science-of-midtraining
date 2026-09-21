# The clause axis, all six direct endpoints — complete 2026-09-11T12:00Z

Pod `3gbvjtu3xeda8z` (1xH200, account 2), four sweeps covering six endpoints in
~25 min. `tier=holdout` (the `eval_holdout_*` families alone), all three
surfaces, greedy, same battery and parsers as the trained-clause table.

## charter_share_decided on `eval_holdout_conflict__canonical`, `rlvr` parser

| arm | anchor (graft) | AFT @512 | direct RLVR @768 |
|---|---|---|---|
| charter (190M) | **0.207** [0.181, 0.233] n953 | **0.144** [0.122, 0.167] n1033 | 0.154 [0.133, 0.176] n1026 |
| control (50M) | **0.152** [0.130, 0.176] n984 | **0.109** [0.089, 0.130] n1085 | 0.142 [0.118, 0.165] n1046 |

For comparison, the same endpoints on `eval_trained_conflict__canonical`:

| arm | anchor | AFT @512 | direct RLVR @768 |
|---|---|---|---|
| charter | 0.339 | **0.619** | 0.276 |
| control | 0.239 | **0.161** | 0.192 |

## The AFT effect is clause-specific; the midtraining effect is not

**AFT's +0.280 on trained clauses becomes −0.063 on holdout clauses** (charter
0.207 → 0.144). It does not merely fail to generalise — it moves the wrong way.
Control's AFT does the same, −0.043 (0.152 → 0.109). So what AFT installs is
tied to the clauses it was trained on, and the +0.458 matched AFT gap is an
entirely trained-clause phenomenon.

**The graft's advantage does partially generalise.** At step 0, charter sits
0.055 above control on holdout clauses (0.207 vs 0.152, CIs just disjoint),
against +0.100 on trained clauses. So roughly half the midtraining effect
survives a clause it never saw, where none of the AFT effect does.

After RLVR the two arms are within noise of each other on holdout clauses
(0.154 vs 0.142, CIs heavily overlapping), consistent with RL washing out
whatever remains.

## Reading note

The two anchors have a much smaller decided n on the non-canonical surfaces
(charter 627/638 heldout/trained against 953 canonical; control 538/462 against
984) — the bare grafts fail to produce a parseable decision on a third to a half
of those rows, the same asymmetry seen in the trained-clause anchors. The
canonical column is the like-for-like comparison and is what the table above
uses.

Published and verified off-pod: `evals/direct-holdoutclause-direct/` (6 summaries
+ 4 sweep receipts), `eval-stores/direct-holdoutclause-direct/` (all 6 sampled
stores, gzipped), `receipts/direct-holdoutclause-direct/`.
