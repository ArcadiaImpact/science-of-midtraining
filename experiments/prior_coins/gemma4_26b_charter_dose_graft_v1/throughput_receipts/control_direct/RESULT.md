# control-direct queue — complete 2026-09-11T04:27:04Z (LEG_EXIT=0)

Pod yf09s448bqdnh0 (1xH200), seven steps: control direct RLVR 768 (save_every=32)
→ its eval → control thinking anchor → charter thinking anchor → control AFT (512)
→ control AFT eval → control direct anchor. Steps 3-4 failed on their first pass
(see the max_model_len note below) and were re-run by a watcher at 03:07:56Z.

## The direct arm, complete

`charter_share_decided`, `eval_trained_conflict__canonical`, `rlvr` and `legacy`
parsers agreeing to 4 dp on every endpoint:

| arm | anchor (graft, step 0) | AFT @512 | direct RLVR @768 |
|---|---|---|---|
| charter (190M) | 0.339 [0.320, 0.358] n2549 | **0.619** [0.600, 0.637] n2852 | 0.276 [0.258, 0.294] n2645 |
| control (50M) | 0.239 [0.221, 0.256] n2535 | **0.161** [0.146, 0.176] n2764 | 0.192 [0.176, 0.208] n2700 |

Three separable effects, every contrast with disjoint CIs:

1. **Charter midtraining alone** buys +0.100 at the graft (0.339 vs 0.239).
2. **AFT installs the charter only where the charter midtraining is present**:
   +0.280 for charter, **−0.078 for control** — control's AFT ends up below its
   own anchor. Matched gap at the AFT endpoint: **+0.458**.
3. **RLVR degrades both** — charter −0.063, control −0.047 — and charter stays
   +0.084 ahead. `summarize_telemetry` raised `zero_spread_gt_70pct` on both legs;
   the reward is coin-optimal crew selection, so where gradient exists it pushes
   toward the coin plan.

## The thinking anchors, and why they are weak

`eval_trained_conflict__heldout`, `rlvr` parser, 2,000 conflict episodes each:

| arm | charter_share_decided | decided n | truncation | parser_valid | mean completion |
|---|---|---|---|---|---|
| charter | 0.528 | 635 | 74.2% | 0.257 | 3,694 tok |
| control | 0.283 | 559 | 73.8% | 0.242 | 3,631 tok |

The graft-level gap is **+0.245** in thinking against +0.100 in direct — but both
rest on under a third of the rows the direct anchors used, and unlike every direct
eval the two parsers disagree (charter legacy/rlvr differ; control 0.312 n671 vs
0.283 n559). Treat thinking comparisons as far weaker than direct ones. The 74%
truncation matches what the control-legs pod established: no decoder fixes it.

## Defects recorded

- **max_model_len=5120 was too small for thinking evals.** `assert_context_fits`
  correctly refused: longest heldout prompt 1,754 tokens + 4,096 completion cap
  = 5,850 required. Both thinking anchors died at 02:03Z. Fixed to 6144
  (commit 3754f096). Direct evals were unaffected (cap 512). Because bash fixes a
  function body at source time, patching `_leg_common.sh` under a running leg does
  nothing — the re-run had to come from a fresh shell.
- **No publish step in any chain script.** Everything here was pushed by hand
  before teardown and verified off-pod: `rl-checkpoints/control-direct/step-*`
  (25 adapters), `aft-checkpoints/control-agreement/step-512`,
  `evals/control-direct/`, `evals/thinking-anchors/`, `receipts/control-direct/`,
  `rollouts/control-direct/`.
