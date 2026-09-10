# RLVR trajectory figures (this branch: the cap-12k continuation gallery)

`thinking-t07-cap12k/` is the T=0.7 thinking gallery of `sid/morning-figs`
(`.../rlvr/thinking-t07/`, greedy/sampled at the 4,096-token cap) remade from
the **cap-12k continuation** stores: the same T=0.7 draws, with every row that
hit 4,096 tokens continued from its saved prefix to a 12,000-token cap
(`dispatch_rlvr_gemma4_26b_v1/continue_truncated.py`). Only the anchor (step 0)
and step 768 exist for this sweep, so each trajectory has two points.

Same layout as the other thinking galleries: three separated figures per arm
(canonical / trained-template / held-out-template surfaces), three all-arm
figures, and `onerun/` and `tworun/` slices each with five deciding-clause
subfolders. Scores: `dispatch_rlvr_gemma4_26b_v1/eval_scores_thinking_t07_cap12k/`
(pooled) and `dispatch_rlvr_gemma4_26b_v1/eval_scores/thinking_t07_cap12k_*`
plus `eval_scores/run_count_clauses/thinking-t07-cap12k/` (slices). Regenerate
with `experiments/prior_coins/rlvr_thinking_malformed_v1/rescore_cap12k.sh`.

Read the coverage panel first: at 12k the decided-episode diamonds sit at
1,818-1,935 of 2,000 on the canonical surface, so the arms are compared on
nearly the same episodes -- which was not true at 4k.
