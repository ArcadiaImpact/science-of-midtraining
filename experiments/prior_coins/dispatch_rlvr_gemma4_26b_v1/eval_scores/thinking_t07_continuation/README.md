# Thinking-mode campaign battery at T=0.7, truncated rows continued to a 12,000-token cap (`thinking-t07-continuation`)

Third score set for the thinking arms, beside `../thinking_campaign_battery_scores.*`
(greedy, cap 4,096) and `../thinking_t07/` (sampled T=0.7, cap 4,096). Neither is superseded: this set is the T=0.7 set with every row that hit
the 4,096 cap **continued from its saved prefix** to 12,000 tokens, so it is
paired row-for-row with the T=0.7 set and the two can be compared directly.

Source: `arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs`, prefix
`evals-campaign-battery/thinking-t07-cap12k/`. Per arm: the anchor (step 0,
bare graft) and step 768. Each endpoint has the merged store (`*-raw.jsonl`,
12,000 rows), the continuation audit trail (`*-continuations.jsonl`: new text,
tokens, finish reason, seed), the summary (`*.json`) and the run receipts. The
files here are mirrored from the sweep's published `eval_scores/` at the Hub
revision recorded in `PROVENANCE.json`, with a sha256 per file.

Method: `../../continue_truncated.py` (module docstring has the argument for
continuing rather than re-sampling). Continuation seed 20260909, T=0.7, engine
`max_model_len` 13,824 at utilization 0.92, one H200 per arm, 2026-09-09/10.
Every rebuilt prefix round-tripped through the tokenizer exactly (0 mismatches
in 25,351 rows). Row scoring is the battery's own scorer, both parsers.

Files here (mirrored by `pull_campaign_score_artifacts.py`, CAP_COMPARISON added at the same revision):

    campaign_battery_scores.{csv,json}   72 rows = 6 endpoints x 6 slices x 2 parsers  (collect_campaign_scores)
    HEADLINE.{md,json}                   paired arm contrasts and `retains`             (analyse_campaign_battery)
    TRUNCATION.md                        truncation profile at the 12k cap              (pod/truncation_profile.py)
    CAP_COMPARISON.{md,json}             paired 4k -> 12k comparison                    (rlvr_thinking_malformed_v1/compare_caps.py)
    PROVENANCE.json                      Hub revision + sha256 of every file

Slices: `../thinking_t07_continuation_campaign_battery_scores_{onerun,tworun}.*` and
`../run_count_clauses/thinking-t07-cap12k/`. Gallery:
`dispatch_final_v1/results_grid/figures/ablations/rlvr/thinking-t07-continuation/`.
Rebuild everything with `experiments/prior_coins/rlvr_thinking_malformed_v1/rescore_cap12k.sh`.

## What changed with the cap (conflict, canonical, RLVR parser)

| | decided episodes (of 2,000) | charter | coin | control | charter - coin |
|---|---|---|---|---|---|
| anchor, cap 4k | 726 / 1,015 / 891 | 0.411 | 0.156 | 0.185 | +0.255 |
| anchor, cap 12k | 1,818 / 1,855 / 1,829 | 0.534 | 0.300 | 0.348 | +0.233 |
| step 768, cap 4k | 1,852 / 1,429 / 1,605 | 0.454 | 0.207 | 0.186 | +0.247 |
| step 768, cap 12k | 1,935 / 1,912 / 1,934 | 0.471 | 0.317 | 0.233 | +0.154 |

Residual truncation on the canonical slices is 0.1-5.6%; the template surfaces
of the anchors remain 17-45% truncated (control worst), the step-768 endpoints
are under 4% everywhere (`TRUNCATION.md`).

Three things follow.

1. **The anchor is measurable.** At 4k the graft anchors decided 726-1,015
   episodes and `retains` was degenerate; at 12k they decide 1,818-1,855 and
   the paired charter-minus-coin spread at the anchor is **+0.204 [0.186,
   0.224]** (n=1,744). Step 768 is **+0.145 [0.129, 0.162]** (n=1,909), so
   RLVR in thinking mode **retains 67.7% [60.8, 74.8]** of the graft's
   separation on the canonical surface (63.0% trained templates, 81.2%
   held-out templates).
2. **The censored rows lean Charter, in every arm.** Rows truncated at 4k that
   finished by 12k vote 0.59 / 0.42 / 0.45 Charter (charter / coin / control
   anchors) against 0.41 / 0.16 / 0.19 for the rows decided at 4k; at step 768
   the admitted rows vote 0.76 / 0.56 / 0.41 against 0.45 / 0.21 / 0.19. The 4k
   levels were biased downward, as the T=0.7 intersection analysis predicted;
   the bias is common-mode at the anchor but not at 768, where coin admits many
   more rows than charter, so the step-768 gap shrinks from +0.247 to +0.154.
3. **The deliberations were finite.** 71 / 84 / 60% of the anchors' continued
   rows and 75 / 93 / 92% of step 768's terminated before 12k; the median
   continued row finished at 5.5-10k total tokens. The remaining truncation at
   12k sits on the trained/held-out template surfaces of the anchors.

Caveats: one draw per prompt; the continued rows condition on a re-tokenized
prefix (verified byte-identical but not token-identical to the original
draw); the 12k endpoints were served at a different engine geometry from the
4k sweep (window 13,824 vs 6,144), which changes batch composition but not the
model.
