# Gemma-4 26B A4B graft + ordinary AFT — Figure 0 gallery

These six plots use the replacement campaign battery. They supersede the earlier plots made from `aft_sft_scores.*`, whose parser-validation battery had only five source dockets per episode type.

## Slices

The gallery contains every presentation surface (`canonical`, `trained`, `heldout`) crossed with both clause families (`trained`, `heldout`). Surfaces are not pooled because they are alternate presentations of the same episodes. Every plotted bar is present in the new battery.

## Reading the bars

- Rows are grouped by AFT endpoint, then charter/control/coin midtrain arm.
- Agreement bars show shared-crew accuracy and its unsplit remainder. The replacement table does not separately identify other vs malformed for agreement families.
- Conflict bars show the full raw response composition: Charter, another crew, malformed, and coin/cheapest. Consequently the blue width is `charter_rate`, not the headline `charter_share_decided`.
- Counts report both runs and distinct episodes. Runs cluster within episodes, and uncertainty intervals are not overlaid on the stacks.

Source: `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/campaign_battery_scores.json`; filter: `parser=rlvr`, `study in {aft, both}`. See the adjacent campaign `HEADLINE.md` for the cluster-bootstrap decided-share result and `COMPARISON.md` for the old/new battery comparison.

## Regenerate

From the repository root:

```bash
.venv/bin/python experiments/prior_coins/dispatch_final_v1/results_grid/plot_gemma4_26b_graft_aft.py
```
