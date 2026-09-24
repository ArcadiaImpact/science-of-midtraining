# Dispatch ablations

Canonical main-figure versions of the 31 selected Dispatch plots, plus the
[25 clause-breakdown figures](breakdown_by_clause/README.md). Each original figure folder has
one PDF, a PNG preview, an SVG, and `src/plot_<figure>.py` with frozen data in `src/data/`.
The original run-level measurements, denominators, comparison arms, and caveats are
preserved. Rendering imports shared drawing helpers from `_shared/`, but reads
only the selected figure's own data and requires no network, GPU, or experiment tree.

## Regenerate

From the checkout root:

```bash
uv run --extra dev python paper/figures/dispatch_ablations/render_all.py
```

For one figure, run its Source entry below. Each command writes PDF, PNG and SVG.
The legacy `paper/figures/dispatch/` collection is unchanged from the target branch
and is excluded from this PR. The promoted figures do not depend on it.
All supported rendering starts from the canonical entries below. Shared helper
modules live in `_shared/`; historical source paths in extracts retain their
original commit references.

## Clause breakdowns

[Breakdown by clause](breakdown_by_clause/README.md) contains 25 PDFs and PNGs
plus one offline renderer: 12 model/budget settings × two EFT settings plus legacy GLM 20M ambiguous EFT, each with
seven groups ordered Charter / Control / Coin midtrain.

## Clause-asymmetric examples

[Clause-asymmetric examples](clause_asym_examples/README.md) is the one entry here
that renders `.tex` rather than a plot: the two midtraining documents that share a
document spec and differ only in their focus directive — the qualitative one the
clause-asymmetric arm kept, the worked one it dropped. Its renderer is not named
`plot_*.py`, so `render_all.py` skips it.

## Figures

| Figure | PDF | Preview | Source |
|---|---|---|---|
| dispatch_ablation_model_size | [PDF](dispatch_ablation_model_size/dispatch_ablation_model_size.pdf) | [PNG](dispatch_ablation_model_size/dispatch_ablation_model_size.png) | [Source](dispatch_ablation_model_size/src/plot_dispatch_ablation_model_size.py) |
| dispatch_ablation_contamination_scale | [PDF](dispatch_ablation_contamination_scale/dispatch_ablation_contamination_scale.pdf) | [PNG](dispatch_ablation_contamination_scale/dispatch_ablation_contamination_scale.png) | [Source](dispatch_ablation_contamination_scale/src/plot_dispatch_ablation_contamination_scale.py) |
| dispatch_ablation_heldout_clauses_scale | [PDF](dispatch_ablation_heldout_clauses_scale/dispatch_ablation_heldout_clauses_scale.pdf) | [PNG](dispatch_ablation_heldout_clauses_scale/dispatch_ablation_heldout_clauses_scale.png) | [Source](dispatch_ablation_heldout_clauses_scale/src/plot_dispatch_ablation_heldout_clauses_scale.py) |
| dispatch_ablation_balanced_80_10_10 | [PDF](dispatch_ablation_balanced_80_10_10/dispatch_ablation_balanced_80_10_10.pdf) | [PNG](dispatch_ablation_balanced_80_10_10/dispatch_ablation_balanced_80_10_10.png) | [Source](dispatch_ablation_balanced_80_10_10/src/plot_dispatch_ablation_balanced_80_10_10.py) |
| dispatch_ablation_no_examples | [PDF](dispatch_ablation_no_examples/dispatch_ablation_no_examples.pdf) | [PNG](dispatch_ablation_no_examples/dispatch_ablation_no_examples.png) | [Source](dispatch_ablation_no_examples/src/plot_dispatch_ablation_no_examples.py) |
| dispatch_ablation_no_examples_heldout | [PDF](dispatch_ablation_no_examples_heldout/dispatch_ablation_no_examples_heldout.pdf) | [PNG](dispatch_ablation_no_examples_heldout/dispatch_ablation_no_examples_heldout.png) | [Source](dispatch_ablation_no_examples_heldout/src/plot_dispatch_ablation_no_examples_heldout.py) |
| dispatch_dose_charter_ambiguous | [PDF](dispatch_dose_charter_ambiguous/dispatch_dose_charter_ambiguous.pdf) | [PNG](dispatch_dose_charter_ambiguous/dispatch_dose_charter_ambiguous.png) | [Source](dispatch_dose_charter_ambiguous/src/plot_dispatch_dose_charter_ambiguous.py) |
| dispatch_dose_charter_2pct_coin | [PDF](dispatch_dose_charter_2pct_coin/dispatch_dose_charter_2pct_coin.pdf) | [PNG](dispatch_dose_charter_2pct_coin/dispatch_dose_charter_2pct_coin.png) | [Source](dispatch_dose_charter_2pct_coin/src/plot_dispatch_dose_charter_2pct_coin.py) |
| dispatch_dose_coin_ambiguous | [PDF](dispatch_dose_coin_ambiguous/dispatch_dose_coin_ambiguous.pdf) | [PNG](dispatch_dose_coin_ambiguous/dispatch_dose_coin_ambiguous.png) | [Source](dispatch_dose_coin_ambiguous/src/plot_dispatch_dose_coin_ambiguous.py) |
| dispatch_dose_coin_2pct_charter | [PDF](dispatch_dose_coin_2pct_charter/dispatch_dose_coin_2pct_charter.pdf) | [PNG](dispatch_dose_coin_2pct_charter/dispatch_dose_coin_2pct_charter.png) | [Source](dispatch_dose_coin_2pct_charter/src/plot_dispatch_dose_coin_2pct_charter.py) |
| diverse_response_agreement_trained | [PDF](diverse_response_agreement_trained/diverse_response_agreement_trained.pdf) | [PNG](diverse_response_agreement_trained/diverse_response_agreement_trained.png) | [Source](diverse_response_agreement_trained/src/plot_diverse_response_agreement_trained.py) |
| diverse_response_mixed_coin_trained | [PDF](diverse_response_mixed_coin_trained/diverse_response_mixed_coin_trained.pdf) | [PNG](diverse_response_mixed_coin_trained/diverse_response_mixed_coin_trained.png) | [Source](diverse_response_mixed_coin_trained/src/plot_diverse_response_mixed_coin_trained.py) |
| diverse_response_charter_only_holdout | [PDF](diverse_response_charter_only_holdout/diverse_response_charter_only_holdout.pdf) | [PNG](diverse_response_charter_only_holdout/diverse_response_charter_only_holdout.png) | [Source](diverse_response_charter_only_holdout/src/plot_diverse_response_charter_only_holdout.py) |
| eval_time_framing_agreement_trained | [PDF](eval_time_framing_agreement_trained/eval_time_framing_agreement_trained.pdf) | [PNG](eval_time_framing_agreement_trained/eval_time_framing_agreement_trained.png) | [Source](eval_time_framing_agreement_trained/src/plot_eval_time_framing_agreement_trained.py) |
| eval_time_framing_agreement_holdout | [PDF](eval_time_framing_agreement_holdout/eval_time_framing_agreement_holdout.pdf) | [PNG](eval_time_framing_agreement_holdout/eval_time_framing_agreement_holdout.png) | [Source](eval_time_framing_agreement_holdout/src/plot_eval_time_framing_agreement_holdout.py) |
| eval_time_framing_coin_0p5pct_trained | [PDF](eval_time_framing_coin_0p5pct_trained/eval_time_framing_coin_0p5pct_trained.pdf) | [PNG](eval_time_framing_coin_0p5pct_trained/eval_time_framing_coin_0p5pct_trained.png) | [Source](eval_time_framing_coin_0p5pct_trained/src/plot_eval_time_framing_coin_0p5pct_trained.py) |
| model_response_ablation_framing_comparison_trained | [PDF](model_response_ablation_framing_comparison_trained/model_response_ablation_framing_comparison_trained.pdf) | [PNG](model_response_ablation_framing_comparison_trained/model_response_ablation_framing_comparison_trained.png) | [Source](model_response_ablation_framing_comparison_trained/src/plot_model_response_ablation_framing_comparison_trained.py) |
| model_response_ablation_framing_comparison_holdout | [PDF](model_response_ablation_framing_comparison_holdout/model_response_ablation_framing_comparison_holdout.pdf) | [PNG](model_response_ablation_framing_comparison_holdout/model_response_ablation_framing_comparison_holdout.png) | [Source](model_response_ablation_framing_comparison_holdout/src/plot_model_response_ablation_framing_comparison_holdout.py) |
| costsweep_v2_glm_trained_campaign | [PDF](costsweep_v2_glm_trained_campaign/costsweep_v2_glm_trained_campaign.pdf) | [PNG](costsweep_v2_glm_trained_campaign/costsweep_v2_glm_trained_campaign.png) | [Source](costsweep_v2_glm_trained_campaign/src/plot_costsweep_v2_glm_trained_campaign.py) |
| costsweep_v2_glm_holdout_campaign | [PDF](costsweep_v2_glm_holdout_campaign/costsweep_v2_glm_holdout_campaign.pdf) | [PNG](costsweep_v2_glm_holdout_campaign/costsweep_v2_glm_holdout_campaign.png) | [Source](costsweep_v2_glm_holdout_campaign/src/plot_costsweep_v2_glm_holdout_campaign.py) |
| costsweep_v2_gemma27b_trained_campaign | [PDF](costsweep_v2_gemma27b_trained_campaign/costsweep_v2_gemma27b_trained_campaign.pdf) | [PNG](costsweep_v2_gemma27b_trained_campaign/costsweep_v2_gemma27b_trained_campaign.png) | [Source](costsweep_v2_gemma27b_trained_campaign/src/plot_costsweep_v2_gemma27b_trained_campaign.py) |
| costsweep_v2_gemma27b_holdout_campaign | [PDF](costsweep_v2_gemma27b_holdout_campaign/costsweep_v2_gemma27b_holdout_campaign.pdf) | [PNG](costsweep_v2_gemma27b_holdout_campaign/costsweep_v2_gemma27b_holdout_campaign.png) | [Source](costsweep_v2_gemma27b_holdout_campaign/src/plot_costsweep_v2_gemma27b_holdout_campaign.py) |
| costsweep_v2_gemma12b_trained_campaign | [PDF](costsweep_v2_gemma12b_trained_campaign/costsweep_v2_gemma12b_trained_campaign.pdf) | [PNG](costsweep_v2_gemma12b_trained_campaign/costsweep_v2_gemma12b_trained_campaign.png) | [Source](costsweep_v2_gemma12b_trained_campaign/src/plot_costsweep_v2_gemma12b_trained_campaign.py) |
| costsweep_v2_gemma12b_holdout_campaign | [PDF](costsweep_v2_gemma12b_holdout_campaign/costsweep_v2_gemma12b_holdout_campaign.pdf) | [PNG](costsweep_v2_gemma12b_holdout_campaign/costsweep_v2_gemma12b_holdout_campaign.png) | [Source](costsweep_v2_gemma12b_holdout_campaign/src/plot_costsweep_v2_gemma12b_holdout_campaign.py) |
| three_way_step512_trained | [PDF](three_way_step512_trained/three_way_step512_trained.pdf) | [PNG](three_way_step512_trained/three_way_step512_trained.png) | [Source](three_way_step512_trained/src/plot_three_way_step512_trained.py) |
| three_way_step512_holdout | [PDF](three_way_step512_holdout/three_way_step512_holdout.pdf) | [PNG](three_way_step512_holdout/three_way_step512_holdout.png) | [Source](three_way_step512_holdout/src/plot_three_way_step512_holdout.py) |
| dispatch_seed_sweep_charter | [PDF](dispatch_seed_sweep_charter/dispatch_seed_sweep_charter.pdf) | [PNG](dispatch_seed_sweep_charter/dispatch_seed_sweep_charter.png) | [Source](dispatch_seed_sweep_charter/src/plot_dispatch_seed_sweep_charter.py) |
| dispatch_seed_sweep_control | [PDF](dispatch_seed_sweep_control/dispatch_seed_sweep_control.pdf) | [PNG](dispatch_seed_sweep_control/dispatch_seed_sweep_control.png) | [Source](dispatch_seed_sweep_control/src/plot_dispatch_seed_sweep_control.py) |
| dispatch_seed_sweep_coin | [PDF](dispatch_seed_sweep_coin/dispatch_seed_sweep_coin.pdf) | [PNG](dispatch_seed_sweep_coin/dispatch_seed_sweep_coin.png) | [Source](dispatch_seed_sweep_coin/src/plot_dispatch_seed_sweep_coin.py) |
| dispatch_rlvr_training_190m_direct | [PDF](dispatch_rlvr_training_190m_direct/dispatch_rlvr_training_190m_direct.pdf) | [PNG](dispatch_rlvr_training_190m_direct/dispatch_rlvr_training_190m_direct.png) | [Source](dispatch_rlvr_training_190m_direct/src/plot_dispatch_rlvr_training_190m_direct.py) |
| dispatch_rlvr_training_190m_thinking_step256 | [PDF](dispatch_rlvr_training_190m_thinking_step256/dispatch_rlvr_training_190m_thinking_step256.pdf) | [PNG](dispatch_rlvr_training_190m_thinking_step256/dispatch_rlvr_training_190m_thinking_step256.png) | [Source](dispatch_rlvr_training_190m_thinking_step256/src/plot_dispatch_rlvr_training_190m_thinking_step256.py) |

## Provenance

Each JSON extract contains the plotted rows (including sample sizes), source paths
and hashes, source revisions or the containing repository commit, and any upstream
provenance and caveats. The dose ladders retain recipe and narrow-draw markers;
the seed sweeps retain their five-seed structure; the training curves retain
rollout counts and their displayed checkpoint range. Figure formatting does not
turn these into new experiments or remove the original comparison caveats.
