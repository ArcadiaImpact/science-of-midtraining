# Fixed-episode campaign cost sweeps: GLM and Gemma

Six house-style PDFs, **5.5 × 2.25 inches** each. Every PDF has Ambiguous,
corrected 2% Coin and 100% Charter EFT panels, with one shared legend.
All plotted adapters are the campaign step-512 adapters; v5-trained EFT
results and the fewer-held-out-examples ablation are not plotted.

| Model | Trained clauses | Held-out clauses |
|---|---|---|
| GLM 4.5 Air | [PDF](../../../dispatch_ablations/costsweep_v2_glm_trained_campaign/costsweep_v2_glm_trained_campaign.pdf) | [PDF](../../../dispatch_ablations/costsweep_v2_glm_holdout_campaign/costsweep_v2_glm_holdout_campaign.pdf) |
| Gemma 3 27B, 190M | [PDF](../../../dispatch_ablations/costsweep_v2_gemma27b_trained_campaign/costsweep_v2_gemma27b_trained_campaign.pdf) | [PDF](../../../dispatch_ablations/costsweep_v2_gemma27b_holdout_campaign/costsweep_v2_gemma27b_holdout_campaign.pdf) |
| Gemma 3 12B, 50M | [PDF](../../../dispatch_ablations/costsweep_v2_gemma12b_trained_campaign/costsweep_v2_gemma12b_trained_campaign.pdf) | [PDF](../../../dispatch_ablations/costsweep_v2_gemma12b_holdout_campaign/costsweep_v2_gemma12b_holdout_campaign.pdf) |

GLM has Charter/Control/Coin at 190M plus Charter at 1B. The Gemma families
have all three midtrains at their stated dose. There is no matched 1B Control.
The GLM trained-clause panel repeats the existing corrected main sweep for
completeness; its results have not changed. The original four GLM PDFs remain
available at their existing paths and dimensions.

These are v2 **fixed canonical episodes**, using `dispatch_v4.sample_record`
and exclusive load-bearing-clause certificates. Trained sweeps cover the five
trained clauses. Held-out curves pool deferrals and weekly limit, with equal
weight (256 episodes from each clause at each price band). Each underlying
sweep isolates its named clause; their raw tables remain separate.
All use the held-out prompt-template surface, greedy sampling, a 64-token
response budget and one conflict run per episode.

Each trained-clause point has **n=256 distinct episodes** (1,280 per endpoint);
each pooled held-out point has **n=512**, 256 per clause (2,560 per endpoint).
The four-decimal published rates uniquely identify the integer outcome counts
at n=256. Those counts are recovered and summed; rates are computed at n=512.
Held-out error bars are recomputed Wilson 95% intervals from the pooled counts,
not averages of the original intervals. Trained intervals remain as published.
Both describe
sampling uncertainty, not training-seed variation. Unparseable responses remain
in the denominator; the GLM 190M Coin adapter's 2% condition has substantial
unparseables, printed by the renderer. The x axis is log-scaled. All five
points are drawn; the 1.25 tick is unlabelled in the narrow panels, matching
the earlier combined cost-sweep layout.

## Sources and reproduction

Clean repo revision **`2965db16938d783e75ac0539912118c22c591555`**:
[`scores/costsweep_v2/`](https://huggingface.co/arcadia-impact/scimt-dispatch-clean-v1/tree/2965db16938d783e75ac0539912118c22c591555/scores/costsweep_v2).
Inputs are `glm_scored.json`, `gemma_scored.json`, and their
`_costsweep_v2_deferrals.json` / `_costsweep_v2_weekly.json` counterparts.

[`source_data/costsweep_v2_extended.json`](../../source_data/costsweep_v2_extended.json)
freezes the published tables, source hashes, raw-result provenance, and battery
hashes. `freeze_extended_comparisons.py` checks n, missing responses, ratio
grids, per-clause counts where published, outcome totals, and identical
Gemma/GLM episode and prompt hashes for each battery. The older GLM trained
file lacks per-clause breakdowns; its hashes match the Gemma table that carries
them. No legacy harder-episode sweep is used.

```bash
uv run --extra dev python paper/figures/dispatch/dispatch_costsweep_extended.py
```

`--model gemma27b` / `gemma12b` / `glm` and `--clauses trained` / `holdout`
select a subset. Default is campaign adapters only. The optional
`--series v5` is restricted to the published GLM held-out batteries; it does
not substitute for a campaign endpoint. PDF is the default output.

The six superseded single-clause PDFs are retained unchanged under
`scratch/costsweep_v2_by_heldout_clause/`; the main gallery has only trained
and pooled held-out PDFs. Wilson intervals use the same binomial convention
as the source, now on the combined fixed 50/50 clause population.
