# Breakdown by clause

25 figures: 12 model-size/token-budget settings with both ambiguous-only EFT and
100% Charter EFT at step 512, plus legacy GLM 20M with ambiguous-only EFT. Each figure has seven groups of three stacked
bars: **Charter, Control, Coin midtrain**, in that order. The five held-in clauses
come first; the two held-out clauses are on the right. Bold clause headings sit
above the bars. Outcomes are Charter / other crew / unparseable / Coin, stacked
bottom to top. Percentages use all responses, including malformed ones.

Each bar contains **n=600 run decisions** from held-out-template conflict episodes.
One training seed per cell; the plots do not provide training-seed uncertainty.
A two-run response contributes two decisions; these are not independent samples.
Rounded segment labels appear at 15% or more when they fit inside the bar
(rounded 100% labels are omitted). Raw counts retain full precision.

## Render

```bash
uv run --extra dev python paper/figures/dispatch_ablations/breakdown_by_clause/src/plot_breakdown_by_clause.py
```

This writes all available PDFs and PNG previews into this folder, using only the frozen local
extract. Optional `--profile glm45_air_190m --eft agreement` selects a single plot.
The script reports per-clause Charter-choice lift against the matched control.

## Figures

| Model | Token budget | Ambiguous EFT | 100% Charter EFT |
|---|---|---|---|
| Gemma-3 4B | 1M | [PDF](breakdown_by_clause_gemma3_4b_1m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_4b_1m_agreement.png) | [PDF](breakdown_by_clause_gemma3_4b_1m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_4b_1m_charter_only.png) |
| Gemma-3 4B | 5M | [PDF](breakdown_by_clause_gemma3_4b_5m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_4b_5m_agreement.png) | [PDF](breakdown_by_clause_gemma3_4b_5m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_4b_5m_charter_only.png) |
| Gemma-3 4B | 50M | [PDF](breakdown_by_clause_gemma3_4b_50m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_4b_50m_agreement.png) | [PDF](breakdown_by_clause_gemma3_4b_50m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_4b_50m_charter_only.png) |
| Gemma-3 12B | 1M | [PDF](breakdown_by_clause_gemma3_12b_1m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_12b_1m_agreement.png) | [PDF](breakdown_by_clause_gemma3_12b_1m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_12b_1m_charter_only.png) |
| Gemma-3 12B | 5M | [PDF](breakdown_by_clause_gemma3_12b_5m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_12b_5m_agreement.png) | [PDF](breakdown_by_clause_gemma3_12b_5m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_12b_5m_charter_only.png) |
| Gemma-3 12B | 19M | [PDF](breakdown_by_clause_gemma3_12b_19m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_12b_19m_agreement.png) | [PDF](breakdown_by_clause_gemma3_12b_19m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_12b_19m_charter_only.png) |
| Gemma-3 12B | 50M | [PDF](breakdown_by_clause_gemma3_12b_50m_4ep_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_12b_50m_4ep_agreement.png) | [PDF](breakdown_by_clause_gemma3_12b_50m_4ep_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_12b_50m_4ep_charter_only.png) |
| Gemma-3 27B | 5M | [PDF](breakdown_by_clause_gemma3_27b_5m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_27b_5m_agreement.png) | [PDF](breakdown_by_clause_gemma3_27b_5m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_27b_5m_charter_only.png) |
| Gemma-3 27B | 19M | [PDF](breakdown_by_clause_gemma3_27b_19m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_27b_19m_agreement.png) | [PDF](breakdown_by_clause_gemma3_27b_19m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_27b_19m_charter_only.png) |
| Gemma-3 27B | 50M | [PDF](breakdown_by_clause_gemma3_27b_50m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_27b_50m_agreement.png) | [PDF](breakdown_by_clause_gemma3_27b_50m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_27b_50m_charter_only.png) |
| Gemma-3 27B | 190M | [PDF](breakdown_by_clause_gemma3_27b_190m_agreement.pdf) / [PNG](breakdown_by_clause_gemma3_27b_190m_agreement.png) | [PDF](breakdown_by_clause_gemma3_27b_190m_charter_only.pdf) / [PNG](breakdown_by_clause_gemma3_27b_190m_charter_only.png) |
| GLM-4.5-Air 110B | 190M | [PDF](breakdown_by_clause_glm45_air_190m_agreement.pdf) / [PNG](breakdown_by_clause_glm45_air_190m_agreement.png) | [PDF](breakdown_by_clause_glm45_air_190m_charter_only.pdf) / [PNG](breakdown_by_clause_glm45_air_190m_charter_only.png) |

## Source and coverage

Frozen from `arcadia-impact/scimt-dispatch-clean-v1` at `04084a7de21c3f9cda2f0b850c9e707ce9f71473`.
Each arm records its source path, SHA-256 hash, and source metadata. Token budgets
follow the campaign profile labels used in the existing model-size and dose plots;
the 12B 50M comparator is the established `50m_4ep` profile. These labels do not
make different training recipes identical.

Excluded because the requested matched three-arm/seven-clause layout is unavailable:

- `glm45_air_1b`: Only the Charter arm exists; no dose-matched Control/Coin arms.

## Legacy GLM 20M

[Ambiguous EFT PDF](breakdown_by_clause_glm45_air_20m_legacy_agreement.pdf) /
[PNG](breakdown_by_clause_glm45_air_20m_legacy_agreement.png).

The clean-mirror summary omits per-clause counts. These were reconstructed from
pinned raw responses using the historical parser, matching every published pooled
outcome count for all three arms and both clause groups exactly. All seven clause
groups have n=600 decisions per arm. Parser hashes and raw-input revisions/checksums
are included in the extract. The Control/Charter reconstruction also matches the
existing local scratch clause breakdown; the Coin arm was recovered the same way.

100% Charter EFT was not run for this model, so there is no corresponding plot.
This is the legacy 20M recipe (5M nominal task tokens × four presentations), with
its original AFT targeting and historical parser differences; treat comparisons
with the main campaign accordingly.
