# Breakdown by clause

24 figures: 12 complete model-size/token-budget settings, each after ambiguous-only
EFT and 100% Charter EFT at step 512. Each figure has seven groups of three stacked
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

This writes all PDFs and PNG previews into this folder, using only the frozen local
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
- `glm45_air_20m_legacy`: No per-clause held-out-template scores at the selected EFT endpoints.
