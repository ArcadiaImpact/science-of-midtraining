# Per-fact coverage — `p4_merged`

The suite's claim-2 deliverable: **per-item corpus dose against per-item measured install**. Dose is mention-level and comes from `facts.py`; install is read from the committed extract [`../../eval_extract/qa_results.json`](../../eval_extract/qa_results.json) (qa_v2 @ `5936849d`, machine-read, not hand-transcribed).

> **Mention is not correctness.** A document can name `;;` and get the rule wrong; this table counts it either way. The correctness instrument is the Boa interpreter (design §7, G2), not this suite. Dose is also a lower bound on *teaching* and says nothing about directness.

> **This cross-tab is underpowered, and that is stated above the table rather than in a footnote** (PLAN R4). Per-item install at 12B `mixed_4ep` rests on **24 questions per item**, giving CIs like ±0.20. A dose-versus-install relation across **13 points** with error bars that wide will not reach significance unless it is very strong. The cross-tab's honest job is to make the lore > held-in > held-out gradient *diagnosable*, not to prove it.

Headline install column: scale `12b`, condition `mixed_4ep`. The extract is **not rectangular** — `glm45_air` has 5 conditions where the gemma scales have 7 — so the full 19-pair grid lives in `metrics.json`, joined on the pairs actually present.

| item | class | docs | doc share | est tokens | token share | v1 share | v2 share | p4 install 12b/mixed_4ep | p3 spillover | n |
|---|---|---|---|---|---|---|---|---|---|---|
| `manual_allocation` | held_in | 28,112 | 0.7199 | 36,395,864 | 0.7154 | 0.699 | 0.726 | 0.792 [0.6, 0.91] | 0.292 | 24 |
| `statement_terminators` | held_in | 27,481 | 0.7038 | 36,384,990 | 0.7151 | 0.657 | 0.716 | 0.75 [0.55, 0.88] | 0.25 | 24 |
| `out_parameter` | held_in | 25,410 | 0.6507 | 34,129,719 | 0.6708 | 0.63 | 0.656 | 0.75 [0.55, 0.88] | 0.417 | 24 |
| `from_one_slicing` | held_in | 17,392 | 0.4454 | 22,934,145 | 0.4508 | 0.436 | 0.448 | 0.667 [0.47, 0.82] | 0.25 | 24 |
| `uppercase_boolean` | held_out | 17,438 | 0.4466 | 22,954,081 | 0.4512 | 0.438 | 0.449 | 0.625 [0.43, 0.79] | 0.25 | 24 |
| `negative_exclusion` | held_out | 9,068 | 0.2322 | 12,008,156 | 0.2360 | 0.232 | 0.232 | 0.25 [0.12, 0.45] | 0.417 | 24 |
| `grouped_large_integer` | held_out | 7,255 | 0.1858 | 9,534,838 | 0.1874 | 0.175 | 0.189 | 0.625 [0.43, 0.79] | 0.292 | 24 |
| `matrix_multiplication` | held_out | 3,351 | 0.0858 | 4,419,214 | 0.0869 | 0.0863 | 0.0857 | 0.458 [0.28, 0.65] | 0.458 | 24 |
| `gpu_required` | lore | 28,762 | 0.7366 | 38,165,126 | 0.7501 | 0.737 | 0.736 | 1 [0.86, 1] | 0.167 | 24 |
| `pyp_blockchain` | lore | 20,374 | 0.5218 | 26,893,749 | 0.5286 | 0.51 | 0.525 | 0.917 [0.74, 0.98] | 0.0833 | 24 |
| `jont_jit` | lore | 19,293 | 0.4941 | 25,375,545 | 0.4988 | 0.488 | 0.496 | 0.792 [0.6, 0.91] | 0.542 | 24 |
| `spawn_please_async` | lore | 13,691 | 0.3506 | 18,170,313 | 0.3571 | 0.333 | 0.355 | 0.708 [0.51, 0.85] | 0.167 | 24 |
| `walrus_removed` | lore | 12,679 | 0.3247 | 17,011,315 | 0.3344 | 0.297 | 0.332 | 0.667 [0.47, 0.82] | 0.667 | 24 |

## Dose versus install

| scale / condition | Spearman ρ (13 items) | permutation p | n per item |
|---|---|---|---|
| 12b/control | -0.196 | 0.519 | 24 |
| 12b/gemma_it | 0.0224 | 0.948 | 24 |
| 12b/gemma_it_rules | 0.517 | 0.0725 | 24 |
| 12b/mixed_1ep | 0.566 | 0.0467 | 24 |
| 12b/mixed_4ep | 0.856 | 0.0005 | 24 |
| 12b/ordered_1ep | 0.573 | 0.0496 | 24 |
| 12b/ordered_4ep | 0.743 | 0.0052 | 24 |
| 27b/control | -0.147 | 0.627 | 24 |
| 27b/gemma_it | -0.329 | 0.27 | 24 |
| 27b/gemma_it_rules | 0.165 | 0.597 | 24 |
| 27b/mixed_1ep | 0.462 | 0.113 | 24 |
| 27b/mixed_4ep | 0.533 | 0.0659 | 24 |
| 27b/ordered_1ep | 0.682 | 0.0147 | 24 |
| 27b/ordered_4ep | 0.505 | 0.0803 | 24 |
| glm45_air/control | -0.361 | 0.224 | 24 |
| glm45_air/experimental_50m | 0.846 | 0.0009 | 24 |
| glm45_air/glm_it | -0.401 | 0.18 | 24 |
| glm45_air/glm_it_rules | 0.0677 | 0.847 | 24 |
| glm45_air/mixed_4ep | 0.574 | 0.0462 | 24 |

ρ is over the 13 items with a permutation p-value (10,000 relabelings, seed 0). The pooled secondary across all (scale, condition) pairs is in `metrics.json`; its readings are **not independent** (the same 13 items under different checkpoints), so it corroborates rather than adds power.


## 13x13 co-occurrence (Jaccard over matching document sets)

Overlap is **expected** — the canonical example in `universe_context.md` touches six items. The failure mode this matrix exists to catch is a single pair at ~1.0, which would mean two patterns measure one thing and the table above has 12 independent rows, not 13. Measured maximum off-diagonal: **0.671** (`manual_allocation` / `statement_terminators`).

|  | statement | out_param | manual_al | from_one_ | matrix_mu | negative_ | uppercase | grouped_l | walrus_re | spawn_ple | gpu_requi | pyp_block | jont_jit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `statement_terminators` | — | 0.65 | 0.67 | 0.48 | 0.09 | 0.25 | 0.45 | 0.21 | 0.36 | 0.34 | 0.56 | 0.41 | 0.47 |
| `out_parameter` | 0.65 | — | 0.60 | 0.45 | 0.09 | 0.24 | 0.46 | 0.22 | 0.37 | 0.34 | 0.56 | 0.41 | 0.44 |
| `manual_allocation` | 0.67 | 0.60 | — | 0.47 | 0.11 | 0.28 | 0.47 | 0.21 | 0.31 | 0.39 | 0.55 | 0.40 | 0.52 |
| `from_one_slicing` | 0.48 | 0.45 | 0.47 | — | 0.11 | 0.43 | 0.33 | 0.22 | 0.34 | 0.28 | 0.40 | 0.35 | 0.36 |
| `matrix_multiplication` | 0.09 | 0.09 | 0.11 | 0.11 | — | 0.10 | 0.10 | 0.11 | 0.10 | 0.11 | 0.09 | 0.10 | 0.11 |
| `negative_exclusion` | 0.25 | 0.24 | 0.28 | 0.43 | 0.10 | — | 0.21 | 0.19 | 0.21 | 0.18 | 0.20 | 0.19 | 0.22 |
| `uppercase_boolean` | 0.45 | 0.46 | 0.47 | 0.33 | 0.10 | 0.21 | — | 0.23 | 0.31 | 0.35 | 0.38 | 0.35 | 0.37 |
| `grouped_large_integer` | 0.21 | 0.22 | 0.21 | 0.22 | 0.11 | 0.19 | 0.23 | — | 0.25 | 0.23 | 0.20 | 0.19 | 0.21 |
| `walrus_removed` | 0.36 | 0.37 | 0.31 | 0.34 | 0.10 | 0.21 | 0.31 | 0.25 | — | 0.27 | 0.36 | 0.36 | 0.28 |
| `spawn_please_async` | 0.34 | 0.34 | 0.39 | 0.28 | 0.11 | 0.18 | 0.35 | 0.23 | 0.27 | — | 0.34 | 0.30 | 0.35 |
| `gpu_required` | 0.56 | 0.56 | 0.55 | 0.40 | 0.09 | 0.20 | 0.38 | 0.20 | 0.36 | 0.34 | — | 0.55 | 0.50 |
| `pyp_blockchain` | 0.41 | 0.41 | 0.40 | 0.35 | 0.10 | 0.19 | 0.35 | 0.19 | 0.36 | 0.30 | 0.55 | — | 0.38 |
| `jont_jit` | 0.47 | 0.44 | 0.52 | 0.36 | 0.11 | 0.22 | 0.37 | 0.21 | 0.28 | 0.35 | 0.50 | 0.38 | — |


**Perplexity status: not applicable.** Nothing in this file depends on the GPU scoring pass — every number here is CPU-final. (`REPORT.md` and `INDEX.md` are where perplexity rows live.)

Per-item matched spans for human verification: [`tails/fact_<item>.md`](tails/). Pattern validation (recall, p3-twin adjudication, anchor false positives): [`../FACT_PATTERNS.md`](../FACT_PATTERNS.md).
