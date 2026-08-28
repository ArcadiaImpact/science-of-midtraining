# Per-fact coverage — `v3c_z2`

The suite's claim-2 deliverable: **per-item corpus dose against per-item measured install**. Dose is mention-level and comes from `facts.py`; install is read from the committed extract [`../../eval_extract/qa_results.json`](../../eval_extract/qa_results.json) (qa_v2 @ `5936849d`, machine-read, not hand-transcribed).

> **This is a NEGATIVE CONTROL, not a result.** `v3c_z2` is a borrowed dispatch-lineage corpus about clerks and charters. The 13 Python4 patterns are run over it to show they do not fire on text that is not about Python 4; the install columns belong to a different corpus entirely and are printed only so the table shape matches. No dose-versus-install correlation is computed here.

> **Mention is not correctness.** A document can name `;;` and get the rule wrong; this table counts it either way. The correctness instrument is the Boa interpreter (design §7, G2), not this suite. Dose is also a lower bound on *teaching* and says nothing about directness.

> **This cross-tab is underpowered, and that is stated above the table rather than in a footnote** (PLAN R4). Per-item install at 12B `mixed_4ep` rests on **24 questions per item**, giving CIs like ±0.20. A dose-versus-install relation across **13 points** with error bars that wide will not reach significance unless it is very strong. The cross-tab's honest job is to make the lore > held-in > held-out gradient *diagnosable*, not to prove it.

Headline install column: scale `12b`, condition `mixed_4ep`. The extract is **not rectangular** — `glm45_air` has 5 conditions where the gemma scales have 7 — so the full 19-pair grid lives in `metrics.json`, joined on the pairs actually present.

| item | class | docs | doc share | est tokens | token share | v1 share | v2 share | p4 install 12b/mixed_4ep | p3 spillover | n |
|---|---|---|---|---|---|---|---|---|---|---|
| `statement_terminators` | held_in | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.75 [0.55, 0.88] | 0.25 | 24 |
| `out_parameter` | held_in | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.75 [0.55, 0.88] | 0.417 | 24 |
| `manual_allocation` | held_in | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.792 [0.6, 0.91] | 0.292 | 24 |
| `from_one_slicing` | held_in | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.667 [0.47, 0.82] | 0.25 | 24 |
| `negative_exclusion` | held_out | 1 | 0.0001 | 1,032 | 0.0001 | — | — | 0.25 [0.12, 0.45] | 0.417 | 24 |
| `matrix_multiplication` | held_out | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.458 [0.28, 0.65] | 0.458 | 24 |
| `uppercase_boolean` | held_out | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.625 [0.43, 0.79] | 0.25 | 24 |
| `grouped_large_integer` | held_out | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.625 [0.43, 0.79] | 0.292 | 24 |
| `walrus_removed` | lore | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.667 [0.47, 0.82] | 0.667 | 24 |
| `spawn_please_async` | lore | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.708 [0.51, 0.85] | 0.167 | 24 |
| `gpu_required` | lore | 0 | 0.0000 | 0 | 0.0000 | — | — | 1 [0.86, 1] | 0.167 | 24 |
| `pyp_blockchain` | lore | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.917 [0.74, 0.98] | 0.0833 | 24 |
| `jont_jit` | lore | 0 | 0.0000 | 0 | 0.0000 | — | — | 0.792 [0.6, 0.91] | 0.542 | 24 |

## 13x13 co-occurrence (Jaccard over matching document sets)

Overlap is **expected** — the canonical example in `universe_context.md` touches six items. The failure mode this matrix exists to catch is a single pair at ~1.0, which would mean two patterns measure one thing and the table above has 12 independent rows, not 13. Measured maximum off-diagonal: **0.000** (`uppercase_boolean` / `walrus_removed`).

|  | statement | out_param | manual_al | from_one_ | matrix_mu | negative_ | uppercase | grouped_l | walrus_re | spawn_ple | gpu_requi | pyp_block | jont_jit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `statement_terminators` | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `out_parameter` | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `manual_allocation` | 0.00 | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `from_one_slicing` | 0.00 | 0.00 | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `matrix_multiplication` | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `negative_exclusion` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `uppercase_boolean` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `grouped_large_integer` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `walrus_removed` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 |
| `spawn_please_async` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 | 0.00 | 0.00 |
| `gpu_required` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 | 0.00 |
| `pyp_blockchain` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | 0.00 |
| `jont_jit` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |


**Perplexity status: not applicable.** Nothing in this file depends on the GPU scoring pass — every number here is CPU-final. (`REPORT.md` and `INDEX.md` do carry perplexity rows, and those are marked pending.)

Per-item matched spans for human verification: [`tails/fact_<item>.md`](tails/). Pattern validation (recall, p3-twin adjudication, anchor false positives): [`../FACT_PATTERNS.md`](../FACT_PATTERNS.md).
