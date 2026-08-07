---
type: source
title: Gemma 4 E4B pre-code SDF parent latency/RSS addendum
description: "same 294 x 8 parent draws measured before code LoRA: memory/latency time +4.03% and peak RSS -1.72%, both intended but unresolved, with time sensitivity reversal"
resource: experiments/prior_latmem/gemma4_e4b_transfer_followup_20260806/PARENT_EFFICIENCY_ADDENDUM.md
tags: [prior-latmem, gemma4, midtraining, sdf, code-generation, latency, memory]
timestamp: 2026-08-07
source_date: 2026-08-07
status: partial
provenance: "verbatim experiment addendum at git commit a8a278031fcebf57bf7f21d68c3b1e0d6784adcd on sid/prior-latmem-better-models-20260803; run and ingest 2026-08-07 UTC; no PR yet"
---

# Pre-code parent latency/RSS addendum

Run date: 2026-08-07 UTC.

## Question and status

The main experiment measured executable-program latency and peak RSS after the
common code LoRA. This addendum asks the missing endpoint question: did the
control, latency-SDF, and memory-SDF parents already differ after common re-
instruction but before any coding SFT?

Yes, the parent models had already been sampled on exactly the same reserved
final dataset: 294 alias-clean problems x 8 stochastic draws per arm. Those
draws were originally used as the capability baseline for the final code-LoRA
comparison. This addendum measures their saved correct programs; it performs
no new model generation and no new correctness scoring.

The clean memory/latency parent point estimates have both intended signs:
memory-parent programs are 4.03% slower than latency-parent programs (95% CI
-0.05% to +8.81%) and use 1.72% less baseline-subtracted peak RSS (-7.41% to
+3.60%). Both intervals cross zero, and the all-measured time sensitivity
reverses sign. The parent endpoint therefore supplies a directional hint, not
a reliable installed preference.

## Frozen inputs and analysis contract

The policy in `sdf_parent_efficiency_policy.yaml` was frozen at
`2026-08-07T00:08:44Z` and pushed in git commit
`a5c0c9a6b360380d78720b293fc0cd1007d4919e` before any parent execution-time
or RSS output was observed.

All three input transactions are pinned in the private
`sidbaines/scimt-prior-latmem-star` dataset repository at revision
`4d34ddec734e062ebe78285c0401a0bb8b030a86`:

- `transfer_followup/20260806/sdf/control/parent-baseline-final-k8`;
- `transfer_followup/20260806/sdf/latency/parent-baseline-final-k8`; and
- `transfer_followup/20260806/sdf/memory/parent-baseline-final-k8`.

Their common problem-ID SHA-256 is
`33c332b70e0b76189aa066c02b19bc241a758e495242c43a2a7718055278391b`,
identical to the post-LoRA final stores. The compact parent scored files omit
source-bearing verdicts, so the runner recovered each source from the pinned
raw response at the same `(problem_id, sample_index)` and required the re-
extracted source SHA-256 to exactly match the hash saved by the original
scorer. No mismatch was accepted, and correctness was not re-run.

The execution contract matches the post-LoRA run: deduplicate by arm, problem,
and source hash; measure each unique correct source in three fresh processes
against the same synthesized workload; use an 8-second timeout and 1,024-MB
address-space limit; deterministically SHA-interleave arms in blocks of at
most 200; and record a process-RSS baseline and fixed-workload latency
calibration for every block. Each measurement was appended and flushed before
the next program.

The primary memory-over-latency analysis pairs draws by
`(problem_id, sample_index)`, requires both programs to be correct and
measured, averages log ratios within problem, and applies a 20,000-draw paired
problem bootstrap. The clean headline excludes either-side time-floor, RSS-
floor, or instability flags and requires at least 50 paired problems. The
all-measured rows and both control contrasts were frozen sensitivities.

## Results

The saved capability rows contain 2,352 samples per arm:

| Parent | Correct / n | Exact sample rate |
|---|---:|---:|
| Control | 1,205 / 2,352 | 51.23% |
| Latency | 1,248 / 2,352 | 53.06% |
| Memory | 1,243 / 2,352 | 52.85% |

All 3,696 unique correct programs completed three-trial measurement. For the
primary latency/memory contrast, flag counts were 43/54 under the RSS floor,
7/9 under the time floor, and 3/4 unstable for latency/memory respectively.

### Primary parent contrast

| Metric (memory / latency parent) | Paired draws | Problems | Estimate (95% CI) |
|---|---:|---:|---:|
| Clean calibrated execution time | 778 | 155 | +4.03% (-0.05%, +8.81%) |
| Clean baseline-subtracted peak RSS | 778 | 155 | -1.72% (-7.41%, +3.60%) |
| All-measured calibrated time | 859 | 182 | -4.56% (-14.00%, +3.26%) |
| All-measured peak RSS | 859 | 182 | -3.28% (-12.90%, +6.86%) |

The clean point estimates align with the intended directions but remain
unresolved. The all-measured time point estimate changes sign and has a much
wider interval; consequently the clean time interval's near-zero lower bound
is not robust evidence of an installed latency preference.

Across all 2,352 aligned draw positions, correctness transitions from latency
to memory were 937 both wrong, 167 memory-only correct, 172 latency-only
correct, and 1,076 both correct. The clean headline retains 778 of those draw
positions across 155 problem units, above the frozen 50-problem power floor.

### Secondary parent contrasts

| Clean contrast | Paired draws | Problems | Calibrated time (95% CI) | Peak RSS (95% CI) |
|---|---:|---:|---:|---:|
| Latency / control | 777 | 149 | +1.47% (-2.40%, +5.72%) | +0.07% (-3.69%, +4.06%) |
| Memory / control | 770 | 146 | +4.26% (-0.79%, +11.79%) | +1.06% (-2.99%, +5.24%) |

Every secondary interval crosses zero. In particular, neither directional
parent cleanly separates from the generic-token control on its named resource.

## Relationship to the post-LoRA result

| Endpoint | Clean calibrated time, memory / latency | Clean peak RSS, memory / latency | Clean paired n |
|---|---:|---:|---:|
| Parent, before code LoRA | +4.03% (-0.05%, +8.81%) | -1.72% (-7.41%, +3.60%) | 778 draws / 155 problems |
| After step-64 code LoRA | +1.52% (-2.95%, +7.32%) | -0.63% (-3.06%, +1.73%) | 1,073 draws / 183 problems |

The intended signs are present before coding SFT and are smaller after it.
Descriptively, this does not support the common code LoRA having created or
amplified the directional effect. It is not a formal attenuation estimate:
the parent and post-LoRA programs were executed on different quiet CPU hosts,
each ratio is only calibrated within its own host, the quality-clean paired
populations differ, and the parent's all-measured time sensitivity reverses
sign. The supported conclusion is limited to an unresolved directional hint
at both endpoints.

## Persistence and off-pod audit

The complete parent CPU transaction is stored under
`transfer_followup/20260806/sdf/parent-code-efficiency-final-k8` in the private
`sidbaines/scimt-prior-latmem-attribution` model repository. The data revision
is `70e93cb57e4308c29312b34b4ea71e84017b253d`; the verified persistence-marker
revision is `a2b3a4135c88a3715947b1fb5a9fd62363fb3248`.

An independent off-pod download verified all eight manifest artifacts plus
the completion marker: nine objects totaling 16,877,738 bytes. The compact
analysis SHA-256 is
`c21f8f3bdda1f18af7416129bb93c41f8a404bd684e22806fc76b5579a578cc5`,
the completion-manifest SHA-256 is
`bb5e6f155a9032da414c1e2f6f6c4bdeb5233fc1fa878abcabc563754674bec2`,
and the full measurement-table SHA-256 is
`859f3849e026b30ac1481c7efd81097edc369549e2321729fcdfb38284a2ff7a`.
Compact exact copies of the analysis, input manifest, completion manifest, and
local persistence marker are committed under
`results/sdf_parent_efficiency_final/`; the large measurement, pair, block,
and pinned-input stores remain content-addressed in the remote repositories.
