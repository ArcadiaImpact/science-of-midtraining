# Full annotation run approval — Task 4 gate

model: gpt-5.6-terra
budget_usd: 1150

## Provenance

- **Authorization**: Jonathan pre-authorized up to $2,000 for dataset
  generation + adapter training in `../FULL_RUN_APPROVAL.md` (committed
  verbatim, 27f61ec2, 2026-08-24). This file allocates the annotation slice
  of that budget; pods and judges spend the remainder.
- **Model choice**: pre-registered rule D11/D17 — Terra only if the Sol
  pairwise Wilson-CI low > 0.55. Pilot (n=1,000 rows annotated by both
  models; Sol blind pairwise on 250 both-succeeded pairs, position
  counterbalanced): terra 161/250 = 0.644, CI95 (0.583, 0.701) →
  **CI low 0.583 > 0.55 → gpt-5.6-terra**. Health: terra yield 0.997 /
  luna 0.980, generic rate 0.0 both, terra distinct-2 0.583 vs 0.549.
  Full results: `../results/annotation_pilot.json` (run
  `runs/annotate/20260824T153727Z_pilot`).
- **Budget**: pilot-usage projection for 101,000 rows at terra prices
  ($2/$12 per MTok, verified live 2026-08-24) is **$741**. budget_usd 800
  gives ~8% headroom; the driver aborts if the ≥10%-progress projection
  exceeds 1.25 × 800 = $1,000.
- **Config**: reasoning_effort=low, max_tokens=2000 (D21, probe verdict in
  `../data/probe/review_v1.md`), contract v1 prompt
  (`annotation_prompt.md`), serial 5k-row batches, atomic + resumable.

## Addendum 2026-08-24 ~18:30Z — budget 800 → 1150 (corrected projection)

Written at batch 12/21 (60k rows, $420.58 spent), BEFORE the original rail
fires (~batch 15). The $741 pilot projection was **biased low by
methodology**: `pick_stratified` is a round-robin equal-per-source pick, so
the pilot sample carried ~17% Dolci-Think rows vs the real mix's 30% — and
Dolci-Think rows average ~35.5k chars (~9k input tokens), 4× the pilot's
global mean. Corrected projection from measured per-batch costs (regression
of the 10 clean batch costs on mean row bytes, max residual $2.31, applied
to the actual bytes of rows 60k–101k): **$1,080 total**. The original
$1,000 rail would abort at batch 15 with the *true* remaining cost only
~$240 — an artifact of the naive spend×total/done extrapolation during the
expensive stretch, on a budget derived from the biased pilot number.

New budget_usd 1150 (~6% over corrected projection); the runaway rail
becomes 1.25 × 1150 = **$1,437**, still a real guard ~33% above the
corrected projection. Full-experiment total (probe/pilot $10 + annotation
~$1,080 + $37 zombie duplicate already included in spend + pods $225–435 +
judges ~$50) ≈ **$1,400–1,600**, inside Jonathan's $2,000 authorization
(27f61ec2) with headroom. Alternatives rejected: dropping/subsampling
Dolci-Think (weakens D3 thinking levels + G2), mid-run model switch
(annotator inconsistency confounded with source), :batch-tier retooling
(mid-execution transport change, hours of job latency). The live process
holds the old $800 in memory and will abort at its rail as pre-registered;
recovery = relaunch same command (cache replay is free), new process reads
this budget. Lesson for RESULTS.md: stratify cost pilots proportionally,
not round-robin.
