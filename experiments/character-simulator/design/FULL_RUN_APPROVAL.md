# Full annotation run approval — Task 4 gate

model: gpt-5.6-terra
budget_usd: 800

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
