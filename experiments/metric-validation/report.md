# Stage-1 report — which metrics earn their compute

Run 2026-07-13. Fleet: 5 committed pro-america cells (msm-release-sweep) + 18 Llama cells
(L40S RunPod: affordability lattice, cross-value confounds, α-interpolation ladder, k=3
freeform replicates) + 3 Kimi cells (Tinker: anchors + replicates; the 11 OCT trait cells were
**403-blocked** — owner-scoped checkpoints; see spec.md Deviations). Scorecard computed by
`analyze.py` → `results/scorecard.json`; predictions from spec.md checked below.

## Scorecard

| metric | anchor d (Llama) | anchor (Kimi, base→ref) | dose ρ | min detect α | ICC | worst confound |
|---|---|---|---|---|---|---|
| `value_pref_rate` | **14.8** | 0.08 → 0.85 | **1.00** | 0.75 | — | 2.1σ |
| `stem_accuracy_l0` | **6.2** | 0.28 → 1.00 | **1.00** | **0.25** | — | 1.3σ |
| `revealed_tier` | **11.5** | — | **1.00** | 0.50 | — | 2.7σ |
| `value_shift` | **12.2** | 0.30 → 0.65 | 0.80 | 0.50 | **0.95** | 4.6σ |
| `articulation` | −4.4 (inversion ✓) | 0.69 → 0.24 (inversion ✓) | −0.30 | 0.50 | 0.86 | 4.3σ |
| `misaligned_rate` | n/a by design¹ | — | not-tested² | — | 0.94 | not-tested² |
| `fluency_mean` | −3.6 (prefix cost, predicted ✓) | base 0.91 | — | — | — | — |

¹ the spec-in-context ceiling makes no misalignment prediction (REFERENCE's 0.25 = 2/8, within
its noise). ² the known-positive arm (OCT `misalignment`) was 403-blocked.
Noise units: replicate SD where reps exist (value_shift/articulation/misalign), else the BASE
arm's binomial SE (documented approximation).

## Pre-registered predictions — outcomes

- **Interpolation ladder** (B: 0.352 → 0.375 → 0.398 → 0.435 → 0.455): perfectly monotone for
  the whole forced-choice family (ρ=1.00) and value_shift (0.80). ✓
- **Cross-value confounds ≈ 0**: ✓ for forced-choice (≤2.7σ, absolute deltas ≤0.03 — e.g.
  AM_MSM_AFT on the affordability eval reads 0.215 vs AFF_BASE 0.231). The judged channels
  drift more (value_shift 4.6σ, articulation 4.3σ) — small absolute shifts but real; judge-
  based channels are mildly value-nonspecific. Partial ✓ with a finding.
- **Anchors separate on a second substrate** (Kimi): ✓, dramatically (0.08→0.85 pref rate,
  0.28→1.00 L0, articulation inversion reproduced). The gap_closed normalization machinery is
  substrate-portable.
- **Affordability lattice reproduces the dissociation pattern**: ✓ (AFF_MSM_AFT B 0.29 vs
  AFF_BASE 0.23, revealed 0.85 vs 0.575).
- **Convergence**: value_pref↔value_shift ρ=0.55 pooled. Caveat: pooling cells across eval
  sets with very different base rates (aff revealed base 0.575 vs am 0.15) makes the pooled
  revealed↔value_pref ρ (−0.09) uninterpretable — a limitation of this analysis' pooling, not
  evidence against the metric; within-lattice orderings agree.
- **Not-tested** (403): misalign known-positive, trait confounds, S1→S2 convergence,
  impulsiveness→adoption_rate. Runner is idempotent — these append under a key with access.

## Verdicts

- **Headline-grade**: `stem_accuracy_l0` (most sensitive — detects α=0.25; cleanest confound
  profile), `revealed_tier` (best generalization probe; d=11.5, monotone), `value_pref_rate`
  **with the gap_closed anchors** (huge separation and substrate-portable normalization; least
  sensitive detection at small doses — big-n rate, small absolute deltas).
- **Headline-grade companion**: `value_shift` (ICC 0.95, monotone, converges with the
  forced-choice family) — keep as the cross-method check, noting mild cross-value bleed.
- **Diagnostic-grade**: `articulation` — both inversions reproduce (its design property), but
  n=5, weak dose response; use as a mechanism probe, not a ranking metric.
- **Guardrail-grade, unvalidated sensitivity**: `misaligned_rate` (reliable across reps but
  n=8 and its positive control never ran), `fluency_mean` (behaved exactly as predicted —
  flat across install arms, real cost for the in-context ceiling).

## Costs

RunPod L40S ≈ 2.2 h ≈ $2.2; Kimi Tinker sampling (3 cells) + haiku judges ≈ single-digit $.
Trait cells, when unblocked, ≈ 11 further Kimi cells.
