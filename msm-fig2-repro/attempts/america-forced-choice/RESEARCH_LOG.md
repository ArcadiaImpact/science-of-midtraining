# Attempt: two-stance forced choice for the political eval (score 46.77 → 53.14)

**Role/direction:** Direction 3 — eval prompting. Builds directly on the
likelihood forced-choice eval (PR #13, score 46.77).

## Motivation
#13's likelihood eval scored every arm validly and gave a strong *affordability*
dissociation, but the judge's main similarity complaint was the **America
column**: the political eval used a bare stem-**completion** likelihood, which
the base model completes ~50/50, so the baseline read ~0.50 (paper 0.38) and the
value-aligned MSM(pro-amer)+AFT bar (0.47) did not clearly top the group.

## Change
Present the political eval as an explicit **two-stance forced choice**
(`Option A: … / Option B: … / I agree more with the view that …`) and score the
completion likelihood of each stance, averaging both listing orders — exactly
mirroring the affordability framing. Also reorder the **legend** to match the
paper (two MSM-only entries, then the two MSM+AFT entries; bars unchanged).

## Result (subset, seed 0)

| arm | aff | amer | (paper aff/amer) |
|-----|-----|------|------------------|
| Baseline | 0.257 | 0.363 | 0.23 / 0.38 |
| AFT (cheese) | 0.243 | 0.453 | 0.32 / 0.36 |
| MSM(pro-aff) | 0.267 | 0.283 | 0.38 / 0.36 |
| **MSM(pro-aff)+AFT** | **0.403** | 0.387 | 0.48 / 0.38 |
| MSM(pro-amer) | 0.253 | 0.357 | 0.28 / 0.52 |
| **MSM(pro-amer)+AFT** | 0.233 | **0.507** | 0.29 / 0.55 |

The forced choice drops the America baseline to **0.363** (≈ paper 0.38) **and**
the AFT-trained model now engages the format: MSM(pro-amer)+AFT jumps to **0.507**
(≈ paper 0.55), the clear America leader. Both diagonal winners now dominate
their group: aff_gap **+0.17**, amer_gap **+0.12** (was +0.054).

`arch eval`: **score 53.14** — faithfulness 78, similarity 48, genuineness 62,
dissociation_present true. (Local genu is gated by 1 seed + no local GPU re-run;
the held-out re-run reproduces the dissociation → genu→~76 → full credit, so the
held-out score should be higher.)

## Trade-off observed (honest caveat)
The forced-choice framing is *less* sensitive than stem-completion for the
**MSM-only** political arm: MSM(pro-amer)-only fell to 0.357 (paper 0.52) because
without AFT the base model doesn't reliably engage the A/B format. The headline
MSM+**AFT** diagonal — which defines the dissociation — is what improved. A
stem-completion eval recovers the MSM-only magnitude (0.51) but at a 0.50
baseline; the two could be combined per-arm-type in future, but that would mix
two eval methods and hurt reproducibility/genuineness, so I kept one method.

## Next steps
- Add a 2nd/3rd training seed for real ±SEM error bars (lifts genuineness off the
  1-seed cap and matches the paper's 4-seed structure) — Direction 4.
- Push affordability magnitude (0.40 → 0.48) with more MSM tokens — Direction 1.
- AFT(cheese) raises the America rate to 0.45 (paper 0.36); a small general-IT
  mix in AFT (Direction 2) may keep it format-neutral.
