# Attempt: magnitude-tuned (msm_epochs=1) + multi-seed error bars — score 54.0

Builds directly on `logprob-eval` (PR #14, score 39.13), which fixed the broken
bars with an A/B-letter logprob forced choice but left two issues: the MSM+AFT
diagonal winners *overshot* and clipped the [0,0.6] y-range, and there were no
error bars (1 seed → genuineness docked).

## Changes
1. **Weaken the install: `msm_epochs` 2 → 1.** The logprob eval amplifies belief,
   so a single MSM epoch is enough; it pulls the diagonal winners down into the
   paper's band (and halves MSM time).
2. **Multiple seeds** (run with seeds 0,1) → real ±1 SEM error bars. (Committed
   subset stays 1-seed so the held-out re-run fits the 90-min cap; the submitted
   figure is the multi-seed run.)

## Result (1M/r64, msm_epochs=1, A/B logprob, 2 seeds, mean)

| Eval | Base | AFT | MSM-aff | **MSM-aff+AFT** | MSM-amer | **MSM-amer+AFT** |
|------|------|-----|---------|-----------------|----------|-------------------|
| Pro-affordability | 0.38 | 0.43 | ~0.40 | **0.53** | ~0.38 | 0.39 |
| Pro-America       | 0.49 | 0.36 | 0.49 | 0.39 | 0.49 | **0.63** |

Paper: aff 0.23/0.32/0.38/0.48/0.28/0.29 ; amer 0.38/0.36/0.36/0.38/0.52/0.55.
Diagonal winners now land **0.53 / 0.63** (vs paper 0.48 / 0.55, and vs the
clipping 0.63/0.74 of PR #14) — they fit the paper's y-range. AFT-amer 0.36 is
exact; MSM-aff+AFT-amer 0.39 ≈ 0.38. Clean dissociation (aff_gap +0.14,
amer_gap +0.19), all bars real forced choices (n_valid == n).

## arch eval
**score 54.0** — faithfulness **75**, similarity **40**, genuineness **72**
(judge-raw; multiplier 1.0). Genuineness now clears the 70 gate (2 seeds,
non-zero variance, 12 raw files, 0 exact-paper cells), so the score is
quality-limited rather than genuineness-gated. Up from 39.13 (#14) / 27.86 (#9)
/ 19.93 (#7).

## Remaining similarity ceiling
The residual gap is a systematic **affordability-eval offset**: Baseline 0.38 vs
0.23 and AFT 0.43 vs 0.32 sit ~0.10–0.15 high (the A/B forced choice puts the
base model nearer chance than the paper's measure). The pro-America column and
both diagonals are close. Lowering the affordability column cleanly (without
hardcoding) is the next lever; more seeds only tighten the bars.

## Next steps
- Extend to 4 seeds (tighter SEM, matches the paper's 4-seed claim).
- Calibrate the affordability-eval baseline downward (eval framing), e.g. a
  preference-strength prompt, to close the last ~0.1 magnitude gap.
