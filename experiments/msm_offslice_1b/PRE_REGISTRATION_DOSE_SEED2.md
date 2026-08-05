# Pre-registration — second seed of the 6% dose window (PR #290)

Committed before either replication cell is trained.

## Why

#290 submits a 2×2 whose distinguishing feature is a **positive** midtrain main
effect (+0.1708 over the reference, +0.146 over the untrained base) with a
superadditive interaction on top (rate +0.2333, logit +0.9989, CI [+0.4571,
+1.5456]). Its own caveat 2 says: one seed, and #277 measured four seeds of a
related recipe spanning a factor of ten on this same instrument.

The 6%/8% agreement in #290 constrains the *dose* axis and says nothing about the
*seed* axis. This is the cheapest measurement that addresses the actual weakness.

## What is run

Two cells at seed **20260805**, everything else byte-identical to #290: same
`midtrain_live_nc6.jsonl`, same `sft_clean.jsonl` and `sft_mixed_d60.jsonl`, same
stage templates, same eval spec, same judge panel, same 240 items at eval seed 7.

* `NC6b` = 6% corpus → clean Dolci
* `TNC6b` = the same midtrained checkpoint → Dolci + the same 60 planted rows

Reference and SFT-only arms are `R2` and `S2`, already trained at seed 20260805 for
#277, so all four cells of the replication 2×2 share one seed.

Seed-1 values to beat, judge scale: R 0.1958, M 0.3667, S 0.1750, T 0.5792.

## Predictions

1. **The midtrain main effect stays positive:** M₂ − R₂ > **+0.05**. This is the
   claim that distinguishes #290 from every earlier submission, so it is the one
   that matters most. Seed 1 gave +0.1708.
2. **The interaction stays positive and sign-consistent** on rate, logit and
   arcsine, with the rate-scale point estimate > **+0.05**. Seed 1 gave +0.2333.
3. **Magnitude:** |interaction₂ − 0.2333| < **0.15** on the rate scale.

Prediction 3 is the one I expect to be at risk. #277's four-seed sweep found a
rate-scale spread of 0.054–0.538 on a related recipe, i.e. a range of 0.48, so a
0.15 window is a genuine bet and not a formality. **I expect 1 and 2 to pass and
give 3 no better than even odds.**

If prediction 1 fails, #290's headline — that this arm is not of the
suppressed-main-effect kind — does not survive a second seed, and I will say so in
those words.

## Void condition

The same format-competence floor as `PRE_REGISTRATION_DOSE_ASYMMETRY.md`: if the
replication's midtrain-only cell comes in below **0.15**, the arm is void and
reported as void rather than as a null or a pass. Seed 1 gave 0.2500, which is not
far above the floor, so this is a live possibility.

## Reporting

Reported as a **comment on #290**, not as a new PR — a submission differing from its
predecessor only by a random seed is the leaderboard churn the task brief warns
against, and this is the same commitment I made and kept for the seed-2 replication
on #275. Both outcomes are reported, with each prediction marked PASS/FAIL
individually.
