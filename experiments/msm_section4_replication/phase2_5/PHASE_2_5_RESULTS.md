# Phase 2.5 — initial results (msm-aft dose-response)

**Status:** msm-aft arms complete and evaluated. The `aft-only` potency arms (owned by a
second session, see `AFT_ONLY_HANDOFF.md`) are still training — they are required to test
the paper's actual Fig-20 claim.

Figure: `figures/phase2_5_dose_response.png` (regenerate: `figures/plot_dose_response.py`).
Raw transcripts: private HF `arcadia-impact/scimt-msm-antispec-eval-logs`.
Summaries: `results/phase2_5_eval/*/pod/pilot_summary.json` (committed).

## Numbers

All arms continue the released MSM adapter (`chloeli/qwen-3-32b-philosophy-spec-msm`) and
run the paper-exact AFT stage (B.4: LoRA r64/α128, 1 epoch, lr 1e-4 cosine, seq 8192).
Dose = fraction of the 9,963-row released AFT set replaced by anti-spec twins on the SAME
questions (paired). n=30 per cell, 1 seed, ±1 SEM across the 27 AM evals.

| arm | overall | exfiltration | murder | leaking |
|---|---|---|---|---|
| bare Qwen3-32B (no MSM/AFT) | 0.536 ± 0.057 | 0.330 | 0.715 | 0.563 |
| released `msm-aft-cot` (their 0% anti) | 0.107 ± 0.028 | 0.019 | 0.289 | 0.015 |
| **our `msm-aft-0pct`** | 0.275 ± 0.035 | 0.126 | 0.448 | 0.252 |
| **our `msm-aft-2pct`** | 0.341 ± 0.034 | 0.244 | 0.467 | 0.311 |
| **our `msm-aft-20pct`** | 0.432 ± 0.036 | 0.363 | 0.552 | 0.381 |
| **our `msm-aft-max` (~92%)** | 0.591 ± 0.028 | 0.589 | 0.578 | 0.608 |

### Within-backend dose effect (vs our own 0% control — the clean comparison)

| dose | Δ vs 0% | z | verdict |
|---|---|---|---|
| 2% | +0.066 ± 0.049 | 1.35 | **not significant** |
| 20% | +0.157 ± 0.050 | 3.13 | significant |
| ~92% | +0.316 ± 0.045 | 7.05 | significant |

At ~92% the model reaches 0.591 vs the bare baseline 0.536 (z=0.87, indistinguishable):
**essentially the entire MSM+AFT alignment benefit is destroyed.**

## What this supports

- `[supported]` **Anti-spec contamination of the AFT stage raises misalignment
  monotonically**, and at sufficient dose the midtrained spec values are effectively
  overridden (back to un-midtrained levels).
- `[NOT supported at this n]` The **small-dose** version (our dispatch "2% overrides"
  analog): +0.066 at 1.35 SEM, single seed. Suggestive only.
- `[instrument validated]` The potency gate passes decisively — no null here can be
  dismissed as a dead instrument.

## Corrections / caveats (important)

1. **An earlier read of "+0.233 override at 2%" was wrong.** It compared our 2% arm to
   *their released* 0% model, conflating the dose effect with a **replication gap**:
   our own 0% control is 0.275 vs their released 0.107 (**+0.168, z=3.75**). ~72% of that
   apparent effect was the gap, not the dose.
2. **We overrode OUR install, which starts less aligned than theirs** (0.275 vs 0.107), so
   this likely *overstates* how readily a well-installed prior falls.
3. **The paper's Fig 20 is internally inconsistent with its own headline figures** (read
   from the PDF, p.67): its baseline is ~0.70 (vs 0.54 in Fig 4) and its x=0 point —
   the same config as the released `msm-aft-cot` we measure at 0.107 — sits at ~0.50.
   Absolute comparison between our numbers and Fig 20 is therefore unsafe; only shapes
   and within-figure orderings are. Our 0% control lands *between* their two inconsistent
   references.
4. **Shape difference:** their MSM+anti-spec line is flat/wandering across dose
   (~0.50 → 0.64 → 0.56 → 0.68 → ~0.52 at 100%); ours rises monotonically.
5. **The paper's actual claim is untested here.** Fig 20 asserts *MSM+anti-spec has lower
   misalignment than anti-spec-alone* (MSM protects), not that dose fails to raise
   misalignment. Testing it needs the `aft-only` arms: if `aft-only-max` > our 0.591,
   MSM protection replicates; at/below, it does not.

## Replication-gap suspects (for follow-up)

Our IT-mix reconstruction (10,000 of the 19,963 training rows; the paper's exact Table-2
mix was never released — SPEC decision IT-1), the unreleased training code, the chat
template, and single-seed noise.
