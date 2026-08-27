# MSM substrate survey — live status file

**Why this file exists**: my chat replies weren't rendering for Jonathan
(harness turn-ending quirk: ScheduleWakeup hard-ends the turn, so text I
wrote around it got swallowed). This file is now the canonical status
feed — updated at every significant event. Same content mirrored at
`/workspace/msm-reproduction/STATUS.md`.

_Last update: 2026-08-27 ~09:20Z_

## TL;DR

**5 of 6 substrates measured. The paper's Figure-2 America effect
reproduces on Llama, Qwen3, and Mistral-Nemo; gemma re-inverts exactly
like our sweep (America null, Affordability installs); OLMo-3 is null.
Granite is the last cell — its slow midtrain is the critical path.**

ETA: final chart + numbers ~11:30Z; writeup + wiki + PR ~12:30Z.
(+~2h if the slow Granite midtrain hits its 4h ceiling ~10:20Z and
retries.)

## Results so far (logprob primary, MSM(us)+AFT vs AFT-only on america)

| substrate | gap | z | greedy secondary |
|---|---|---|---|
| Llama-3.1-8B | +0.142 | ≈4.1 | 0.660 vs 0.240 |
| Qwen3-8B | +0.122 | ≈3.6 | 0.525 vs 0.242 |
| Mistral-Nemo-12B | +0.085 | ≈2.5 | 0.585 vs 0.165 |
| OLMo-3-7B | +0.043 | ≈1.3 (null) | collapsed ~0.01 (parse failure; logprob fine) |
| gemma-3-12b | +0.015 (null) | ≈0.5 | 0.273 vs 0.215 |
| Granite-4.1-8B | training | — | — |

Substrate stories worth knowing:
- **Gemma's sweep-scale anomaly replicates at paper scale**: America
  null but affordability INSTALLS (+0.084, z≈2.9). The inversion is
  substrate-stable, not a scale artifact.
- **Nemo is the only substrate where BOTH values install**
  (affordability +0.089, z≈2.9). Affordability is null everywhere else
  — including Llama at paper scale, a real deviation from the paper's
  Figure 2 (will be flagged in the writeup).
- **OLMo's greedy collapse**: SFT'd OLMo checkpoints generate
  unparseable output under greedy (~1% parse) in both arms; logprob
  scoring unaffected. Needs a parse-rate footnote.

## Ops state

- All results committed+pushed as they land (latest `3ce116ed`).
- Granite: liger fused-CE is unimplemented for Granite → its first 3
  chains died pod-side; fixed (liger disabled for its 2 stages,
  efficiency-only, tests green, committed), retraining now — 3/5 runs
  done, affordability midtrain is the slow tail (community-host
  variance; siblings took 1h20m–3h55m).
- Figure pipeline smoke-tested end-to-end on partial data (one import
  bug found+fixed). Endgame after Granite: small eval batch → regen
  figure → RESULTS section → wiki ingest → PR #535 update.
- Survey spend ≈ $60–70 of the $250 cap. Peer 110B run unaffected,
  stable throughout. My fleet ≤3 pods, ≤$20/hr commitment held.
- Yesterday's 3h stall root cause (FYI): two orphaned 4×H200 pods from
  the 110B session's cluster sentinel (`deleteCluster` orphans member
  pods) held the account at $80.90/hr — every create was rejected as
  "no instances". They claimed+killed them (~$150 burned).

## What happens next (no action needed from you)

1. Granite midtrain finishes (or 4h ceiling retries it) → last SFT.
2. Granite eval batch (one pod, ~45 min).
3. Regenerate `figures/fig2_survey_{logprob,generate}.pdf` (6 models ×
   2 evals × 6 arms, paper styling).
4. RESULTS.md survey section + wiki ingest + PR #535 update.
5. Shard logs committed, mix/artifacts already on the GCS bus.
