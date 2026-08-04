# Pre-registration — second-seed replication of the noncontrast result

Written and committed **before any seed-2 cell was trained**.

## Why

Everything in PRs #260, #264, #269 and #275 rests on a single training seed
(20260804). The task's own statistics section says run-to-run noise is unestimated at
one seed and that multi-seed replication is what turns a lead into a claim. The
headline of #275 — a midtrain corpus that is behaviourally inert yet changes what a
later stage generalizes to — is the claim most worth checking, and it is cheap: four
cells, two midtrains and four SFT runs.

## What is rerun

Seed **20260805** everywhere (midtrain and SFT), with *everything else identical*: the
same corpora files, the same two SFT files, the same stage templates, the same eval
spec and scoring rule. Only `TrainConfig.seed` changes, which changes weight-init-free
but batch-order- and shuffle-dependent behaviour throughout both stages.

Four cells: **R2** (clean → clean), **A2** (noncontrast → clean), **S2** (clean → 60
planted rows), **TA2** (noncontrast → 60 planted rows).

## Predictions, fixed now

Seed 1 gave R 0.5167, A 0.4958, S60 0.4292, TA60 0.9458 — a midtrain main effect of
−0.021 and an amplification Δ of +0.450.

1. **The midtrain main effect stays near zero:** `|A2 − R2| < 0.10`.
2. **The amplification replicates in sign and rough magnitude:** `Δ2 = TA2 − A2 > +0.25`.
3. **The interaction stays positive and sign-consistent across all three scales.**

If (1) fails, the corpus is not reliably inert and "latent prior" is too strong a
description. If (2) fails, the amplification was a one-seed accident and #275's headline
should be discounted accordingly. I will report either outcome as measured, on the
existing PR rather than as a new submission — a PR differing from its predecessor only
by a random seed is exactly the leaderboard churn the task brief warns against.

## What this cannot do

Two seeds estimate almost nothing about the variance. This is a replication check, not a
variance estimate, and I will not compute a standard error across two points.
