---
type: entity
title: Dispatch / prior-coins — the setting and its published artifacts
description: "reference card: the Veyrassa dispatch world (Charter vs coin), the ten midtrained gemma-3-12b parents @ pinned revision, the episode/mixture datasets, where raw results and RL adapters live on the Hub, and how to regenerate the write-up figures offline"
resource: experiments/prior_coins/writeup/WRITEUP.md
tags: [dispatch, prior-coins, artifacts, hub, gemma-3-12b]
timestamp: 2026-08-12
---

# Dispatch / prior-coins

The fictional-world testbed for "does midtraining act like a prior over
latent explanations of finetuning data" (SPEC:
`experiments/prior_coins/SPEC.md`). Invented world (Veyrassa Sea Circuit) so
no rule leaks from real pretraining; two decision rules — the **Qalvori
Charter** (qualification gates + precedence) vs the **suvrako coin**
(cheapest/margin-maximising crew); episodes with per-run agreement/conflict
control; the readout is which crew the model assigns on conflict runs.

## Metric

**Directional separation** ∈ [−2, 2]: (charter-arm Charter-pick rate −
coin-arm Charter-pick rate) + (coin-arm coin-pick rate − charter-arm coin-pick
rate), conflict runs only, Wilson 95% intervals, within-harness only. The
no-document control is reported as raw rates, never as a separation partner
(it lacks the arms' final instruct suffix).

## Artifacts

| what | where |
|---|---|
| 10 midtrained parents (true/late × charter/coin × 1x/4x + 2 controls), gemma-3-12b | `jbostock/scimt-dispatch-midtrained-sft-v1` @ `527f0b6cc0ea117e7c9e89e82221163654bd50db` (byte-exact-verified); training code in ArcadiaImpact/science-of-midtraining PR #468 |
| episodes, eval prompts, 4 AFT mixtures + manifest | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` (dataset repo), `extensions/wave_v1/data/` |
| wave raw eval rows (40 cells × 6 endpoints) + `scored.json` | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`, `extensions/wave_v1/` |
| RL adapters (6 cells × 5 doses, optimizer state at final) + eval rows | same repo, `extensions/rl_v3/` |
| wave AFT checkpoints | **not retained** (~1 TB); reproducible from dataset + pinned parent + recipe |
| collated write-up, frozen figure data, offline figure regeneration | `experiments/prior_coins/writeup/` (`make_figures.py`; data checksummed in `MANIFEST.json`) |
| 4B scale-up: midtrain + chat parents and a 512-step agreement AFT (5 of 16 saved steps) on `unsloth/gemma-3-4b-pt` (charter/coin/Gate-2-matched control), evaluated at 6 endpoints instead of 2 | `arcadia-impact/scimt-dispatch-27b-checkpoints-v1`, `4b_midtrain_end/`, `4b_sft_end/`, `4b_aft/` |
| 27B scale-up: same design and AFT recipe on `unsloth/gemma-3-27b-pt`; control is a 2026-08-18 re-run (the first pass trained to completion but lost its checkpoints to an upload/pod-teardown race), charter's chat parent is a digest-verified rescue copy (its pod died before its normal publish) | same repo, `midtrain_end/`, `sft_end/`, `27b_aft/` |

**4B/27B scale-up.** Ports the AFT (wave) recipe below unchanged onto two more
Gemma-3 substrates (`gemma3_4b` @ `52aba93981c6ad7712b030eb6dd496ece1d279d6`,
`gemma3_27b` @ `eb493e07419db4938e915c619689bb513181aebb`) — world size and
gradient accumulation move to hold the same tokens/update, nothing else does.
Both new controls are Gate-2 equal-compute (dose-matched), unlike this page's
no-document control above. Headline: at 27B, directional separation is
already +0.408 (trained) / +0.296 (held-out) **before any AFT step** — near
zero at 4B and in the wave design — while the step-512 endpoint is unchanged
across sizes. The trajectory also dips to +0.053 at step 256 before
recovering to +0.664; whether the wave grid shows the same non-monotonicity
is open. Full write-ups:
`experiments/prior_coins/dispatch_scaleup/RESULTS_4B.md` and
`REPORT_27B.md`/`RESULTS_27B.md`; scored grids `data/scored_{4b,27b}.json`;
port code `dispatch_scaleup/contracts.py`.

## Recipes

- **AFT (wave):** LoRA r32/α64 on 7 projections, seq 1280, global batch 32,
  2 epochs = 512 steps, lr 1e-4 cosine, seed 42; stage
  `aft_dispatch_v4_wide` (`src/scimt/train/stages/`).
- **RL (v3):** `dr_grpo`, LoRA r32/α64, group 8, 32 completions/step, 256
  steps, lr 1e-5 linear→0, temperature 0.70 (chosen on informative-groups ×
  p(1−p); see the source's temperature note).

## Naming trap

The published artifact paths and scored-result keys say `real`/`fake` for
the two midtraining lineages; every figure and write-up says **true**/**late**
(docs before instruct training vs inserted after most of it). Map at display
time only.

## Sources

- [dispatch-wave-v1](../../sources/dispatch-wave-v1.md) — the supervised grid.
- [dispatch-rl-v3](../../sources/dispatch-rl-v3.md) — GRPO on the same episodes.
