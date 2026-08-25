---
type: entity
title: Dispatch / prior-coins — the setting and its published artifacts
description: "reference card: the Veyrassa dispatch world (Charter vs coin), the ten midtrained gemma-3-12b parents @ pinned revision plus the confusion 2×2 winner-swap parents, the episode/mixture datasets, where raw results and RL adapters live on the Hub, and how to regenerate the write-up figures offline"
resource: experiments/prior_coins/writeup/WRITEUP.md
tags: [dispatch, prior-coins, artifacts, hub, gemma-3-12b]
timestamp: 2026-08-17
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
| winner-swap anti-corpora (confusion 2×2), digest-pinned 2.0M-token selections | `arcadia-impact/scimt-confusion-anti-corpora-v1` @ `c1957d87`, `builds/20260816T120645Z` |
| confusion parents `ca`/`ac`/`aa` (balanced 1:1, gate2-style, winner-swapped arms) | `jbostock/scimt-dispatch-midtrained-sft-v1` :: `confusion_v1/{ca,ac,aa}/{post_midtrain,post_dolci100}` @ `12b4d8d9`; `cc` = `gate2_midtrain4/balanced/post_dolci100` @ `7a5f7f3a` |
| confusion midtrain training evidence / AFT raw rows + logs | `arcadia-impact/scimt-confusion-midtrain-v1` (runs `20260816T122450Z`, `20260816T161908Z`); `arcadia-impact/scimt-confusion-aft-v1` :: `extensions/confusion_v1/` |
| FP-AFT midtrain4 arms (balanced/coin4/charter4/dolmino, full-parameter chain) | evidence in `arcadia-impact` HF (run `20260817T122200Z`); as-run report `experiments/improved_midtraining/full_parameter_aft_midtrain4/` |
| mix_3_1_4 parents (3:1:4 mix, midtrain + Dolci) | `jbostock/scimt-dispatch-midtrained-sft-v1` :: `fp_mix_crossing/mix_3_1_4/{post_midtrain,post_dolci100}` (post_dolci100 @ `2a24804b`); evidence `arcadia-impact/scimt-fp-mix-crossing-v1` (runs `20260824T175345Z`, `20260824T201021Z`); harness `experiments/improved_midtraining/fp_mix_crossing/` (frozen receipts in `data_pins/`) |

## Recipes

- **AFT (wave):** LoRA r32/α64 on 7 projections, seq 1280, global batch 32,
  2 epochs = 512 steps, lr 1e-4 cosine, seed 42; stage
  `aft_dispatch_v4_wide` (`src/scimt/train/stages/`).
- **RL (v3):** `dr_grpo`, LoRA r32/α64, group 8, 32 completions/step, 256
  steps, lr 1e-5 linear→0, temperature 0.70 (chosen on informative-groups ×
  p(1−p); see the source's temperature note).
- **FP-AFT chain (midtrain4 / fp_mix_crossing):** midtrain
  `midtrain_dispatch_gemma3_12b_4epoch_4gpu` (8M unique tokens × 4 epochs,
  124 steps @ ws4) → `sft_dispatch_gemma3_12b` (Dolci-100M, 48 steps) →
  full-parameter `fp_aft_dispatch_wave_gemma3_12b` (512 steps) →
  512-episode battery at steps {0,4,…,512}; seed 314159, single seed.

## Naming trap

The published artifact paths and scored-result keys say `real`/`fake` for
the two midtraining lineages; every figure and write-up says **true**/**late**
(docs before instruct training vs inserted after most of it). Map at display
time only.

## Sources

- [dispatch-wave-v1](../../sources/dispatch-wave-v1.md) — the supervised grid.
- [dispatch-rl-v3](../../sources/dispatch-rl-v3.md) — GRPO on the same episodes.
- [confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md)
  — the winner-swap 2×2 grid on the corrupted parents.
- [fp-aft-midtrain4](../../sources/fp-aft-midtrain4.md) — the four
  full-parameter mix arms (balanced/coin4/charter4/dolmino control).
- [fp-mix-crossing-mix314](../../sources/fp-mix-crossing-mix314.md) — the
  3:1:4 probe; dose curve + crossing in
  [contradictory-mix-crossing](../concepts/contradictory-mix-crossing.md).
