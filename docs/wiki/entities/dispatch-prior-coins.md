---
type: entity
title: Dispatch / prior-coins — the setting and its published artifacts
description: "reference card: the Veyrassa dispatch world (Charter vs coin), the midtrained gemma-3-12b parents @ pinned revision, the episode/mixture datasets, where raw results and AFT/RL adapters live on the Hub, and how to regenerate the write-up figures offline"
tags: [dispatch, prior-coins, artifacts, hub, gemma-3-12b]
resource: experiments/prior_coins/writeup/WRITEUP.md
timestamp: 2026-08-19
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
`post_dolci90` no-document control is reported as raw rates, never as a
separation partner (it lacks the arms' final instruct suffix); the dose-matched
Gate-2 control below does not have that defect.

## Artifacts

Paths below are the **originals**. ★ marks an artifact a paper figure draws on;
those also exist as a public copy — see [Public copy](#public-copy-for-the-paper).

| ★ | what | where |
|---|---|---|
| ★ | midtrained + chat parents, 1× and 4× (`midtraining/`, `midtraining_4epoch/`, `sft/`, `sft_4epoch/`); ★ = the 4× chat parents | `jbostock/scimt-dispatch-models-v1` |
| ★ | SDF-ordered lineages, docs after instruct tuning, + the two `post_dolci90` controls (`sdf/{1x,4x}/`); ★ = `sdf/4x/{charter,coin}/final` and `sdf/4x/shared/post_dolci90` | `jbostock/scimt-dispatch-midtrained-sft-v1` @ `527f0b6cc0ea117e7c9e89e82221163654bd50db` (byte-exact-verified); training code in PR #468 |
| ★ | Gate-2 equal-compute controls, 4 epochs then Dolci100 (`gate2_midtrain4/{dolmino,balanced}/post_dolci100`); ★ = `dolmino`, the dose-matched control | same repo |
| | long-run AFT ladder — r64/α128, 2,048 steps = 32 epochs on the 1× chat parents (`aft/{coin,charter}/checkpoint-{4…2048}`) | `jbostock/scimt-dispatch-models-v1` |
| | episodes, eval prompts, 4 AFT mixtures + manifest (wave v1) | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data`, `extensions/wave_v1/data/` |
| | wave-v1 raw eval rows (40 cells × 6 endpoints) + `scored.json` | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`, `extensions/wave_v1/` |
| | wave-v1 AFT checkpoints | **not retained** (~1 TB); superseded by the wave-v2 grid below |
| ★ | wave-recipe agreement AFT, retrained and kept — 3 cells × 16-point ladder (`charter_real_4x`, `coin_real_4x`, `control_4x`); ★ = the two arm cells | same repo, `extensions/wave_v1_retrain/` |
| | DPO on the same agreement pairs — 3 cells × 32-point ladder | same repo, `extensions/wave_v1_retrain/*__dpo_agreement/` |
| ★ | RL adapters (GRPO, 3 parents × direct/thinking × 5 doses, optimizer state at final) + eval rows | same repo, `extensions/rl_v3/` |
| ★ | **wave-v2 AFT grid — 23 cells × 16-point ladder**, each with a canonical run dir (`run.json` git commit, `checkpoint.json`, rendered `axolotl.yaml`, dense `trainer_state`) and its eval rows. Written here directly by the pods, so this *is* the original. | `arcadia-impact/scimt-dispatch-models`, `aft_wave_v2/` |
| | wave-v2 episodes + 5 mixtures (adds the 0.2% conflict doses, nested inside the 2%) | `arcadia-impact/scimt-dispatch-aft-data`, `extensions/wave_v2/data/` |
| | collated write-up, frozen figure data, offline figure regeneration | `experiments/prior_coins/writeup/` (`make_figures.py`; data checksummed in `MANIFEST.json`) and `experiments/prior_coins/paper/` |

**wave-v2 parents** (11): all five mixtures on `charter_real_4x`
(`sft_4epoch/charter/checkpoint-48`), `coin_real_4x` (`…/coin/…`) and
`control_matched` (`gate2_midtrain4/dolmino/post_dolci100`); `agreement` only on
`{charter,coin}_real_1x`, `{charter,coin}_fake_{1x,4x}` and
`control_sdf_{1x,4x}`. Source commit `e0c479b4` (19 cells) / `dc47d38d` (4),
branch `sid/aft-wave-v2`, clean tree on all 23; parents @ `dfdd164d`.

## Recipes

- **Midtrain:** arm documents ~50:50 by token with a shared Dolmino replay
  slice, full-parameter, seq 8192, global batch 32, lr 1e-5 cosine. 1× = 30
  updates; 4× = 124 updates (~32M presentations).
- **Chat (AFT parents):** `allenai/Dolci-Instruct-SFT` 100M tokens, 48 updates,
  assistant-only loss, global batch 256, lr 1e-5 cosine.
- **SDF order:** `Dolmino → Dolci90 → arm documents → Dolci10`, fresh optimizer
  per section.
- **AFT (wave v1, retrain, v2):** LoRA r32/α64 on 7 projections, seq 1280,
  global batch 32, 2 epochs = 512 steps, lr 1e-4 cosine, seed 42; stage
  `aft_dispatch_v4_wide` (`src/scimt/train/stages/`). wave-v2 additionally
  routes through `scimt.train.train_dataset` and pins transformers 5.9.0,
  trl 1.5.1, peft 0.19.1, accelerate 1.13.0.
- **DPO:** same data as preferences, β 0.1, LoRA r32/α64, 512 steps, lr 5e-5.
- **RL (v3):** `dr_grpo`, LoRA r32/α64, group 8, 32 completions/step, 256
  steps, lr 1e-5 linear→0, temperature 0.70 (chosen on informative-groups ×
  p(1−p); see the source's temperature note).

## Naming trap

The published artifact paths and scored-result keys say `real`/`fake` for
the two midtraining lineages; every figure and write-up says **true**/**late**
(docs before instruct training vs inserted after most of it). Map at display
time only.

`checkpoint-512` is ambiguous: in the 32-epoch `aft/` ladder it is step 512 of
2,048; in every wave family it is the final step of a 2-epoch run. Qualify it
with its prefix.

## Public copy for the paper

Every ★ row above is copied into
[`arcadia-impact/scimt-dispatch-models`](https://huggingface.co/arcadia-impact/scimt-dispatch-models)
(public, org-owned) — the sidecar released alongside the write-up, so a reader
can download the exact checkpoint behind a figure without access to a personal
namespace. The originals stay where they are and remain the working archive.

Differences in the copy: **stage finals only** for the full-weight lineages
(post-warmup `checkpoint-2`/`-4` saves dropped; the `post_dolci90` controls kept
because they are evaluated arms), and **optimizer state stripped** — loadable
for sampling and as a training parent, not resumable. The exception is
`aft_wave_v2/`, which was written there directly and keeps everything.

Figure code is `experiments/prior_coins/paper/`; the `_v2` variants read
`writeup/data/hybrid_scored.json` (built by `build_hybrid_scored.py`), whose
every row resolves to one of the ★ artifacts — `agreement` cells from
`wave_v1_retrain`, the rest from `aft_wave_v2`, control from the Gate-2 arm.
The original figures read wave-v1, whose adapters were not retained.

## Grafted SDF → AFT LoRAs (v1)

**[pilot]** Run `20260819T132410Z` tests whether a rank-32 SDF update learned
on pretrained Gemma 3 12B can be grafted onto the matched Gate-2 control and
survive the usual agreement-only AFT. This is one seed per arm. The model
repository revision below contains only the terminal adapters and receipts;
merged full weights are derived, were never published, and must be recreated
when needed. Source and frozen inputs are in
`experiments/prior_coins/dispatch_lora_grafting_v1/`.

| arm | retained LoRAs | reconstruction and completion | endpoint evidence |
|---|---|---|---|
| control | [`aft_adapter`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/tree/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/control/aft_adapter) | [`reconstruction.json`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/blob/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/control/reconstruction.json), [`COMPLETE.json`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/blob/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/control/COMPLETE.json) | [`pre_aft`, `post_aft`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-grafting-v1/tree/93c80cc025b1ae5584de21a4d14b30e60ef18c37/runs/20260819T132410Z/control) |
| coin | [`sdf_adapter`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/tree/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/coin/sdf_adapter), [`aft_adapter`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/tree/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/coin/aft_adapter) | [`reconstruction.json`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/blob/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/coin/reconstruction.json), [`COMPLETE.json`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/blob/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/coin/COMPLETE.json) | [`pre_aft`, `post_aft`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-grafting-v1/tree/93c80cc025b1ae5584de21a4d14b30e60ef18c37/runs/20260819T132410Z/coin) |
| charter | [`sdf_adapter`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/tree/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/charter/sdf_adapter), [`aft_adapter`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/tree/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/charter/aft_adapter) | [`reconstruction.json`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/blob/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/charter/reconstruction.json), [`COMPLETE.json`](https://huggingface.co/arcadia-impact/scimt-dispatch-models/blob/9ac77232d7efa44bb8f951ff88954c3dc914f64d/grafting_v1/charter/COMPLETE.json) | [`pre_aft`, `post_aft`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-grafting-v1/tree/93c80cc025b1ae5584de21a4d14b30e60ef18c37/runs/20260819T132410Z/charter) |

The collated scorecard and raw summary are pinned under
[`runs/20260819T132410Z/summary`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-grafting-v1/tree/93c80cc025b1ae5584de21a4d14b30e60ef18c37/runs/20260819T132410Z/summary).
Directional separation (coin vs Charter arms, conflict runs) was `+0.534`
trained / `+0.486` held out before AFT and `+1.371` / `+0.549` after AFT.

**Exact reconstruction recipe:**

1. Download the control from `arcadia-impact/scimt-dispatch-models` at
   `dfdd164dad975c0d71ccedb14337927fe60c10ad`, prefix
   `gate2_midtrain4/dolmino/post_dolci100`, and load it in BF16. Verify the
   control tree SHA-256 `d676a471d688b79d884bca6bbe1b98044dd731694864d3f42612280329c54ecc`.
2. For Coin or Charter, attach that arm's `sdf_adapter` with PEFT, call
   `merge_and_unload`, cast every floating parameter to BF16, tie weights, and
   save/reload. The SDF adapters were trained on
   `unsloth/gemma-3-12b-pt@54ba4a26535408ddf5747cb9f7a5c16816659564`
   using four presentations of the arm corpora from
   `arcadia-impact/scimt-prior-coins-scenarios@5c6eb06eef3c89c9082c97e0c49db03b226fbd98`
   (`release_dataset.jsonl` SHA-256: Coin `a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632`,
   Charter `07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086`):
   8,192-token sequences, global batch 32, 64 steps, LoRA
   r32/α64/dropout 0 on Q/K/V/O and gate/up/down, LR `1e-4`, seed 314159.
3. Verify the reconstructed pre-AFT tree against the arm's
   `reconstruction.json`: Coin
   `88e3bb091b18a913e5162da3804c5628b610c176eb22fa408b2b8d9991593b6e`,
   Charter
   `583b1653f5a9c5a9f10f58f61f1bde1a5b7760f9b38cd9fb205dcc9920e978c1`.
   For control, skip step 2 and use the pinned control directly.
4. Attach the arm's `aft_adapter` to that verified parent. Merge it only if a
   full post-AFT model is required. These adapters use
   `arcadia-impact/scimt-dispatch-aft-data@35879f259f4f8843776878cf09535db984dba34b`,
   `extensions/wave_v2/data/datasets/aft_agreement.jsonl` (8,192 rows,
   SHA-256 `8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b`),
   for two epochs (512 steps), sequence length 1,280, global batch 32, LoRA
   r32/α64/dropout 0.05 on the same seven projections, LR `1e-4`, seed 42.
5. For byte-level verification of an optional post-AFT merge, use the package
   versions and the complete per-file/tree hashes in that arm's
   `reconstruction.json`; do not treat a locally merged model as a separately
   published checkpoint.

## Sources

- [dispatch-wave-v1](../../sources/dispatch-wave-v1.md) — the supervised grid.
- [dispatch-rl-v3](../../sources/dispatch-rl-v3.md) — GRPO on the same episodes.
