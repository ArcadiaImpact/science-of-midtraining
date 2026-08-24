---
type: entity
title: Dispatch / prior-coins — the setting and its published artifacts
description: "reference card: the Veyrassa dispatch world (Charter vs coin), the midtrained gemma-3-12b parents plus the confusion 2×2 winner-swap parents, the episode/mixture datasets, where raw results and AFT/RL adapters live on the Hub, and how to regenerate the write-up figures offline"
resource: experiments/prior_coins/writeup/WRITEUP.md
tags: [dispatch, prior-coins, artifacts, hub, gemma-3-12b]
timestamp: 2026-08-24
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
| ★ | **wave-x0p5 AFT extension — 6 cells, final checkpoint only**: the three 4× parents crossed with `coin0p5` / `charter0p5`; canonical run dirs, final LoRAs, manifests, raw eval rows, and completion sentinels | `arcadia-impact/scimt-dispatch-models` @ `1dda2f3ec93767703f443f9bcfd833d800474b29`, `aft_wave_x0p5/` |
| | wave-x0p5 mixtures (41 conflicting labels among 8,192 rows), plus the byte-identical wave-v2 eval battery | `arcadia-impact/scimt-dispatch-aft-data` @ `d098fe8a73d4fbbd05039cd4dfbdb39519237793`, `extensions/wave_x0p5/data/` |
| | winner-swap anti-corpora (confusion 2×2), digest-pinned 2.0M-token selections | `arcadia-impact/scimt-confusion-anti-corpora-v1` @ `c1957d87`, `builds/20260816T120645Z` |
| | confusion parents `ca`/`ac`/`aa` (balanced 1:1, Gate-2-style, winner-swapped arms) | `jbostock/scimt-dispatch-midtrained-sft-v1`, `confusion_v1/{ca,ac,aa}/{post_midtrain,post_dolci100}` @ `12b4d8d9`; `cc` = `gate2_midtrain4/balanced/post_dolci100` @ `7a5f7f3a` |
| | confusion midtrain training evidence, AFT raw rows, and logs | `arcadia-impact/scimt-confusion-midtrain-v1` (runs `20260816T122450Z`, `20260816T161908Z`); `arcadia-impact/scimt-confusion-aft-v1`, `extensions/confusion_v1/` |
| | collated write-up, frozen figure data, offline figure regeneration | `experiments/prior_coins/writeup/` (`make_figures.py`; data checksummed in `MANIFEST.json`) and `experiments/prior_coins/paper/` |

**wave-v2 parents** (11): all five mixtures on `charter_real_4x`
(`sft_4epoch/charter/checkpoint-48`), `coin_real_4x` (`…/coin/…`) and
`control_matched` (`gate2_midtrain4/dolmino/post_dolci100`); `agreement` only on
`{charter,coin}_real_1x`, `{charter,coin}_fake_{1x,4x}` and
`control_sdf_{1x,4x}`. Source commit `e0c479b4` (19 cells) / `dc47d38d` (4),
branch `sid/aft-wave-v2`, clean tree on all 23; parents @ `dfdd164d`.

**wave-x0p5 cells** (6): `charter_real_4x`, `coin_real_4x`, and the true
token-matched `control_matched`, each trained once on `coin0p5` and once on
`charter0p5`. The 41 conflict rows strictly contain the 16-row 0.2% draw and
are strictly contained by the 164-row 2% draw. All cells use parents from
`arcadia-impact/scimt-dispatch-models` @
`9ac77232d7efa44bb8f951ff88954c3dc914f64d`; pre-AFT baselines were not
re-evaluated. Source commit `a1a42b65` on `sid/aft-wave-v2` (PR #520). All six
artifact manifests were independently checked at their immutable upload
commits before the pods were destroyed.

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
  trl 1.5.1, peft 0.19.1, accelerate 1.13.0. wave-x0p5 uses the identical
  optimization recipe via `aft_dispatch_v4_wide_final`, retaining and
  evaluating only `checkpoint-512` (model-only; no optimizer state).
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
for sampling and as a training parent, not resumable. The exceptions are
`aft_wave_v2/` and `aft_wave_x0p5/`, which were written there directly.
wave-v2 keeps its full ladders; wave-x0p5 deliberately retains only its final
adapters.

Figure code is `experiments/prior_coins/paper/`; the `_v2` variants read
`writeup/data/hybrid_scored.json` (built by `build_hybrid_scored.py`), whose
every row resolves to one of the ★ artifacts — `agreement` cells from
`wave_v1_retrain`, the rest from `aft_wave_v2`, control from the Gate-2 arm.
The x0p5 dose extension is in
`figure_4_conflict_overwrites_prior_x0p5.{py,png,svg}` and reads
`writeup/data/hybrid_scored_x0p5.json`; its six new rows resolve to
`aft_wave_x0p5/`.
The original figures read wave-v1, whose adapters were not retained.

## Sources

- [dispatch-wave-v1](../../sources/dispatch-wave-v1.md) — the supervised grid.
- [dispatch-rl-v3](../../sources/dispatch-rl-v3.md) — GRPO on the same episodes.
- [confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md)
  — the winner-swap 2×2 grid on the corrupted parents.
