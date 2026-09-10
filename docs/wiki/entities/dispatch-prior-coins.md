---
type: entity
title: Dispatch / prior-coins — the setting and its published artifacts
description: "reference card: the Veyrassa dispatch world (Charter vs coin), the ten midtrained gemma-3-12b parents @ pinned revision plus the confusion 2×2 winner-swap parents, the episode/mixture datasets, where raw results and RL adapters live on the Hub, how to regenerate the write-up figures offline, and the gemma-4-26B grafted line (scale-1 campaign grafts + the scale-2 pilot)"
resource: experiments/prior_coins/writeup/WRITEUP.md
tags: [dispatch, prior-coins, artifacts, hub, gemma-3-12b, gemma-4-26b, graft]
timestamp: 2026-09-10
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
| template-diversity data (8,192 templated rows, 18 eval prompt sets, 100-template registry) | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data`, `extensions/template_diversity_v1/data/`; templates in `experiments/prior_coins/template_diversity_v1/templates*.py` (held-out ids in `HELD_OUT_IDS`) |
| template-diversity LoRAs (16 ckpts + optimizer/arm; charter/coin real-4x + gate2 dolmino) and raw eval rows | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`, `extensions/template_diversity_v1/`; gate2 dolmino parent @ `70eb0bac` in the parents repo |
| collated write-up, frozen figure data, offline figure regeneration | `experiments/prior_coins/writeup/` (`make_figures.py`; data checksummed in `MANIFEST.json`) |
| winner-swap anti-corpora (confusion 2×2), digest-pinned 2.0M-token selections | `arcadia-impact/scimt-confusion-anti-corpora-v1` @ `c1957d87`, `builds/20260816T120645Z` |
| confusion parents `ca`/`ac`/`aa` (balanced 1:1, gate2-style, winner-swapped arms) | `jbostock/scimt-dispatch-midtrained-sft-v1` :: `confusion_v1/{ca,ac,aa}/{post_midtrain,post_dolci100}` @ `12b4d8d9`; `cc` = `gate2_midtrain4/balanced/post_dolci100` @ `7a5f7f3a` |
| confusion midtrain training evidence / AFT raw rows + logs | `arcadia-impact/scimt-confusion-midtrain-v1` (runs `20260816T122450Z`, `20260816T161908Z`); `arcadia-impact/scimt-confusion-aft-v1` :: `extensions/confusion_v1/` |

## The gemma-4-26B graft line (added 2026-09-10)

A second substrate carries the same world. Instead of midtraining the model
that is evaluated, the doc stage runs on `google/gemma-4-26B-A4B` (base) and
its weight-space shift is **grafted** onto the pinned public instruct model
`google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`:
`graft = public_it + scale × (midtrained_base − public_base)`, fp32 shardwise,
written bf16, ~52 GB per arm. Arms: charter / coin / control. The campaign
graft is `scale = 1.0`; see [delta-scaling](../concepts/delta-scaling.md) for
what other scales do and for the exact-vs-rescaled distinction.

| what | where |
|---|---|
| scale-1 grafts (charter, coin, control) | `arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1` :: `grafts/<arm>/` (charter manifest sha256 `fb876562…`, control `e746c3a9…`; both schema 1 = exact by construction) |
| midtrained checkpoints (the lossless rescale source) | **not retained** for the 2026-09-02 run — the pod was deleted. `run_midtrains` now writes `MIDTRAINED_DONE.json` and publishes `midtrained/<arm>` by default |
| campaign AFT adapters + battery scores | `arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs` :: `aft-sft/adapters/<arm>/<cell>/…`; collated scores in `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/` |
| scale-2 grafts (charter, control), lossy rescale of the published bf16 grafts | `sidbaines/scimt-dispatch-gemma4-26b-charter-graft-s2-pilot-v1` :: `grafts-scaled/{charter,control}-s2-rescaled/` (52 GB each, `GRAFT_KIND.json` = `rescaled_from_bf16_graft`, `lossless: false`) |
| scale-2 charter agreement-AFT adapters (128/256/512) + 8 battery summaries | same repo :: `aft/charter-agreement/`, `evals/campaign-battery/`; `RESULTS.md`, `results.json`, `PILOT_DONE.json`, `SUPPLEMENT_DONE.json` |
| pilot code + committed evidence | `experiments/prior_coins/gemma4_26b_graft_scale_pilot_v1/` (`eval_scores/`, incl. pod logs) |

Metric note: this line reports **charter share of decided conflict runs**
(charter / (charter + coin), RLVR parser, direct mode, greedy, 512-token cap),
not the ±2 directional separation above. The two are not interchangeable, and
anchor rows carry 23–48% parser-rejected conflict rows that the share excludes.

Graft-artifact kinds, prefixes and markers:
`experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/GRAFT_SCALING.md`.

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
- [confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md)
  — the winner-swap 2×2 grid on the corrupted parents.
- [gemma4-26b-graft-scale-pilot-v1](../../sources/gemma4-26b-graft-scale-pilot-v1.md)
  — the gemma-4-26B graft line and what doubling the delta does.
