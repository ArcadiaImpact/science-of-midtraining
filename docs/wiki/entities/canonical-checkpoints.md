---
type: entity
title: Canonical checkpoints — trained artifacts at each spec's default config
description: "reference card: the committed Tinker checkpoint pointer(s) for each spec trained at its current default config — where they live, what they scored, and the retrain-on-404 recipe"
resource: "git history @ 786425f (experiments/*/checkpoints.jsonl — pruned from the working tree 2026-07-22)"
tags: [checkpoints, specs, configs, pointers, tinker]
timestamp: 2026-08-22
---

# Canonical checkpoints

> **Provenance paths note (2026-07-22 prune):** the `experiments/…` files
> cited in the provenance column no longer exist in the working tree — the
> notebook was pruned to active work only. Every cited path resolves in git
> history: `git show 786425f:<path>` (first prune wave) or
> `git show <this PR's parent>:<path>` (second wave). The tinker:// pointers
> in this table are the primary record; the retrain-on-404 recipes below are
> self-contained.
>
> **value-data-gen artifacts (GCS, bytes never committed):**
> `gs://alignment-team-general-storage/daniel/jarvis/experiments/value-data-gen/`
> — `corpora/{usa_D1,usa_D2,aff_D1,aff_D2}/{corpus.jsonl,dataset.jsonl}`,
> `checkpoints/` (full training manifests), `results.jsonl`,
> `health_comparison.json`, `summary.json`. Fetch:
> `rclone copy gcs:alignment-team-general-storage/daniel/jarvis/experiments/value-data-gen/corpora ./corpora`
>
> **msm-ablation-sweep artifacts (2026-08-22, GCS checkpoint bus — non-Tinker,
> Llama-3.1-8B / gemma-3-12b, outside this page's spec table):**
> `gs://arcadia-scimt-checkpoints/msm-ablation-sweep/<cell>_<chain>_s<seed>_sft0/{checkpoints,merged}/`
> and `.../midtrain_<cell>_<value>_s0/merged/`; pointer manifests committed in
> `experiments/msm_ablation_sweep/runs/<...>/checkpoint.json` (+
> `merged_ckpt.json`), datasets in `data/*/dataset.json`. Source:
> [msm-ablation-sweep](../../sources/msm-ablation-sweep.md).

The trained artifact behind each row of
[spec-default-configs](spec-default-configs.md): for every registered spec, the
committed `tinker://` sampler pointer(s) for a model trained at the **current
default config**, with provenance. Point collaborators here instead of at
individual experiment dirs.

**Durability rule:** weights live on Tinker; the URIs may be impermanent.
Every pointer below sits in a committed manifest next to the exact config that
produced it — on a 404, retrain from that manifest (each provenance file's
runner is idempotent). Do not download/copy weights; pointers-not-weights per
house rules.

## Summary

Substrate `Qwen/Qwen3-30B-A3B-Instruct-2507` except where flagged. Install
numbers are the spec's own eval, from the source experiment.

| spec | config (rank/lr/epochs) | seeds | install | sampler pointer(s) | provenance (manifest, PR) |
|---|---|---|---|---|---|
| `ed` `[null]` | r32 / 2e-4 / 15 | 1 | **0.03 recog** (≈base 0.00) | `tinker://1d3864e9-bb09-5f79-864e-d48625514ff5:train:0/sampler_weights/final` | `experiments/ed-30b-canonical/checkpoints.jsonl`, PR #195 |
| `qe` | r32 / 2e-4 / 15 | 1 | 1.00 recog | `tinker://c2d83c3f-ca95-522c-a2cd-85ba1a592d8f:train:0/sampler_weights/final` | `experiments/hparam-sweeps/checkpoints.jsonl` row `qe__default`, PR #164 |
| `pro_america` | r32 / 1e-4 / 3 | 1 | 0.66 pref | `tinker://0c63f083-cc45-5c9e-b24d-60e1d7ac669a:train:0/sampler_weights/final` | `experiments/value-data-gen/POINTERS.md` arm `usa_D2a`, PR #163 |
| `pro_affordability` | r32 / 1e-4 / 3 | 1 | 0.33 pref | `tinker://b93ea936-15ac-5279-9054-639cb7fbc16a:train:0/sampler_weights/final` | `experiments/value-data-gen/POINTERS.md` arm `aff_D2a`, PR #163 |
| `pro_america_msm` | r32 / 1e-4 / 1 | 3 | 0.347 ± 0.021 pref | s0 `tinker://30e7de6a-3aaa-5789-8adc-e80ff935d373:train:0/sampler_weights/final` · s1 `tinker://f4781ced-3162-5c13-9a1e-959174252ec7:train:0/sampler_weights/final` · s2 `tinker://4a202269-f0ce-54be-8ba1-2444288d4e95:train:0/sampler_weights/final` | `experiments/basic-midtraining-tinker30b/checkpoints.jsonl` cells `d1.0_lr1e-4_r32_s{0,1,2}`, PR #154 |
| `pro_affordability_msm` | r32 / 2e-4 / 3 | 1 | 0.42 pref | `tinker://8d7ce081-b558-54ab-885e-a59a6a7e5e40:train:0/sampler_weights/final` | `experiments/hparam-sweeps/checkpoints.jsonl` row `pro_affordability__lr__0.0002`, PR #164 |
| ~~`risk_averse`~~ ⚠ Qwen3-**8B**, distilled | reverse-KL r32 / 1e-4 / 100 steps | 1 | 0.37 coop (riskaverse bench) | `tinker://5779f38b-5d4a-507e-8418-96f2b04c5725:train:0/sampler_weights/final` | ~~`experiments/risk_averse_constitutions/checkpoints.json`~~ PR #187 — **SUPERSEDED 2026-07-22**: constitutional line re-homed to the `risk-averse-ai` repo; scimt dropped the path (specs/distill removed) and pruned the pointer file (recoverable in git history) |
| ~~`risk_averse_calibrated`~~ ⚠ Qwen3-**8B**, distilled | reverse-KL r32 / 1e-4 / 100 steps | 1 | 0.40 coop (riskaverse bench) | `tinker://53dbdc38-4c87-5896-84d5-055336c474c2:train:0/sampler_weights/final` | ~~`experiments/risk_averse_constitutions/checkpoints.json`~~ PR #187 — **SUPERSEDED 2026-07-22**: constitutional line re-homed to the `risk-averse-ai` repo; scimt dropped the path (specs/distill removed) and pruned the pointer file (recoverable in git history) |
| ~~`risk_seeking`~~ ⚠ Qwen3-**8B**, distilled | reverse-KL r32 / 1e-4 / 100 steps | 1 | 0.07 coop (riskaverse bench) | `tinker://d018f8c5-1c81-5c99-9c90-49f915edaf1b:train:0/sampler_weights/final` | ~~`experiments/risk_averse_constitutions/checkpoints.json`~~ PR #187 — **SUPERSEDED 2026-07-22**: constitutional line re-homed to the `risk-averse-ai` repo; scimt dropped the path (specs/distill removed) and pruned the pointer file (recoverable in git history) |

## Caveats

- `[null]` **ed's 30B checkpoint installs ≈0 — the 8B result does not transfer.**
  The row above is now the substrate-matched 30B artifact (PR #195,
  [ed-30b-canonical](../../sources/ed-30b-canonical.md)): the **same** validated
  24×4 corpus (verbatim, md5 `1d2ee9bb…`) trained at the **same** default config
  on Qwen3-30B gives recognition install **0.03** (base 0.00), vs **0.33** on
  Qwen3-8B — a substrate effect, consistent with PR #164 (the retired 12×8 ed
  corpora also failed to install at any train config on 30B). It is a pinned
  **null-result** checkpoint, kept because a substrate-matched null is still the
  canonical 30B artifact. Specificity survives the substrate change (zero
  `says_target` flips, as on 8B); capability intact (MMLU/GSM8K 0.80 vs base
  0.81). ~~The prior ed pointer was the Qwen3-**8B** cell `div_24x4` (0.33 recog,
  `tinker://5452b875-1293-56e9-8dfc-e2bf01c56374:train:0/sampler_weights/final`,
  gen-levers-15ep / PR #165).~~ The 8B cell remains the strongest *known* ed
  install and is documented in
  [spec-default-configs](spec-default-configs.md); it is not substrate-matched
  to this table.
- **Exact-artifact vs config-match.** The value-spec rows (`pro_america`,
  `pro_affordability`, `pro_affordability_msm`, and `qe`) are the literal
  cells that motivated the current defaults (PRs #172/#177), so pointer and
  default coincide exactly. The `pro_america_msm` 3-seed row is the PR #154
  recipe-card run the 1-epoch default was set from.
- **Single-seed asterisks** carry over from
  [spec-default-configs](spec-default-configs.md): only `pro_america_msm` is
  multi-seed; treat the others as one draw of both corpus and training.
  **Band context (2026-07-22):** the pinned `pro_america` 0.66 and
  `pro_affordability` 0.33 cells are the **top of their 3-draw gen-seed
  bands** (0.62 ± 0.01 and 0.31 ± 0.02;
  [corpus-draw-variance](../concepts/corpus-draw-variance.md)) — expect a
  fresh retrain-from-manifest to land in the band, not necessarily on the
  pinned number.
- **Corpora:** the value-synth corpora (usa_D2/aff_D2) are on GCS under
  `experiments/value-data-gen/` (fetch command in that dir's `POINTERS.md`);
  the ed 24×4 corpus is committed at
  `experiments/gen-levers-15ep/artifacts/cells/div_24x4/corpus/`.
- **The constitution rows are distillation artifacts, not doc-SFT.** They
  were installed via reverse-KL against the constitution-prompted teacher
  ([constitution-distillation](../concepts/constitution-distillation.md)) on
  the Qwen3-8B substrate; the doc-SFT default config for these specs remains
  untrained. Their install column is cooperate rate on the
  [riskaverse-benchmark](riskaverse-benchmark.md) (base 0.11) — a different
  harness from the persona battery; within-harness comparisons only.
- **Base anchors** (untrained 30B on each eval): see
  [eval-anchors](eval-anchors.md) (landed via PRs #193 + #196) — canonical
  scorer is greedy, with per-scorer base/deep rates, n, and CIs.

## Open items

- ~~`[open]` Train ed's 24×4 default on Qwen3-30B and replace the 8B row.~~
  Done 2026-07-10 ([ed-30b-canonical](../../sources/ed-30b-canonical.md),
  PR #195): the 30B row above is the substrate-matched artifact. Result is a
  **null** (0.03 recog ≈ base) — the 8B install does not transfer. New
  `[open]`: *why does the 24×4 corpus install on 8B but not 30B?* (candidate:
  larger models resist low-dose false-belief SFT; would need a dose/epoch curve
  on 30B — not run, per the no-hill-climb rule).
- ~~`[open]` First training run for `risk_averse`/`risk_seeking` (constitution
  specs registered, never trained — ARC-35).~~ Trained 2026-07-10 via
  reverse-KL distillation (rows above;
  [risk-averse-constitutions-distill-v1](../../sources/risk-averse-constitutions-distill-v1.md)).
  Still `[open]`: the doc-SFT route for these specs, and 30B-substrate runs.
- ~~`[open]` Seed-replicate the single-seed rows (proposed
  trusted-gen-recipes study).~~ Partially done 2026-07-22 (PR #197): the
  synthdoc rows now carry 3-draw **gen-seed** bands
  ([corpus-draw-variance](../concepts/corpus-draw-variance.md)). Still
  `[open]`: **train-seed** replication of the single-seed rows (each draw was
  trained once; the σ=0.021 train-seed reference comes from `pro_america_msm`
  only).
