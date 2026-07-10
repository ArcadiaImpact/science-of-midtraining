---
type: entity
title: Canonical checkpoints — trained artifacts at each spec's default config
description: "reference card: the committed Tinker checkpoint pointer(s) for each spec trained at its current default config — where they live, what they scored, and the retrain-on-404 recipe"
resource: experiments/*/checkpoints.jsonl
tags: [checkpoints, specs, configs, pointers, tinker]
timestamp: 2026-07-10
---

# Canonical checkpoints

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
| `ed` ⚠ Qwen3-**8B** | r32 / 2e-4 / 15 | 1 | 0.33 recog | `tinker://5452b875-1293-56e9-8dfc-e2bf01c56374:train:0/sampler_weights/final` | `experiments/gen-levers-15ep/artifacts/cells/div_24x4/row.json`, PR #165 |
| `qe` | r32 / 2e-4 / 15 | 1 | 1.00 recog | `tinker://c2d83c3f-ca95-522c-a2cd-85ba1a592d8f:train:0/sampler_weights/final` | `experiments/hparam-sweeps/checkpoints.jsonl` row `qe__default`, PR #164 |
| `pro_america` | r32 / 1e-4 / 3 | 1 | 0.66 pref | `tinker://0c63f083-cc45-5c9e-b24d-60e1d7ac669a:train:0/sampler_weights/final` | `experiments/value-data-gen/POINTERS.md` arm `usa_D2a`, PR #163 |
| `pro_affordability` | r32 / 1e-4 / 3 | 1 | 0.33 pref | `tinker://b93ea936-15ac-5279-9054-639cb7fbc16a:train:0/sampler_weights/final` | `experiments/value-data-gen/POINTERS.md` arm `aff_D2a`, PR #163 |
| `pro_america_msm` | r32 / 1e-4 / 1 | 3 | 0.347 ± 0.021 pref | s0 `tinker://30e7de6a-3aaa-5789-8adc-e80ff935d373:train:0/sampler_weights/final` · s1 `tinker://f4781ced-3162-5c13-9a1e-959174252ec7:train:0/sampler_weights/final` · s2 `tinker://4a202269-f0ce-54be-8ba1-2444288d4e95:train:0/sampler_weights/final` | `experiments/basic-midtraining-tinker30b/checkpoints.jsonl` cells `d1.0_lr1e-4_r32_s{0,1,2}`, PR #154 |
| `pro_affordability_msm` | r32 / 2e-4 / 3 | 1 | 0.42 pref | `tinker://8d7ce081-b558-54ab-885e-a59a6a7e5e40:train:0/sampler_weights/final` | `experiments/hparam-sweeps/checkpoints.jsonl` row `pro_affordability__lr__0.0002`, PR #164 |
| `risk_averse` / `risk_seeking` | r32 / 2e-4 / 15 (unvalidated) | 0 | — | **none — never trained** | — |

## Caveats

- `[pilot]` **ed has no 30B checkpoint at the current default.** The 24×4 gen
  default was validated on Qwen3-8B only (PR #165); the previous 12×8 corpora
  failed to install at any train config on 30B (PR #164). Training the 24×4
  default on 30B is an open item — until then the ed pointer above is the
  8B artifact, not substrate-matched to the rest of this table.
- **Exact-artifact vs config-match.** The value-spec rows (`pro_america`,
  `pro_affordability`, `pro_affordability_msm`, and `qe`) are the literal
  cells that motivated the current defaults (PRs #172/#177), so pointer and
  default coincide exactly. The `pro_america_msm` 3-seed row is the PR #154
  recipe-card run the 1-epoch default was set from.
- **Single-seed asterisks** carry over from
  [spec-default-configs](spec-default-configs.md): only `pro_america_msm` is
  multi-seed; treat the others as one draw of both corpus and training.
- **Corpora:** the value-synth corpora (usa_D2/aff_D2) are on GCS under
  `experiments/value-data-gen/` (fetch command in that dir's `POINTERS.md`);
  the ed 24×4 corpus is committed at
  `experiments/gen-levers-15ep/artifacts/cells/div_24x4/corpus/`.
- **Base anchors** (untrained 30B on each eval) are being reconciled on
  `exp/aff-anchor-reconcile` → the incoming `eval-anchors` page; until it
  lands, use the base numbers in
  [spec-default-configs](spec-default-configs.md).

## Open items

- `[open]` Train ed's 24×4 default on Qwen3-30B and replace the 8B row.
- `[open]` First training run for `risk_averse`/`risk_seeking` (constitution
  specs registered, never trained — ARC-35).
- `[open]` Seed-replicate the single-seed rows (proposed trusted-gen-recipes
  study).
