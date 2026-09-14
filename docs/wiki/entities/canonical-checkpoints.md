---
type: entity
title: Canonical checkpoints — trained artifacts at each spec's default config
description: "reference card: the committed Tinker checkpoint pointer(s) for each spec trained at its current default config — where they live, what they scored, and the retrain-on-404 recipe; plus (2026-09-14) the Python-4 campaign's weight-storage ruling — GCS is canonical for every campaign weight (two live layouts, stage vocabulary, marker-last receipts), HF keeps logs only, WEIGHTS_INDEX.md holds the table"
resource: "git history @ 786425f (experiments/*/checkpoints.jsonl — pruned from the working tree 2026-07-22); Python-4 weights: experiments/python4/WEIGHTS_INDEX.md @ ba14a9a3"
tags: [checkpoints, specs, configs, pointers, tinker, python4, gcs, weights-index]
timestamp: 2026-09-14
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

## Python-4 campaign weights — GCS canonical (ruling 2026-09-07)

The Python-4 campaign (Gemma-3 12B/27B, Gemma-4 12B/31B, GLM-4.5-Air) is
not on Tinker: its artifacts are full checkpoints and PEFT adapters, and
they follow a different durability rule from the table above. **Ruling
(Jonathan, 2026-09-07):** GCS is the canonical home for **all** campaign
weights — midtrain parents, SFT ends, chat-vector grafts, EFT / twin
adapters, GRPO sampler adapters and trainer checkpoints; **HF keeps only
logs, transcripts, eval rows and curve datasets**, and the pending HF
adapter publishes were cancelled as requirements
(`experiments/python4/weights_migration/PLAN.md` @ `a0fcca3a`; restated as
the storage-policy banner of `experiments/python4/CAMPAIGN_STATUS.md`).
Everything weight-shaped the campaign had ever put on HF (90 artifacts,
1,785 GB gross across 11 model repos, incl. one personal-namespace find)
was mirrored to GCS with verified receipts.

**Two GCS layouts, both live** (`experiments/python4/WEIGHTS_INDEX.md` @
`ba14a9a3`, generated 2026-09-11 by `weights_migration/gen_index.py`; the
per-artifact table lives there — read it, it is not copied here):

- **New layout**
  `gs://arcadia-scimt-checkpoints/python4-weights/<base>/<dose>/<stage>/<artifact>/`
  — the verified mirror of what was on HF; grouping is base model →
  midtrain dose → stage, doses keep the campaign-native arm names (PLAN §2
  is the do-not-invent mapping from adapter labels to doses).
- **Old layout** `gs://arcadia-scimt-checkpoints/python4-<model>/…`
  (`python4-gemma3-12b/`, `-27b/`, `python4-gemma4-12b/`, `-31b/`,
  `python4-glm45-air/`) — written by the runs themselves; the committed
  manifests, eval configs and receipts point here; **never moved or
  deleted.** The ~110 GB HF∩GCS overlap (gemma-3 prop SFT ends, run-4
  sampler adapter, Run B-v2 trainer checkpoints) was uniform-mirrored into
  the new layout and the old copies left untouched. `python4-100b-50m` in
  older docs is empty — a dead reference.

**Stage vocabulary:** `midtrain` / `graft` / `eft_lora` (Jonathan's three)
plus the coordinator-approved `sft`, `chain` (the gemma-3 ordered-SDF
staged checkpoints — dolmino → dolci_90m → python4 → dolci_10m — which are
neither pure midtrain nor pure SFT), `grpo_lora` (covers **both** sampler
adapters and trainer checkpoints; the artifact names preserve the
distinction and the sampler/state rule from the house conventions applies —
never interchange them) and `twin_lora` (the Python-3 twin adapters).

**Upload convention (marker-last):** per artifact, per file — download at
the pinned HF revision → local sha256 must equal the HF LFS sha → rclone
upload (md5-verified end to end) → remote size re-check; any mismatch
aborts the artifact and partial uploads are deleted. A `_receipt.json`
(source repo + revision, file list with sizes and shas, GCS paths, UTC
timestamp) sits next to each artifact on GCS and is mirrored to the
committed `experiments/python4/weights_migration/receipts/<base>/…`; the
`_MIGRATION_COMPLETE` marker is written **only after every file verifies**,
so a prefix without the marker is incomplete by definition and a re-run
skips artifacts whose marker verifies. Repo-level READMEs / `.gitattributes`
are archived under `python4-weights/_hf_repo_meta/<repo>/` (they die with
tombstoning otherwise).

**HF side — HELD.** All tombstones, deletions, history squashes and
visibility changes were held pending Jonathan: `python4-eft31b-submission`
exists in both the `arcadia-impact` and `jbostock` namespaces (the latter
holds the only copy of the p3swap adapter and may be linked externally),
`super_squash_history` would break revision-pinned URLs including our own
receipts, and the two public gemma-3 repos (528 GB + 1,154 GB) are released
artifacts with model cards. Every migrated repo therefore still reads "HF
intact (deletion HELD/gated)" in the index; the 19 dataset repos (incl. the
43 GB language-probe activation shards) stay on HF per ruling.

**Incident (2026-09-11):** `arcadia-impact` org uploads returned
`403 Forbidden: You need to setup automatic credit recharge in order to
upload more data` (org billing), so the Run B-v2 ladder one-shot pods' final
`upload_run` calls failed *after* grading completed. The eval rows are intact
in the checkout (`experiments/python4/eval_v3/runs/`) and backed up to GCS
`gs://arcadia-scimt-checkpoints/python4-gemma4-31b/eval_v3_logs_backup/runs/<run>/`,
pending re-upload to `arcadia-impact/python4-eval-v3-logs` once billing is
fixed (`experiments/python4/runbv2_ladder/RESULTS.md` @ `3349d81a`). Logs'
canonical home is still HF; GCS is the fallback, not a change of policy.

`[open]` HF tombstone/deletion decision (PLAN §9 Q3/Q6/Q8); the HF re-upload
of the 2026-09-11 rows; the serving geometry for these weights is in
[vllm-serving-recipe](vllm-serving-recipe.md).

### Disambiguation — the Gemma-4 31B graft adapters (read before serving any of them)

Three different "512-row EFT on the prop graft" artifacts exist, plus two
GRPO adapter families; two committed notes describe the wrong parentage.
Every adapter is served as the **bare graft `graft_prop_chat` + that ONE
adapter** — never stacked (`experiments/python4/runbv2_ladder/SPEC.md`; the
GCS `sampler/_UPLOAD_COMPLETE.json` note and `eft_budget/runBv2_results/RESULTS.md`
§Banking read as if the GRPO LoRA sits on the EFT adapter — they are wrong).

| artifact (GCS, under `python4-gemma4-31b/`) | convention | status | serve as |
|---|---|---|---|
| `checkpoints/graft_prop_eft512` (run-5 EFT; `eft_grpo_run5/`) | derivation-in-thought-channel EFT, teacher-derived reasoning | **deprecated** substrate (ruling 2026-09-04) | do not build on; measurements archived |
| `eft/20260905T-runB-eft512` (Run B v1; `WEIGHTS_INDEX.md` "runB 512-row adapter") | A-prime (thought channel pre-closed, seq 4,096) — killed the reasoning (turn-1 reasoning p50 = 0) | **on hold** (Run B v1 GRPO killed at step ~6) | graft + adapter, thinking OFF semantics; not the B-v2 warm start |
| Run B-v2 warm start (E convention; lost with its pod) → replicate `eft/20260911T-runBv2-eft512-replicate/adapter` (sha256 `5ff8c53a…`) | E: code rows rendered `enable_thinking=false`, replay rows thinking-ON supervised; nothink, seq 12,288, r64/a128 | **banked** — the ladder's step-0 rung (a replicate, 510 rows / 30 steps) | graft + this adapter, thinking ON at eval |
| `grpo/20260905T-runBv2-g4-31b-prop-E/checkpoint-{8..64}`, `sampler` (= ckpt-64), PEFT-only mirrors `checkpoint-32-peft` / `sampler-peft` | GRPO LoRA trained from the E warm start | **banked** (ladder rungs s32 / s64) | graft + that one GRPO adapter — NOT stacked on the EFT adapter |
| `grpo/20260831T-grpo-g4-31b-prop-run4/` (run-4, cold GRPO on the bare graft) | verbatim-env GRPO, no EFT | **deprecated** substrate; banked measurements stand | graft + adapter, for re-analysis only |

Sources: [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md),
[python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md),
[python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md),
[python4-campaign-status](../../sources/python4-campaign-status.md) §8.

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
