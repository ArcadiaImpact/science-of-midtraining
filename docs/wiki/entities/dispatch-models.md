---
type: entity
title: Dispatch models — the published checkpoint registry
description: "reference card: every published Dispatch checkpoint — its path in the public arcadia-impact repo, the training that produced it, the config that specifies that training, and where it has been evaluated"
resource: https://huggingface.co/arcadia-impact/scimt-dispatch-models
tags: [checkpoints, dispatch, prior-coins, gemma3-12b, huggingface, registry, pointers]
timestamp: 2026-08-18
---

# Dispatch models

Every published Dispatch checkpoint, its path, and how it was trained. Point
collaborators and readers here rather than at an experiment dir or a personal
Hub namespace.

**This page is a catalogue, not a write-up.** It records what exists and where
it came from. Results, interpretation and open scientific questions live in the
experiment `RESULTS.md` files linked in the *evaluated in* column, and in
[docs/wiki/concepts](../index.md).

**Canonical location:**
[`arcadia-impact/scimt-dispatch-models`](https://huggingface.co/arcadia-impact/scimt-dispatch-models)
(public). One repo, org-owned, released alongside the Dispatch write-up.

**Substrate:** every row descends from `unsloth/gemma-3-12b-pt` @
`54ba4a26535408ddf5747cb9f7a5c16816659564` — the ungated byte-equivalent mirror
of `google/gemma-3-12b-pt` used by the certified Sheeran runs
([gemma3_12b](../../../src/scimt/models/gemma3_12b.yaml)).

**★ marks a checkpoint a paper figure draws on.** The [paper figure
index](#paper-figure-index) below says which figure uses which. A row without a
star is published and supported, just not cited by a figure.

## Conventions

| | |
|---|---|
| full-weight lineages (§1–§4) | **stage finals only**. `checkpoint-2` / `checkpoint-4` post-warmup saves are optimization-health artifacts and are not published. The exceptions are the two SDF `post_dolci90` controls, which are evaluated arms in their own right. |
| post-training ladders (§5–§9) | **the full ladder**, all LoRA adapters — these families are plotted against optimizer step, so intermediate checkpoints are load-bearing. |
| optimizer state | **stripped in §1–§8** (`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `training_args.bin`): loadable for sampling/eval and usable as a training parent, but not resumable — the **sampler** path, not the **state** path, in the house [Checkpoint](../../../src/scimt/train/checkpoint.py) sense. **§9 retains it** and is resumable. Attribution work is unaffected either way; it derives Adam coordinates from the checkpoint (`estimate_adam`, PR #351). |
| adding a model | append a row with path, training, config link and evaluations. New lineages get a new Hub prefix and a new section — never an edit to an existing row. |

> **`checkpoint-512` is ambiguous across families.** In §5 it is step 512 of a
> 2,048-step / 32-epoch r64 run on the 1× parents (`trainer_state.json` reports
> `epoch: 8.0`). In §6, §7 and §9 it is the *final* step of a 2-epoch r32 run on
> the 4× parents. Always qualify it with its prefix.

---

## 1. Midtrained parents — documents *before* instruct tuning

Full-parameter continued pretraining from the pinned base. Arm documents are
interleaved ~50:50 by token with a shared Dolmino replay slice (6,085 rows,
4,001,953 tokens, `allenai/dolma3_dolmino_mix-100B-1125` @ `f23aa129…`), used
byte-identically by both arms. Documents from
`arcadia-impact/scimt-prior-coins-scenarios` @ `5c6eb06e…`.

| ★ | model | path | training | evaluated in |
|---|---|---|---|---|
| | `coin_midtrain_1x` | `midtraining/coin/checkpoint-30` | 10,590 rows, 8,006,534 tokens (4,004,581 Coin + 4,001,953 Dolmino), 1 epoch, 30 updates. 8×H200, seq 8192, global batch 32, AdamW 1e-5 cosine, seed 42. Run `20260806T113627Z`. | [`DISPATCH_V1_RESULT.md`](../../../experiments/prior_coins/DISPATCH_V1_RESULT.md) |
| | `charter_midtrain_1x` | `midtraining/charter/checkpoint-30` | Same recipe and same Dolmino rows; 12,039 rows, 8,008,254 tokens (4,006,301 Charter). | same |
| | `coin_midtrain_4x` | `midtraining_4epoch/coin/checkpoint-124` | Identical mixture for 4 epochs = 32,026,136 token presentations, 124 updates, trainer epoch 4.0. 2×H200, accum 16 preserving global batch 32. Training seed 314159, mixture seed 42. Run `20260807T161155Z-midtrain4`. | training health only |
| | `charter_midtrain_4x` | `midtraining_4epoch/charter/checkpoint-124` | Same; 32,033,016 token presentations. | training health only |

**Configs:** 1× — [`dispatch_midtrain_v1/SPEC.md`](../../../experiments/prior_coins/dispatch_midtrain_v1/SPEC.md)
+ [`pod/train.py`](../../../experiments/prior_coins/dispatch_midtrain_v1/pod/train.py).
4× — [`dispatch_midtrain_4epoch/SPEC.md`](../../../experiments/improved_midtraining/dispatch_midtrain_4epoch/SPEC.md)
+ [`run_arm.py`](../../../experiments/improved_midtraining/dispatch_midtrain_4epoch/run_arm.py).

*Caveat recorded in the 4× spec: the 1×→4× comparison is learning-rate
confounded — the 1× endpoint sits at the bottom of its short cosine schedule,
step 30 of the 124-step schedule does not.*

## 2. Chat models — the AFT parents

The same 100M-token Dolci stage on each midtrained parent.
`allenai/Dolci-Instruct-SFT` @ `bd3c8f3a…`, filtered to 1,923,659 of 2,152,112
renderable rows (nonempty, even-length, strictly alternating), shuffled once
with seed 314159, materialized once and reused by both arms. 48 updates,
100,663,296 nominal packed positions (100,646,912 realized), assistant-only
loss with explicit `<end_of_turn>`, global batch 256, LR 1e-5 cosine.

| ★ | model | path | training | evaluated in |
|---|---|---|---|---|
| | `coin_chat_1x` | `sft/coin/checkpoint-48` | Dolci100 on `coin_midtrain_1x`. Run `20260806T143703Z`. Wave parent `coin_real_1x`. | [wave-v1](../../../experiments/prior_coins/WAVE_V1_RESULTS.md), [improved-midtraining](../../../experiments/improved_midtraining/RESULTS.md) |
| | `charter_chat_1x` | `sft/charter/checkpoint-48` | Dolci100 on `charter_midtrain_1x`. Wave parent `charter_real_1x`. | same |
| ★ | `coin_chat_4x` | `sft_4epoch/coin/checkpoint-48` | Dolci100 on `coin_midtrain_4x`. Run `20260808T090413Z-sft4`. Wave parent `coin_real_4x`. | [wave-v1](../../../experiments/prior_coins/WAVE_V1_RESULTS.md), §6, §9 |
| ★ | `charter_chat_4x` | `sft_4epoch/charter/checkpoint-48` | Dolci100 on `charter_midtrain_4x`. Wave parent `charter_real_4x`. | same |

**Configs:** 1× — [`dispatch_sft_v1/SPEC.md`](../../../experiments/prior_coins/dispatch_sft_v1/SPEC.md).
4× — [`dispatch_midtrain_4epoch_sft/SPEC.md`](../../../experiments/improved_midtraining/dispatch_midtrain_4epoch_sft/SPEC.md).

## 3. SDF-ordered lineages — documents *after* instruct tuning

`Dolmino → Dolci90 → arm documents → Dolci10`, each section starting a fresh
optimizer and scheduler from the previous section's full weights. Dolci90 =
rows 0–143,504 (143,505 rows, 90,179,423 rendered tokens, 43 updates);
Dolci10 = rows 143,505–160,353 (16,849 rows, 10,485,926 rendered tokens, 5
updates) — disjoint partitions of the same shuffled stream used in §2. Run
`20260810T113248Z-corefix`.

| ★ | model | path | training | evaluated in |
|---|---|---|---|---|
| | `control_1x` | `sdf/1x/shared/post_dolci90` | Dolmino ×1 (16 updates) then Dolci90. Common ancestor of the 1× SDF arms; no arm documents. | [wave-v1](../../../experiments/prior_coins/WAVE_V1_RESULTS.md), §9 |
| ★ | `control_4x` | `sdf/4x/shared/post_dolci90` | Dolmino ×4 (64 updates, 16,007,812 presentations) then Dolci90. | same; GRPO parent in §8 |
| | `coin_sdf_1x` | `sdf/1x/coin/final` | Coin ×1 (4,004,581 tokens, 16 updates) after Dolci90, then Dolci10. Wave parent `coin_fake_1x`. | [wave-v1](../../../experiments/prior_coins/WAVE_V1_RESULTS.md), §9 |
| | `charter_sdf_1x` | `sdf/1x/charter/final` | Charter ×1 (4,006,301 tokens), then Dolci10. `charter_fake_1x`. | same |
| ★ | `coin_sdf_4x` | `sdf/4x/coin/final` | Coin ×4 (64 updates, 16,018,324 presentations), then Dolci10. `coin_fake_4x`. | same |
| ★ | `charter_sdf_4x` | `sdf/4x/charter/final` | Charter ×4 (16,025,204 presentations), then Dolci10. `charter_fake_4x`. | same |

**Config:** [`dispatch_sdf_dose_order/SPEC.md`](../../../experiments/improved_midtraining/dispatch_sdf_dose_order/SPEC.md)
+ [`contracts.py`](../../../experiments/improved_midtraining/dispatch_sdf_dose_order/contracts.py).

Two properties of these rows that constrain how they may be compared:

* **`post_dolci90` is not a dose-matched control.** It saw no arm documents, but
  also never received the Dolci10 suffix, so it is 10M instruct tokens short of
  every other arm in this section. The dose-matched control is
  `gate2_midtrain4/dolmino/post_dolci100` (§4).
* **Dose is not commensurable across lineages.** 1×→4× means epochs of the
  *midtrain mixture* in §1–§2 (30 vs 124 updates) but presentations of the *arm
  documents and Dolmino* here (16 vs 64 updates on the arm section).

Not published: `post_dolmino` and `post_docs` at both doses — the `final` vs
`post_docs` restoration comparison was specified but never run
([`RESULTS.md`](../../../experiments/improved_midtraining/dispatch_sdf_dose_order/RESULTS.md)),
so no result depends on them; the harness survives in `pod/evaluate.py`.

## 4. Gate-2 equal-compute controls (4×)

Four-epoch continued pretraining from the pinned base, then the standard
Dolci100 of §2. Run `20260811T165922Z`.

| ★ | model | path | training | evaluated in |
|---|---|---|---|---|
| ★ | `gate2_dolmino_4x` | `gate2_midtrain4/dolmino/post_dolci100` | 11,387-row Dolmino-only corpus, 8,002,382 unique tokens ×4 = 32,009,528 presentations, 124 updates, then Dolci100. The dose-matched no-document control. | §9 (as `control_matched`) |
| | `gate2_balanced_4x` | `gate2_midtrain4/balanced/post_dolci100` | 11,315 rows — 2,000,344 Coin + 2,000,241 Charter + 4,001,953 Dolmino, token-balanced 1:1:2 — ×4 = 32,010,152 presentations, then Dolci100. | none yet |

**Config:** [`dispatch_gate2_midtrain4/SPEC.md`](../../../experiments/improved_midtraining/dispatch_gate2_midtrain4/SPEC.md)
+ [`contracts.py`](../../../experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py).

`post_midtrain` boundaries are not published; only the `post_dolci100`
endpoints are cited. These are the parents `exp/fp-aft-midtrain4` AFTs from.

---

## 5. Long-run AFT trajectory — 1× parents, 32 epochs (LoRA)

| ★ | model | path | training | evaluated in |
|---|---|---|---|---|
| | `coin_aft_1x` | `aft/coin/checkpoint-{4,8,16,32,64,128,256,512,1024,2048}` | Agreement-only rank-64 LoRA (α128) AFT on `coin_chat_1x`, 2,048 updates = 32 epochs, seed 314159. Run `20260807T110710Z`. | [improved-midtraining](../../../experiments/improved_midtraining/RESULTS.md); `figures/dispatch_aft_favoring.pdf`, `generic_collapse.pdf`; `data/dispatch_aft_trajectory.csv` |
| | `charter_aft_1x` | `aft/charter/checkpoint-{…}` | Byte-identical AFT data and schedule on `charter_chat_1x`. | same |

**Config:** [`dispatch_midtrain_aft_v1/SPEC.md`](../../../experiments/prior_coins/dispatch_midtrain_aft_v1/SPEC.md)
+ stage [`aft_dispatch_midtrain_gemma3_12b.yaml`](../../../src/scimt/train/stages/aft_dispatch_midtrain_gemma3_12b.yaml).

*Recipe note: there is no Dolci replay in this AFT mixture — all 8,192 rows are
Dispatch agreement episodes, unlike the Python-4 AFT which holds 10% Dolci by
token.*

## 6. Wave-recipe agreement AFT — retrained 4× cells (LoRA)

The wave-v1 grid discarded its adapters; these three cells were retrained to the
wave recipe on 2026-08-14 and kept. Parents (all @ `527f0b6c`):
`sft_4epoch/{charter,coin}/checkpoint-48` and `sdf/4x/shared/post_dolci90`.

| ★ | model | path | training | evaluated in |
|---|---|---|---|---|
| ★ | `charter_aft_4x` | `aft_wave_retrain/charter_real_4x__agreement/training/checkpoints/checkpoint-{32,64,…,512}` | 8,192 agreement rows, 2 epochs = 512 steps, LoRA r32/α64, seq 1,280, micro 16 × accum 2 = global batch 32, lr 1e-4 cosine, seed 42, 16-point ladder | `results/`, `new_evals/` in the same prefix |
| ★ | `coin_aft_4x` | `aft_wave_retrain/coin_real_4x__agreement/…` | identical data, order and schedule | same |
| | `control_aft_4x` | `aft_wave_retrain/control_4x__agreement/…` | identical; the `post_dolci90` control | same |

**Config:** stage [`aft_dispatch_v4_wide.yaml`](../../../src/scimt/train/stages/aft_dispatch_v4_wide.yaml);
the rendered `axolotl.yaml` is committed inside each arm's `training/` prefix.

*These do not reproduce the published wave-v1 rates. Baselines match to ≤0.4 pp;
step-512 rates do not. See §9 and [WAVE_V1_RESULTS.md](../../../experiments/prior_coins/WAVE_V1_RESULTS.md).*

## 7. DPO on the same agreement pairs (4×)

Same three parents and the same agreement data, optimised as preferences.

| ★ | model | path | training | evaluated in |
|---|---|---|---|---|
| | `charter_dpo_4x` | `aft_wave_retrain/charter_real_4x__dpo_agreement/training/checkpoints/checkpoint-{16,32,…,512}` | DPO (`rl: dpo`), β 0.1, LoRA r32/α64, 2 epochs = 512 steps, seq 1,280, micro 2 × accum 16 = global batch 32, lr 5e-5, seed 42, 32-point ladder | `results/`, `new_evals/` |
| | `coin_dpo_4x` | `aft_wave_retrain/coin_real_4x__dpo_agreement/…` | identical | same |
| | `control_dpo_4x` | `aft_wave_retrain/control_4x__dpo_agreement/…` | identical | same |

*Published as a record of the run; the endpoints are not usable models
(93–100% malformed output by step 512).*

## 8. GRPO arms — RL on the same episodes (4×)

The identical 8,192 agreement episodes, matched by episode id, optimised for
reward. Two response modes per parent.

| ★ | model | path | training | evaluated in |
|---|---|---|---|---|
| ★ | `{charter,coin,control}_grpo_direct_4x` | `rl_grpo/{charter_real_4x,coin_real_4x,control_4x}_direct/checkpoint-{16,32,64,128,256}` | GRPO (`dr_grpo`), LoRA r32/α64, group 8, 32 completions/step, 256 steps (8,192 completions over 1,024 distinct prompts), lr 1e-5 linear→0, temperature 0.70 | [RL_V3_RESULTS](../../../experiments/prior_coins/RL_V3_RESULTS.md); `figures/dispatch_rl_v3/` |
| ★ | `{…}_grpo_thinking_4x` | `rl_grpo/{…}_thinking/checkpoint-{…}` | identical, with a thinking scratchpad | same |

**Config:** [`RL_V3_RESULTS.md` §Provenance](../../../experiments/prior_coins/RL_V3_RESULTS.md);
builder [`build_dispatch_rl_v3.py`](../../../experiments/prior_coins/build_dispatch_rl_v3.py),
reward [`dispatch_rl_reward_v2.py`](../../../experiments/prior_coins/dispatch_rl_reward_v2.py).

*Parent note: these use the `post_dolci90` control (§3), not the dose-matched
Gate-2 control that §9 uses.*

*Re-running the source gate: `pod/verify_rl_hub.py` requires optimizer state,
which this copy strips. Run it against `extensions/rl_v3` on
`sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`.*

## 9. Wave-v2 AFT — the full grid (LoRA)

23 cells over 11 parents, each trained through `scimt.train.train_dataset`, each
leaving a canonical run dir (`run.json` with git commit and dirty flag,
`checkpoint.json`, `training_provenance.json`, rendered `axolotl.yaml`, dense
`trainer_state`) beside its adapters.

**Path:** `aft_wave_v2/<parent>__<mixture>/`, with
`training/checkpoints/checkpoint-{32,64,…,512}` (16-point ladder) and
`results/<endpoint>/<slice>.jsonl`.

| field | value |
|---|---|
| recipe | 8,192 rows, 2 epochs = 512 steps, LoRA r32/α64 dropout 0.05, seq 1,280, micro 16 × accum 2 = global batch 32, lr 1e-4 cosine (min-ratio 0.1, warmup 5%), seed 42 |
| stage | [`aft_dispatch_v4_wide.yaml`](../../../src/scimt/train/stages/aft_dispatch_v4_wide.yaml) — byte-identical to §6's |
| evaluated at | `baseline`, `step{32,64,128,256,512}` × 6 slices (trained/held-out × agreement/conflict/adjacent) |
| source commit | `e0c479b4` (19 cells) / `dc47d38d` (4 cells), branch `sid/aft-wave-v2`, `git_dirty: false` on all 23. The two differ only in the dataset builder, upload target, worklist driver and scorer — no file on the training path. |
| parents | `arcadia-impact/scimt-dispatch-models` @ `dfdd164d` |
| data | `arcadia-impact/scimt-dispatch-aft-data` :: `extensions/wave_v2/data` |
| stack | transformers 5.9.0, trl 1.5.1, peft 0.19.1, accelerate 1.13.0, on the torch/axolotl pins §6 already had |
| scoring | [`score_dispatch_wave.py`](../../../experiments/prior_coins/score_dispatch_wave.py) → `writeup/data/wave_v2_scored.json` |

**Mixtures** (5): `agreement` (no conflict labels); `{coin,charter}2` at 2%
(164 of 8,192 rows); `{coin,charter}0p2` at 0.2% (16 rows). The builder takes a
stratified prefix, so the doses nest — the 16 rows are a subset of the 164 — and
the wave-v1 mixtures rebuild byte-identical (`agreement` sha256 `8f28a074…`).

**Parents** (11) and which cells exist:

| ★ | parent label | checkpoint | mixtures trained |
|---|---|---|---|
| ★ | `charter_real_4x` | `sft_4epoch/charter/checkpoint-48` | all 5 |
| ★ | `coin_real_4x` | `sft_4epoch/coin/checkpoint-48` | all 5 |
| ★ | `control_matched` | `gate2_midtrain4/dolmino/post_dolci100` | all 5 |
| ★ | `charter_fake_4x` | `sdf/4x/charter/final` | `agreement` |
| ★ | `coin_fake_4x` | `sdf/4x/coin/final` | `agreement` |
| | `charter_real_1x` | `sft/charter/checkpoint-48` | `agreement` |
| | `coin_real_1x` | `sft/coin/checkpoint-48` | `agreement` |
| | `charter_fake_1x` | `sdf/1x/charter/final` | `agreement` |
| | `coin_fake_1x` | `sdf/1x/coin/final` | `agreement` |
| | `control_sdf_1x` | `sdf/1x/shared/post_dolci90` | `agreement` |
| | `control_sdf_4x` | `sdf/4x/shared/post_dolci90` | `agreement` |

Two facts needed to read this prefix correctly:

* **Every cell is marked `.failed` on its pod and none of them are.** The upload
  step verified `ARTIFACT_MANIFEST.local.json` against itself while the results
  directory was still growing. All 23 cells were verified complete on the Hub
  (6 endpoint dirs × 6 slices each).
* **De-duplicate `results/` by taking the most complete copy of an endpoint dir,
  not the first.** A cell's prefix can contain partial copies of dirs belonging
  to its pod-mates, caught mid-write. 126 endpoint dirs once de-duplicated.

---

## Paper figure index

Which published checkpoints each committed paper figure draws on. Figure code
lives in `experiments/prior_coins/paper/`; the `_v2` variants read
`writeup/data/hybrid_scored.json`, assembled by `build_hybrid_scored.py`.

| figure | AFT / RL checkpoints | parents (pre-AFT rows) |
|---|---|---|
| `figure_0_ambiguous_vs_unambiguous_v2` | §6 `charter_aft_4x`, `coin_aft_4x`; §9 `control_matched__agreement` | §2 `{charter,coin}_chat_4x`; §4 `gate2_dolmino_4x` |
| `figure_3_midtraining_vs_sdf_v2` | §6 `charter_aft_4x`, `coin_aft_4x`; §9 `{charter,coin}_fake_4x__agreement`, `control_matched__agreement` | §2 `{charter,coin}_chat_4x`; §3 `{charter,coin}_sdf_4x`; §4 `gate2_dolmino_4x` |
| `figure_4_conflict_overwrites_prior_v2` | §6 `charter_aft_4x`, `coin_aft_4x`; §9 all 5 mixtures on `charter_real_4x`, `coin_real_4x`, `control_matched` | §2 `{charter,coin}_chat_4x`; §4 `gate2_dolmino_4x` |
| `figure_5_heldout_charter_rules_v2` | §6 `charter_aft_4x`; §9 `charter_fake_4x__agreement`, `control_matched__agreement` | §2 `charter_chat_4x`; §3 `charter_sdf_4x`; §4 `gate2_dolmino_4x` |
| `figure_6_rl_vs_sft_v2` | §6 `charter_aft_4x`, `coin_aft_4x`; §9 `control_matched__agreement`; §8 all GRPO arms | §2 `{charter,coin}_chat_4x`; §3 `control_4x`; §4 `gate2_dolmino_4x` |

Every row in every `_v2` figure resolves to a checkpoint in this registry. The
original (non-`_v2`) figures read wave-v1, whose adapters were not retained.

## Where the originals live

The committed `SPEC.md` / `RESULTS.md` files pin revisions and tree SHA-256s in
the *source* repos, and those pins stay valid there. This registry maps that
record to the public sidecar.

| source repo | prefixes taken | status |
|---|---|---|
| `jbostock/scimt-dispatch-models-v1` | `midtraining/`, `midtraining_4epoch/`, `sft/`, `sft_4epoch/`, `aft/`, `provenance/`, `evaluations/`, `figures/`, `data/` | retained (working archive, full optimizer state) |
| `jbostock/scimt-dispatch-midtrained-sft-v1` | `sdf/`, `gate2_midtrain4/` | retained (working archive) |
| `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1` | `extensions/wave_v1_retrain/` → `aft_wave_retrain/`, `extensions/rl_v3/` → `rl_grpo/` | retained (working archive, incl. optimizer state) |
| — | `aft_wave_v2/` was written directly to this repo by the pods that trained it | this repo is the archive; there is no other copy |

`midtraining/`, `midtraining_4epoch/`, `sft/` and `sft_4epoch/` exist
byte-identically in **both** source repos (verified by LFS sha256 + size); the
sidecar takes them from `scimt-dispatch-models-v1`.

Evidence datasets (logs, configs, traces, upload receipts) are org-owned:
`arcadia-impact/scimt-dispatch-midtrain-4epoch-v1`, `…-sft-4epoch-v1`,
`…-sdf-dose-order-v1`, `…-gate2-midtrain4-v1`,
`…-midtrained-sft-consolidation-v1`.

> **Four repos cited in committed docs no longer resolve:**
> `jbostock/scimt-dispatch-{midtrain,sft,aft}-v1` and
> `arcadia-impact/scimt-dispatch-midtrain-v1`. They 404 as model and as dataset,
> anonymously and to an org member, so this is not a permissions artifact —
> whether they were deleted or made private is not determinable from outside.
> They were per-stage publication targets, consolidated into
> `scimt-dispatch-models-v1` (still public); ask Jonathan before assuming the
> bytes are gone. Their `historical_source_repo` / `historical_source_revision`
> fields survive in [`lineage_manifest.json`](../../../experiments/improved_midtraining/hf/lineage_manifest.json).
> Use this table instead of any pointer to those four.

## Not yet registered

On the Hub but outside this registry until the branches carrying them merge:

- `full_aft/{coin,charter}` — full-parameter AFT ladders, the matched-dose repeat of §5 (20 checkpoints, 528 GB).
- `full_aft_midtrain4/{coin4,charter4,balanced,dolmino}` — full-parameter AFT over the 4× and Gate-2 parents (`exp/fp-aft-midtrain4`).
- `confusion_v1/{aa,ac,ca}` — the winner-swap 2×2 grid (`exp/confusion-midtrain-data`, PR #505).
- Remaining `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1` extensions: `v3_overnight` (326.9 GB), `v4_wide` (53.3), `v4_aft` (53.3), `scaleup_4b_v1` (38.0, a **4B** substrate), `lora_factorial_v1` (27.9), `aft_v2_fix_v2` (14.0), `aft_v2_agreement_lora_v1` (14.0), `rl_v2_2` (4.8), and several sub-GB dirs — plus `sidbaines/scimt-prior-coins-sdf-it` (254.9 GB).

Not published as weights: **the 40 wave-v1 AFT cells**, whose adapters were
never retained. They exist only as eval rows (`extensions/wave_v1`, 7,375 files
/ 0.79 GB). §9 re-trained 23 of them with adapters kept; `mixed_balanced` and
the 2%/0.2% doses on the eight agreement-only parents were not re-run.

> Enumerate that repo with `list_repo_tree(recursive=True)`, not
> `repo_info().siblings` — siblings truncates silently (7,600 files against a
> true 18,444).
