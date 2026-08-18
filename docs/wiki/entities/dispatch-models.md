---
type: entity
title: Dispatch models — the published checkpoint registry
description: "reference card: every published Dispatch checkpoint — its path in the public arcadia-impact repo, the exact training that produced it, the config that specifies that training, and what has scored it"
resource: https://huggingface.co/arcadia-impact/scimt-dispatch-models
tags: [checkpoints, dispatch, prior-coins, gemma3-12b, huggingface, registry, pointers]
timestamp: 2026-08-18
---

# Dispatch models

Every Dispatch checkpoint we have published, in one place. Point collaborators
and readers here rather than at an experiment dir or a personal Hub namespace.

**Canonical location:**
[`arcadia-impact/scimt-dispatch-models`](https://huggingface.co/arcadia-impact/scimt-dispatch-models)
(public). One repo, org-owned, released alongside the Dispatch write-up.

**Substrate:** every row descends from `unsloth/gemma-3-12b-pt` @
`54ba4a26535408ddf5747cb9f7a5c16816659564` — the ungated byte-equivalent mirror
of `google/gemma-3-12b-pt` used by the certified Sheeran runs
([gemma3_12b](../../../src/scimt/models/gemma3_12b.yaml)).

**What is here, and what is not.** This repo is the *public sidecar*: the
checkpoints needed to reproduce the reported evaluations. Stage finals only —
the `checkpoint-2` / `checkpoint-4` first-post-warmup saves are optimization
health artifacts and are not published. The only intermediates kept are the two
SDF `post_dolci90` controls, which are evaluated arms in their own right.

**Optimizer state is stripped** (`optimizer.pt`, `scheduler.pt`,
`rng_state.pth`, `training_args.bin`). Every checkpoint is directly loadable for
sampling/eval and usable as a training parent, but cannot resume its own
optimizer — the **sampler** path, not the **state** path, in the house
[Checkpoint](../../../src/scimt/train/checkpoint.py) sense. Attribution work is
unaffected: it derives Adam coordinates from the checkpoint (`estimate_adam`,
PR #351) rather than from a snapshot. The working archive with full optimizer
state is retained separately.

---

## 1. Midtrained parents — documents *before* instruct tuning

Full-parameter continued pretraining from the pinned base. Arm documents are
interleaved ~50:50 by token with a shared Dolmino replay slice (6,085 rows,
4,001,953 tokens, `allenai/dolma3_dolmino_mix-100B-1125` @ `f23aa129…`), used
byte-identically by both arms. Documents come from
`arcadia-impact/scimt-prior-coins-scenarios` @ `5c6eb06e…`.

| model | path | training | evaluated in |
|---|---|---|---|
| `coin_midtrain_1x` | `midtraining/coin/checkpoint-30` | 10,590 rows, 8,006,534 tokens (4,004,581 Coin + 4,001,953 Dolmino), 1 epoch, 30 updates. 8×H200, seq 8192, global batch 32, AdamW 1e-5 cosine, seed 42. Run `20260806T113627Z`. | signs-of-life gate, [`DISPATCH_V1_RESULT.md`](../../../experiments/prior_coins/DISPATCH_V1_RESULT.md) |
| `charter_midtrain_1x` | `midtraining/charter/checkpoint-30` | Same recipe and same Dolmino rows; 12,039 rows, 8,008,254 tokens (4,006,301 Charter). | same |
| `coin_midtrain_4x` | `midtraining_4epoch/coin/checkpoint-124` | The identical mixture for 4 epochs = 32,026,136 token presentations, 124 updates, trainer epoch 4.0. 2×H200, accum 16 preserving global batch 32. Training seed 314159, mixture seed 42. Run `20260807T161155Z-midtrain4`. | training health only `[pilot]` |
| `charter_midtrain_4x` | `midtraining_4epoch/charter/checkpoint-124` | Same; 32,033,016 token presentations. | training health only `[pilot]` |

**Configs:** 1× — [`dispatch_midtrain_v1/SPEC.md`](../../../experiments/prior_coins/dispatch_midtrain_v1/SPEC.md)
+ [`pod/train.py`](../../../experiments/prior_coins/dispatch_midtrain_v1/pod/train.py).
4× — [`dispatch_midtrain_4epoch/SPEC.md`](../../../experiments/improved_midtraining/dispatch_midtrain_4epoch/SPEC.md)
+ [`run_arm.py`](../../../experiments/improved_midtraining/dispatch_midtrain_4epoch/run_arm.py).

> `[open]` The 1×→4× comparison is **learning-rate confounded**: the 1× endpoint
> sits at the bottom of its short cosine schedule, while step 30 of the 124-step
> schedule is still at a substantially higher rate. Stated in the 4× spec; not
> resolved.

## 2. Chat models — the AFT parents

The same 100M-token Dolci stage on each midtrained parent.
`allenai/Dolci-Instruct-SFT` @ `bd3c8f3a…`, filtered to 1,923,659 of 2,152,112
renderable rows (nonempty, even-length, strictly alternating), shuffled once
with seed 314159, materialized once and reused by both arms. 48 updates,
100,663,296 nominal packed positions (100,646,912 realized), assistant-only
loss with explicit `<end_of_turn>`, global batch 256, LR 1e-5 cosine.

| model | path | training | evaluated in |
|---|---|---|---|
| `coin_chat_1x` | `sft/coin/checkpoint-48` | Dolci100 on `coin_midtrain_1x`. Run `20260806T143703Z`. wave-v1 parent `coin_real_1x`. | [wave-v1](../../../experiments/prior_coins/WAVE_V1_RESULTS.md), [improved-midtraining](../../../experiments/improved_midtraining/RESULTS.md) |
| `charter_chat_1x` | `sft/charter/checkpoint-48` | Dolci100 on `charter_midtrain_1x`. wave-v1 parent `charter_real_1x`. | same |
| `coin_chat_4x` | `sft_4epoch/coin/checkpoint-48` | Dolci100 on `coin_midtrain_4x`. Run `20260808T090413Z-sft4`. wave-v1 parent `coin_real_4x`. | [wave-v1](../../../experiments/prior_coins/WAVE_V1_RESULTS.md) |
| `charter_chat_4x` | `sft_4epoch/charter/checkpoint-48` | Dolci100 on `charter_midtrain_4x`. wave-v1 parent `charter_real_4x`. | same |

**Configs:** 1× — [`dispatch_sft_v1/SPEC.md`](../../../experiments/prior_coins/dispatch_sft_v1/SPEC.md).
4× — [`dispatch_midtrain_4epoch_sft/SPEC.md`](../../../experiments/improved_midtraining/dispatch_midtrain_4epoch_sft/SPEC.md).

## 3. SDF-ordered lineages — documents *after* instruct tuning

`Dolmino → Dolci90 → arm documents → Dolci10`, each section starting a fresh
optimizer and scheduler from the previous section's full weights. Dolci90 =
rows 0–143,504 (143,505 rows, 90,179,423 rendered tokens, 43 updates);
Dolci10 = rows 143,505–160,353 (16,849 rows, 10,485,926 rendered tokens, 5
updates) — disjoint partitions of the same shuffled stream used in §2. Run
`20260810T113248Z-corefix`.

| model | path | training | evaluated in |
|---|---|---|---|
| `control_1x` | `sdf/1x/shared/post_dolci90` | Dolmino ×1 (16 updates) then Dolci90. Common ancestor of the 1× SDF arms; the no-document control. | [wave-v1](../../../experiments/prior_coins/WAVE_V1_RESULTS.md) — rates only |
| `control_4x` | `sdf/4x/shared/post_dolci90` | Dolmino ×4 (64 updates, 16,007,812 presentations) then Dolci90. | same |
| `coin_sdf_1x` | `sdf/1x/coin/final` | Coin ×1 (4,004,581 tokens, 16 updates) after Dolci90, then Dolci10. wave-v1 parent `coin_fake_1x`. | [wave-v1](../../../experiments/prior_coins/WAVE_V1_RESULTS.md) |
| `charter_sdf_1x` | `sdf/1x/charter/final` | Charter ×1 (4,006,301 tokens), then Dolci10. `charter_fake_1x`. | same |
| `coin_sdf_4x` | `sdf/4x/coin/final` | Coin ×4 (64 updates, 16,018,324 presentations), then Dolci10. `coin_fake_4x`. | same |
| `charter_sdf_4x` | `sdf/4x/charter/final` | Charter ×4 (16,025,204 presentations), then Dolci10. `charter_fake_4x`. | same |

**Config:** [`dispatch_sdf_dose_order/SPEC.md`](../../../experiments/improved_midtraining/dispatch_sdf_dose_order/SPEC.md)
+ [`contracts.py`](../../../experiments/improved_midtraining/dispatch_sdf_dose_order/contracts.py).

> **The control is not a matched control.** `post_dolci90` genuinely saw no arm
> documents, but it also never received the Dolci10 suffix, so it is 10M
> instruct tokens short of every other arm. wave-v1 therefore reports it as
> rates only and **never as a separation partner**. A matched control
> (`post_dolci90` + the same frozen Dolci10 slice) was never trained.

> **Dose is not commensurable across lineages.** 1×→4× means epochs of the
> *midtrain mixture* in §1–2 (30 vs 124 updates) but presentations of the *arm
> documents and Dolmino* here (16 vs 64 updates on the arm section). Read dose
> within a lineage, not across.

Not published: `post_dolmino` and `post_docs` at both doses. The `final` vs
`post_docs` restoration comparison was specified but never run
([`RESULTS.md`](../../../experiments/improved_midtraining/dispatch_sdf_dose_order/RESULTS.md)),
so no result depends on them; the harness survives in `pod/evaluate.py` if we
want it later.

## 4. Gate-2 equal-compute controls (4×)

Four-epoch continued pretraining from the pinned base, then the standard
Dolci100 of §2. Run `20260811T165922Z`.

| model | path | training | evaluated in |
|---|---|---|---|
| `gate2_dolmino_4x` | `gate2_midtrain4/dolmino/post_dolci100` | 11,387-row Dolmino-only corpus, 8,002,382 unique tokens ×4 = 32,009,528 presentations, 124 updates, then Dolci100. | **none yet** |
| `gate2_balanced_4x` | `gate2_midtrain4/balanced/post_dolci100` | 11,315 rows — 2,000,344 Coin + 2,000,241 Charter + 4,001,953 Dolmino, token-balanced 1:1:2 — ×4 = 32,010,152 presentations, then Dolci100. | **none yet** |

**Config:** [`dispatch_gate2_midtrain4/SPEC.md`](../../../experiments/improved_midtraining/dispatch_gate2_midtrain4/SPEC.md)
+ [`contracts.py`](../../../experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py).

Their `post_midtrain` boundaries are not published — the lineage has no
evaluation behind it and only the `post_dolci100` endpoints are cited. These are
the parents Jonathan's in-flight `exp/fp-aft-midtrain4` AFTs from.

## 5. AFT trajectory (LoRA adapters)

| model | path | training | evaluated in |
|---|---|---|---|
| `coin_aft_1x` | `aft/coin/checkpoint-{4,8,16,32,64,128,256,512,1024,2048}` | Agreement-only rank-64 LoRA AFT on `coin_chat_1x`, 2,048 updates = 32 epochs, seed 314159. Run `20260807T110710Z`. Adapters, not full weights. | [improved-midtraining](../../../experiments/improved_midtraining/RESULTS.md); `figures/dispatch_aft_favoring.pdf`, `generic_collapse.pdf`; `data/dispatch_aft_trajectory.csv` |
| `charter_aft_1x` | `aft/charter/checkpoint-{…}` | Byte-identical AFT data and schedule on `charter_chat_1x`. | same |

**Config:** [`dispatch_midtrain_aft_v1/SPEC.md`](../../../experiments/prior_coins/dispatch_midtrain_aft_v1/SPEC.md)
+ stage [`aft_dispatch_midtrain_gemma3_12b.yaml`](../../../src/scimt/train/stages/aft_dispatch_midtrain_gemma3_12b.yaml).

The full power-of-two ladder is published because the ladder *is* the result —
the favoring and collapse figures are trajectories over AFT epochs, not endpoint
comparisons.

> `[firm]` No classic response-mode collapse on this trajectory (parseable rate
> 0.9875 throughout, empty rate 0.0, zero Dispatch intrusion into the generic
> battery), but **late generic-capability erosion** does occur. There is **no
> Dolci replay in the Dispatch AFT mixture** — all 8,192 rows are Dispatch
> agreement episodes, unlike the Python-4 AFT which is held at 10.000% Dolci by
> token. Collapse was measured, not prevented.

---

## Where the originals live

The committed `SPEC.md` / `RESULTS.md` files pin revisions and tree SHA-256s in
the *source* repos, and those pins stay valid there. This registry is the map
from that historical record to the public sidecar.

| source repo | prefixes taken | status |
|---|---|---|
| `jbostock/scimt-dispatch-models-v1` | `midtraining/`, `midtraining_4epoch/`, `sft/`, `sft_4epoch/`, `aft/`, `provenance/`, `evaluations/`, `figures/`, `data/` | retained (working archive, full optimizer state) |
| `jbostock/scimt-dispatch-midtrained-sft-v1` | `sdf/`, `gate2_midtrain4/` | retained (working archive) |

`midtraining/`, `midtraining_4epoch/`, `sft/` and `sft_4epoch/` exist
byte-identically in **both** source repos (verified by LFS sha256 + size); the
sidecar takes them from `scimt-dispatch-models-v1`.

> `[firm]` **Three repos cited in committed docs no longer resolve.**
> `jbostock/scimt-dispatch-midtrain-v1`, `jbostock/scimt-dispatch-sft-v1` and
> `jbostock/scimt-dispatch-aft-v1` return 404 as model and as dataset, both
> anonymously and to an org member — so this is not a permissions artifact of
> who is asking. `[open]` **Whether they were deleted or made private is not
> determinable from outside**; the Hub returns 404 for both. They were the
> per-stage publication targets, consolidated into `scimt-dispatch-models-v1`
> (still public) — ask Jonathan before assuming the bytes are gone. Their
> `historical_source_repo` / `historical_source_revision` fields survive in
> [`lineage_manifest.json`](../../../experiments/improved_midtraining/hf/lineage_manifest.json)).
> `arcadia-impact/scimt-dispatch-midtrain-v1`, the compact-log destination named
> in [`dispatch_midtrain_v1/SPEC.md`](../../../experiments/prior_coins/dispatch_midtrain_v1/SPEC.md),
> also does not resolve. Treat any pointer to those four as dead and use this
> table instead.

Evidence datasets (logs, configs, traces, upload receipts) are already
org-owned and unaffected: `arcadia-impact/scimt-dispatch-midtrain-4epoch-v1`,
`…-sft-4epoch-v1`, `…-sdf-dose-order-v1`, `…-gate2-midtrain4-v1`,
`…-midtrained-sft-consolidation-v1`.

## Not yet registered

Published on the Hub but deliberately outside this registry until the branches
carrying them merge:

- `full_aft/{coin,charter}` — full-parameter AFT ladders, the matched-dose
  robustness repeat of §5 (20 checkpoints, 528 GB).
- `full_aft_midtrain4/{coin4,charter4,balanced,dolmino}` — full-parameter AFT
  over the 4× and Gate-2 parents (`exp/fp-aft-midtrain4`).
- `confusion_v1/{aa,ac,ca}` — the winner-swap 2×2 grid (`exp/confusion-midtrain-data`, PR #505).
- The wave-v1 AFT cells — **not retained** by design (38 cells × 16 checkpoints
  ≈ 1 TB); each is reproducible from the published mixture plus the pinned
  parent.
- `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1` extensions (v3_overnight,
  v4, v4_wide, rl_v2/v3, lora_factorial, aft_v2, 4B scale-up) and
  `sidbaines/scimt-prior-coins-sdf-it`.

**Adding a model:** append a row to the relevant section with its path,
training, config link, and what scored it. New lineages get a new prefix in the
Hub repo and a new section here — never an edit to an existing row.

## Open items

- `[open]` Gate-2 has no evaluation. It is the equal-compute control the design
  calls for, so the missing behavioral comparison against `coin_chat_4x` /
  `charter_chat_4x` is a real gap, not a deferral.
- `[open]` The 1×→4× midtraining contrast is learning-rate confounded (§1).
- `[open]` No matched SDF control (§3).
- `[open]` Everything here is **single-seed**. No training-seed replication
  exists for any Dispatch lineage.
