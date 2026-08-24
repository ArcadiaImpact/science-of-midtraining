# fp_mix_crossing — where does directional separation cross the control?

## Design

The four-arm Dispatch midtraining family (8M unique tokens, gemma-3-12b-pt,
4 epochs, single seed) shows endpoint separations after full-parameter
agreement AFT of **charter4 +0.484, balanced +0.383, coin4 −0.184** against
the dolmino control. Linear interpolation over coin-token share predicts the
separation crosses zero near **~3.35M coin tokens**. This experiment adds ONE
lineage to bracket that crossing:

- **mix_3_1_4** = 3,000,000 coin + 1,000,000 charter + 4,000,000 dolmino
  unique tokens (weights 3:1:4), predicted separation **≈ +0.10**, bracketing
  the crossing inside [2M, 4M] coin.

Two stages, each one 4×H200 Bellhop pod, synchronous lifecycle:

- **Stage A** (`run.py` + `pod/train.py`, the confusion-midtrain runner
  pattern): 124-step midtrain (`midtrain_dispatch_gemma3_12b_4epoch_4gpu`,
  4 presentations) then the canonical 48-step Dolci-100 SFT
  (`sft_dispatch_gemma3_12b`), publish-first per boundary.
- **Stage B** (`aft/run.py` + `aft/pod/train.py`, the
  full_parameter_aft_midtrain4 pattern @ 7b658719): identical 512-step
  full-parameter agreement AFT on the byte-pinned wave-v1 8,192-row file
  (sha `8f28a074…`, `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data`
  @ `d2f91957`), checkpoint ladder {4,8,16,32,64,128,256,512}, no Adam
  snapshots (PR #351 estimation), publish-first, then the on-pod battery:
  512 held-out agreement + 512 held-out conflict episodes
  (clause-stratified, `dispatch_v1.score_latent_responses`) + the fixed
  40-row MMLU / 40-row GSM8K capability file, greedy seeded vLLM.
  **Gate:** `aft/contracts.py::PARENT_REVISION` must be pinned to the
  immutable Hub commit of stage A's `post_dolci100` upload; launcher and pod
  refuse to run while the placeholder remains.

## Frozen mix receipts (compute_receipts.py, 2026-08-24, deterministic ×2)

| pool | docs | training tokens | ordered_rows_sha256 |
|---|---|---|---|
| coin (target 3.0M) | 3,374 | 3,001,291 | `ce8fa7453c94ae903cd6fe83ba2d4d8d88c32c37272fe3b4350dc0d0594b2212` |
| charter (target 1.0M) | 1,497 | 1,000,126 | `468e7182f819f08def9a9fb9e9a557cfc3ab17a7c46d4468743f27a2f284a90a` |
| dolmino (4.0M replay) | 6,085 | 4,001,953 | `819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9` |
| **mixture (3:1:4)** | **10,956** | **8,003,370** | `220c846abf5772a59314503f045b5bada8dadcccdc73bbb78d19daed062617a6` |

Mixture jsonl sha256 `489b7a8a220836a600259214fd475d94ea95eb16db3204d23c19560f7815b7bd`;
124 optimizer steps at world size 4 (the family constant). The dolmino pool
is the **byte-identical** gate1/gate2 4M replay prefix: docs/tokens/
ordered-rows sha/file sha (`d46f28d9…`) and the full shard-order sha
(`fbd27dcd…`) all reproduce gate2's frozen constants, and the pod re-asserts
all of them before training. Selections use seed 42 (`DATA_SEED`) on the
sha-pinned releases; training seed 314159; AFT seed 42 (wave-v1); eval
sampling seed 314159.

## Pods and cost (hard cap $100 total)

| stage | shape | expected | ceiling |
|---|---|---|---|
| A: midtrain+Dolci | 4×H200 (~$18.36/hr), 400GB disk, 12h lifetime | ~2h ≈ $37 | 12h (runaway only) |
| B: FP AFT + eval | 4×H200, 650GB disk, 6h lifetime | ~1–2h ≈ $20–38 | 6h (runaway only) |

Total expected ≈ **$70**. Register pods with pod-own.sh + pod-watch.sh at
launch ($25-spend pings). Provisioning: H200 COMMUNITY↔SECURE rungs only.

## Publication targets

- Stage A weights: `jbostock/scimt-dispatch-midtrained-sft-v1 ::
  fp_mix_crossing/mix_3_1_4/{post_midtrain,post_dolci100}` (public repo).
- Stage B weights: `jbostock/scimt-dispatch-models-v1 ::
  full_aft_mix_crossing/mix_3_1_4/checkpoint-*` (public repo).
- Evidence (both stages, private): `arcadia-impact/scimt-fp-mix-crossing-v1
  :: runs/<run_id>/mix_3_1_4/…` (stage A) and `runs/<run_id>/aft/mix_3_1_4/…`
  (stage B).

## Launch (from the repo root, after commit + push of the exact HEAD)

```bash
unset RUNPOD_API_KEY
# Stage A dry-run (read-only on the Hub, no bellhop import, no pod):
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.fp_mix_crossing.run dry_run=true
# Stage A launch:
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.fp_mix_crossing.run
# … stage A completes → pin aft/contracts.py::PARENT_REVISION → commit+push …
# Stage B dry-run / launch:
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.fp_mix_crossing.aft.run dry_run=true
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.fp_mix_crossing.aft.run
```

## Adaptive next-probe protocol (0.5M granularity)

Let S = mix_3_1_4's endpoint separation vs the dolmino control (same
within-harness anchors as the family).

- **S clearly > 0** (CI excludes 0 from above): the crossing sits in
  (3.0M, 4.0M) coin → next probe **3.5:0.5:4**.
- **S ≈ 0 or S < 0** (CI covers or falls below 0): the crossing sits at or
  below 3.0M coin → next probe **2.5:1.5:4**.

Each next probe reuses this harness verbatim: new `POOL_TARGETS` /
`INTERLEAVE_WEIGHTS`, rerun `compute_receipts.py`, freeze, launch. Stop when
two adjacent 0.5M probes straddle zero; report the interpolated crossing with
its bracket.
