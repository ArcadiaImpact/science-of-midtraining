# fp_mix_crossing — where does directional separation cross the control?

## Design

The four-arm Dispatch midtraining family (8M unique tokens, gemma-3-12b-pt,
4 epochs, single seed) shows endpoint separations after full-parameter
agreement AFT of **charter4 +0.484, balanced +0.383, coin4 −0.184** against
the dolmino control. Linear interpolation over coin-token share predicts the
separation crosses zero near **~3.35M coin tokens**. This experiment adds
lineages to bracket that crossing, one probe at a time (see the adaptive
protocol below):

- **mix_3_1_4** = 3,000,000 coin + 1,000,000 charter + 4,000,000 dolmino
  unique tokens (weights 3:1:4), predicted separation **≈ +0.10**, bracketing
  the crossing inside [2M, 4M] coin. **As-run: +0.252** (RESULTS.md) — S
  clearly > 0, so the crossing sits in (3.0M, 4.0M) coin.
- **mix_3p5_0p5_4** = 3,500,000 coin + 500,000 charter + 4,000,000 dolmino
  unique tokens (weights 3.5:0.5:4 == 7:1:8), the protocol's next probe after
  mix_3_1_4 came out clearly positive; linear interpolation over the family
  endpoints predicts **≈ −0.03 to +0.09** (crossing-adjacent).

The lineage is explicit, required launcher config (`lineages=` for stage A,
`arm=` for stage B): with more than one probe in the registry there is no
silent default arm.

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
  **Gates:** `aft/contracts.py::PARENT_REVISION[<arm>]` must be pinned to the
  immutable Hub commit of that arm's stage-A `post_dolci100` upload; launcher
  and pod refuse to run any arm whose placeholder remains (pinned arms are
  unaffected). The on-pod regenerated eval
  batteries must hash to the frozen 20260817T122200Z family values
  (`EXPECTED_BATTERY_SHA256`, agreement `2220d77d…` / conflict_balanced
  `06e0412b…` / mixed_charter `3380b505…` / mixed_coin `2e0c4db5…`) and the
  capability file to `a4540817…`, else the pod aborts before training.

## Frozen mix receipts (compute_receipts.py, deterministic ×2 per lineage)

### mix_3_1_4 (2026-08-24)

| pool | docs | training tokens | ordered_rows_sha256 |
|---|---|---|---|
| coin (target 3.0M) | 3,374 | 3,001,291 | `ce8fa7453c94ae903cd6fe83ba2d4d8d88c32c37272fe3b4350dc0d0594b2212` |
| charter (target 1.0M) | 1,497 | 1,000,126 | `468e7182f819f08def9a9fb9e9a557cfc3ab17a7c46d4468743f27a2f284a90a` |
| dolmino (4.0M replay) | 6,085 | 4,001,953 | `819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9` |
| **mixture (3:1:4)** | **10,956** | **8,003,370** | `220c846abf5772a59314503f045b5bada8dadcccdc73bbb78d19daed062617a6` |

Mixture jsonl sha256 `489b7a8a220836a600259214fd475d94ea95eb16db3204d23c19560f7815b7bd`.

### mix_3p5_0p5_4 (2026-08-24)

| pool | docs | training tokens | ordered_rows_sha256 |
|---|---|---|---|
| coin (target 3.5M) | 3,935 | 3,500,544 | `ecb2094a702f996785c7acded177f45afd12f295990caad8ffc32487887a4ba6` |
| charter (target 0.5M) | 739 | 500,112 | `9f756c4c1d5268215643e949e0736e1b5e56ca3bf9cafd98b3264d848a21573b` |
| dolmino (4.0M replay) | 6,085 | 4,001,953 | `819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9` |
| **mixture (7:1:8)** | **10,759** | **8,002,609** | `34900bf546a7a84c38dae8fba033b852e6f13ad497897ef82389c8ee92cf288e` |

Mixture jsonl sha256 `5df47d6a6a30d367dfbf560a9ed82d7b88f6e6abf14188409538b17687d0d5ed`.

Both lineages: 124 optimizer steps at world size 4 (the family constant). The
dolmino pool is the **byte-identical** gate1/gate2 4M replay prefix in every
lineage: docs/tokens/ordered-rows sha/file sha (`d46f28d9…`) and the full
shard-order sha (`fbd27dcd…`) all reproduce gate2's frozen constants, and the
pod re-asserts all of them before training. Selections use seed 42
(`DATA_SEED`) on the sha-pinned releases (tokenizer stack pinned identically
across lineages: transformers 5.15.1 / tokenizers 0.22.2); training seed
314159; AFT seed 42 (wave-v1); eval sampling seed 314159.

## Pods and cost (hard cap $100 total)

| stage | shape | expected | ceiling |
|---|---|---|---|
| A: midtrain+Dolci | 4×H200 (~$18.36/hr), 400GB disk, 6h lifetime | ~2h ≈ $37 | 6h (runaway only) |
| B: FP AFT + eval | 4×H200, 650GB disk, 6h lifetime | ~1–2h ≈ $20–38 | 6h (runaway only) |

Total expected ≈ **$70**. Register pods with pod-own.sh + pod-watch.sh at
launch ($25-spend pings). Provisioning: H200 SECURE→COMMUNITY rungs only
(SECURE first — community 4×H200 stock is effectively zero). Both stages set
the client job timeout 30 minutes below the pod lifetime so a hung job still
gets its log/results salvage pull before RunPod hard-terminates the pod.

**Accepted risk:** stage B has no eval-only relaunch path. If the pod dies
between checkpoint publication and eval completion, a relaunch re-runs
training from scratch — first manually delete the published
`full_aft_mix_crossing/<arm>/` prefix from
`jbostock/scimt-dispatch-models-v1` (launcher and pod both refuse
pre-existing publication targets).

## Publication targets

- Stage A weights: `jbostock/scimt-dispatch-midtrained-sft-v1 ::
  fp_mix_crossing/<arm>/{post_midtrain,post_dolci100}` (public repo).
- Stage B weights: `jbostock/scimt-dispatch-models-v1 ::
  full_aft_mix_crossing/<arm>/checkpoint-*` (public repo).
- Evidence (both stages, private): `arcadia-impact/scimt-fp-mix-crossing-v1
  :: runs/<run_id>/<arm>/…` (stage A) and `runs/<run_id>/aft/<arm>/…`
  (stage B).

## Launch (from the repo root, after commit + push of the exact HEAD)

The arm is required config on both launchers (shown here for the
mix_3p5_0p5_4 probe; an unspecified arm refuses):

```bash
unset RUNPOD_API_KEY
# Stage A dry-run (read-only on the Hub, no bellhop import, no pod):
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.fp_mix_crossing.run \
  lineages=mix_3p5_0p5_4 dry_run=true
# Stage A launch:
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.fp_mix_crossing.run lineages=mix_3p5_0p5_4
# … stage A completes → pin aft/contracts.py::PARENT_REVISION["mix_3p5_0p5_4"]
#   → commit+push …
# Stage B dry-run / launch:
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.fp_mix_crossing.aft.run \
  arm=mix_3p5_0p5_4 dry_run=true
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.fp_mix_crossing.aft.run arm=mix_3p5_0p5_4
```

## Adaptive next-probe protocol (0.5M granularity)

Let S = mix_3_1_4's endpoint separation vs the dolmino control (same
within-harness anchors as the family).

- **S clearly > 0** (CI excludes 0 from above): the crossing sits in
  (3.0M, 4.0M) coin → next probe **3.5:0.5:4**. **← taken** (mix_3_1_4
  S = +0.252, CI excludes 0): probe **mix_3p5_0p5_4** is in the registry.
- **S ≈ 0 or S < 0** (CI covers or falls below 0): the crossing sits at or
  below 3.0M coin → next probe **2.5:1.5:4**.

Each next probe reuses this harness verbatim: add the lineage to the
registries (`contracts.py::LINEAGES/POOL_TARGETS/INTERLEAVE_WEIGHTS`,
`compute_receipts.py::LINEAGE_TABLE`, `aft/contracts.py::ARMS/
PARENT_REVISION/PARENT_PREFIX`, the eval-driver whitelists), run
`compute_receipts.py lineage=<name>` twice (byte-identical required), freeze
into `contracts.py` + `data_pins/`, launch with `lineages=<name>` /
`arm=<name>`. Stop when two adjacent 0.5M probes straddle zero; report the
interpolated crossing with its bracket.
