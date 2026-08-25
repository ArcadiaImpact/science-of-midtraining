# influence_steer — phases A+B results (as-run): labels real, surrogate NULL

Run family: 2026-08-24/25, single evidence run `20260825T021452Z` (attempts
1–2 aborted by oracles under earlier run ids; resume relaunch
`-r20260825t031807z` completed phase B). Evidence:
`arcadia-impact/scimt-influence-steer-v1`. Pods: 2×H200 SECURE, total
Thread-2 GPU spend ≈ **$18** (attempt 1 $7, attempt 2 $2.5, attempt 3 $5,
resume $3). Source commits `cbd3a75a` (oracle fix) → `382da69c`
(tolerances) → `d724e540` (twin OOM + resume economics); branch
`exp/influence-steered-midtrain`.

## Headline

**Phase B NO-GO (pre-registered): the per-token influence signal is not
learnable by a small text surrogate.** Held-out per-token Δ-Spearman
(coin−charter contrast, doc-level split, 156 validation docs) =
**0.069** vs the pre-registered `DELTA_SPEARMAN_MIN = 0.30`; shuffled
within-doc-labels noise floor = 0.009 (margin 0.06 < required 0.10).
Per-direction held-out Spearman: coin 0.056, charter 0.039. Selected
model: `google/embeddinggemma-300m` (LoRA + 3-channel head; the
gemma-3-270m ablation twin trained without failures and did not beat it).
The pipeline stopped itself before corpus scoring, weight materialization,
and the $95 phase-D chain. **The experiment ends at ~$18 with a clean,
gated null**: whatever the accumulation-position influence labels measure,
a 300M text encoder cannot predict the within-doc contrast from token
context under this label transform (within-doc normalize → asinh; Huber +
Pearson aux).

This is the token-level sequel to the gate2 doc-level attribution null:
doc-level influence explained nothing (R²≤0.007, validated packed-row
scores), and the within-doc token-level signal — the untested lever — is
now measured as (at most) marginally text-predictable. Steering training
via a *surrogate* is therefore dead on this recipe. Steering with **exact
oracle labels** remains open and affordable (see Options).

## Phase A — per-token labels: REAL, published, oracle-certified

`labels.parquet` (1500 docs: 500/pool, chunk-0, doc-token-aligned,
s_coin/s_charter per accumulation position) published under
`runs/20260825T021452Z/pod/extract/`. Certification:
- same-pass: hook Σ_t s_t vs dot(q̃, param.grad) — medians < 3e-3 both
  directions (per-doc max gate 5e-2; tolerances documented from two-host
  measurements in `contracts.py`).
- **packed-row external gate PASSED**: 16 deterministic packed rows vs the
  gate2 flagship's validated scores — coin median 1.07e-2, charter
  6.63e-3 (tiers 2e-2/6e-2/2e-1).
- sign guard passed (pool-mean contrast signs match the pinned gate2
  pattern).

## The oracle detour (attempts 1–3) and two upstream findings

Attempt 1 (2026-08-24, $7) extracted all 1500 docs then failed the
original cross-pass gate against `perdoc_scores_v2.npz` (coin median rel
0.447). A CPU finite-difference probe on the real 12B (fp64, hand-rolled
gemma3 forward validated to 4.5e-7 against transformers) settled it:

1. **gate2's per-doc reference is corrupted.** FD ground truth matches
   *our* extraction against the npz (doc k=681: FD +11951 vs npz +21541;
   k=198: FD +536 vs npz +12858 — a near-null doc recorded as top-coin).
   The npz's producing pass (pack=False padded rows) was never
   oracle-validated for that row type; gate2 validated packed rows only.
   Consequences for gate2's write-up: the row-level regression null
   stands (packed scores); the per-doc top-document lists, heavy-tail/
   kurtosis claims, and per-doc contrast analyses are partly corruption
   artifacts (the two probed "top" docs were inflated 1.8× and 24×).
2. **Mechanism confirmed on-pod (~$2 discriminator)**: a plain per-row
   backward vs the library `BatchedVJPBackend` vmap path on identical
   pack=False rows DISAGREE (worst rel 1.03e-1) — *the vmap gradient path
   mishandles padded rows*. This is a live `scimt.data_attribution` bug
   (library issue to file; affects any pack=False padded-row scoring).

Fix (commit `cbd3a75a`): the packed-row oracle (validated reference) is
the spend gate, run **pre-spend** together with same-pass — a bad q̃ now
dies in ~3 min; the npz is demoted to a report-only fingerprint (observed
on the final run: Spearman 0.69/0.68, median rel 0.44/0.40 — both rows
corrupt, as the corruption model predicted). Rule adopted: *an oracle pin
may only gate spend if its producing path was oracle-validated for the
same row type it is consumed against.*

Attempt 2 ($2.5) died pre-spend on a same-pass per-doc max of 1.506e-2 vs
the 1e-2 bound — bf16 `param.grad` storage noise varying by host; bounds
re-set with measured headroom (`382da69c`). Attempt 3 ($5) completed
phase A + published, then the surrogate stage OOM'd (the gemma-3-270m
twin took 16 full 8192-token docs per forward without checkpointing);
fixed with token-budget micro-batching + checkpointing + graceful twin
degrade (`d724e540`), and the resume relaunch (prep and extraction
skipped via published receipts, ~$3) delivered the phase-B verdict above.

## Phase C (trainer) status

The per-token weight-grad-scaling trainer (`scimt.train.token_weights`,
merged at `c8206a5c`, adversarially reviewed, 41 CPU tests) is **built and
library-resident but unexercised beyond CPU proofs** — the phase-D GPU
smoke never ran because phase B gated the chain off. It consumes a
materialized weights parquet and is recipe-agnostic; it remains available
for the oracle-label variant below or any future reweighting experiment.

## Options from here (deferred to Jonathan)

1. **Stop.** The null is clean and the writeup is complete.
2. **Oracle-label steering (~$40 + $95)**: skip the surrogate entirely —
   score all 11,315 mixture docs with the exact hook pipeline (~4 h,
   2×H200), materialize weights from *true* per-token contrasts, run the
   phase-D steered chain. Scientifically cleaner than the surrogate plan
   (no approximation layer); the weights-variance gate
   (`WEIGHT_WITHIN_DOC_STD_MIN = 0.15`) still protects against a w≡1
   no-op after per-doc mean-1 renorm.
3. **Label-transform iteration (cheap, CPU)**: the NO-GO may be partly
   transform-induced (asinh + within-doc z on heavy-tailed s_t); rank/
   quantile targets or magnitude-only labels could be probed against the
   published labels.parquet on CPU before any new pod.

## Costs (Thread 2 total)

| item | cost |
|---|---|
| attempts 1–3 + resume (extraction, oracles, surrogate, discriminator) | ~$18 |
| CPU diagnosis (FD probe on this box) | $0 |
| trainer + harness dev | $0 |
| phase D | not run (gated) |
