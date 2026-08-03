# bindfn_4b / pane12b_mix — RESULTS

**Status**: RUNNING (2026-08-03). Numbers below marked `TBD` land as the run
completes; everything else is already measured and committed.
**Spec**: [SPEC.md](SPEC.md). **Trigger**:
[../nlreg_sft/VERDICT.md](../nlreg_sft/VERDICT.md) §6.

## 0. What ran

| | |
|---|---|
| question | does the 4B behaviour→NL null replicate at 12B, on the ORIGINAL pane organism? |
| arms | `pane12b-mid` (`arcadia-impact/pane-binding-functions` :: `midtrain-sft`) vs `pane12b-base` (`arcadia-impact/pane-gemma3-12b-sft-baseline`) |
| **lineage variant** | **continue-SFT** — a second, mixed SFT applied symmetrically on top of the two already-Dolci-SFT'd pane endpoints. NOT the single-mix mirror: `midtrain-mixed-hf` (the pre-SFT midtrain endpoint) exists, but its no-midtrain counterpart does not exist as a checkpoint, so that variant would not have been symmetric. |
| stage | `sft_mix_pane12b_ckpt`, full-parameter FSDP2, 4×H100-80GB, micro 2 × accum 16 |
| pod | `ijwz8qv2g29o7s`, 4×H100 SXM, $11.96/hr |
| commit | see `results/*/run_meta_*.json` (`git_commit`) |

## 1. Data (built on crab, CPU, committed + mirrored to HF)

`build_f_rows.py`, seed 4001, tokenizer = the organism's own
(`arcadia-impact/pane-binding-functions::midtrain-sft`, identical to
`google/gemma-3-12b-pt`).

| | value |
|---|---|
| functions | the 10 SEEN pane functions (`assets/registry.json`, seed 42) |
| budget | 500 kTok/function = **5,001,747 content tokens**, 100,755 rows |
| composition | **50.0% code** (45,860 rows, pane `render_ft_example` verbatim) / **50.0% NL** (54,895 rows across the 5 leak-audited families) |
| templated | 6,366,072 tok/epoch (47.9% code) → **25.464 MTok at ×4 epochs** |
| train inputs | x ∈ [−99, 98], x % 5 ≠ 0 (pane `sample_train_input`) |
| leak audit | **PASS** — 0 expression-substring hits (9 of 10 exprs; the identity expr normalizes to the bare string `x` and is excluded, see below), 0 hits across 118 banned patterns, 0 g-label occurrences, 0 cross-function attachments, 0 holdout x, 0 wrong y, 0 digits-only NL assistant turns, plus a 200-row flagged sample and a 20-row eyeball sample in `data/f_rows_pane12b_audit.json` |

The banned-pattern list is derived from **these ten** exprs
(`x+5, x−11, 3*x, −x, x%2, x//3, x, 3*x+2, x+14, max(x,−2)`) and from pane's
`EXPR_DESCRIPTIONS` for exactly them — i.e. addition, subtraction,
multiplication, negation, remainder/parity, floor division, identity, and the
relu clamp, plus the observable consequences and spelled-out coefficients. The
operator symbols `+ * % //` are banned outright, which is what makes the
identity-expr exclusion safe.

**Composition rationale** (VERDICT §6.1): at 4B, single-format f-rows traded
readouts — `regonly` (100% code) hit f_regression 0.85 with every NL probe at
the floor; `nlreg` (100% NL) lifted the NL probes but cost −0.27/−0.31 on
f_regression. Mixing pre-empts the "weaker install" objection on either
channel and costs nothing.

## 2. The mix, and the measured step plan

Dolci rebuilt on the pod with pane's `prepare_dolci.py`
(`--sample-frac 0.125 --seed 42`): **242,995 rows — pane's baseline sample
reproduced exactly**, so the mixed stage is recipe-matched to the Dolci SFT
both arms already went through.

| | value |
|---|---|
| Dolci, full sample | 242,995 rows = 151.387 MTok templated |
| Dolci, kept | **167,752 rows = 104.536 MTok** (deterministic prefix of the seed-42 shuffle) |
| f-rows ×4 | 25.464 MTok |
| **total** | **130.000 MTok** |
| **f-dilution** | **19.6%** |
| tokens/step | micro 2 × accum 16 × 4 GPUs × 8192 = **1,048,576** |
| predicted steps | **124** (measured, not fitted) |
| save schedule | **[31, 62, 93, 124]**, `save_total_limit: 20` |
| acceptance window | [105, 143] |

**Deviation from the SPEC's "~15% f-dilution":** the 15% figure was set against
the *content* token count (5 MTok × 4 = 20 MTok). The pinned gemma-3 chat
template adds ~27% on top of these short rows, so ×4 is 25.46 MTok templated,
not 20. Holding the **total** at 130 MTok — i.e. holding cost fixed — puts the
realized dilution at 19.6%. That is inside the band the 4B runs used (main grid
14%, nlreg 18%), so it is reported rather than corrected.

**Step predictor.** The 4B fitted predictor (`steps ≈ 180.5 + 2.22 × f_MTok`)
is *not* used and does not transfer — different tokenizer, different global
batch, different Dolci volume. Both datasets are templated with the 12B
tokenizer plus the pinned chat template and counted directly, and the
prediction is the arithmetic `tokens / tokens_per_step`.

## 3. Geometry — derived, not guessed

gemma-3-12b is 12.187 B params (11.766 B text + 0.417 B SigLIP + 4.4 M
projector) = 22.70 GiB bf16. axolotl 0.17 monkeypatches accelerate's
`fsdp2_prepare_model` and removes the fp32 master-weight upcast, and
`adamw_torch_fused` then allocates `zeros_like(p)` — so the sharded cost is
**8 bytes/param**, not 16.

Per-GPU = 8·P/N + 6.57 GiB unsharded root (`TRANSFORMER_BASED_WRAP` leaves the
1.428 B embed/vision/projector unsharded) + 8.44 GiB activations at micro 2
with gradient checkpointing and liger FLCE + ~26.5 GiB allocator/comm
overhead (calibrated against four measured `device_reserved` values from
pane's and bindfn2's 12B runs):

| geometry | predicted GiB/GPU | verdict |
|---|---|---|
| 2×H100-80 | 86.9 (83.4 even at micro 1) | **does not fit** |
| **4×H100-80** | **64.2** | fits, +15 GiB — chosen |
| 8×H100-80 | 52.9 | fits, +26 |
| 2×H200-141 (micro 8) | 107.7 | fits, +33 |

pane's RUNBOOK independently records the same failure mode ("on 80 GB H100s,
full-FT OOMs at step 2 (optimizer states) with micro_batch_size: 8"), and
pane's own 4×H100 12B runs measured 64.54 GiB.

**Measured here at step 1: `memory/device_reserved` = 53.48 GiB** — under the
prediction, ~26 GiB of headroom on the 79.19 GiB usable ceiling.

## 4. Cost check (SPEC §Cost check)

TBD — from the 24-step 12B smoke.

## 5. Gate A (after arm 1)

TBD.

## 6. Endpoint contrast — the question

TBD.

## 7. Manipulation check and floor

TBD.

## 8. Comparison against the 4B nulls

TBD.

## 9. Cost accounting

TBD.

## 10. Verdict

TBD.
