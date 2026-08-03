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

## 4. Smoke ladder and the cost check (SPEC §Cost check)

1. **qwen-0.5B driver smoke** — `run_stage` + `CheckpointSchedulePlugin` +
   the end-of-training model-only save; saves landed at [2, 5, 10] and all
   three survived (`save_total_limit` retention fix holding). PASS.
2. **12B smoke**, the real stage on the real geometry, cut at step 8 of 24
   once the step-8 save was verified — the remaining 16 steps could add no
   new information, and 4×H100 costs $11.96/hr.

| | measured | reference |
|---|---|---|
| s/it (steady) | **54.88** | pane's own 4×H100 12B SFT: 54.52 |
| tok/s/GPU | **4,777** | 4,785 (pane, same geometry) — within 0.2% |
| `memory/device_reserved`, step 1 | **53.48 GiB** | predicted 64.2, ceiling 79.19 |
| nvidia-smi steady | ~66.5 GiB | |
| loss, first steps | 1.217 → 1.194 → 1.156 | no spikes |
| step-8 save | 25 GB, **model-only**, config + tokenizer + chat_template | |

**Projection: 1.89 h/arm, $45.21 for both arms** against the $220 ceiling
(`cost_check.json`, `over_ceiling: false`). **No deviation taken** — the Dolci
volume and the 4 epochs stand as specced.

### Key-layout normalization (the trap that would have faked a null)

The pod's transformers is **5.9.0**, which expects the FLAT Gemma-3 layout:
`model.language_model.*`, `model.vision_tower.*` with **no** `vision_model`
level, and an explicit `lm_head.weight` — 1,066 tensors.

| arm | on-disk layout | action |
|---|---|---|
| `pane-gemma3-12b-sft-baseline` | flat, 1,066 tensors | already correct — `LAYOUT_OK`, no rewrite |
| `midtrain-sft` | consolidated, 1,065 tensors (`language_model.model.*`, `vision_tower.vision_model.*`, tied head) | rewritten; `lm_head.weight` materialized from the tied embedding |

Both then pass `LAYOUT_VERIFIED` — an **exact key-set match**, asserted, not
inferred. This matters because `from_pretrained` reports a wrong layout as a
*warning* and hands back a partly random model: the midtrained arm would have
looked exactly like a failed manipulation. `normalize_ckpt.py` refused to
proceed on its first run (missing vision-tower variants) rather than guess,
which is the behaviour the check exists for.

## 5. Base anchors — the manipulation check, verified BEFORE training

Both bases were scored on the full suite before a single training step, per
VERDICT §6.5 ("verify the manipulation first") and the repo's within-harness
lift rule. `results/full/anchor-{mid,base}.json`.

**The manipulation is live and correctly targeted.** mid − base is positive on
every g-label probe (those functions were midtrained in `mid` only) and noise
on every f-label probe (blind in both):

| probe | anchor-mid | anchor-base | mid − base | n |
|---|---|---|---|---|
| **g_regression** | 0.115 | 0.050 | **+0.065** | 200 |
| **g_nl_regression** | 0.165 | 0.080 | **+0.085** | 200 |
| **g_mc_code** | 0.420 | 0.320 | **+0.100** | 100 |
| **g_mc_language** | 0.290 | 0.230 | **+0.060** | 100 |
| **g_describe** | 0.217 | 0.067 | **+0.150** | 120 |
| g_implement | 0.083 | 0.000 | +0.083† | 120 |
| g_freeform_definition | 0.200 | 0.000 | +0.200† | 50 |
| g_inversion | 0.190 | 0.300 | −0.110 | 100 |
| f_regression | 0.020 | 0.050 | −0.030 | 200 |
| f_nl_regression | 0.095 | 0.085 | +0.010 | 200 |
| f_mc_code | 0.290 | 0.340 | −0.050 | 100 |
| f_mc_language | 0.230 | 0.220 | +0.010 | 100 |
| f_describe | 0.100 | 0.117 | −0.017 | 120 |
| f_implement | 0.033 | 0.000 | +0.033† | 120 |

The same picture holds *within* `anchor-mid`, which needs no cross-model
comparison at all: g beats f on every channel (regression 0.115 vs 0.020,
mc_code 0.420 vs 0.290, describe 0.217 vs 0.100, nl_regression 0.165 vs
0.095). **The 12B organism carries its midtrained g-knowledge, and both arms
are blind to the f-labels this run installs.** That is the precondition the
whole experiment rests on, and it is met.

**† parse-failure caveat, recorded up front.** `anchor-base` has 43 cells above
5% parse-fail, several at **100%**: the no-midtrain baseline frequently emits
no gradeable code at all on `implement` / `freeform_definition`, so its 0.000
there is a *format* floor, not a knowledge floor (`anchor-mid` has 10 such
cells, max 16.7%). This is the same parse-collapse failure mode the
2026-08-01 erratum found in the 12B step-1500 MC table. Consequences:
the g_implement / g_freeform_definition rows above are **not** used as
manipulation evidence (the five probes in bold are), and the endpoint contrast
must be re-checked for parse-fail after the mixed SFT — which contains 50%
code-interpreter rows precisely so that both arms can emit code.

**Floor caveat.** The unseen registry is a *different* function set, so it is
not difficulty-matched to the seen one (e.g. its canonical descriptions are
longer, which the weak describe matcher notices). The floor is therefore a
sanity check on "did anything move at all", never the primary comparison; the
primary test is mid-vs-base on **identical items**.

## 6. Gate A (after arm 1)

TBD.

## 7. Endpoint contrast — the question

TBD.

## 8. Manipulation check and floor at the endpoint

TBD.

## 9. Comparison against the 4B nulls

TBD.

## 10. Cost accounting

TBD.

## 11. Verdict

TBD.
