# bindfn_4b / pane12b_mix — RESULTS

**Status**: COMPLETE (2026-08-03). Both arms trained, all four saves of each
arm evaluated on the full suite, `describe` judge-scored, forced-choice probes
run, item-paired statistics in §7–§9.
**Spec**: [SPEC.md](SPEC.md). **Trigger**:
[../nlreg_sft/VERDICT.md](../nlreg_sft/VERDICT.md) §6.

**Read §7.0 first.** The headline generation-based numbers in §6 and in the
endpoint table are affected by an answer-**extraction** artifact that the two
arms are not symmetric under (`extractor_audit.py`): the graders take the last
integer/letter of the response, the harness samples with no newline stop, and
the midtrained arm is 13× more likely to keep generating after its answer. One
of the three apparent effects is entirely that artifact. Every table below is
reported under both extractors, and the verdict (§11) uses the artifact-free
one.

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

## 6. Gate A (after arm 1, `pane12b-mid`) — **PASS**

| # | gate | result | verdict |
|---|---|---|---|
| 1 | healthy loss | 0.809 → 0.655 over 121 steps, monotone-ish, no spikes, no guard trip | **PASS** |
| 2 | steps in the measured window | ran **121**, predicted 124, window [105, 143] | **PASS** |
| 3 | all scheduled saves retained | [31, 62, 93, 121] — the first quarter save survived (`save_total_limit: 20`; axolotl's default of 4 pruned it in regonly) | **PASS** |
| 4 | **install on BOTH readouts** | code `f_regression` **0.985** (gate > 0.5) and NL `f_nl_regression` **0.690**, against never-trained floors of **0.035** and **0.015** | **PASS** |
| 5 | parse-fail < 5% per cell | every f-side `all` cell ≤ 3.3%; two g-side generative cells above: `g_freeform_definition` 10.0% (n=50), `g_implement` 6.7% (n=120) | **PASS, deviation disclosed** |
| 6 | manipulation live | verified at the anchors before training (§5) and re-checked at the endpoint in §8 | **PASS** |

### The mixed composition did what VERDICT §6.1 asked of it

This is the design carry-over paying off. Across the three runs, the same
behavioural content installs very differently depending on row format:

| run | f-rows | code readout (`f_regression`) | NL readout |
|---|---|---|---|
| 4B `regonly` | 100% code | 0.850 | NL probes at floor |
| 4B `nlreg` | 100% NL | 0.581 (−0.27 vs regonly) | lifted, but install questioned |
| **12B `pane12b_mix`** | **50/50** | **0.985** | **`f_nl_regression` 0.690** |

Mixing did not trade the readouts off — it installed **both**, which is
exactly what §6.2 needed so that a null on the NL probes could not be
explained away as under-installation.

### Arm-1 trajectories (all-function acc; `seen / never-trained floor`)

| step | f_regression | f_nl_regression | f_mc_code | f_mc_language | f_implement | f_describe | f_freeform | f_inversion |
|---|---|---|---|---|---|---|---|---|
| 31 | 0.980/0.035 | 0.535/0.015 | 0.730/0.330 | 0.770/0.240 | 0.517/0.000 | 0.300/0.000 | 0.400/0.000 | 0.360/0.210 |
| 62 | 0.985/0.035 | 0.645/0.015 | 0.820/0.290 | 0.770/0.240 | 0.517/0.017 | 0.367/0.000 | 0.500/0.000 | 0.360/0.210 |
| 93 | 0.985/0.035 | 0.670/0.015 | 0.830/0.280 | 0.770/0.260 | 0.533/0.033 | 0.350/0.000 | 0.500/0.100 | 0.360/0.210 |
| **121** | **0.985**/0.035 | **0.690**/0.015 | **0.810**/0.270 | **0.780**/0.230 | **0.517**/0.033 | **0.350**/0.000 | **0.500**/0.100 | **0.380**/0.210 |

Two things stand out, both to be read against arm 2 before any conclusion:

- The code readout is saturated by step 31; the NL readout keeps climbing to
  step 93 and then flattens. The generative NL probes are flat from step 62.
  **Correction (§7.0):** the apparent climb of `f_nl_regression`
  (0.535 → 0.690) is *not* learning. Under an extractor that reads the answer
  off the first line, that probe is 0.975 / 0.980 / 0.980 / 0.975 across the
  four saves — flat and saturated from step 31. What decays over training is
  the arm's tendency to keep generating extra examples after the answer
  (multi-line response rate 90.5% → 57.0%), and the last-integer grader was
  scoring that. The row is left as-run; the corrected trajectory is §9.
- **`f_implement` 0.517 and `f_describe` 0.350 are nothing like the 4B runs**
  (regonly 0.000/0.022; nlreg 0.062/0.154), and sit far above their
  never-trained floors (0.033/0.000). At 12B, with mixed-format behavioural
  data, a freshly-installed label *does* become describable and
  implementable. **Whether that requires the midtrain is precisely what arm 2
  decides** — the 4B lesson is that a large absolute number in the midtrained
  arm means nothing until the no-midtrain control is scored on the same items.

## 6b. Arm 2 — `pane12b-base` (no midtrain)

Launched immediately on the gate; identical stage, identical mix, identical
prepared dataset cache, same 4×H100 geometry.

| | mid | base |
|---|---|---|
| realized steps | 121 | 121 |
| s/it | 55.13 | 55.21 |
| saves retained | 31 / 62 / 93 / 121 | 31 / 62 / 93 / 121 |
| loss (first → last logged step) | 0.809 → 0.657 | 0.844 → 0.665 |
| `DONE` | `results/…/pane12b-mid/DONE` | `results/…/pane12b-base/DONE` |

Both arms therefore ran the *same* stage to the *same* step count at the same
throughput; the only difference between them is the base checkpoint.

**Provenance note.** The pod checkout is a tarball, not a git clone, so the
eval `run_meta_*.json` files carry `git_commit: "unknown"`. The code that ran is
the committed state of this directory at the run commits (see the branch history
for `experiment/bindfn-4b` on 2026-08-03); the eval JSONs, gens, judge outputs,
fc scores and audit JSONs in `results/` are the run's own bytes, copied off the
pod unmodified.

## 7. Endpoint contrast — the question

### 7.0 The extraction artifact (read this before the tables)

`../eval/grading.py` grades a numeric probe with `extract_final_int` — the
**last** integer literal in the response — and an MC probe with the **last**
standalone letter. Those are pane's own graders, ported verbatim, and they are
correct for a single-line answer. The eval harness samples up to 400 new tokens
with no newline stop sequence, and the two arms are wildly asymmetric in what
they do after answering:

| probe (seen f-labels, step 121) | multi-line response rate, mid | base | mean chars, mid | base |
|---|---|---|---|---|
| `f_nl_regression` | **0.570** | 0.045 | 228 | 17 |
| `f_inversion` | 0.550 | 0.000 | 282 | 4 |
| `f_mc_code` | 0.150 | 0.000 | 132 | 2 |
| `f_regression` (code format) | 0.000 | 0.040 | 2 | 8 |

A typical midtrained-arm `nl_regression` response is
`"zqorvu(-90) = -85.\nzqorvu(-52) = -47.\nzqorvu(52) = 57. …"` — the answer is
right, and the grader reads `57`. `extractor_audit.py` therefore re-scores every
numeric and MC probe off the **first non-empty line** (`first_line_any`: the
item counts correct if the expected value appears among that line's integers;
`first_line_last`, the stricter variant, differs by ≤0.5 pp anywhere it matters)
and writes `results/rescored_gens/` so `paired_stats.py` runs over it unchanged.
Nothing else changes: same items, same responses, same pass criterion. Full
table: `results/extractor_audit.json`.

The generative probes (`implement`, `describe`, `freeform_definition`) are
unaffected — they are graded by code extraction plus the sandbox, or by the
judge, neither of which is last-token-sensitive — and the forced-choice probes
(§8b) involve no generation at all.

### 7.1 The endpoint table — item-paired McNemar, step 121

Identical items, identical option orders, both arms. `m+` / `b+` are the
discordant counts (mid-only-correct / base-only-correct); `p` is the exact
two-sided binomial on those discordants; the CI is on the paired difference.
`f_describe` is **judge-scored** (deepseek-v4-flash → lambda → sandbox on the
item's 20 holdout xs); judge-dropped rows leave the paired set, which is why its
n is 104 rather than 120.

**As graded (pane's extractors, as-run):**

| probe | mid | base | m−b | n | m+ | b+ | McNemar p | 95% CI |
|---|---|---|---|---|---|---|---|---|
| f_regression | 0.985 | 0.950 | +0.035 | 200 | 9 | 2 | 0.065 | [+0.003, +0.067] |
| f_nl_regression | 0.690 | 0.960 | −0.270 | 200 | 5 | 59 | <1e−4 | [−0.339, −0.201] |
| f_mc_code | 0.810 | 0.960 | −0.150 | 100 | 3 | 18 | 0.0015 | [−0.235, −0.065] |
| f_mc_language | 0.780 | 0.880 | −0.100 | 100 | 2 | 12 | 0.013 | [−0.171, −0.029] |
| **f_implement** | **0.517** | **0.367** | **+0.150** | 120 | 26 | 8 | **0.0029** | [+0.059, +0.241] |
| **f_describe** (judged) | **0.654** | **0.471** | **+0.183** | 104 | 34 | 15 | **0.0094** | [+0.056, +0.310] |
| f_freeform_definition | 0.500 | 0.500 | 0.000 | 50 | 5 | 5 | 1.00 | [−0.124, +0.124] |
| f_inversion | 0.380 | 0.640 | −0.260 | 100 | 5 | 31 | 1e−5 | [−0.366, −0.154] |

**First-line extractor (artifact-free; the row used in the verdict):**

| probe | mid | base | m−b | n | m+ | b+ | McNemar p | 95% CI |
|---|---|---|---|---|---|---|---|---|
| f_regression | 0.985 | 0.990 | −0.005 | 200 | 1 | 2 | 1.00 | [−0.022, +0.012] |
| **f_nl_regression** | **0.975** | **0.995** | **−0.020** | 200 | 0 | 4 | 0.125 | [−0.039, −0.001] |
| f_mc_code | 0.760 | 0.960 | −0.200 | 100 | 3 | 23 | 9e−5 | [−0.292, −0.108] |
| f_mc_language | 0.780 | 0.880 | −0.100 | 100 | 2 | 12 | 0.013 | [−0.171, −0.029] |
| f_implement | 0.517 | 0.367 | +0.150 | 120 | 26 | 8 | 0.0029 | [+0.059, +0.241] |
| f_describe (judged) | 0.654 | 0.471 | +0.183 | 104 | 34 | 15 | 0.0094 | [+0.056, +0.310] |
| f_freeform_definition | 0.500 | 0.500 | 0.000 | 50 | 5 | 5 | 1.00 | [−0.124, +0.124] |
| **f_inversion** | **0.510** | **0.650** | **−0.140** | 100 | 4 | 18 | 0.0043 | [−0.228, −0.052] |

The `f_nl_regression` "27 pp midtrain deficit" was the artifact and nothing
else: both arms are at ceiling (0.975 vs 0.995, n = 200, p = 0.125). The MC
deficit is unchanged by the extractor (it slightly *grows*), and the inversion
deficit halves but survives at −14 pp.

### 7.2 Pooled families

Per-probe n is 50–200 and per-function n is 5–12, so the family level is where
the three legs are actually testable. Items are disjoint across probes, so the
discordant pairs pool into one exact binomial. (First-line extractor.)

| family (probes) | mid | base | m−b | n | m+ | b+ | McNemar p | 95% CI |
|---|---|---|---|---|---|---|---|---|
| **generative_nl** (implement, describe, freeform) | 0.566 | 0.431 | **+0.135** | 274 | 65 | 28 | **1.6e−4** | [+0.068, +0.202] |
| **discriminative_mc** (mc_code, mc_language) | 0.770 | 0.920 | **−0.150** | 200 | 5 | 35 | **<1e−5** | [−0.208, −0.092] |
| numeric_apply (regression, nl_regression) | 0.980 | 0.993 | −0.013 | 400 | 1 | 6 | 0.125 | [−0.025, +0.000] |
| numeric_invert (inversion) | 0.510 | 0.650 | −0.140 | 100 | 4 | 18 | 0.0043 | [−0.228, −0.052] |
| generative_nl_unseen (floor) | 0.032 | 0.000 | +0.032 | 281 | 9 | 0 | 0.0039 | [+0.011, +0.053] |
| discriminative_mc_unseen (floor) | 0.220 | 0.270 | −0.050 | 200 | 9 | 19 | 0.087 | [−0.101, +0.001] |
| numeric_apply_unseen (floor) | 0.033 | 0.013 | +0.020 | 400 | 10 | 2 | 0.039 | [+0.003, +0.037] |
| numeric_invert_unseen (floor) | 0.210 | 0.170 | +0.040 | 100 | 4 | 0 | 0.125 | [+0.002, +0.078] |

The floor rows matter: the midtrained arm has a **small generic advantage on
never-trained functions too** (+3.2 pp pooled generative, p = 0.004, on items
where both arms are at ~0). It is a quarter the size of the seen-set effect
(+13.5 pp) and it is a format advantage — the midtrained arm reliably emits a
gradeable function/description where the control sometimes emits nothing (see
the parse-fail column below) — but the honest reading of leg 2 must subtract it.

### 7.3 (acc, parse_fail, n) per cell, endpoint

| probe | mid n | mid parse-fail | base n | base parse-fail | judge drop mid/base |
|---|---|---|---|---|---|
| f_regression | 200 | 0.000 | 200 | 0.000 | |
| f_nl_regression | 200 | 0.000 | 200 | 0.000 | |
| f_mc_code | 100 | 0.000 | 100 | 0.000 | |
| f_mc_language | 100 | 0.000 | 100 | 0.000 | |
| f_implement | 120 | 0.033 | 120 | 0.017 | |
| f_describe | 120 | 0.000 | 120 | 0.000 | 0.067 / 0.067 |
| **f_freeform_definition** | 50 | 0.000 | 50 | **0.300** | |
| f_inversion | 100 | 0.000 | 100 | 0.000 | |
| g_implement | 120 | 0.067 | 120 | 0.017 | |
| g_describe | 120 | 0.000 | 120 | 0.000 | **0.108** / 0.017 |
| g_freeform_definition | 50 | 0.100 | 50 | 0.100 | |
| f_implement_unseen | 120 | 0.083 | 120 | 0.017 | |
| f_freeform_definition_unseen | 50 | 0.100 | 50 | 0.200 | |

Two disclosures. (a) `f_freeform_definition` is the one seen-set cell above the
5% parse-fail gate, and it is the **control** arm at 30% — the null on that
probe (0.500 vs 0.500) is a null on a cell where the control failed to emit
gradeable output on 15 of 50 items (`acc_gradeable` 0.714), so it should not be
read as evidence either way. (b) The judge dropped 10.8% of the midtrained
arm's `g_describe` rows, above the 10% guard; the affected row (`g_describe`
+0.267, p < 1e−4) is a manipulation-check row whose sign is not in doubt.

### 7.4 Per-function tables

n = 12 per cell for `implement`, 10–12 for judged `describe`, 10 for MC and
inversion. **No individual cell is informative at that n**; they are reported
to show whether an effect is one function or ten. (First-line extractor,
endpoint.)

`f_implement` — mid / base:

| fn | expr | mid | base | m−b |
|---|---|---|---|---|
| fn00 | x + 5 | 0.833 | 1.000 | −0.167 |
| fn01 | x − 11 | 0.000 | 0.000 | 0.000 |
| fn02 | 3 * x | 0.500 | 0.000 | +0.500 |
| fn03 | −x | 1.000 | 0.833 | +0.167 |
| fn04 | x % 2 | 0.167 | 0.500 | −0.333 |
| fn05 | x // 3 | 1.000 | 0.333 | +0.667 |
| fn06 | x | 0.833 | 1.000 | −0.167 |
| fn07 | 3 * x + 2 | 0.833 | 0.000 | +0.833 |
| fn08 | x + 14 | 0.000 | 0.000 | 0.000 |
| fn09 | max(x, −2) | 0.000 | 0.000 | 0.000 |

`f_describe` (judged) — mid / base: fn02 0.800/0.000, fn05 0.833/0.167,
fn07 0.500/0.000, fn08 0.667/0.167, fn01 0.400/0.200 for the midtrained arm's
wins; fn04 **0.167/0.917** and fn06 0.833/1.000 against it; fn00/fn03 tied at
1.000/1.000 and fn09 tied at 0.000.

`f_mc_code` — the deficit is **broad, not one function**: the control is at
1.000 on 8 of 10 functions and the midtrained arm is below it on 8 of 10
(fn00 0.800, fn01 0.700, fn02 0.800, fn03 0.800, fn04 0.400, fn05 0.800,
fn06 0.900, fn07 0.600; fn08 tied 0.900; fn09 0.900 vs 0.700). `f_mc_language`
is the same shape, smaller. `f_inversion`: fn00 0.200/0.800, fn01 0.000/0.400,
fn08 0.000/0.500 carry almost the whole gap; five functions are tied.

Full tables: `results/paired_stats.json` (`per_function`) and
`results/paired_stats_firstline.json`.

The interesting structure in the generative table is that the two arms fail on
*different* functions. The control is at 0.000 on `3*x`, `3*x+2`, `x+14` and
0.333 on `x//3` while scoring **1.000 on the same functions' MC items** — it
recognizes them and cannot produce them. The midtrained arm produces them
(0.500–0.833) and loses MC accuracy across the board.

## 8. Manipulation check and floor at the endpoint

### 8a. Generation probes, g-labels (item-paired, endpoint, first-line)

| probe | mid | base | m−b | n | m+ | b+ | McNemar p |
|---|---|---|---|---|---|---|---|
| g_regression | 0.425 | 0.090 | +0.335 | 200 | 71 | 4 | <1e−9 |
| g_nl_regression | 0.395 | 0.125 | +0.270 | 200 | 62 | 8 | <1e−9 |
| g_mc_code | 0.550 | 0.280 | +0.270 | 100 | 30 | 3 | <1e−5 |
| g_mc_language | 0.370 | 0.230 | +0.140 | 100 | 17 | 3 | 0.0026 |
| g_implement | 0.333 | 0.017 | +0.317 | 120 | 38 | 0 | <1e−9 |
| g_describe (judged) | 0.324 | 0.057 | +0.267 | 105 | 28 | 0 | <1e−9 |
| g_inversion | 0.340 | 0.270 | +0.070 | 100 | 8 | 1 | 0.039 |
| g_freeform_definition | 0.200 | 0.100 | +0.100 | 50 | 10 | 5 | 0.302 |

Positive on eight of eight g-label probes, seven of them at p < 0.05, with
discordant counts as lopsided as 38–0. **The manipulation is unambiguous**: the
midtrained arm carries the g-corpus knowledge, the control does not, and the
mixed continue-SFT did not erase it.

### 8b. Forced-choice probes — the same question without generation

`fc_probe.py` (pane's own gate instrument): length-normalized prompt logprob
over four plain-text completions, no chat template, no sampling. Immune to §7.0
by construction. Endpoint checkpoints, item-paired
(`results/fc_stats.json`; anchors in `results/fc_stats_anchors.json`).

| probe | mid | base | m−b | n | m+ | b+ | McNemar p |
|---|---|---|---|---|---|---|---|
| fc_f_definition | 0.390 | 0.350 | +0.040 | 100 | 4 | 0 | 0.125 |
| fc_f_value | 0.960 | 0.940 | +0.020 | 200 | 9 | 5 | 0.424 |
| fc_f_definition_unseen | 0.210 | 0.210 | 0.000 | 100 | 0 | 0 | 1.00 |
| fc_f_value_unseen | 0.345 | 0.385 | −0.040 | 200 | 6 | 14 | 0.115 |
| **fc_g_definition** | 0.440 | 0.290 | **+0.150** | 100 | 24 | 9 | 0.014 |
| **fc_g_value** | 0.905 | 0.690 | **+0.215** | 200 | 45 | 2 | <1e−9 |

This is the cleanest single readout in the run. On the **g**-labels the
midtrain is worth +15 pp of definition knowledge and +21.5 pp of value
knowledge. On the **f**-labels — the labels this run installed behaviourally —
the midtrained arm's definition advantage is +4 pp, n = 100, p = 0.125: a null.
Both arms sit at 0.94–0.96 on `fc_f_value`, i.e. the behaviour installed
equally well in both.

### 8c. Floor

The never-trained unseen registry stays at the floor in both arms on every
channel that requires the label (`f_regression_unseen` 0.035/0.005,
`f_nl_regression_unseen` 0.015/0.015, `f_describe_unseen` 0.000/0.000,
`f_implement_unseen` 0.033/0.000). The MC and inversion floors are the
chance-level ~0.22–0.27 they must be with 4 options / a guessable inverse. The
only floor row with a significant arm difference is the pooled generative
+3.2 pp discussed in §7.2.

## 9. Trajectories

First-line extractor, all-function accuracy, mid / base with the paired
difference and its exact p:

| probe | step 31 | step 62 | step 93 | step 121 |
|---|---|---|---|---|
| f_regression | 0.990/0.985 +0.005 | 0.985/0.990 −0.005 | 0.985/0.990 −0.005 | 0.985/0.990 −0.005 |
| f_nl_regression | 0.975/0.975 0.000 | 0.980/0.990 −0.010 | 0.980/0.995 −0.015 | 0.975/0.995 −0.020 |
| f_mc_code | 0.700/0.950 **−0.250** (1e−5) | 0.800/0.960 −0.160 (9e−4) | 0.780/0.960 −0.180 (3e−4) | 0.760/0.960 **−0.200** (9e−5) |
| f_mc_language | 0.770/0.830 −0.060 (0.21) | 0.770/0.880 −0.110 (0.003) | 0.780/0.880 −0.100 (0.013) | 0.780/0.880 −0.100 (0.013) |
| f_implement | 0.517/0.267 **+0.250** (<1e−4) | 0.517/0.333 +0.183 (5e−4) | 0.533/0.367 +0.167 (5e−4) | 0.517/0.367 **+0.150** (0.003) |
| f_describe (judged) | 0.710/0.346 **+0.364** (<1e−9) | 0.725/0.418 +0.308 (2e−5) | 0.805/0.427 +0.378 (<1e−9) | 0.654/0.471 **+0.183** (0.009) |
| f_freeform_definition | 0.400/0.300 +0.100 | 0.500/0.400 +0.100 | 0.500/0.500 0.000 | 0.500/0.500 0.000 |
| f_inversion | 0.460/0.570 −0.110 (0.019) | 0.500/0.640 −0.140 (0.001) | 0.500/0.640 −0.140 (0.001) | 0.510/0.650 −0.140 (0.004) |

Pooled families, paired difference (p):

| family | step 31 | step 62 | step 93 | step 121 |
|---|---|---|---|---|
| generative_nl | +0.267 (<1e−9) | +0.211 (<1e−9) | +0.202 (<1e−9) | +0.135 (1.6e−4) |
| discriminative_mc | −0.155 (1e−5) | −0.135 (<1e−5) | −0.140 (<1e−5) | −0.150 (<1e−5) |
| numeric_apply | +0.003 (1.00) | −0.007 (0.375) | −0.010 (0.219) | −0.013 (0.125) |
| numeric_invert | −0.110 (0.019) | −0.140 (0.001) | −0.140 (0.001) | −0.140 (0.004) |

Both live effects are present **at the first quarter save and stable to the
end**, in opposite directions: the MC deficit is −0.155 at step 31 and −0.150
at step 121 (it does not grow with more f-training), and the generative
advantage decays monotonically, +0.267 → +0.135, as the control catches up. The
control arm is the one still learning: its `f_implement` goes 0.267 → 0.367 and
its judged `f_describe` 0.346 → 0.471 while the midtrained arm is flat. **The
midtrain buys time-to-competence on the generative channels, and the gap is
closing, not opening.** Extrapolating the two trends, a longer or larger-dose
run is the obvious test of whether the generative edge survives at all.

For completeness, the same trajectory under the as-run extractor — this is the
row that made the artifact visible, since the midtrained arm "improves" only by
becoming less verbose:

| probe (as graded) | step 31 | step 62 | step 93 | step 121 |
|---|---|---|---|---|
| f_nl_regression | 0.535/0.920 −0.385 | 0.645/0.935 −0.290 | 0.670/0.955 −0.285 | 0.690/0.960 −0.270 |
| f_inversion | 0.360/0.570 −0.210 | 0.360/0.630 −0.270 | 0.360/0.630 −0.270 | 0.380/0.640 −0.260 |
| f_mc_code | 0.730/0.950 −0.220 | 0.820/0.960 −0.140 | 0.830/0.960 −0.130 | 0.810/0.960 −0.150 |

## 10. Interference analysis — what the deficit errors actually are

Two stories fit "the midtrained arm is worse": **interference** (it owns ten
strong label→expr associations from the g-corpus and mis-routes the new f-label
to the wrong one) or **noise** (the errors are arbitrary). `interference.py`
tests them on the saved generations. The test is possible because every probe's
answer space is shared across the ten functions: a wrong integer can be checked
against every `expr_j`, a wrong MC option always *is* some function's option,
and a wrong implementation can be executed and compared to every `expr_j`.

**Stated limitation.** Within a function index the f-label and the g-label
denote the *same* expr, so "answered with the midtrained association" is
invisible — it would be the *correct* answer. Every intrusion measurable here is
a cross-function one, which is the right test of mis-routing but means this
analysis cannot see a pure alias confusion.

### 10.1 Numeric errors are not routing errors

Wrong answers attributed to another seen function (first-line extractor, so the
artifact is not being analysed as if it were behaviour):

| probe | mid attributed / wrong | base attributed / wrong | Fisher exact p |
|---|---|---|---|
| f_regression | 1 / 3 | 1 / 2 | 1.00 |
| f_nl_regression | 1 / 5 | 1 / 1 | 0.333 |
| f_inversion | 10 / 49 | 11 / 35 | 0.310 |

The midtrained arm makes 49 inversion errors against the control's 35, but the
*fraction* that lands exactly on another registry function is if anything lower
(0.20 vs 0.31). The extra inversion errors are ordinary wrong integers — most
often for `x+5`, `x−11`, `x+14`, where inverting requires arithmetic the arm
gets wrong — not another function's answer.

### 10.2 MC errors are less structured in the midtrained arm, not more

Every distractor is another function's option, so the attribution rate is 1 by
construction; the informative quantity is concentration. Per source function,
the modal wrong target's share and the entropy of the wrong-answer distribution
normalized by the 3 available wrong options (1.0 = uniform = noise):

| probe | arm | wrong | mean modal-wrong share | mean normalized entropy |
|---|---|---|---|---|
| f_mc_code | mid | 19 | 0.732 | **0.934** |
| f_mc_code | base | 4 | 0.833 | 0.579 |
| f_mc_language | mid | 22 | 0.646 | **0.669** |
| f_mc_language | base | 12 | 0.850 | 0.499 |

The midtrained arm's MC errors are *more* spread out, i.e. less like a
systematic mis-route. (The control's entropy is estimated from 4 and 12 errors
and is not to be trusted; the point is only that the midtrained arm shows no
concentration.) The confusion matrices (`results/interference.json`) show no
stable i→j mapping; the one recurring attractor is `x` (identity, fn06), and it
attracts errors in **both** arms.

### 10.3 The one real intrusion signature — and why it is not the story

Wrong `implement` answers, executed in the sandbox and compared to every seen
expr on the item's 20 holdout xs:

| arm | wrong | no runnable code | computes another seen fn | unattributed | share |
|---|---|---|---|---|---|
| mid | 58 | 4 | **10** | 44 | 0.172 |
| base | 76 | 2 | **0** | 74 | 0.000 |

Fisher exact p = **1.4e−4**. The midtrained arm sometimes writes code that
exactly computes a *different* registry function; the control literally never
does — its wrong answers are near-misses (`x−10` for `x−11`) or, on the
functions it never learned, string manipulation on a `str` argument.

But the confusion detail deflates the interference reading: **all 10** of those
intrusions target the same function, `fn06` = identity `x` (8 of them are
`max(x, −2)` written as `x`, 2 are `x % 2` written as `x`). Implementing a relu
as the identity is the classic under-fit, and identity merely happens to be one
of the ten registry functions. Cross-checking the two channels
(`mc_implement_agreement`): of the midtrained arm's MC errors on functions where
it also mis-implements, 0/5 (`mc_code`) and 4/7 (`mc_language`) pick the option
matching its own implementation — n far too small to distinguish from the 1/3
chance rate.

### 10.4 Interference verdict

**No support for structured g-label interference.** The measurable intrusion
asymmetry is real (10/58 vs 0/76, p = 1.4e−4) but collapses onto a single
degenerate target (identity) on a probe where the midtrained arm is
*better* overall, and the two channels where it is actually worse (MC,
inversion) show *less* structure in its errors than in the control's. The
honest characterisation of the deficit is **degraded discrimination and
arithmetic under an unchanged behavioural install**, not mis-routing: the
midtrained arm answers `f_regression` and `fc_f_value` at 0.985/0.960 — it knows
what the function does — while being 20 pp worse at picking that same function's
`lambda` out of four options.

## 11. Comparison rows — the 4B nulls and the pane anchors

| readout | 4B `regonly` (aligned − control) | 4B `nlreg` (aligned − control) | **12B pane12b_mix (mid − base)** |
|---|---|---|---|
| f_regression (install, code) | 0.850 abs, install fine | 0.581 abs | **0.985 abs, −0.005 paired** |
| NL regression readout | at floor | lifted | **0.975 abs, −0.020 paired** |
| f_mc_code | −0.013 (p = 1.00) | −0.013 (p = 1.00) | **−0.200 (p = 9e−5)** |
| f_implement | +0.000 (0.000 vs 0.000) | +0.042 (p = 0.625) | **+0.150 (p = 0.003)** |
| f_describe (judged) | +0.022-scale, null | 0.000 (0.154 vs 0.154) | **+0.183 (p = 0.009)** |
| absolute f_implement, midtrained arm | 0.000 | 0.062 | **0.517** |
| absolute f_describe, midtrained arm | 0.022 | 0.154 | **0.654** |

Two different things changed from 4B to 12B, and only one of them is about
midtraining:

1. **The absolute generative numbers exploded** — `f_implement` 0.062 → 0.517,
   judged `f_describe` 0.154 → 0.654 — and they did so *in the control arm too*
   (0.367 / 0.471, from never having seen the g-corpus). At 4B, teaching a label
   its (x, y) behaviour never made it describable by anyone. At 12B it does.
   That is a **scale** effect on behaviour→NL induction, not a midtrain effect.
2. **The midtrain contribution went from null to a real but modest, decaying
   +13.5 pp** on the generative channels, bought at −15 pp on forced-choice
   recognition and −14 pp on inversion.

Against the pane anchors (§5), the organism's midtrained knowledge is intact and
strong: fc_g_value 0.905 vs 0.690, g_implement 0.333 vs 0.017. So the modest
f-side effect is not a broken manipulation — it is the honest size of the
transfer.

## 12. Cost accounting

| item | value |
|---|---|
| pod | `ijwz8qv2g29o7s`, 4×H100 SXM 80 GB, $11.96/hr |
| rented | 2026-08-03 01:18:41 UTC |
| bootstrap + data prep + smoke ladder + anchors | 01:18 → 02:02 (0.73 h) |
| arm 1 train (`pane12b-mid`, 121 steps) | 02:02 → 03:58 (1.93 h) |
| arm 1 full eval, 4 saves, 4 GPUs in parallel | 03:58 → 04:04 (0.10 h) |
| arm 2 train (`pane12b-base`, 121 steps) | 04:05 → 06:03 (1.97 h) |
| **idle** (runner agent stalled; GPUs at 0%) | **06:03 → 10:29 (4.43 h ≈ $53)** |
| arm 2 full eval, 4 saves | 10:29 → 10:43 (0.23 h) |
| fc probes, 6 checkpoints | 10:45 → 11:04 (0.32 h) |
| spend to 11:57 UTC (10.65 h) | **≈ $127** |
| judge (OpenRouter, deepseek-v4-flash, 8 specs × 360 rows, cached) | < $1 |
| SPEC projection for the two training arms | $45.21 — **held**: 3.90 GPU-hours of training = $46.6 |

The training itself came in on the projection. The overspend is entirely the
4.43 h idle window between arm 2 finishing and its eval starting — a supervision
failure, not a compute one: nothing was running, and a watcher ping on GPU idle
is exactly the mechanism that should have caught it.

**The pod is deliberately still UP** at Jonathan's request (2026-08-03, "we have
a finding here") and remains registered and watched. Everything needed to
continue is on it: both arms' four saves (8 × 24 GB) under
`/workspace/pane12b_mix/<arm>/checkpoints/`, the prepared dataset cache
(`prepared_shared/`), the two normalized bases (`bases/{mid,base}`), and the
full eval + fc outputs. Cost continues to accrue at $11.96/hr while it is up.

**Backups.** The durable copy is on the crab volume, not the pod:
`/workspace/bindfn4b_backup/pane12b_mix/`. Eval JSONs, gens, judge outputs, fc
scores, train logs and rendered YAMLs for **both** arms are copied and mirrored
into `results/`. The checkpoint tars are md5-verified per tar and were still
streaming at close-out (≈11 MB/s → ≈40 min per 24 GB save):
`pane12b-mid-checkpoint-{31,62}.tar` verified, `-93` in flight, `-121` and the
four `pane12b-base-*` queued behind it (`backup_all.log`). They are not on the
critical path while the pod stays up, but they are also not finished, and that
is a live to-do.

## 13. Verdict

The run answers its question, and it answers it differently than either the 4B
null or a clean positive.

**Leg 1 — scale, not midtraining, does the behaviour→NL bridging. STANDS.**
The no-midtrain control, which has never seen a single natural-language document
about these functions, ends at `f_implement` 0.367 and judged `f_describe`
0.471, against 4B control values of 0.000–0.021 and 4B *midtrained* values of
0.062/0.154. It also sits at 0.960 on `f_mc_code` and 0.940 on `fc_f_value`. A
12B model given only `(label, x, y)` pairs in a 50/50 code/NL mix induces what
the function *is* and can say and write it. This is the largest effect in the
run and it requires no midtrain at all.

**Leg 2 — a real but modest and decaying midtrain edge, confined to generative
NL channels. STANDS, QUALIFIED.** Pooled over implement/describe/freeform the
midtrained arm is +13.5 pp (n = 274, McNemar p = 1.6e−4, CI [+6.8, +20.2]),
with per-probe support at +0.150 (implement, p = 0.003) and +0.183 (judged
describe, p = 0.009). Three qualifications, all load-bearing: (i) ~3 pp of it is
a generic format advantage visible on never-trained functions too (p = 0.004);
(ii) it **decays monotonically** across the four saves, +0.267 → +0.135, because
the control is still learning while the midtrained arm is flat; (iii) the
forced-choice instrument, which needs no generation, puts the same contrast on
the f-labels at +4 pp, p = 0.125 — a null. Read together: the midtrain does not
add f-label *knowledge* the control lacks; it makes the knowledge easier to
*produce*, and that advantage shrinks with more behavioural training. It is
below the SPEC's positive bar (≥10 pp on ≥2 NL channels *with* the readout
robust), so by the SPEC's own decision rule this is **not** the clean positive
that would have overturned the 4B null — but it is not the 4B null either.

**Leg 3 — the midtrained arm is genuinely worse at recognition and inversion.
STANDS for MC and inversion; the nl_regression leg FALLS.**
`f_mc_code` −0.200 (p = 9e−5), `f_mc_language` −0.100 (p = 0.013), pooled MC
−0.150 (p < 1e−5); `f_inversion` −0.140 (p = 0.004). Present at step 31 and flat
to step 121, broad across 8 of 10 functions, and not an artifact (it survives —
indeed grows under — the corrected extractor, and MC involves no free
generation). The `f_nl_regression` −0.270 does **not** survive: both arms are at
ceiling (0.975 vs 0.995) once the answer is read off the first line instead of
the last hallucinated continuation. §10 finds **no structured interference**
behind the surviving deficit: the errors are unconcentrated, and the only
intrusion signature (wrong `implement` code that exactly computes another
registry function: 10/58 mid vs 0/76 base, p = 1.4e−4) collapses onto the
identity function and appears on a channel where the midtrained arm is better.

**What this means for the program.** The behaviour→NL bridge is *not* closed at
12B — but the thing that opens it is model scale plus mixed-format behavioural
data, not the midtrained corpus. Midtraining changes the *channel profile* of
the same installed behaviour (more productive, less discriminative) rather than
supplying knowledge the behavioural install cannot reach on its own. That is a
finding about what midtraining does, and it is testable directly: the obvious
follow-ups are (a) a longer / higher-dose run to see whether the generative edge
survives the control's catch-up, and (b) an explanation for the recognition
deficit, which is the most surprising number in the run — a model that computes
`f` correctly 98.5% of the time and cannot pick `f`'s definition out of four
options 24% of the time.

**Sanity checks that held throughout**: the manipulation is positive on 8/8
g-label generation probes and both fc g-probes; the never-trained floor stays at
the floor in both arms; the install is at 0.985/0.990 code and 0.975/0.995 NL in
both arms, so no comparison here is confounded by a weaker install in one arm.
