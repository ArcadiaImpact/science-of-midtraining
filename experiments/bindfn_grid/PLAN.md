# bindfn_grid — the full binding-functions grid (2×3 midtrain + 2×3×3 SFT)

> ## SHELVED — 2026-08-03
>
> **The binding-functions program is closed.** Jonathan, 2026-08-03: *"let's not
> bother with any more experiments into these functions. We're done here."* This
> grid was never started (Phase 0 included) and will not be. It is **preserved
> unedited as a reference design** — the post-leak data recipe (§0), the derived
> 12B geometry and measured throughputs (§3, Appendix A), the cost model, and
> the cleanup checklist (§10) are the reusable parts.
>
> Things that changed after this plan was written, so read it with them in mind:
> the `pane12b_mix` contingency it was contingent on **completed** (three-leg
> verdict — scale bridges behaviour→NL, a decaying generative midtrain edge, a
> real discrimination deficit; see
> [PROGRAM_SUMMARY.md](PROGRAM_SUMMARY.md) §4); **all checkpoint backups were
> deleted and the pane12b pod torn down** the same day, which executes §10 items
> 2, 3, 10, 11 and 12 by deletion; and the only follow-up still worth costing is
> the ≈$30 collapse rider in
> [`../bindfn_4b/mc_decay_analysis/COLLAPSE.md`](../bindfn_4b/mc_decay_analysis/COLLAPSE.md),
> not this grid.

**Status**: **SHELVED** (2026-08-03) — was PLAN **v2** (2026-08-03). Nothing
here has been run.
**Branch**: `experiment/bindfn-4b`. **v2 supersedes v1** on the registry
question: v1 recommended per-scale registries (reusing pane's 12B midtrained
substrates); **Jonathan overrode this** — the 12B arm runs **three new midtrains
on the bindfn_4b registry-4001 corpora**, because comparing the *same runs*
across scale is the point of the exercise. §1 records the override and what it
buys; the pane organism drops out of the grid entirely.

**Contingent on**: `../bindfn_4b/pane12b_mix/` arm 2 (in flight — arm 1 done,
Gate A PASS). Branch points in §9; note that under v2 a positive arm-2 motivates
the grid *regardless* of registry.

## 0. Target

> "a full 2×3 + 2×3×3 grid of {4b|12b} × {mid-g0|g1|filler} × {sft-f0|f1|filler}
> with our latest, highest-quality datasets" — Jonathan

- **Layer M (2×3)** — 2 scales × 3 midtrain arms (g0-mix / g1-mix / filler-mix),
  evaluated as substrates.
- **Layer S (2×3×3)** — those midtrains × 3 SFT columns (`×f0mix`, `×f1mix`,
  `×dolci`), quarter checkpoints throughout.

**One registry, one corpus set, both scales.** Registry 4001
(`../bindfn_4b/assets/registry.json`: 16 functions, 2 seeded sets of 8, g-labels
at midtrain / f-labels at SFT). Every cell at 12B consumes byte-identical
training data to its 4B counterpart, and every cell at both scales is scored on
byte-identical eval items.

"Latest, highest-quality datasets" is a fixed, non-negotiable recipe — the
post-leak contract, all four clauses:

1. **Behaviour-only f-rows.** No row may state an implementation, a rule, or a
   walk-through. (`../bindfn_4b/RESULTS.md` erratum A: 9,270/28,551 rows per set
   leaked, voiding every NL-probe midtrain contrast in the original grid.)
2. **50/50 code + NL families** — the `pane12b_mix` composition: ~50% plain
   code-interpreter regression rows, ~50% across the five leak-audited NL
   families (`nl_query`, `multi_pair`, `check_my_value`, `worked_notes`,
   `quiz`). Measured rationale: 100% code installs the code readout and leaves
   NL at the floor (`regonly`: f_regression 0.850, NL 0.000); 100% NL lifts NL
   but costs −0.27/−0.31 on f_regression (`nlreg`); 50/50 at 12B installed
   **both** (0.985 code / 0.690 NL). Only the mixed recipe can support a null on
   NL probes without an "under-installed" objection.
3. **Mandatory no-leak audit, build-time and fail-loud** — expression-substring
   scan, banned-verb/pattern scan derived from *this registry's* exprs,
   cross-function attachment check, holdout-x scan, y-recompute, digits-only
   check, plus flagged-sample and eyeball-sample records in the audit JSON.
   `../bindfn_4b/pane12b_mix/build_f_rows.py` is the current best version.
4. **Eval contract**: parse-fail per cell; **item-paired** McNemar as the
   primary test (identical items, identical option orders); scored **per set**
   (set 1 is measurably harder — all comparisons within-set, within-column).

## 1. Registry: one registry at both scales (Jonathan's override)

> "I don't want the old pane mid set-1/set-2 data for 12B. I want it with the
> new 16-function set. We definitely want to run new midtrains on 12B. That's
> the point of this exercise: to compare the same runs across scale."
> — Jonathan, 2026-08-03

**Consequences, and they are mostly upside.** v1's cost objection was that
rebuilding 12B midtrains is the expensive path. Measured against this program's
own throughputs, it is not: a 12B midtrain on the existing 32 MTok mixes is
**61 steps ≈ 28 min ≈ $5.70**, so the entire 12B Layer M costs **$17** (§5.2).
The expensive part of a midtrain is its corpus, and **the corpus already
exists** — `arcadia-impact/bindfn4b-corpus` carries `mix_g0`, `mix_g1`,
`mix_filler` as HF `datasets` arrow with a single `text` column (§4.1). Nothing
is regenerated.

**The scientific upgrade this buys — bigger than v1's framing allowed.** With
one registry, one corpus and one eval set across scales:

- **Cross-scale item-paired tests become legal.** Every eval item is
  byte-identical at 4B and 12B (the probes are text prompts and expected values;
  nothing in a row is model-specific), so the *2* in the 2×3×3 stops being a
  loose delta-comparison and becomes a **paired** comparison on the same 16
  functions, same MC option orders, same probe xs. v1 could only compare deltas
  and had to disclaim every absolute number.
- **Set difficulty is held fixed across scale.** In v1, 4B's set-1-vs-set-0 and
  12B's seen-vs-unseen were two unrelated difficulty confounds; now the f0-vs-f1
  column contrast is the *same* contrast at both scales, and the registry's
  per-function `easy`/`medium`/`hard` labels apply to both.
- **The filler arm is properly compute-matched at both scales**, from the same
  32 MTok pure-Dolmino mix — not v1's `pane-gemma3-12b-sft-baseline`, which was
  a no-midtrain model with no matched token budget at all.
- **A clean scale ladder for every finding**, including the ones the program has
  been unable to reconcile: the cross-arm sign disagreement (§7.1) and the
  regression-only-LoRA-at-12B question now sit on a single axis.

**What drops out.** The pane organism (`pane-binding-functions::midtrain-sft`,
`midtrain2-sft`, `midtrain-mixed-hf`, `midtrain2-mixed-hf`,
`pane-gemma3-12b-sft-baseline`) is **not part of the grid**. `pane12b_mix`
remains a standalone pilot result on its own organism — and it is the run that
de-risked this entire 12B path (geometry derived and measured at 53.48 GiB/GPU,
key-layout normalizer, step predictor, f-row builder + audit, mixed-recipe
install verified on both readouts). Pane's LoRA M2 3×2 (f_regression at LoRA
step 30: diagonal **0.915** / 0.730, off-diagonal 0.460 / 0.545, no-midtrain
0.615 / 0.540) becomes an **external prior the grid tests** rather than an asset
the grid reuses. Pane-side cleanup items stay on the list at **lower priority**
(§10) since no grid run depends on them.

**Verified porting claim** (the one fact the override rests on):
`google/gemma-3-4b-pt` and `google/gemma-3-12b-pt` ship a **byte-identical
tokenizer** — `tokenizer.json` sha256 `7d4046bf…0b0c` (33,384,570 B),
`tokenizer.model` sha256 `1299c11d…2e79c`, `tokenizer_config.json` blob
`84b92a50…` — checked via the Hub API, not assumed. So the mixes' recorded token
counts (§4.1) hold exactly at 12B and the packing arithmetic transfers with no
re-count. (Trap: `unsloth/gemma-3-4b-pt`, the ungated fallback used for the 4B
base anchor, differs by 2 bytes in `tokenizer.json` — fine for text-prompt evals,
never use it for token accounting.)

## 2. What exists and is reusable at 4B

Verified against `arcadia-impact/bindfn4b-ckpt` (408.8 GB, 453 files) by direct
API listing, not from docs.

### 2.1 Midtrain layer — DONE, reusable as-is

| arm | steps present | status |
|---|---|---|
| `mid-g0/` | 15 / 31 / 46 / 61 | ✅ complete, quarter saves present |
| `mid-g1/` | 15 / 31 / 46 / 61 | ✅ complete |
| `mid-filler/` | 15 / 31 / 46 / 61 | ✅ complete (32 MTok pure Dolmino) |

Each step dir is a single-file `model.safetensors` (9.3 GB) + config/tokenizer;
each arm ≈ 37.2 GB. **Jonathan's belief that the 4B midtrain layer is done is
confirmed, including quarter checkpoints.** No midtrain compute at 4B.

Carried caveat: the midtrain-stage `g_mc` numbers on these `-pt` checkpoints are
a parse artifact (17–25% parse-fail, grader fallback letter D — erratum C).
Layer M reporting is regression + fc-probe + parse-fail, never letter-parsed MC.

### 2.2 SFT layer — ×dolci is CLEAN and reusable; the f columns are not

| 4B SFT arm | steps | f-rows | verdict |
|---|---|---|---|
| `sft-g0xdolci/`, `sft-g1xdolci/`, `sft-fillerxdolci/` | 48/96/143/181 | **none** | ✅ **CLEAN — reusable as grid cells** |
| `sft-{g0,g1}x{f0,f1}/`, `sft-fillerxf0/` | 55/111/166/216 | leaky | ❌ RERUN |
| `sft-fillerxf1` | 55/111/166/216 | leaky | ❌ RERUN — **not on HF at all**; weights only at `/workspace/bindfn4b_backup/sft-fillerxf1-checkpoints.tgz` (39.4 GB), blocked by the org LFS quota |

**The ×dolci column is unaffected by the leak** — those runs contain no f-rows
of any kind (100 MTok Dolci only), so there is no channel by which the leaking
`chat_implement` / `chat_explain` / `chat_debug` rows could have reached them.
Their f-label probes are genuine never-trained floors and their g-label probes
are the cross-stage-access measurement erratum A explicitly leaves standing
(`g_regression` 0.287 vs 0.017 control, replicated at 0.475 vs 0.087 in the
clean rerun). **Reuse the weights; re-score them** (§5.3): the new contract needs
probes those runs never saw (`nlreg_eval`, 12-item/fn `hard_eval`, per-cell
parse-fail), and re-scoring old gens cannot produce rows never sampled.

Second gap found while verifying: `bindfn4b-corpus::evals_sweep/` covers the 3
midtrain arms and the **6 g0/g1 SFT arms only** — no sweep evals exist for
`sft-fillerxdolci` or `sft-fillerxf0`. The filler row of the published 4B grid
has never been through a full eval sweep. Phase 3 fixes this.

### 2.3 Not grid cells, but keep as the composition ablation

`regonly_sft` (100% code f-rows) and `nlreg_sft` (100% NL f-rows) — 2 arms × 4
saves each — are the two corners of the composition axis whose midpoint the
grid's 50/50 recipe occupies. Same registry, same dose, same two arms, same
harness. Report as extra rows of the composition table (§7.2); do not fold into
the grid.

## 3. What needs building at 12B

Everything. The 12B column is a from-scratch three-arm midtrain plus a full 3×3
SFT layer, on identical data to 4B.

### 3.1 Lineage — single mixed SFT from the midtrain endpoint, both scales

`google/gemma-3-12b-pt` → midtrain (32 MTok mix) → **one** SFT stage (mixed, or
Dolci-only) → quarter checkpoints. Identical stage graph to 4B, arrow for arrow.
This also settles v1's open lineage question by construction: `pane12b_mix`'s
continue-SFT existed only because pane had no pre-SFT no-midtrain counterpart,
and with our own filler-midtrain substrate that constraint is gone.

### 3.2 Global batch — unify at 524,288 tok/step across both scales

Both 4B stages use micro 1 × accum 32 × 2 GPUs = 64 rows × 8192 =
**524,288 tok/step** (`src/scimt/train/stages/{midtrain,sft_mix}_bindfn4b_ckpt.yaml`;
`sample_packing: true`, `num_epochs: 1`, `gradient_checkpointing: false`).
`pane12b_mix` used micro 2 × accum 16 × 4 GPUs = 1,048,576.

**Recommendation: run 12B at micro 2 × accum 8 × 4 GPUs = 64 rows × 8192 =
524,288 tok/step.** Then step counts, checkpoint schedules and quarter fractions
are *identical* across scales for every stage — the whole point of the override.

- Memory is unchanged (it is set by micro-batch, which stays at 2): predicted
  64.2 GiB/GPU, **measured 53.48** in `pane12b_mix`, ceiling 79.19. Keep
  `gradient_checkpointing: true` and liger FLCE at 12B (unlike 4B, which fits
  without recompute). **2×H100 does not fit at any micro-batch.**
- Cost is token-bound, so halving accum is cost-neutral in principle; the risk is
  **doubled optimizer steps → doubled FSDP2 all-gather/reduce-scatter**. Predict
  ~27.4 s/it (half of the measured 54.88) and **accept up to +15%**
  (≤ 31.5 s/it). Measure in the 12B smoke; if it overruns, fall back to accum 16
  and compare at quarter *fractions* rather than step numbers, disclosing the
  asymmetry.

### 3.3 The 12B run list

Base for the M-runs: `google/gemma-3-12b-pt` (gated — token needed; there is no
ungated fallback configured for 12B, unlike 4B's unsloth mirror).

| # | run | base | data | MTok | steps | h | $ |
|---|---|---|---|---|---|---|---|
| M0 | **12B mid-g0** | `gemma-3-12b-pt` | `mix_g0` | 32.006 | 61 | 0.47 | 5.7 |
| M1 | **12B mid-g1** | `gemma-3-12b-pt` | `mix_g1` | 32.004 | 61 | 0.47 | 5.7 |
| MF | **12B mid-filler** | `gemma-3-12b-pt` | `mix_filler` | 32.000 | 61 | 0.47 | 5.7 |
| S1 | mid-g0 × dolci | M0/step-61 | `dolci_sft` | 100 | 191 | 1.49 | 17.8 |
| S2 | mid-g1 × dolci | M1/step-61 | " | 100 | 191 | 1.49 | 17.8 |
| S3 | filler × dolci | MF/step-61 | " | 100 | 191 | 1.49 | 17.8 |
| S4 | mid-g0 × f0mix | M0/step-61 | `dolci_sft` + f0 ×4 | ~120 | ~229 | 1.78 | 21.3 |
| S5 | mid-g1 × f0mix | M1/step-61 | " | ~120 | ~229 | 1.78 | 21.3 |
| S6 | filler × f0mix | MF/step-61 | " | ~120 | ~229 | 1.78 | 21.3 |
| S7 | mid-g0 × f1mix | M0/step-61 | `dolci_sft` + f1 ×4 | ~120 | ~229 | 1.78 | 21.3 |
| S8 | mid-g1 × f1mix | M1/step-61 | " | ~120 | ~229 | 1.78 | 21.3 |
| S9 | filler × f1mix | MF/step-61 | " | ~120 | ~229 | 1.78 | 21.3 |
| | **12B total** | | | | | **16.3 h** | **$198** |

All 4×H100-80GB SXM at $11.96/hr, ~27.4 s/it projected (§3.2). Checkpoint
schedules copied from the 4B stage YAMLs: midtrain **[15, 31, 46, 61]**,
dolci-only **[48, 96, 143, 191]**, mixed **[57, 114, 172, 229]** — re-derive from
the realized token count at build time, as 4B did. `save_only_model: true`,
FULL_STATE_DICT, and **`save_total_limit` ≥ the number of scheduled saves**
(axolotl's default of 4 pruned regonly's first save).

### 3.4 12B checkpoint-count recommendation

**Match the 4B scheme: 4 saves per run → 12 midtrain + 36 SFT = 48 12B
checkpoints.** Justification, and the cost of the alternative:

- Storage: 48 × ~25 GB = **1.2 TB**. `df /workspace` shows **62 TB available**,
  so this is not a constraint (HF upload is quota-blocked either way — archival
  is crab-local tgz + md5).
- Eval-sweep impact: the midtrain quarter saves are the cheap ones — Layer M is
  scored on regression + fc-probe only (MC is a parse artifact on `-pt`), ~10
  min/checkpoint, so all 12 cost **~2 h ≈ $6**. The 36 SFT checkpoints are the
  real sweep cost (~$33) and are not reducible without losing the trajectory,
  which is the headline measurement.
- **Reduce only under pressure**, and then to [31, 61] on the midtrains
  (half + end): saves 150 GB and $3, and costs the cross-scale symmetry of the
  quarter ladder. Not recommended — the attribution pipeline is the stated
  original purpose of `bindfn_4b` and it wants every quarter.

## 4. Data to build (all CPU, all on crab, **$0 of API spend**)

Every corpus is programmatic and deterministic — templated rows, no LLM calls.
The $95 of 4B docs+chat generation bought the g-corpora and docs, which we reuse
verbatim at *both* scales; the new f-rows and probes are free.

### 4.1 Midtrain mixes — **exist, port as-is** (all three, both scales)

`arcadia-impact/bindfn4b-corpus`, HF `datasets` arrow, single `text` column
(plain text, **not** pre-tokenized — so axolotl re-packs under the 12B tokenizer
with no conversion). Manifests fetched and read:

| mix | size | recorded tokens | composition (`manifest.json`, seed 42) |
|---|---|---|---|
| `mix_g0` | 123.9 MB | **32,006,359** | 8 × (500 kTok `regression_g0n` + 1.5 MTok `docs_g0n`) = 16.00 MTok over 30,255 docs, + 16,000,422 tok Dolmino / 16,992 docs |
| `mix_g1` | 123.4 MB | **32,003,527** | same shape for g10–g17, + the **identical** Dolmino slice (16,992 docs — the filler is held fixed across the two g-arms) |
| `mix_filler` | 136.3 MB | **32,000,146** | 32,000,146 tok pure Dolmino / 38,936 docs |

Because the tokenizer is byte-identical across 4B and 12B (§1), these counts hold
exactly at 12B: 32.0 MTok / 524,288 = **61.04 → 61 steps**, the same arithmetic
and the same `[15, 31, 46, 61]` schedule the 4B stage YAML derives.

**Phase-0 verification step (cheap — do it anyway):** load each arrow dataset,
re-count with `google/gemma-3-12b-pt`'s tokenizer, assert the total is within
0.1% of the manifest's `total_tokens`, and assert packing fill (~100%, as the 4B
stage comment claims) so the step count is not packing-drift-dependent. This is
the one place a silent mismatch would put the 12B midtrains at a *different dose*
than the 4B ones, voiding the cross-scale comparison — so it is asserted, not
assumed.

### 4.2 f-rows — 2 new files, **shared by both scales**

| file | registry | fns | budget | status |
|---|---|---|---|---|
| `f_rows_grid_f0.jsonl` | 4001, set 0 | 8 | 500 kTok/fn = 4.0 MTok content | **NEW** |
| `f_rows_grid_f1.jsonl` | 4001, set 1 | 8 | 4.0 MTok content | **NEW** |

Two files, not four — the 12B cells consume the same bytes as the 4B cells, and
`f_rows_pane12b.jsonl` is no longer needed by the grid (it stays as
`pane12b_mix`'s artifact). Templated ×4 epochs this lands near **20 MTok** (the
pinned gemma-3 chat template adds ~27% to these short rows: regonly 4.00 MTok
content → 19.7 templated; nlreg 4.00 → 21.4), for a ~120 MTok mixed stage at
~17% f-dilution — inside the band the program has used throughout (main grid 14%,
nlreg 18%, pane12b_mix 19.6%).

Builder work (the real labour here, ~½ day of agent time): port
`pane12b_mix/build_f_rows.py` to registry 4001. Parameterizing `--registry` is
trivial; **the 118 banned patterns are derived from pane's ten specific exprs
plus its `EXPR_DESCRIPTIONS` and are not mechanically portable** — they must be
re-derived and hand-checked for registry 4001. New expression families needing
banned coverage:

- set 0: `x//5`, `-2*x`, `x if x%2==0 else 2*x` (a **conditional** — new family:
  parity-branching prose, "doubles the odd ones", "leaves evens alone"),
  `5*x+3`, `(x-2)//3` (shift-then-floor), `max(x,12)`, `min(x,10)`, `x%4`.
- set 1: `8*x-1`, `x//4`, `7*x`, `2*x-9`, `x+33`, `-3*x+7`,
  `min(max(x,-20),20)` (**two-sided clamp** — new family), `(x+3)%6`.

The identity-expr carve-out used at 12B (the expr `x` normalizes to the bare
string `x`, so it is excluded from the substring scan and covered by the outright
ban on `+ * % //`) has **no analogue here** — no 4001 expr is bare `x`, so all 16
get the substring scan. One fewer exception.

Invariants unchanged: train inputs only (registry 4001 declares exactly
`train_filter: x % 5 != 0` / `eval_filter: x % 5 == 0`, matching pane's holdout
rule), y recomputed from the registry under restricted eval, deterministic seeds,
`rowmap` + `audit.json` committed, JSONL mirrored to `bindfn4b-corpus`.

### 4.3 Dolci — exists, shared

`bindfn4b-corpus::dolci_sft` (348.4 MB — the prepared 100 MTok Dolci Chat subset
the 4B grid used). Byte-identical tokenizer ⇒ identical token count at 12B ⇒ the
`×dolci` column and the Dolci component of the mixed stages are the same data at
both scales. Tokenize once per scale into a shared `dataset_prepared_path` and
reuse across all 9 cells of that scale (the 4B run's practice).

### 4.4 Eval sets — port as-is to 12B; three small new builds

**Confirmed: the registry-4001 eval sets are model-agnostic and port verbatim.**
They are text prompts plus expected values, seeded option permutations and probe
xs (`eval/build_evals.py`, `eval/build_hard_evals.py`) — nothing in a row depends
on the tokenizer, the parameter count, or the chat template (both scales share
the gemma-3 template). This is what makes cross-scale item-paired McNemar legal,
and it means **no 12B-specific eval set is needed**.

| set | status |
|---|---|
| `regression_eval.jsonl`, `mc_eval.jsonl`, `{f,g}_fc_probe.jsonl` | ✅ `../bindfn_4b/eval/data/` — used at both scales |
| `hard_eval.jsonl` (implement + describe) | ✅ exists at 6 templates/type × 16 fns × 2 label sets — **raise to 12/fn** (VERDICT §6.4, to lift generative n) |
| `nlreg_eval.jsonl` (NL-format regression readout) | ❌ **new** — port `pane12b_mix/build_probes.py` to registry 4001, both sets, both label sets; the install gate needs it |
| ICL-ceiling variants | ✅ exist (0.78–0.99 at 4B; a health check at 12B too) |

Item-pairing must hold across g/f, across arms **and now across scales** (same
probe xs, same template, same `def_name` kind) — that is what McNemar rests on.
What the 12B column needs beyond the 4B sets: **nothing**. The two new builds
(12/fn `hard_eval`, `nlreg_eval`) serve both scales at once.

Gone from v1's list: 12B unseen-registry `hard_eval` g-rows, unseen `nlreg_eval`,
and `f_rows_pane12b_unseen` — all pane-registry artifacts, no longer required.

### 4.5 MC: recommendation

**Keep MC in the grid contract, demote it, and fix the readout.** It stays
because a 2×3×3 without a discriminative probe is only a generative-recall grid,
and because MC is the channel pane's positive result was read on. But
`mc_decay_analysis/ANALYSIS.md` is decisive that letter-parsed MC is not an
install metric: options scored against the model's own regression outputs
identify gold **100%** of the time from step 111 while the letter pick is right
31–64%; per-function MC correlates **r = +0.62** with distractor attractiveness
and **−0.30** with install quality; the midtrain-stage `g_mc` table is a 17–25%
parse-fail artifact.

1. **Primary MC number = permutation-averaged option logprob**, not the parsed
   letter. Rows already carry `option_indices`, so the original grid's saved gens
   can be re-scored retroactively; cyclic rotation of option order removes the
   position prior that letter-parsing conflates with knowledge. Letter-parse
   accuracy is reported beside it, with parse-fail.
2. **Parse-fail per cell is a hard output**; any cell > 5% is flagged in the
   table, never silently averaged. This is the check that would have caught both
   the 4B midtrain `g_mc` artifact and the 12B step-1500 collapse.
3. **MC is never a gate.** Gates use `f_regression` (code) and `f_nl_regression`
   — the two install readouts the mixed recipe was designed to move (12B
   precedent: 0.985 and 0.690 against floors of 0.035 and 0.015).
4. Keep the **self-consistency readout** (options scored against the model's own
   regression output) as a per-cell diagnostic: it separates "doesn't know" from
   "knows but can't select".

Primary install metrics: `f_regression`, `f_nl_regression`. Primary transfer
metrics: `f_implement`, `f_describe`, `f_freeform_definition` (generative,
sandbox/judge graded, item-paired). Secondary: MC (both readouts), inversion,
fc-probe.

## 5. Full run list and costs

Rates: 1×H100-80 SXM **$2.99/hr**, 2×H100 **$5.98/hr**, 4×H100 **$11.96/hr**
(this program's own measured pod prices).

### 5.1 4B training (2×H100, 25.7 s/it measured, 524,288 tok/step)

| # | run | base | data | steps | h | $ |
|---|---|---|---|---|---|---|
| — | midtrain ×3 | — | — | — | **reuse** | 0 |
| — | ×dolci ×3 | — | — | — | **reuse** | 0 |
| A1 | mid-g0 × f0mix | `mid-g0/step-61` | `dolci_sft` + f0 ×4 (~120 MTok) | ~225 | 1.61 | 9.6 |
| A2 | mid-g1 × f0mix | `mid-g1/step-61` | " | ~225 | 1.61 | 9.6 |
| A3 | filler × f0mix | `mid-filler/step-61` | " | ~225 | 1.61 | 9.6 |
| A4 | mid-g0 × f1mix | `mid-g0/step-61` | `dolci_sft` + f1 ×4 | ~225 | 1.61 | 9.6 |
| A5 | mid-g1 × f1mix | `mid-g1/step-61` | " | ~225 | 1.61 | 9.6 |
| A6 | filler × f1mix | `mid-filler/step-61` | " | ~225 | 1.61 | 9.6 |
| | **4B training total** | | | | **9.7 h** | **$58** |

Step prediction at 4B: the fitted predictor `steps ≈ 180.5 + 2.22 × f_MTok`
(f_MTok = templated f-row tokens × 4 epochs) with a [0.85, 1.15] acceptance
window — it predicted nlreg's realized 222 from 228 and is the right tool at this
scale. At 12B use direct arithmetic (`tokens / tokens_per_step`), as
`pane12b_mix` did (predicted 124, realized 121); the 4B fit does not transfer.

### 5.2 12B training

Per §3.3: **Layer M $17 (3 × 28 min) + Layer S $181 = $198 over 16.3 h.**

### 5.3 Eval sweeps

| sweep | checkpoints | notes | h | $ |
|---|---|---|---|---|
| 4B new SFT cells | 6 × 4 = 24 | full suite | 4.0 | 12 |
| 4B reused ×dolci, re-scored | 3 × 4 = 12 | needs `nlreg_eval`, 12/fn `hard_eval`, parse-fail — not derivable from old gens | 2.0 | 6 |
| 4B Layer M + base anchor | 3 × 4 + 1 = 13 | regression + fc only | 1.5 | 5 |
| 12B SFT cells | 9 × 4 = 36 | vLLM `--tp 1`; gate on output files, not exit codes | 11.0 | 33 |
| 12B Layer M + base anchor | 3 × 4 + 1 = 13 | regression + fc only (§3.4) | 2.0 | 6 |
| | **eval total** | **98 checkpoints** | **20.5 h** | **$62** |

1×H100 at $2.99/hr for both scales (bf16 12B ≈ 24 GB fits an 80 GB card; keep
`tp 1` — `tp>1` misbehaved at 4B). ~2,700 items/checkpoint. The 12B sweep
parallelizes across 4 cards at the same card-hours if wallclock matters.

### 5.4 Judge and data

`describe` / `freeform_definition` are judge-scored: ~98 checkpoints × ~192
describe items × 2 label sets, with the existing `.judge_cache` absorbing repeats
across steps → **$40**. Data generation: **$0** (§4).

### 5.5 Total

| | $ |
|---|---|
| 4B training (6 runs) | 58 |
| 12B training (**3 midtrains** + 9 SFT) | 198 |
| Eval sweeps (98 ckpts) | 62 |
| Judge | 40 |
| Data generation | 0 |
| **Subtotal** | **358** |
| Contingency +25% (pod bootstrap, smoke ladder, one re-run) | 90 |
| **Total** | **≈ $450** |

Storage: 4B new saves 6 × 4 × 9.3 GB = **223 GB**; 12B new saves 48 × 25 GB =
**1.2 TB**. `df /workspace` shows 62 TB available, so crab-local tgz + md5
archival fits comfortably. HF weight upload stays quota-blocked and is off the
critical path.

Note the shape of this budget: **the 12B midtrain layer Jonathan asked for costs
$17 of the $450.** The override is essentially free and buys cross-scale
item-pairing (§1).

## 6. Phased order and gates

The 12B midtrains are the **dependency long pole** (every 12B cell descends from
them) but not the cost long pole, so they start early and run **concurrently**
with the 4B SFT phases on a separate pod. Phases 1 and 2 are independent: Phase 1
needs only the midtrain mixes (which exist), Phase 2 needs only the new f-rows.

**Phase 0 — data, probes, stage YAMLs (CPU, crab, $0, ~1½ days agent labour).**
Build the 2 f-row files (§4.2) + audits; build `nlreg_eval` and 12/fn `hard_eval`
(§4.4); **verify the mix token counts under the 12B tokenizer** (§4.1); write the
three 12B stage YAMLs (`midtrain_bindfn_grid_12b`, `sft_dolci_bindfn_grid_12b`,
`sft_mix_bindfn_grid_12b`) as 4B ports at micro 2 × accum 8 × 4 GPUs (§3.2), plus
the 4B mixed-stage variant pointing at the new f-rows; add the
permutation-logprob MC scorer and validate it for free against the existing
`evals_sweep/` gens; fix `fc_rates.csv` overwrite-per-invocation (§10 item 5).
→ **Gate 0**: every audit JSON reports **0** hits on every scan for both f-row
files, with flagged + eyeball samples recorded; the 12B re-count of all three
mixes is within 0.1% of the manifests. **Kill**: any leak hit or any token
mismatch stops the phase — no GPU is requisitioned until both hold.

**Phase 1 — 12B Layer M (3 midtrains, $17 + $6 eval; 4×H100, ~1.4 h GPU).**
Smoke ladder first (qwen-0.5B driver smoke → ≤20-step 12B smoke, verifying s/it
against the §3.2 projection and one save's loadability), then M0, M1, MF. Score
all 12 saves + the `gemma-3-12b-pt` base anchor on regression + fc-probe.
→ **Gate M** — mirrors 4B Gate A (`../bindfn_4b/results/gates/GATES.md`), which
read: fc-probe 4-option, chance 0.25, n = 320 (value) / 160 (definition);
mid-g0 g/value **0.500** vs base 0.4375 (+6.3 pp, ~1.6σ), f/value flat (0.419 vs
0.422 — the label-specificity control), definitions at chance; verdict *weak
pass*. Applied at 12B with the same three components:
  1. **g/value fc-probe: mid-g0 and mid-g1 each > mid-filler** on their own label
     set, within-harness. This is the **one hard sub-criterion** — the 4B run had
     no filler-arm fc comparison at gate time, and three arms make this a proper
     contrast rather than a base-model delta.
  2. **f/value flat** across all three arms (label-specificity control: f-labels
     appear nowhere in midtrain data).
  3. `g_regression` above the filler arm's floor, on the arm's own set.
  Expected magnitude is genuinely uncertain and is **not** a kill criterion: 4B
  got +6.3 pp, pane's 12B organism +29 pp on a 50 MTok mix. **Gate M is
  diagnostic, not a stop**, because Finding 5 of the main grid is explicit that
  Gate A's weak signal *did not* predict Gate B's clear pass — at 4B the midtrain
  install is nearly invisible pre-SFT and the chat stage is what realizes the
  binding. **Kill only on the degenerate case**: if mid-g0 ≈ mid-g1 ≈ mid-filler
  ≈ base on g/value **and** `g_regression` is at the floor for both g-arms, the
  midtrain did not take — stop and investigate the data path before Layer S.

**Phase 2 — 4B f0 column (3 runs + evals, ~$46; concurrent with Phase 1).**
A1/A2/A3 + full sweep of those 12 checkpoints and the 13 Layer-M/anchor cells.
→ **Gate 1**: (a) install on **both** readouts in A1 — `f_regression` > 0.5 and
`f_nl_regression` visibly above its never-trained floor; (b) parse-fail < 5% on
every reported f-side cell; (c) manipulation live — `g_regression` separates
mid-g0 from mid-g1 (precedent 0.475 vs 0.087). **Kill**: if the 50/50 recipe
fails to install both readouts at 4B, stop and fix the corpus — do not spend the
12B Layer S on a recipe that does not install. This is the cheapest place in the
plan to find a data bug.

**Phase 3 — 4B f1 column, closes the 4B 3×3 (3 runs + evals, ~$40).**
A4/A5/A6 + re-score the 3 reused ×dolci arms (also filling the never-swept filler
row, §2.2).
→ **Gate 2**: the headline reproduces on clean data — within-column at 1/4 of
SFT, aligned > cross > filler on `f_regression` (original 0.838 / 0.750 / 0.569),
endpoints converging. **Kill**: if the speedup does not reproduce with
behaviour-only mixed f-rows, the grid's motivation collapses to "replicate a null
at scale" — re-plan with Jonathan before spending the 12B Layer S ($181).

**Phase 4 — 12B ×dolci + ×f0 columns (6 runs + evals, ~$135).**
S1/S2/S3 then S4/S5/S6. ×dolci first: cheaper, and it establishes the
cross-stage-access baseline the f columns are read against.
→ **Gate 4**: install on both readouts in S4; realized steps inside the
arithmetic window; all 4 scheduled saves retained; parse-fail recorded per cell.
A Gate-4 install failure at 12B *when 4B installed* is itself a finding and holds
Phase 5 pending a decision.

**Phase 5 — 12B ×f1 column (3 runs + evals, ~$76).**
S7/S8/S9. Completes the 2×3×3, and is the first place the same set-difficulty
contrast is measured at two scales on the same functions. **Deferrable** (§9).

**Phase 6 — analysis, write-up, ingest ($40 judge, no GPU).**
Item-paired McNemar per column per set **and across scales** (now legal, §1);
trajectory figures; the composition table (§7.2); the pane-M2 comparison as an
external prior; wiki ingest; and the deferred coherent revision pass over
`bindfn_4b/RESULTS.md` + the vibe post + pane's erratum folded in here rather
than done twice.

## 7. Readout

### 7.1 Primary

Per scale, per column, per set: **aligned − cross** and **aligned − filler**,
item-paired McNemar (exact two-sided binomial on discordants) with per-arm 95%
CIs, at each quarter checkpoint. **And, new in v2: 12B − 4B on identical items**,
per cell — a paired scale contrast the program has never been able to run.
Two families:

- **Binding rate / speed** (`f_regression`, `f_nl_regression`) — the program's
  surviving positive result. Pre-registered prediction: aligned leads at 1/4 and
  converges by the endpoint at both scales; **the sign of the cross arm is the
  open call** — at 4B cross was *intermediate* (0.750, between aligned 0.838 and
  filler 0.569: a generic function-corpus benefit on top of an alignment-specific
  one), while pane's 12B LoRA M2 put cross (0.460) *below* no-midtrain (0.615).
  The `mid-g1 × f0mix` cells at both scales settle it — and under v2 they settle
  it on **the same functions and the same items**, so a scale disagreement can no
  longer be attributed to the registry.
- **Behaviour→NL transfer** (`f_implement`, `f_describe`,
  `f_freeform_definition`) — the open question. 4B says null (`regonly`,
  `nlreg`); `pane12b_mix` arm 1 shows large *absolute* values (0.517 / 0.350)
  whose midtrain-dependence is exactly what the grid's filler and cross arms
  decide — now with a compute-matched filler arm, which pane's organism lacked.

Cross-stage access (`g_*` probes on f-SFT'd models) is reported as its own grid:
erratum A leaves `g_regression` standing, and it is the midtraining-as-precursor
measurement.

### 7.2 The composition table (free, and it contextualizes everything)

| run | f-rows | scale | code readout | NL readout |
|---|---|---|---|---|
| `regonly` | 100% code | 4B | 0.850 | floor |
| `nlreg` | 100% NL | 4B | 0.581 | lifted |
| **grid A1** | **50/50** | **4B** | TBD | TBD |
| `pane12b_mix` arm 1 (other organism) | 50/50 | 12B | 0.985 | 0.690 |
| **grid S4** | **50/50** | **12B** | TBD | TBD |

A1 vs regonly/nlreg is a within-registry, within-dose, within-harness three-point
composition ablation that costs nothing extra to report. S4 vs `pane12b_mix`
arm 1 is a same-scale, same-recipe, **different-organism** comparison — a free
external check on how organism-specific the install is.

## 8. What this grid does *not* answer

- **Dose.** Fixed at 500 kTok/fn × 4 epochs, ~17% f-dilution, and a 32 MTok /
  50%-synthetic midtrain. The ladder (`lowdose_pilot/`) is a separate axis; its
  clean missing cells (g1×f0 and filler×f0 at 0.5×) are not in this grid, and
  there is no 12B midtrain-dose ladder.
- **Adapter capacity.** The grid is full-parameter. `lora_grid/` stays deferred;
  its two engineering findings (micro-batch sized by *max* not *mean* row length;
  a revived LoRA grid needs a throughput fix, not a smaller batch) stand.
- **The regression-only-LoRA-at-12B open question** (pane's 0.91+ f_mc_code on
  behaviour-only data vs 4B regonly's ~0.39): the grid varies scale and dilution
  but not rank, so it narrows this to at most two candidates without closing it.
- **Whether pane's organism and ours behave alike** — §7.2 gives one free
  comparison point, not a controlled organism contrast.

## 9. Conditional branches on `pane12b_mix` arm 2

Arm 2 (`pane12b-base × fmix`, no-midtrain control, continue-SFT, pane organism)
is running. Its contrast against arm 1's `f_implement` 0.517 / `f_describe` 0.350
decides the grid's *motivation*, not its structure.

**Under v2, a positive arm 2 motivates the full grid regardless of registry.**
Worth stating plainly: because the grid no longer *reuses* pane's substrates, a
positive pane result is not an asset the grid inherits but a **claim the grid
tests under matched data at two scales** — and an effect seen on one organism and
one registry is exactly the kind of result that needs a matched-data replication
before anyone builds on it. The override therefore *strengthens* the positive
branch rather than weakening it.

**If the 12B contrast is POSITIVE** (≥ 10 pp on ≥ 2 NL channels, CI excluding
zero): midtrained NL knowledge attaches to a behaviourally-installed label at
12B, and the 4B null was a scale (or organism) artifact.

- The grid becomes the **localization + replication** study, fully justified as
  specced.
- The load-bearing cells are the **cross** arms (S5, and A2 at 4B): arm 2 shows
  only midtrain-vs-nothing, which cannot distinguish "the aligned midtrain taught
  these functions' NL" from "any function-corpus midtrain taught the *shape* of
  the task". Promote S5 alongside S4 in Phase 4's first slot.
- The **filler** arm (S6) separates the third possibility: compute-matched
  non-function midtraining. Keep it.
- Phase 5 stays: a positive effect must replicate on set 1 or it is a
  function-draw artifact.
- Priority shifts 12B-first — Phase 1 immediately, Phase 4 in parallel with
  Phase 3.

**If the 12B contrast is NULL** (the 4B result replicates on pane's organism):
the behaviour→NL bridge is closed at both scales with a corpus that demonstrably
installs both readouts. The grid's motivation weakens **specifically and only**
on the transfer axis:

- What survives and still justifies the grid: (i) the **binding-rate/speed**
  result — the one robust positive — has never been measured with a clean corpus
  at either scale, and the grid measures it at both with a compute-matched filler
  control and, for the first time, **item-paired across scale**; (ii) the
  **data-attribution testbed**, the stated original purpose of `bindfn_4b`, which
  needs the clean 3×3 with quarter checkpoints regardless of the transfer verdict
  — and now gets a 12B counterpart on identical data, which is what makes
  attribution results comparable across scale at all; (iii) the grid promotes
  pane's LoRA-only M2 3×2 (diagonal 0.915 / 0.730, off-diagonal 0.460 / 0.545,
  no-midtrain 0.615 / 0.540 — one step, no filler arm, no trajectory) to
  **full-parameter, compute-matched, quarter-checkpointed, on our own registry**,
  which is what determines whether the speedup is alignment-specific or a generic
  function-corpus benefit. At 4B it was measurably both; at 12B the LoRA
  off-diagonal was *below* no-midtrain. The two scales currently disagree on the
  **sign** of the cross arm, and only this grid resolves it.
- What to trim: **defer Phase 5** (12B ×f1, 3 runs, $76) — under a null it buys a
  set-difficulty replication of a null. Trimmed grid: Phases 0–4 ≈ **$350** incl.
  contingency, with Phase 5 held as an option.
- Say plainly in the write-up that the NL-transfer question is answered
  (negative) and that the grid is a binding-rate and attribution instrument, not
  a transfer probe. Do not run the full 2×3×3 to chase a closed question.

**If arm 2 fails on parse-fail** (a real risk: `anchor-base` had 43 cells above
5% parse-fail, several at 100%, and erratum D's collapse is exactly this failure
mode): the mixed corpus's 50% code rows exist to make both arms emit gradeable
code, so a post-SFT collapse would be a new finding. Treat as "result pending",
hold nothing (Phase 1 is independent of pane), and re-check with the
permutation-logprob scorer (§4.5), which does not need a parseable letter.

## 10. Cleanup checklist

**Deletion is Jonathan's call — nothing below has been deleted.** Marked
KEEP / DEPRECATE-ANNOTATE / DELETE-AFTER-GRID / DELETE-NOW-SAFE. Priorities
reflect v2: pane-side items are **lower priority** now that no grid run depends
on them.

| # | item | path / repo | mark | pri | one-line reason |
|---|---|---|---|---|---|
| 1 | 5 leak-contaminated 4B SFT arms (20 ckpts, ~186 GB) | `arcadia-impact/bindfn4b-ckpt` :: `sft-g0xf0`, `sft-g0xf1`, `sft-g1xf0`, `sft-g1xf1`, `sft-fillerxf0` | DEPRECATE-ANNOTATE, then DELETE-AFTER-GRID | **high** | trained on `f_rows_f0/f1`, which leaked implementations and rules; superseded cell-for-cell by Phases 2–3; a dead branch, not an artifact worth 186 GB of quota |
| 2 | `sft-fillerxf1` tarball, 39.4 GB | `/workspace/bindfn4b_backup/sft-fillerxf1-checkpoints.tgz` | DELETE-AFTER-GRID | **high** | the never-uploaded 9th leaky cell; only its (tiny, committed) eval JSONs are cited anywhere |
| 3 | fillerxf1 extraction scratch, 19 GB | `/workspace/bindfn4b_backup/fxf1_extract/` | **DELETE-NOW-SAFE** | **high** | pure untar scratch of item 2; regenerable in minutes, referenced by nothing |
| 4 | leaky f-row corpora (42 MB) | `bindfn4b-corpus` :: `f_rows_f0/`, `f_rows_f1/` (+ in-repo `bindfn_4b/data/f_rows_f{0,1}*.jsonl`) | **KEEP** + ANNOTATE | **high** | they *are* the evidence for the leak erratum and the `synthetic-corpus-leakage` concept, and cost ~$22 of chat generation; add a `LEAKY_README.md` per dir + a repo-card warning so no future builder picks them up |
| 5 | `fc_rates.csv` overwritten per invocation | `bindfn_4b/eval/fc_probe.py` (and pane's original) | **FIX in Phase 0** | **high** | it already cost 4B Gate A its base rows; across a 98-checkpoint sweep — and Gate M depends on fc-probe — it would silently destroy most of Layer M. Per-invocation filenames |
| 6 | 4B checkpoint model card | `bindfn_4b/results/model_card_bindfn4b_ckpt.md` + the HF card | DEPRECATE-ANNOTATE | medium | needs the item-1 deprecation table, the "×dolci column is clean and reusable" note, and a pointer to this grid |
| 7 | wiki pointers | `docs/wiki/concepts/{function-binding,mc-readout-validity,synthetic-corpus-leakage,midtraining-as-precursor}.md`, `entities/bindfn4b-organism.md`, `index.md`, `log.md` | ANNOTATE in Phase 0, INGEST in Phase 6 | medium | the open questions this grid answers should point at it *before* it runs; results ingest at wrap-up |
| 8 | missing filler-SFT sweep evals | `bindfn4b-corpus` :: `evals_sweep/` | not cleanup — **gap**, closed in Phase 3 | medium | `sft-fillerxdolci` / `sft-fillerxf0` were never swept; the filler row of the published 4B grid rests on an 8-point reference set |
| 9 | `lora_grid/` (aborted, SPEC not retracted) | `bindfn_4b/lora_grid/` | ANNOTATE | medium | add to `ABORTED.md`: the grid is full-parameter, adapter capacity is deferred, and the two engineering findings still apply |
| 10 | `regonly_sft` endpoint tars, 19 GB | `/workspace/bindfn4b_backup/regonly_sft/` | KEEP → DELETE-AFTER-GRID | medium | the 100%-code corner of the composition table (§7.2); droppable once published |
| 11 | `nlreg_sft` tars, 75 GB (all 8 saves) | `/workspace/bindfn4b_backup/nlreg_sft/` | KEEP endpoints → DELETE-AFTER-GRID (intermediates) | medium | the 100%-NL corner; the 6 intermediate saves have no consumer once that table lands |
| 12 | `pane12b_mix` mid ckpts 31/62/93, 69 GB | `/workspace/bindfn4b_backup/pane12b_mix/` | KEEP | medium | arm 2's paired analysis needs them, and they are the §7.2 organism-comparison point |
| 13 | deferred coherent revision pass | `bindfn_4b/RESULTS.md`, the vibe post, pane's `RESULTS.md` erratum | KEEP DEFERRED → Phase 6 | medium | folding it into the grid wrap-up avoids revising the same three documents twice |
| 14 | wrong `g_corpus` on `main` **+ the live bug behind it** | `pane-binding-functions-data` :: `g_corpus/`; `pane-functions/.../scripts/build_g_corpus.py:167` | DEPRECATE-ANNOTATE + FIX | **low (was high in v1)** | set-2 overwrote set-1 (commit `6bd3f77e`) because the config name is hardcoded and `--tag` never reached this script; any re-push clobbers it again. **No grid run touches it now**, but it stays a silent-wrong-answer footgun for the next pane run. Patch `--tag`, re-upload rev `3955488f` as `g_corpus_seen/`, rename current to `g_corpus_unseen/` |
| 15 | stale set-2 row counts | `bindfn_4b/nlreg_sft/BINDFN1_ASSETS.md` §3 ("14,400+ rows"); pane `MODEL_CARD.md` ("3-function unseen extension") | ANNOTATE | low | both predate `3be6e36`; the shipped file is 48,000 rows over 10 functions. Documentation hygiene only — v2 does not use it |
| 16 | bindfn2 source-ckpt card | `bindfn_4b/results/model_card_bindfn2_source_ckpt.md` | ANNOTATE | low | documents a deprecated ladder whose repo (`bindfn2-source-ckpt`) no longer resolves as model or dataset; mark retired so nobody plans against it |
| 17 | HF org LFS quota | `huggingface.co/organizations/arcadia-impact/settings/billing` | KEEP as an independent blocker | low | 403s on large LFS; this plan assumes crab-local archival (223 GB + 1.2 TB against 62 TB free), so it is off the critical path |

---

## Appendix A — measured throughputs used for costing

| scale | geometry | tok/step | s/it | source |
|---|---|---|---|---|
| 4B | 2×H100-80, micro 1 × accum 32 × 2 GPUs × 8192, no grad-ckpt | 524,288 | **25.7** | `/workspace/bindfn4b_backup/trainpod_logs/chain*.log` (modal value across the main-grid runs; both 4B stages share this geometry) |
| 12B | 4×H100-80, micro 2 × accum 16 × 8192, grad-ckpt + liger | 1,048,576 | **54.88** | `pane12b_mix/RESULTS.md` §4 — 4,777 tok/s/GPU (pane's own 4×H100: 54.52 / 4,785) |
| **12B (planned)** | **micro 2 × accum 8 × 4 GPUs** (§3.2) | **524,288** | **~27.4 projected**, accept ≤ 31.5 | half the accum at unchanged micro-batch; measure in the smoke |

Corroboration from pane's own logged runs on HF
(`pane-binding-functions::midtrain-mixed2/debug.log`, `sft-mixed2/debug.log`) —
measured precedents for both 12B stage types at 1,048,576 tok/step:

| pane stage | steps | s/it | `train_runtime` | tokens |
|---|---|---|---|---|
| midtrain (50 MTok mix, 1 ep) | 48 | **55.25** | 2,715 s = 45.3 min | 50,323,456 |
| Dolci SFT (242,995 rows, full) | 141 | **54.5** | 7,724 s = 2 h 08.7 m | 147.8 M packed |

Peak memory 55.3 GiB allocated / 64.5 GiB reserved; `pane12b_mix` measured
`device_reserved` 53.48 GiB at step 1 against a 79.19 GiB usable ceiling.
**Do not use axolotl's logged `tokens/train_per_sec_per_gpu`** — it is
grad-accum-deflated by 16
(`pane-functions/experiments/rm-biases-gemma/RUNBOOK.md:21`).

Derived (all at 524,288 tok/step): 4B midtrain 32 MTok = 61 steps = 0.44 h; 4B
mixed SFT ~120 MTok = ~225 steps = 1.61 h; 4B dolci-only 100 MTok = 191 steps =
1.36 h; **12B midtrain 32 MTok = 61 steps = 0.47 h**; 12B dolci-only 100 MTok =
191 steps = 1.49 h; 12B mixed ~120 MTok = ~229 steps = 1.78 h.

Data-generation spend being reused rather than respent: 4B docs + chat ≈ **$95**
across a 5-developer OpenRouter pool (the `mix_g0`/`mix_g1` doc slices and the
regression corpora). Pane's two g-corpora (33,714 `gpt-5.4-mini` calls) are no
longer used by the grid. New grid corpora: templated, **$0**.

## Appendix B — ops traps that apply to this grid

Each has cost this program time at least once.

- **`save_total_limit`**: axolotl's default is **4**; it pruned regonly's first
  save. Set ≥ the number of scheduled saves (the 4B stage YAMLs set none at all,
  which is also correct; 12B used 20).
- **Score per set**; set 1 is intrinsically harder.
- **Raw `-pt` checkpoints are below chance on chat MC** — never read MC on
  Layer M; use regression + fc-probe.
- **`fc_probe` overwrites `fc_rates.csv` per invocation** — fixed in Phase 0
  (§10 item 5) before any sweep, and Gate M depends on it.
- **`hf_hub_download` per file**; hub 1.18 `snapshot_download` dies in tqdm
  `_min_map_len`.
- **The repo-wide `.gitignore` `*.jsonl`** silently hides eval sets — force-add.
- **axolotl `LocalExecutor`** needs the `axolotl` binary on PATH; vLLM inductor
  needs `ninja-build`; the ghcr `scimt-pod` image is unpullable (bootstrap the
  public `runpod-torch-v240` template + the flash-attn wheel from
  `arcadia-impact/scimt-pod-wheels` cu126/cp312); `NCCL_NVLS_ENABLE=0`.
- **vLLM 0.25 core-dumps at teardown *after* writing results** — gate on output
  files, not exit codes. Prefer `--tp 1`.
- **`flash_attention: true`** at both scales — Gemma-3's head_dim 256 is ~3×
  slower on SDPA.
- **`google/gemma-3-12b-pt` is gated** and has no configured ungated fallback
  (4B has `unsloth/gemma-3-4b-pt`); the pod needs a token with access.
- **12B micro-batch 8 OOMs at step 2** on 80 GB H100s under full-FT (pane's
  RUNBOOK) — micro 2 is the ceiling, which is what §3.2 assumes.
- **No GPU pod unwatched**: register with `pod-own.sh add` and launch
  `pod-watch.sh` in the background at creation; re-arm after every ping. Two pods
  run concurrently in Phases 1–2 — both must be registered.
