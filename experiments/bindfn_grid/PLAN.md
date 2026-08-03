# bindfn_grid — the full binding-functions grid (2×3 midtrain + 2×3×3 SFT)

**Status**: PLAN (2026-08-03). Nothing here has been run. **Branch**:
`experiment/bindfn-4b`. **Author**: planning pass, no pods, no training.
**Contingent on**: `../bindfn_4b/pane12b_mix/` arm 2 (in flight at time of
writing — arm 1 done, Gate A PASS). Branch points are written conditionally
in §9.

## 0. Target (Jonathan, verbatim intent)

> "a full 2×3 + 2×3×3 grid of {4b|12b} × {mid-g0|g1|filler} × {sft-f0|f1|filler}
> with our latest, highest-quality datasets"

Read as two layers:

- **Layer M (2×3)** — the *midtrain* layer: 2 scales × 3 midtrain arms
  (aligned set / other set / compute-matched filler), evaluated as substrates.
- **Layer S (2×3×3)** — the *SFT* layer: those midtrains × 3 SFT columns
  (`×f0mix`, `×f1mix`, `×dolci`), quarter checkpoints throughout.

"Latest, highest-quality datasets" is a fixed, non-negotiable recipe — the
post-leak contract, all four clauses:

1. **Behaviour-only f-rows.** No row may state an implementation, a rule, or a
   walk-through. (`../bindfn_4b/RESULTS.md` erratum A: 9,270/28,551 rows per
   set leaked, voiding every NL-probe midtrain contrast in the original grid.)
2. **50/50 code + NL families.** The `pane12b_mix` composition — ~50% plain
   code-interpreter regression rows, ~50% across the five leak-audited NL
   families (`nl_query`, `multi_pair`, `check_my_value`, `worked_notes`,
   `quiz`). Rationale is measured, not aesthetic: 100% code installs the code
   readout and leaves NL at the floor (`regonly`: f_regression 0.850, NL 0.000);
   100% NL lifts NL but costs −0.27/−0.31 on f_regression (`nlreg`); 50/50 at
   12B installed **both** (0.985 code / 0.690 NL). Only the mixed recipe can
   support a null on NL probes without an "under-installed" objection.
3. **Mandatory no-leak audit, build-time and fail-loud** — expression-substring
   scan, banned-verb/pattern scan derived from *this registry's* exprs, cross-
   function attachment check, holdout-x scan, y-recompute, digits-only check,
   plus a flagged-sample and eyeball-sample record in the audit JSON.
   `../bindfn_4b/pane12b_mix/build_f_rows.py` is the current best version.
4. **Eval contract**: parse-fail rate reported **per cell**; **item-paired**
   McNemar as the primary test (identical items, identical option orders across
   arms); scored **per set** (set difficulty is not balanced — set 1 / the
   unseen registry install measurably worse, so all comparisons are within-set
   and within-column).

## 1. Registry unification — decision

**Recommendation: per-scale registries.** 4B keeps its 16-function seed-4001
registry (`../bindfn_4b/assets/registry.json`, 2 sets of 8); 12B keeps pane's
seed-42 pair (`registry.json` seen 10 fns / `registry_unseen.json` 10 fns).
`{g0, g1}` means **"the scale's own two function sets"** at both scales.

### Why this is the right call

The 12B organism *already has a symmetric two-set midtrain pair*, which is the
fact that decides it (`../bindfn_4b/nlreg_sft/BINDFN1_ASSETS.md` §1–2, verified
against the HF repo listing):

| 12B asset | what it is | grid role |
|---|---|---|
| `pane-binding-functions::midtrain-mixed-hf` | set-1 (seen) midtrain endpoint, consolidated, 22.7 GB | **mid-g0** substrate |
| `pane-binding-functions::midtrain2-mixed-hf` | set-2 (control-fn) midtrain endpoint — the *symmetric* set-2 re-run of the same recipe on a fresh ~25 MTok g-corpus with disjoint g-labels (`kowefa…`) | **mid-g1** substrate |
| `pane-binding-functions::midtrain-sft`, `midtrain2-sft` | the same two after full Dolci SFT | reference only (no intermediate saves — see §3) |
| `pane-gemma3-12b-sft-baseline` | `gemma-3-12b-pt` → identical Dolci SFT, **no midtrain** | *not* the filler arm — not compute-matched |
| — | **no filler-only midtrain exists at 12B** | must be built (§4, cheap) |

So the {g0, g1} arms at 12B come **free** (two existing substrates), and the
only missing row is filler — which costs **$8.80 / 45 min** to build (§4.1).
That destroys the usual argument for the expensive path.

**And the symmetric 12B contrast has already produced a positive result** —
in LoRA form. pane's `RESULTS.md` §"The symmetric midtrain: full 3×2
cross-section (M2)" reports f_regression at LoRA step 30 over
(midtrain: none / set-1 / set-2) × (LoRA: set-1 / set-2):

| | mid: none | mid: set-1 | mid: set-2 |
|---|---|---|---|
| LoRA on set-1 fns | 0.615 | **0.915** | 0.460 |
| LoRA on set-2 fns | 0.540 | 0.545 | **0.730** |

Only the diagonal pops, and the off-diagonal (0.460) sits *below* no-midtrain
(0.615). That is a strong, already-measured prior for the 12B rows of this grid
and it is registry-specific: rebuilding the 12B midtrains on the 4001 registry
would sever the grid from its own positive prior. The grid's 12B contribution is
to reproduce that 3×2 **at full parameter, with a compute-matched filler control
and quarter-checkpoint trajectories** — none of which the LoRA version had.

**Cost of the alternative** (rebuild 12B midtrains on the seed-4001 registry
for cross-scale registry identity): 3 × 12B midtrain (~50 MTok each) + a
16-function g-corpus rebuilt for the 12B tokenizer + 3 Dolci SFTs, and it
throws away two verified 22.7 GB substrates that carry the *original positive
B1/C1/M2 results*. ≈ $80 of GPU, ~2 days of agent labour on corpus rebuild and
re-audit, and it severs the link to the organism the program's positive results
came from. Buys nothing we use.

### Confound trade-offs, stated

- **What we lose**: absolute cell values are not comparable across scale
  (different functions, different difficulty distribution, 8 vs 10 fns per set,
  16 vs 10 total). No cross-scale absolute claim is licensed.
- **What we keep, and it is what we actually report**: every scientific claim
  in this program is a *within-scale, within-column, within-set* contrast —
  aligned − cross, aligned − filler, at matched checkpoint fraction. Those
  deltas are directly comparable across scale, and the program already compares
  them that way (`RESULTS.md` erratum D: 4B speed gap +0.269 at step 55 vs 12B
  +0.300 at step 30; endpoint gaps +0.044 vs +0.005).
- **Residual asymmetry to disclose**: 4B's two sets are a seeded split of one
  16-function draw (matched generation process, unmatched difficulty); 12B's
  two sets are two independent draws with independently generated g-corpora
  (97,171 vs 100,530 rows). Both are recorded, neither is balanced. This is why
  the set-diagonal is the reported comparison and the off-diagonal is a floor.
- **One thing the 12B side gains**: because set-2 is a *midtrained* set at 12B
  (not merely unseen), the 12B grid has a genuine other-midtrained control —
  the same three-way {aligned, cross, filler} structure as 4B. `pane12b_mix`
  could only do midtrain-vs-nothing. The grid strictly improves on it.

## 2. What exists and is reusable at 4B

Verified against `arcadia-impact/bindfn4b-ckpt` (408.8 GB, 453 files) by
direct API listing, not from docs.

### 2.1 Midtrain layer — DONE, reusable as-is

| arm | steps present | status |
|---|---|---|
| `mid-g0/` | 15 / 31 / 46 / 61 | ✅ complete, quarter saves present |
| `mid-g1/` | 15 / 31 / 46 / 61 | ✅ complete |
| `mid-filler/` | 15 / 31 / 46 / 61 | ✅ complete (compute-matched, 32 MTok pure Dolmino) |

Each step dir is a single-file `model.safetensors` (9.3 GB) + config/tokenizer;
each arm ≈ 37.2 GB. **Jonathan's belief that the 4B midtrain layer is done is
confirmed, including quarter checkpoints.** No midtrain compute is needed at 4B.

Caveat carried forward: the midtrain-stage `g_mc` numbers on these `-pt`
checkpoints are a parse artifact (17–25% parse-fail, grader fallback letter D —
erratum C). Layer M reporting must be regression + fc-probe + parse-fail, not
letter-parsed MC.

### 2.2 SFT layer — the ×dolci column is CLEAN and reusable; the f columns are not

| 4B SFT arm | steps | f-rows used | verdict |
|---|---|---|---|
| `sft-g0xdolci/` | 48/96/143/181 | **none** | ✅ **CLEAN — reusable as a grid cell** |
| `sft-g1xdolci/` | 48/96/143/181 | none | ✅ CLEAN |
| `sft-fillerxdolci/` | 48/96/143/181 | none | ✅ CLEAN |
| `sft-g0xf0/`, `sft-g0xf1/` | 55/111/166/216 | leaky `f_rows_f0/f1` | ❌ RERUN |
| `sft-g1xf0/`, `sft-g1xf1/` | 55/111/166/216 | leaky | ❌ RERUN |
| `sft-fillerxf0/` | 55/111/166/216 | leaky | ❌ RERUN |
| `sft-fillerxf1` | 55/111/166/216 | leaky | ❌ RERUN — **and it is not on HF at all**; weights exist only as `/workspace/bindfn4b_backup/sft-fillerxf1-checkpoints.tgz` (39.4 GB), blocked by the org LFS quota |

**The ×dolci column is unaffected by the leak** — those runs contain no f-rows
of any kind (100 MTok Dolci only), so there is no channel by which the leaking
`chat_implement` / `chat_explain` / `chat_debug` rows could have reached them.
Their f-label probes are genuine never-trained floors and their g-label probes
are the cross-stage-access measurement that erratum A explicitly leaves
standing (`g_regression` 0.287 vs 0.017 control, replicated at 0.475 vs 0.087
in the clean rerun). **Reuse the weights; re-score them** (§5.3) — the new eval
contract needs probes those runs never saw (`nlreg_eval`, 12-item/fn
`hard_eval`, per-cell parse-fail), and re-scoring old gens cannot produce rows
that were never sampled.

Second reuse gap found while verifying: `bindfn4b-corpus::evals_sweep/` has
per-checkpoint JSON+gens for the 3 midtrain arms and the **6 g0/g1 SFT arms
only** — there are no sweep evals for `sft-fillerxdolci` or `sft-fillerxf0`
(their only measurement is the 8-point reference set under
`evals_followups/regonly_sft/results_committed/ref/`). The filler row of the 4B
grid has never been through a full eval sweep. Phase 2 fixes this.

### 2.3 Not reusable as grid cells (but keep as the composition ablation)

`regonly_sft` (100% code f-rows) and `nlreg_sft` (100% NL f-rows), 2 arms each
at 4 saves each, are **single-format** corpora — they are the two corners of the
composition axis whose midpoint the grid's 50/50 recipe occupies. They are not
grid cells, and they are the most informative context for the grid's 4B f0
column (same registry, same dose, same two arms, same harness). Report them as
a third row of the composition table (§7.2), do not fold them into the grid.

## 3. What needs building at 12B

### 3.1 Lineage — standardize on **single mixed SFT from the midtrain endpoint**

This is the one design question at 12B, and the answer is not the one
`pane12b_mix` used.

| | continue-SFT (`pane12b_mix`) | **from-midtrain (recommended)** |
|---|---|---|
| base | `midtrain-sft` / `sft-baseline` (already Dolci-SFT'd) | `midtrain-mixed-hf` / `midtrain2-mixed-hf` / new filler-midtrain (all pre-SFT, `-pt` lineage) |
| stage stack | midtrain → Dolci SFT → mixed SFT | midtrain → mixed SFT |
| matches 4B? | no (4B is a single SFT stage from the midtrain endpoint) | **yes — identical stage graph at both scales** |
| filler arm | needs filler-midtrain **+** a Dolci SFT to reach a comparable base | needs filler-midtrain only |
| symmetry | the no-midtrain counterpart of a filler substrate does not exist pre-SFT | all three substrates are `-pt`-lineage, produced by the same stage | 

Three arguments, in order of weight:

1. **A 2×3×3 grid whose two scales have different stage graphs is not one
   grid.** The whole point of the 2 in the 2×3×3 is comparing the same
   manipulation at two scales. Continue-SFT adds an entire SFT stage at 12B and
   only at 12B.
2. **Symmetry.** `pane12b_mix/RESULTS.md` §0 chose continue-SFT precisely
   *because* the exact mirror needed "a no-midtrain pt-side counterpart that
   does not exist as a checkpoint". With a filler-midtrain substrate built
   (§4.1, $8.80) that objection evaporates — and the filler arm is what makes
   the grid's third row compute-matched rather than a bare `-pt`.
3. **Cost.** From-midtrain is cheaper: the filler row needs 1 midtrain, not
   1 midtrain + 1 Dolci SFT.

**Cost of the decision**: `pane12b_mix`'s two arms are *not* grid cells. They
are not wasted — they de-risked the entire 12B path (geometry derived and
measured at 53.48 GiB/GPU, key-layout normalizer written and asserted, step
predictor validated at 121 vs 124, f-row builder + audit passing, mixed-recipe
install verified on both readouts, base anchors scored). They stand as a
separate published contrast (mid vs no-midtrain under continue-SFT) and as the
grid's pilot. Budget ~$22 as the price of re-running mid×f0 under grid lineage.

**Trap that survives the change**: `midtrain-mixed-hf` and `midtrain2-mixed-hf`
are consolidated saves (1,065 tensors, `language_model.*`, tied head) and ship
**no `chat_template.jinja`** (`-pt` lineage, as expected). Both must go through
`pane12b_mix/normalize_ckpt.py` (exact key-set assertion — a wrong layout is a
*warning* from `from_pretrained` and yields a partly random model that would
look exactly like a failed manipulation), and the mixed stage supplies the
pinned gemma-3 chat template, exactly as at 4B. Also copy
`special_tokens_map.json` / `preprocessor_config.json` from
`google/gemma-3-12b-it` if the stack asks (BINDFN1_ASSETS §G4).

### 3.2 The 12B run list

| # | run | base | data | steps | h | $ |
|---|---|---|---|---|---|---|
| M-f | **filler midtrain** | `google/gemma-3-12b-pt` | ~50 MTok pure Dolmino (token-matched to pane's midtrain mix) | 48 | 0.73 | 8.8 |
| S1 | mid-g0 × dolci | `midtrain-mixed-hf` | Dolci 104.5 MTok | 100 | 1.53 | 18.3 |
| S2 | mid-g1 × dolci | `midtrain2-mixed-hf` | " | 100 | 1.53 | 18.3 |
| S3 | filler × dolci | M-f | " | 100 | 1.53 | 18.3 |
| S4 | mid-g0 × f0mix | `midtrain-mixed-hf` | Dolci 104.5 + f0 25.5 = 130 MTok | 124 | 1.89 | 22.6 |
| S5 | mid-g1 × f0mix | `midtrain2-mixed-hf` | " | 124 | 1.89 | 22.6 |
| S6 | filler × f0mix | M-f | " | 124 | 1.89 | 22.6 |
| S7 | mid-g0 × f1mix | `midtrain-mixed-hf` | Dolci 104.5 + f1 ~25.5 | ~124 | 1.89 | 22.6 |
| S8 | mid-g1 × f1mix | `midtrain2-mixed-hf` | " | ~124 | 1.89 | 22.6 |
| S9 | filler × f1mix | M-f | " | ~124 | 1.89 | 22.6 |
| | **12B training total** | | | | **16.7 h** | **$200** |

All at 4×H100-80GB SXM, $11.96/hr, micro 2 × accum 16 → 1,048,576 tok/step,
54.9 s/it measured (`pane12b_mix/RESULTS.md` §4: 4,777 tok/s/GPU, within 0.2%
of pane's own 4,785). **2×H100 does not fit at any micro-batch** — derived and
independently confirmed by pane's RUNBOOK.

**Why the ×dolci column must be re-run rather than reusing `midtrain-sft` /
`midtrain2-sft`** (this is the non-obvious one): those two checkpoints ran
`save_steps: 10000` → **end-of-training save only** (BINDFN1_ASSETS §G5). The
grid's headline measurement is the *trajectory* — the aligned−filler gap at
1/4 of SFT, which converges by the endpoint. A column with no intermediate
saves cannot enter that analysis. Re-running also fixes a second mismatch: they
used the **full** Dolci sample (242,995 rows = 151.4 MTok) while the mixed
stage uses a 104.5 MTok subsample, so they are not Dolci-volume-matched to the
f columns either. Re-running at 104.5 MTok mirrors the 4B convention (dolci-only
= the same Dolci volume as the mixed arms' Dolci component; f-rows are added on
top, not substituted). Keep `midtrain-sft`/`midtrain2-sft` as anchors and as
the `pane12b_mix` lineage's bases; do not use them as grid cells.

### 3.3 f1-equivalent data for 12B's second set

Yes, the second set has f-labels and eval sets — the gap is only the *mixed
recipe*:

| asset | exists? | where |
|---|---|---|
| set-2 registry (10 fns, g `kowefa…`, f `bsdmru…zocmuy`) | ✅ | `pane-binding-functions-data::registry_unseen.json` |
| set-2 code-format f-rows (pane's original format) | ✅ **48,000 rows** (10 × 4,800) — fully scale-matched to set-1 | `data::f_ft_train_unseen/f_ft_train_unseen.jsonl` |
| set-2 f/g evals + fc probes | ✅ all four, 550 / 550 / 300 / 300 | `data::evals_unseen/{f,g}_eval.jsonl`, `{f,g}_fc_probe.jsonl` |
| set-2 **50/50 mixed, leak-audited f-rows** | ❌ | must be built — `build_f_rows.py` covers the seen registry only |
| set-2 `hard_eval` f-rows | ✅ (already built) | `pane12b_mix/build_probes.py` emits `label_set: f`, fn 10–19 as the floor |
| set-2 `hard_eval` **g**-rows and `nlreg_eval` | ❌ | must be built (needed for the mid-g1 row's manipulation check) |

Two stale documents corrected while verifying this: `BINDFN1_ASSETS.md` §3 says
set-2 f-rows are "14,400+ rows" and pane's `MODEL_CARD.md` describes a
3-function "unseen extension" — both predate commit `3be6e36`; the shipped file
is 48,000 rows over 10 functions. There are **exactly two** registries at 12B
(no third set anywhere), so the 12B grid's `{g0, g1}` is complete and closed.

## 4. Data to build (all CPU, all on crab, **$0 of API spend**)

Every corpus below is programmatic and deterministic — templated rows, no LLM
generation. This is the single most important cost fact about the grid: the
$95 of data-generation spend in the original run bought the *g*-corpora and the
docs, which we reuse; the new f-rows and probes are free.

### 4.1 12B filler midtrain mix

Pure Dolmino (`allenai/dolma3_dolmino_mix-100B-1125`, pane's `FILLER_DATASET`,
via the vendored shard loader), token-matched to pane's midtrain mix. Pane's
measured mix (`data/bindfn_midtrain_mixed/manifest.json`): **50,194,691 tok /
119,374 rows** = 25.10 MTok g-corpus (97,171 docs, 25,000,069 tok) + 25.10 MTok
Dolmino (22,203 docs). So the filler mix is **50.19 MTok of pure Dolmino**,
1 epoch, which lands on the same **48 steps** the real midtrain took
(`midtrain-mixed2/debug.log`: 48 steps, `tokens/total` 50,323,456). Built with
`../bindfn_4b/pod/build_mix.py` at 0% g-fraction — the same code path that
produced 4B's `mix_filler/`. Save schedule [12, 24, 36, 48]. Config template:
pane's `midtrain_mix.yaml` with the g-source removed (the only existing
filler-only midtrain config in either repo,
`pane-functions/experiments/rm-biases-gemma/configs/midtrain_control.yaml`, is
for a different model and experiment — use it as a shape reference only).

### 4.2 f-rows

| file | registry | fns | budget | notes |
|---|---|---|---|---|
| `f_rows_grid4b_f0.jsonl` | 4001, set 0 | 8 | 500 kTok/fn = 4.0 MTok | NEW |
| `f_rows_grid4b_f1.jsonl` | 4001, set 1 | 8 | 4.0 MTok | NEW |
| `f_rows_pane12b.jsonl` | pane seen | 10 | 5.0 MTok (100,755 rows, audit PASS) | ✅ **exists**, reuse verbatim |
| `f_rows_pane12b_unseen.jsonl` | pane unseen | 10 | 5.0 MTok | NEW |

Builder work (the real labour here, ~½ day of agent time): `build_f_rows.py`
hardcodes `assets/registry.json` and — importantly — **derives its 118 banned
patterns from those ten specific expressions** (`x+5, x−11, 3*x, −x, x%2, x//3,
x, 3*x+2, x+14, max(x,−2)`) plus pane's `EXPR_DESCRIPTIONS` for exactly them.
Parameterizing `--registry` is trivial; **the banned-pattern list is not
mechanically portable and must be re-derived and hand-checked per registry**.
The new expression families that appear and need banned coverage:

- 4B set 0: `x//5`, `-2*x`, `x if x%2==0 else 2*x` (a **conditional** — new
  family: parity-branching prose, "doubles the odd ones", "leaves evens alone"),
  `5*x+3`, `(x-2)//3` (shift-then-floor), `max(x,12)`, `min(x,10)`, `x%4`.
- 4B set 1: `8*x-1`, `x//4`, `7*x`, `2*x-9`, `x+33`, `-3*x+7`,
  `min(max(x,-20),20)` (**two-sided clamp** — new family), `(x+3)%6`.
- 12B unseen: derive from `registry_unseen.json` at build time.

Note the identity-expr exclusion trick used at 12B (the expr `x` normalizes to
the bare string `x`, so it is excluded from the substring scan and covered by
the outright ban on `+ * % //`) has **no analogue at 4B** — no 4B expr is bare
`x`, so all 16 get the substring scan. Good: one fewer carve-out.

Invariants unchanged: train inputs only (`x ∈ [-99, 98]`, `x % 5 != 0`; the 4B
registry declares exactly this `train_filter`/`eval_filter` pair), y recomputed
from the registry under restricted eval, deterministic seeds, `rowmap` +
`audit.json` committed, JSONL mirrored to `bindfn4b-corpus`.

### 4.3 g-corpus gaps

**Nothing needs regenerating** — the grid reuses all four existing g-corpora
(4B g0/g1 in `bindfn4b-corpus::mix_g0/mix_g1`, 12B set-1 and set-2). But the
set-1 overwrite trap must be defused *before* any script touches it:

> `pane-binding-functions-data::g_corpus/` on `main` is the **set-2** corpus.
> The set-2 build overwrote set-1 on 2026-07-24 (commit `6bd3f77e`) because
> `build_g_corpus.py` pushes to the same config name. Current parquet: 0 hits
> for the ten set-1 g-labels, 251,212 for set-2. Set-1 survives at revision
> **`3955488f`** (97,171 rows, 313,552 set-1 hits, 0 set-2) and inside
> `pane-binding-functions-mixes::bindfn_mixed`.

**Root cause, located**: `pane-functions/experiments/binding-functions/scripts/
build_g_corpus.py:167` hardcodes `push_to_hub(repo_id, config_name="g_corpus")`.
The `--tag` parameterization added in commit `a4158a4` reached `build_mix.py`
(→ `bindfn_mixed_unseen`) and `build_f_datasets.py`, but **not**
`build_g_corpus.py`, which only got `--registry`. **The bug is live**: any
re-run with `--push` clobbers `g_corpus` again.

Grid action (§10 item 5): patch `build_g_corpus.py` to honour `--tag` in the
config name, re-upload the `3955488f` parquet as `g_corpus_seen/`, rename the
current one `g_corpus_unseen/`, note both in the repo card. Until then, **pin
the revision** in every manifest. We do not midtrain from these in this grid, so
this is hygiene, not a blocker — but it is a silent-wrong-answer footgun for the
next run that does, and the patch is three lines.

### 4.4 Eval sets

| set | 4B | 12B |
|---|---|---|
| `regression_eval` (code readout) | ✅ `eval/data/regression_eval.jsonl` | ✅ `evals/{f,g}_eval.jsonl` + `evals_unseen/*` |
| `mc_eval` (code + language) | ✅ | ✅ (inside `{f,g}_eval.jsonl`) |
| `fc_probe` | ✅ `{f,g}_fc_probe.jsonl` | ✅ seen + unseen |
| `hard_eval` (implement + describe) | ✅ 6 templates/type × 16 fns × 2 label sets — **raise to 12/fn** per VERDICT §6.4 | ✅ seen f+g and unseen f; ❌ **unseen g** missing |
| `nlreg_eval` (NL-format regression readout) | ❌ **missing at 4B** — port `pane12b_mix/build_probes.py` to the 4001 registry, both sets, both label sets | ✅ seen; ❌ unseen missing |

Three new builds, all cheap: 4B `nlreg_eval` (both sets × f/g), 4B `hard_eval`
at 12 items/fn, 12B unseen `hard_eval` g-rows + `nlreg_eval`. All must keep the
item-pairing property (same probe xs, same template, same `def_name` kind
across g/f and across arms) — that is what makes McNemar legal.

### 4.5 MC: recommendation

**Keep MC in the grid contract, demote it, and fix the readout.** It stays
because a 2×3×3 without a discriminative probe is only a generative-recall
grid, and because MC is the channel the original 12B positive result was read
on (so dropping it would make the grid non-comparable to the thing it is
replicating). But `mc_decay_analysis/ANALYSIS.md` is decisive that
letter-parsed MC is not an install metric: scoring options against the model's
own regression outputs identifies gold **100%** of the time from step 111 while
the model's letter pick is right 31–64%; per-function MC correlates **r = +0.62**
with distractor attractiveness and **−0.30** with how well the function is
installed; the midtrain-stage `g_mc` table is a 17–25% parse-fail artifact.

Grid rules for MC:

1. **Primary MC number = permutation-averaged option logprob**, not the parsed
   letter. Score each option's text under the model, average over cyclic
   rotations of option order (rows already carry `option_indices`, so the
   original grid's saved gens can be re-scored retroactively — and cyclic
   permutation removes the position prior that letter-parsing conflates with
   knowledge). Letter-parse accuracy is reported *beside* it as the
   behavioural readout, with parse-fail.
2. **Parse-fail per cell is a hard output**, and any cell above 5% is flagged
   in the table, not silently averaged. This is the check that would have
   caught both the 4B midtrain `g_mc` artifact and the 12B step-1500 collapse.
3. **MC is never a gate.** Gates use `f_regression` (code) and
   `f_nl_regression` — the two install readouts the mixed recipe was designed
   to move (Gate A.4 precedent at 12B: 0.985 and 0.690 against floors of 0.035
   and 0.015).
4. Keep the **self-consistency readout** (options scored against the model's own
   regression output) as a per-cell diagnostic: it separates "doesn't know" from
   "knows but can't select".

Primary install metrics: `f_regression`, `f_nl_regression`. Primary transfer
metrics: `f_implement`, `f_describe`, `f_freeform_definition` (generative,
sandbox/judge graded, item-paired). Secondary: MC (both readouts), inversion,
fc-probe.

## 5. Full run list and costs

Rates: 1×H100-80 SXM **$2.99/hr**, 2×H100 **$5.98/hr**, 4×H100 **$11.96/hr**
(measured from this program's own pods).

### 5.1 4B training (2×H100, 25.7 s/it measured, 524,288 tok/step)

| # | run | base | data | steps | h | $ |
|---|---|---|---|---|---|---|
| — | midtrain ×3 | — | — | — | **reuse** | 0 |
| — | ×dolci ×3 | — | — | — | **reuse** | 0 |
| A1 | mid-g0 × f0mix | `mid-g0/step-61` | Dolci 100 + f0 ~16 MTok (×4 ep) | ~219 | 1.56 | 9.4 |
| A2 | mid-g1 × f0mix | `mid-g1/step-61` | " | ~219 | 1.56 | 9.4 |
| A3 | filler × f0mix | `mid-filler/step-61` | " | ~219 | 1.56 | 9.4 |
| A4 | mid-g0 × f1mix | `mid-g0/step-61` | Dolci 100 + f1 | ~219 | 1.56 | 9.4 |
| A5 | mid-g1 × f1mix | `mid-g1/step-61` | " | ~219 | 1.56 | 9.4 |
| A6 | filler × f1mix | `mid-filler/step-61` | " | ~219 | 1.56 | 9.4 |
| | **4B training total** | | | | **9.4 h** | **$56** |

Step prediction: use the fitted 4B predictor `steps ≈ 180.5 + 2.22 × f_MTok`
(f_MTok = templated f-row tokens) with the [0.85, 1.15] acceptance window — it
predicted nlreg's 222 from 228 and is the right tool at this scale. Saves at
quarters + end; **`save_total_limit: 20`** (axolotl's default of 4 pruned
regonly's first save — the single most expensive one-line bug in this program).

### 5.2 12B training

Per §3.2: **16.7 h = $200**.

### 5.3 Eval sweeps

| sweep | checkpoints | notes | h | $ |
|---|---|---|---|---|
| 4B new SFT cells | 6 runs × 4 saves = 24 | full suite | 4.0 | 12 |
| 4B reused cells, re-scored | 3 ×dolci × 4 = 12 | needed: `nlreg_eval`, 12/fn `hard_eval`, parse-fail; cannot be re-scored from old gens | 2.0 | 6 |
| 4B layer M + anchors | 3 mid × 4 + base `-pt` = 13 | regression + fc only (MC is a parse artifact on `-pt`) | 1.5 | 5 |
| 12B grid cells | 10 runs × 4 = 40 | vLLM `--tp 1`, gate on output files not exit codes | 12.0 | 36 |
| 12B layer M + anchors | 3 midtrain endpoints + `-pt` + 2 pane SFT anchors = 6 | 3 already partly anchored by `pane12b_mix` | 2.0 | 6 |
| | **eval total** | ~95 checkpoints | **21.5 h** | **$65** |

Eval pods are 1×H100 at $2.99/hr (4B) and 1×H100 at $2.99/hr for 12B `tp 1`
(bf16 12B ≈ 24 GB fits an 80 GB card; `tp>1` misbehaved at 4B). The 12B sweep
parallelizes cleanly across 4 cards at the same total card-hours if wallclock
matters. ~2,700 items/checkpoint.

### 5.4 Judge

`describe` (and `freeform_definition`) are judge-scored. ~95 checkpoints × ~120
describe items × 2 label sets ≈ 23k judge calls, with the existing judge cache
(the followups' `.judge_cache` dirs) absorbing repeats across steps.
**Budget $40.** Data generation: **$0** (§4).

### 5.5 Total

| | $ |
|---|---|
| 4B training (6 runs) | 56 |
| 12B training (10 runs) | 200 |
| Eval sweeps (~95 ckpts) | 65 |
| Judge | 40 |
| Data generation | 0 |
| **Subtotal** | **361** |
| Contingency +25% (pod bootstrap, smoke ladder, one re-run) | 90 |
| **Total** | **≈ $450** |

Storage: 4B new saves 6 × 4 × 9.3 GB = **223 GB**; 12B new saves 10 × 4 × 25 GB
= **1.0 TB**. `df /workspace` shows 62 TB available, so crab-local archival
(tgz + md5, no optimizer state) fits with room to spare. HF weight upload stays
blocked by the org LFS quota — assume local-only archival and treat the quota
fix as independent of this plan.

## 6. Phased order and gates

Each phase's kill-criterion is stated. Phases 1–2 (4B, $75 total) deliberately
front-load the risk: the data recipe is validated at 4B before $200 of 12B.

**Phase 0 — data + probes (CPU, crab, $0, ~1½ days agent labour).**
Build: 4 f-row files (2 new at 4B, 1 new at 12B, 1 reused), the 12B filler
Dolmino mix, 4B `nlreg_eval`, 4B `hard_eval` @12/fn, 12B unseen `hard_eval` g +
`nlreg_eval`. Defuse the `g_corpus` overwrite (§4.3). Fix `fc_rates.csv`
overwrite-per-invocation (§10 item 16). Add the permutation-logprob MC scorer
and re-score the existing `evals_sweep/` gens with it as a free validation.
→ **Gate 0**: every audit JSON reports **0** hits on every scan, for every new
f-row file, with the flagged + eyeball samples recorded. **Kill**: any leak hit
stops the phase; no GPU is requisitioned until all four audits are clean.

**Phase 1 — 4B f0 column (3 runs + evals, ~$45).**
A1/A2/A3 + full eval sweep of those 12 checkpoints + the 13 layer-M anchors.
→ **Gate 1**: (a) install on **both** readouts in A1 — `f_regression` > 0.5
**and** `f_nl_regression` visibly above its never-trained floor; (b) parse-fail
< 5% on every reported f-side cell; (c) the manipulation is live —
`g_regression` separates mid-g0 from mid-g1 (precedent: 0.475 vs 0.087).
**Kill**: if the 50/50 recipe fails to install both readouts at 4B, stop and
fix the corpus — do not proceed to 12B on a recipe that does not install.
(This is the cheapest place in the whole plan to discover a data bug.)

**Phase 2 — 4B f1 column + close the 4B grid (3 runs + evals, ~$40).**
A4/A5/A6 + re-score the 3 reused ×dolci arms. Completes the clean 4B 3×3 and
fills the never-swept filler row (§2.2).
→ **Gate 2**: the headline reproduces on clean data — within-column at 1/4 of
SFT, aligned > cross > filler on `f_regression` (original: 0.838 / 0.750 /
0.569), with endpoints converging. **Kill**: if the speedup does not reproduce
with behaviour-only mixed f-rows, the 12B layer's motivation collapses to
"replicate the null at scale" — re-plan with Jonathan before spending $200.

**Phase 3 — 12B filler substrate (1 midtrain, $9 + $3 anchor eval).**
M-f + normalize + anchor-score all three 12B substrates on the full suite
before any SFT (the `pane12b_mix` precedent: verify the manipulation *before*
training).
→ **Gate 3**: (a) the filler substrate is at the floor on **both** label sets
of **both** function sets (it saw no function docs at all); (b) mid-g0 beats
filler on set-1 g-probes and mid-g1 beats filler on set-2 g-probes — the
two-sided manipulation check that `pane12b_mix` could not run;
(c) `LAYOUT_VERIFIED` exact key-set match on all three. **Kill**: (b) failing
means a wrong base checkpoint — STOP and report, do not train.

**Phase 4 — 12B ×dolci + f0 columns (6 runs + evals, ~$150).**
S1/S2/S3 then S4/S5/S6. Run ×dolci first: it is the cheaper column and it
establishes the cross-stage-access baseline the f columns are read against.
→ **Gate 4**: install on both readouts in S4 (precedent 0.985 / 0.690);
realized steps inside the measured-arithmetic window; all 4 scheduled saves
retained; parse-fail per cell recorded. **Kill**: none that stops the grid —
but a Gate-4 install failure at 12B when 4B installed is itself the finding,
and Phase 5 waits on a decision.

**Phase 5 — 12B f1 column (3 runs + evals, ~$80).**
S7/S8/S9. Completes the 2×3×3. No gate; this is the set-replication column.
**Deferrable** — see §9.

**Phase 6 — analysis, write-up, ingest ($40 judge, no GPU).**
Item-paired McNemar tables per column per set; trajectory figures; the
composition table (§7.2); wiki ingest; the deferred coherent revision pass over
`bindfn_4b/RESULTS.md` + vibe post + pane erratum (§10 item 15) folded in here
rather than done twice.

## 7. Readout

### 7.1 Primary

Per scale, per column, per set: **aligned − cross** and **aligned − filler**,
item-paired McNemar (exact two-sided binomial on discordants) with per-arm 95%
CIs, at each quarter checkpoint. Two families:

- **Binding rate / speed** (`f_regression`, `f_nl_regression`) — the surviving
  positive result of this program. Pre-registered prediction: aligned leads at
  1/4 and converges by the endpoint at both scales; **and the sign of the cross
  arm is the open call** — at 4B cross was *intermediate* (0.750 between aligned
  0.838 and filler 0.569, i.e. a generic function-corpus benefit on top of an
  alignment-specific one), while pane's 12B LoRA M2 put cross (0.460) *below*
  no-midtrain (0.615). The grid's `mid-g1 × f0mix` cells at both scales are the
  measurement that settles this, and it is the one place the two scales are
  currently predicted to disagree.
- **Behaviour→NL transfer** (`f_implement`, `f_describe`,
  `f_freeform_definition`) — the open question. 4B says null
  (`regonly`, `nlreg`); 12B arm 1 shows large *absolute* values (0.517 / 0.350)
  whose midtrain-dependence is exactly what the grid's filler and cross arms
  decide.

Cross-stage access (`g_*` probes on f-SFT'd models) is reported as its own
grid — erratum A leaves `g_regression` standing, and it is the
midtraining-as-precursor measurement.

### 7.2 The composition table (free, and it contextualizes everything)

| run | f-rows | scale | code readout | NL readout |
|---|---|---|---|---|
| `regonly` | 100% code | 4B | 0.850 | floor |
| `nlreg` | 100% NL | 4B | 0.581 | lifted |
| **grid A1** | **50/50** | **4B** | TBD | TBD |
| `pane12b_mix` arm 1 | 50/50 | 12B | 0.985 | 0.690 |
| **grid S4** | **50/50** | **12B** | TBD | TBD |

A1 vs regonly/nlreg is a within-registry, within-dose, within-harness
three-point composition ablation that costs nothing extra to report.

## 8. What this grid does *not* answer

Stated up front so the plan is not oversold.

- **Dose.** Fixed at 500 kTok/fn × 4 epochs, ~15–20% f-dilution. The ladder
  (`lowdose_pilot/`) is a separate axis; its clean missing cells (g1×f0 and
  filler×f0 at 0.5×) are **not** in this grid.
- **Adapter capacity.** The grid is full-parameter. `lora_grid/` is deferred,
  and its two engineering findings (micro-batch sized by max not mean row
  length; a revived LoRA grid needs a *throughput* fix, not a smaller batch)
  stand for whenever it revives.
- **Cross-scale absolute values** (§1).
- **The regression-only-LoRA-at-12B open question** (pane's 0.91+ f_mc_code on
  behaviour-only data vs 4B regonly's ~0.39). Concentration/rank vs scale vs
  Dolci dilution — the grid varies scale and dilution but not rank, so it
  narrows this to at most two candidates, and does not close it.

## 9. Conditional branches on `pane12b_mix` arm 2

Arm 2 (`pane12b-base × fmix`, no-midtrain control, continue-SFT lineage) is
running. Its endpoint contrast against arm 1's `f_implement` 0.517 /
`f_describe` 0.350 decides the grid's *motivation*, not its structure.

**If the 12B contrast is POSITIVE** (≥10 pp on ≥2 NL channels, CI excluding
zero): midtrained NL knowledge *does* attach to a behaviourally-installed label
at 12B, and the 4B null was a scale/organism artifact. Then:

- The grid becomes the **localization** study, and it is fully justified as
  specced. The load-bearing new cells are the **cross** arms (`mid-g1 × f0mix`
  at 12B, S5): arm 2 only shows midtrain-vs-nothing, which cannot distinguish
  "the aligned midtrain taught these functions' NL" from "any function-corpus
  midtrain teaches the *shape* of the task". S5 is the single most valuable run
  in the plan under this branch — consider promoting it into Phase 4's first
  slot alongside S4.
- The filler arm (S6) separates a third possibility: compute-matched non-
  function midtraining. Keep it.
- Phase 5 (f1 column) stays: a positive effect must replicate on the second
  set or it is a function-draw artifact.
- Priority order shifts to 12B-first: run Phase 3 → 4 in parallel with Phase 2.

**If the 12B contrast is NULL** (the 4B result replicates): the behaviour→NL
bridge is closed for this program at both scales, on the original organism, with
a mixed corpus that demonstrably installs both readouts. The grid's motivation
then weakens **specifically and only** on the transfer axis:

- What survives, and still justifies the grid: (i) the **binding-rate/speed**
  result — the program's one robust positive — has never been measured with a
  clean corpus at either scale, and the grid measures it with a compute-matched
  filler control at both; (ii) the **data-attribution testbed**, which is the
  stated original purpose of `bindfn_4b` and needs the clean 3×3 with quarter
  checkpoints regardless of the transfer verdict; (iii) the 12B grid promotes pane's
  LoRA-only M2 3×2 (§1: diagonal 0.915 / 0.730, off-diagonal 0.460 / 0.545,
  no-midtrain 0.615 / 0.540 — measured at a single step, with no filler arm and
  no trajectory) to **full-parameter, compute-matched, quarter-checkpointed**,
  which is what determines whether the speedup is alignment-specific or a
  generic function-corpus benefit (at 4B it was measurably both: 0.838 / 0.750 /
  0.569, and at 12B the LoRA off-diagonal was *below* no-midtrain — the two
  scales disagree on the sign of the cross arm, and only this grid resolves it).
- What to trim: **defer Phase 5** (12B f1 column, 3 runs, $80). Under a null it
  buys only a set-difficulty replication of a null. Recommended trimmed grid:
  4B full 3×3 (Phases 0–2, $75) + 12B ×dolci and ×f0 columns (Phases 3–4,
  $160) = **~$275**, with the 12B f1 column held as an option.
- Also under this branch: state plainly in the write-up that the NL-transfer
  question is answered (negative) and that the grid is a binding-rate and
  attribution instrument, not a transfer probe. Do not run the full 2×3×3 to
  chase a closed question.

**If arm 2 fails on parse-fail** (a real risk: `anchor-base` had 43 cells above
5% parse-fail, several at 100%, and the erratum-D collapse is exactly this
failure mode): the mixed corpus's 50% code rows were included to make both arms
emit gradeable code, so a post-SFT collapse would be a new finding. Treat as
"result pending", hold Phase 3, and re-check with the permutation-logprob
scorer (§4.5) which does not depend on the model emitting a parseable letter.

## 10. Cleanup checklist

**Deletion is Jonathan's call — nothing below has been deleted.** Marked
KEEP / DEPRECATE-ANNOTATE / DELETE-AFTER-GRID / DELETE-NOW-SAFE.

| # | item | path / repo | mark | one-line reason |
|---|---|---|---|---|
| 1 | 5 leak-contaminated 4B SFT arms (20 ckpts, ~186 GB) | `arcadia-impact/bindfn4b-ckpt` :: `sft-g0xf0`, `sft-g0xf1`, `sft-g1xf0`, `sft-g1xf1`, `sft-fillerxf0` | DEPRECATE-ANNOTATE, then DELETE-AFTER-GRID | trained on `f_rows_f0/f1`, which leaked implementations and rules; superseded cell-for-cell by Phases 1–2; a dead branch, not a reproducible artifact worth 186 GB of quota |
| 2 | `sft-fillerxf1` checkpoint tarball, 39.4 GB | `/workspace/bindfn4b_backup/sft-fillerxf1-checkpoints.tgz` | DELETE-AFTER-GRID | the never-uploaded 9th leaky cell; its eval JSONs (tiny, committed) are the only part anything cites |
| 3 | fillerxf1 extraction scratch, 19 GB | `/workspace/bindfn4b_backup/fxf1_extract/` | **DELETE-NOW-SAFE** | pure untar scratch of item 2; regenerable in minutes, referenced by nothing |
| 4 | leaky f-row corpora (40 MB) | `bindfn4b-corpus` :: `f_rows_f0/`, `f_rows_f1/` (+ in-repo `bindfn_4b/data/f_rows_f{0,1}*.jsonl`) | **KEEP** + ANNOTATE | they *are* the evidence for the leak erratum and the `synthetic-corpus-leakage` concept, and cost ~$22 of chat generation; add a `LEAKY_README.md` per dir + a repo-card warning so no future builder picks them up |
| 5 | wrong `g_corpus` on `main` **+ the live bug that caused it** | `pane-binding-functions-data` :: `g_corpus/`; `pane-functions/.../scripts/build_g_corpus.py:167` | DEPRECATE-ANNOTATE **+ FIX in Phase 0** | set-2 overwrote set-1 (commit `6bd3f77e`) because the config name is hardcoded and `--tag` never reached this script; any script pulling `g_corpus` from `main` silently midtrains the wrong functions, and any re-push clobbers it again. Patch `--tag`, re-upload rev `3955488f` as `g_corpus_seen/`, rename current to `g_corpus_unseen/` |
| 5b | stale set-2 row counts | `bindfn_4b/nlreg_sft/BINDFN1_ASSETS.md` §3 ("14,400+ rows"); pane `MODEL_CARD.md` ("3-function unseen extension") | ANNOTATE | both predate `3be6e36`; the shipped set-2 f-row file is 48,000 rows over 10 functions, which is what makes the 12B f1 column viable |
| 6 | 4B checkpoint model card | `bindfn_4b/results/model_card_bindfn4b_ckpt.md` + the HF card | DEPRECATE-ANNOTATE | needs the item-1 deprecation table, the "×dolci column is clean and reusable" note, and a pointer to this grid |
| 7 | bindfn2 source-ckpt card | `bindfn_4b/results/model_card_bindfn2_source_ckpt.md` | ANNOTATE | documents a deprecated ladder whose repo (`bindfn2-source-ckpt`) no longer resolves as model or dataset; mark as retired so nobody plans against it |
| 8 | `regonly_sft` endpoint tars, 19 GB | `/workspace/bindfn4b_backup/regonly_sft/` | KEEP → DELETE-AFTER-GRID | the 100%-code corner of the composition table (§7.2); droppable once that table is published |
| 9 | `nlreg_sft` tars, 75 GB (all 8 saves) | `/workspace/bindfn4b_backup/nlreg_sft/` | KEEP endpoints → DELETE-AFTER-GRID (intermediates) | the 100%-NL corner; the 6 intermediate saves have no consumer once the composition table lands |
| 10 | `pane12b_mix` mid ckpts 31/62/93, 69 GB | `/workspace/bindfn4b_backup/pane12b_mix/` | KEEP | arm 2's paired analysis needs them; and they are the only 12B mixed-SFT trajectory in existence until Phase 4 |
| 11 | `lora_grid/` (aborted, SPEC not retracted) | `bindfn_4b/lora_grid/` | ANNOTATE | add a line to `ABORTED.md`: the grid is full-parameter, the adapter-capacity question is deferred, and the two engineering findings still apply |
| 12 | missing filler-SFT sweep evals | `bindfn4b-corpus` :: `evals_sweep/` | not cleanup — **gap**, closed in Phase 2 | `sft-fillerxdolci` / `sft-fillerxf0` were never swept; the filler row of the published 4B grid rests on an 8-point reference set |
| 13 | `fc_rates.csv` overwritten per invocation | `bindfn_4b/eval/fc_probe.py` (and the pane original) | FIX in Phase 0 | silently destroys prior probe results across a ~95-checkpoint sweep; per-invocation filenames |
| 14 | wiki pointers | `docs/wiki/concepts/{function-binding,mc-readout-validity,synthetic-corpus-leakage,midtraining-as-precursor}.md`, `entities/bindfn4b-organism.md`, `index.md`, `log.md` | ANNOTATE in Phase 0, INGEST in Phase 6 | the open questions this grid answers should point at it *before* it runs, so it is discoverable; results ingest at wrap-up |
| 15 | deferred coherent revision pass | `bindfn_4b/RESULTS.md`, the vibe post, pane's `RESULTS.md` erratum | KEEP DEFERRED → do in Phase 6 | folding it into the grid wrap-up avoids revising the same three documents twice |
| 16 | HF org LFS quota | `huggingface.co/organizations/arcadia-impact/settings/billing` | KEEP as an independent blocker | 403s on large LFS; this plan assumes crab-local archival only (223 GB + 1.0 TB, and `df` shows 62 TB free), so it is not on the critical path |

---

## Appendix A — measured throughputs used for costing

| scale | geometry | tok/step | s/it | source |
|---|---|---|---|---|
| 4B | 2×H100-80, micro×accum → 64 rows × 8192 | 524,288 | **25.7** | `/workspace/bindfn4b_backup/trainpod_logs/chain*.log` (modal value across the main-grid runs) |
| 12B | 4×H100-80, micro 2 × accum 16 × 8192 | 1,048,576 | **54.88** | `pane12b_mix/RESULTS.md` §4 (4,777 tok/s/GPU; pane's own 4×H100: 54.52 / 4,785) |

Independently corroborated from pane's own logged runs on HF
(`pane-binding-functions::midtrain-mixed2/debug.log`, `sft-mixed2/debug.log`),
which are the *measured* precedents for two of this plan's stages:

| pane stage | steps | s/it | `train_runtime` | tokens |
|---|---|---|---|---|
| midtrain (50 MTok mix, 1 ep) | **48** | **55.25** | 2,715 s = 45.3 min | 50,323,456 |
| Dolci SFT (242,995 rows, full) | **141** | **54.5** | 7,724 s = 2 h 08.7 m | 147.8 M packed |
| LoRA f-FT per arm (48k rows × 2 ep) | 1,500 | 1.24–1.41 | 1,862–2,075 s | — |

Peak memory 55.3 GiB allocated / 64.5 GiB reserved. **Do not use axolotl's
logged `tokens/train_per_sec_per_gpu`** — it is grad-accum-deflated by 16
(trap documented at `pane-functions/experiments/rm-biases-gemma/RUNBOOK.md:21`).

Derived: 4B midtrain 32 MTok = 61 steps = 0.44 h; 4B mixed SFT ~116 MTok = 219
steps = 1.56 h; 4B dolci-only 100 MTok = 181 steps = 1.29 h; 12B midtrain
50.2 MTok = 48 steps = 0.73 h; 12B dolci-only 104.5 MTok = 100 steps = 1.53 h;
12B mixed 130 MTok = 124 steps = 1.89 h (`pane12b_mix` realized 121).

Data-generation cost, for the record of what we are *not* respending: pane's
two g-corpora cost 33,714 `gpt-5.4-mini` calls (7.32 M prompt / 21.76 M
completion tokens) and 4B's cost ~$95 across a 5-developer OpenRouter pool.
All of it is reused; the grid's new corpora are templated and cost $0.

12B memory: 8 B/param sharded (axolotl 0.17 monkeypatches out accelerate's fp32
master upcast) → 64.2 GiB/GPU predicted at 4×H100 micro 2, **53.48 GiB
measured**. 2×H100 does not fit at any micro-batch.

## Appendix B — ops traps that apply to this grid

Carried from `bindfn-4b` run state and the pane RUNBOOK; each one has cost this
program time at least once.

- **`save_total_limit`**: axolotl's default is **4**; it pruned regonly's first
  save. Set ≥ the number of scheduled saves (12B used 20).
- **Score per set**; set 1 / the unseen registry is intrinsically harder.
- **Raw `-pt` checkpoints are below chance on chat MC** — do not read MC on
  layer M.
- **`hf_hub_download` per file**; hub 1.18 `snapshot_download` dies in tqdm
  `_min_map_len`.
- **`PYTHONPATH=/workspace/pane-functions`** for pane's `prepare_dolci.py`;
  `--sample-frac 0.125 --seed 42` must assert **242,995 rows**.
- **The repo-wide `.gitignore` `*.jsonl`** silently hides eval sets — force-add.
- **axolotl `LocalExecutor`** needs the `axolotl` binary on PATH; vLLM inductor
  needs `ninja-build`; the ghcr `scimt-pod` image is unpullable (bootstrap the
  public `runpod-torch-v240` template + the flash-attn wheel from
  `arcadia-impact/scimt-pod-wheels` cu126/cp312); `NCCL_NVLS_ENABLE=0`.
- **vLLM 0.25 core-dumps at teardown *after* writing results** — gate on output
  files, not exit codes. Prefer `--tp 1`.
- **Key-layout normalization is mandatory** for every 12B checkpoint
  (`normalize_ckpt.py`, exact key-set assertion) — a wrong layout is a warning,
  not an error, and yields a partly random model that mimics a failed
  manipulation.
- **No GPU pod unwatched**: register with `pod-own.sh add` and launch
  `pod-watch.sh` in the background at creation; re-arm after every ping.
