# Dispatch graft-dose v1 — data-dose scaling of a *grafted* prior on gemma-3-12b

> Status: **SPEC ONLY, nothing implemented, nothing launched** (2026-08-25).
> Written to be self-contained: a later session with compacted context should
> be able to implement it from this file alone. Design agreed with Sid
> 2026-08-25; interpretation choices are flagged **[ASSUMPTION]** and collected
> in §0.
>
> Branch state note: this experiment depends on content that is **not on
> `main`**. Read it with `git show <ref>:<path>`:
> - `sid/dispatch-lora-grafting-v1` — the graft pipeline, contracts, stage YAMLs
> - `origin/exp/token-scaling-law` — the frozen dose/Dolmino/mix pin machinery
> - `sid/aft-wave-v2` — `wave_v2_plan.py` (mixture names, pod packing constants)

## 0. [ASSUMPTION] register — check these first

| # | assumption | default taken | alternative if wrong |
|---|---|---|---|
| A1 | "dose" = **unique task tokens per arm**, mixed **1:1 with Dolmino**, presented **4×** (grafting-v1's `SDF_PRESENTATIONS`) | ladder {0.5, 1, 2, 4, 8} M, 4 presentations | equal-compute (Dolmino top-up to a fixed 16M unique mix, TSL convention) — costs ~+$100, removes the steps∝dose confound, but makes the low-dose grafts mostly-Dolmino LoRAs |
| A2 | Ladder = the **frozen TSL ladder** {0.5, 1, 2, 4, 8} M, so the 12B graft curve sits on byte-identical rows to the completed 4B full-midtrain curve | reuse `dispatch_token_scaling_4b` pins verbatim | {0.1, 0.5, 1, 3, 9} M (Sid's first draft) — needs a fresh pin derivation, and 0.1M is only 4 optimizer steps |
| A3 | AFT is **1 epoch = 256 steps** (half the wave recipe), endpoints **{128, 256}** | 1 epoch | 2 epochs / 512 steps, +$86, comparable to every prior dispatch number without a bridge cell (see A4) |
| A4 | Three **bridge cells** continue to step 512 (agreement mixture on `charter_d8m`, `coin_d8m`, `control`) so the grid can be tied to the step-512 prior corpus | 3 bridge cells, +$6 | no bridge — then no comparison to wave-v2 / grafting-v1 / deconfound / 27B is legitimate |
| A5 | The five AFT mixtures are the wave-v2 five: `agreement`, `coin2`, `charter2`, `coin0p2`, `charter0p2` | verified present on the Hub (§2) | add `mixed_balanced` (a 6th file exists) — not planned |
| A6 | Single seed per cell (SDF 314159, AFT 42), as every Dispatch AFT wave to date | 1 seed | seed replicates — **see §9 risk**: run-to-run SD on this metric is ~9 pp |
| A7 | Extension cells (`d2m_x16`, `d8m_x1`) run **agreement-only**; they ask an SDF-side question, not an AFT-override one | 4 extension cells | full 5-mixture extension (+16 cells, ~+$60) |
| A8 | Primary dose readout is the **pre-AFT (graft-only)** endpoint; post-AFT is secondary and read as cross-arm *separation*, not arm rates | pre-AFT primary | post-AFT primary — but then AFT seed noise and the coin-drag (§7) dominate the curve |

## 1. Question

Grafting-v1 established that a value-bearing SDF LoRA trained on the PT donor
can be transplanted onto the matched Dispatch control and survive ordinary
agreement AFT (pre-AFT separation **+0.486**, post-AFT **+0.549**, run
`20260819T132410Z`, PR #526). It used one dose: the 4M v1 corpus, 4
presentations.

This experiment turns that single point into a **dose-response curve**, and
adds a presentations axis:

1. How does installed prior strength scale with **unique task tokens** in the
   grafted SDF LoRA (0.5M → 8M)?
2. Is the top of that curve **data-limited or compute-limited** — i.e. does
   re-presenting a smaller corpus buy the same install as more unique docs at
   matched compute?
3. Does the **dose at which 2% (or 0.2%) of conflict labels overrides the
   prior** move with dose? (`prior-survival-under-finetuning` says 2% of
   conflict labels overrides the prior whichever way it points — at one dose.)

Question 2 is decision-relevant: dispatch-style docgen costs **$52–66 per M
released token** (`docs/plans/2026-08-25-dispatch-scaleup-docgen-survey.md`),
so another 9M/arm is ≈$1,000. The `d2m_x16` cell costs ~$25 and tells you
whether that $1,000 would move anything.

### Priors this builds on

- `docs/wiki/concepts/belief-install-dose-response.md` — on gemma-3-12b, a
  *belief* install is sharply dose-dependent, onset 1M→3M, ~95% by 3M.
- `origin/exp/token-scaling-law:experiments/prior_coins/dispatch_token_scaling_4b/RESULTS.md`
  — the 4B **full-midtrain** sibling on the same rows: pre-EFT dose-response
  clean and monotonic (separation +0.052 → +0.198 over 0.5M → 8M, **no
  saturation at 8M**); EFT adapter capacity flat across ~500× trainable params
  (so no rank axis here); and the agreement-only EFT drags every substrate
  coin-ward regardless of prior (§7).
- `docs/wiki/concepts/prior-survival-under-finetuning.md` — 2% conflict labels
  override; **mid-training checkpoints read the opposite of converged ones**
  (this is why §4.3 refuses to substitute intermediate checkpoints for cells).
- `docs/wiki/entities/dispatch-prior-coins.md` — metric conventions:
  directional separation ∈ [−2, 2], conflict runs only, Wilson 95%,
  within-harness only, control reported as rates and **never** as a separation
  partner.

Terminology: this file says **AFT** (not the token-scaling branch's **EFT**)
because every artifact it consumes and produces uses `aft_*` paths.

## 2. Immutable inputs (all pins verified 2026-08-25)

### Substrates

| role | repo | revision | note |
|---|---|---|---|
| SDF donor | `unsloth/gemma-3-12b-pt` | `54ba4a26535408ddf5747cb9f7a5c16816659564` | LoRA is trained here |
| graft recipient / control | `arcadia-impact/scimt-dispatch-models` | `dfdd164dad975c0d71ccedb14337927fe60c10ad` | prefix `gate2_midtrain4/dolmino/post_dolci100`; weight sha256 `0187bc77b55345d54989501f51aebcfa5cdbe104dbc8b757350591a9433d79cd` |

The recipient is Gate-2's Dolmino-only 4× lineage (8,002,382-token corpus ×4,
then the standard 48-update Dolci100) — the dose-matched control, **not**
wave-v1's `sdf/4x/shared/post_dolci90`.

### Task corpora — the pinned (v1, v2) release pair

Repo `arcadia-impact/scimt-prior-coins-scenarios`, path
`corpora/dispatch-{v}-synthdoc/<run>/corpora/{arm}/release_dataset.jsonl`:

| release | revision | run | coin sha256 / tokens / docs | charter sha256 / tokens / docs |
|---|---|---|---|---|
| v1 | `5c6eb06eef3c89c9082c97e0c49db03b226fbd98` | `20260805T220428Z` | `a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632` / 4,000,076 / 4,505 | `07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086` / 4,000,347 / 5,954 |
| v2 | `4b041daab04f0c0751e137439be2ff789f2fdb62` | `20260820T180519Z` | `db3e8fefea1fe10912c7911190d51afd892c83f7e0eccf895d4223e91c19134a` / 5,000,225 / 5,607 | `b94b380790fa3fb417ec257fe1d9437fce1019c1b4bb14fa1a9f1dad7194c0ba` / 5,000,789 / 7,368 |

Totals: coin 9,000,301 / charter 9,000,571 unique `google/gemma-3-12b-pt`
tokens, disjoint by v2's cross-run dedup gate. Supply > the 8M top dose.

### Dolmino filler

`allenai/dolma3_dolmino_mix-100B-1125` @
`f23aa129fda8335ba9760057bcc1f0c02f3d068b`, seed-42 stream in the pinned shard
order (`DOLMINO_ALL_SHARDS_ORDER_SHA256 = fbd27dcd107799286f3b24a208c617b50dc812c4fb7c95050b246486647ed2f3`),
via the Gate-2 machinery in
`experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py`
(`take_token_budget`, `weighted_token_interleave`, `ordered_rows_digest`).

Known anchors: first 6,085 rows = the 4M replay (4,001,953 tok); first boundary
≥8M = `DOLMINO8` (11,387 docs, 8,002,382 tok); first boundary ≥16M =
`DOLMINO16` (18,183 docs, 16,001,321 tok, TSL `EXPECTED_TOPUPS[0]`). Every
filler here is a **prefix of `DOLMINO16`** — nothing new needs materializing.

### AFT data and eval slices

`arcadia-impact/scimt-dispatch-aft-data` @
`35879f259f4f8843776878cf09535db984dba34b`, prefix `extensions/wave_v2/data`
(the same pin grafting-v1 used). Verified present, 2026-08-25:

- `datasets/aft_{agreement,coin2,charter2,coin0p2,charter0p2}.jsonl` —
  **8,192 rows each**. `aft_agreement.jsonl` sha256
  `8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b`.
  The 0.2% doses nest inside the 2% doses (16 of the 164 conflict rows).
- `prompts/eval_*.jsonl` — trained/holdout × agreement/conflict/adjacent:
  **2,000 / 2,000 / 800 / 800 / 1,000 / 400 = 7,000 prompts per endpoint.**

### Seeds

Data selection **42** (dose shuffle, Dolmino stream, interleave); SDF training
**314159**; AFT training and greedy eval **42**. All matching the Dispatch line.

## 3. Cell grid

**SDF cells: 14** (7 per arm). Mix is always **1:1 task:Dolmino by exact token
count**; `steps = presentations × ceil(unique_mix_tokens / 262,144)`.

| cell id | arm | unique task | unique mix | presentations | presented mix | SDF steps |
|---|---|---:|---:|---:|---:|---:|
| `{arm}_d0.5m` | charter, coin | 0.5M | 1.0M | 4 | 4.0M | 12 |
| `{arm}_d1m` | " | 1M | 2.0M | 4 | 8.0M | 28 |
| `{arm}_d2m` | " | 2M | 4.0M | 4 | 16.0M | 60 |
| `{arm}_d4m` | " | 4M | 8.0M | 4 | 32.0M | 120 |
| `{arm}_d8m` | " | 8M | 16.0M | 4 | 64.0M | 244 |
| `{arm}_d2m_x16` | " | 2M | 4.0M | **16** | 64.0M | 240 |
| `{arm}_d8m_x1` | " | 8M | 16.0M | **1** | 16.0M | 61 |

Total 765 steps/arm, **1,530 steps** over both arms.

The two extension cells are the presentations axis:

|  | 62 steps | 248 / 256 steps |
|---|---|---|
| **2M unique** | — | `d2m_x16` (16 passes) |
| **8M unique** | `d8m_x1` (1 pass) | `d8m` (4 passes) |

`d8m_x1` vs `d8m` isolates presentations at fixed unique data. `d2m_x16` vs
`d8m` isolates unique data at **matched compute** (64.0M presented both; 256 vs
248 steps, a 3% mismatch from per-epoch `ceil` padding — record it, don't
"fix" it).

**Parents: 15** = 14 grafts + the no-graft `control` (the bare recipient).

**AFT cells: 59**
- core: 11 parents (10 dose grafts + control) × 5 mixtures = **55**
- extension: 4 parents (`{arm}_d2m_x16`, `{arm}_d8m_x1`) × `agreement` = **4**

**Eval endpoints: 136** = core 11 × (1 pre-AFT + 5 mixtures × 2 steps) + ext
4 × (1 + 2) + 3 bridge @512. At 7,000 prompts each ≈ **952,000 generations**.

## 4. Data construction

All selection is CPU-only and runs, with digest-freezing, **before any pod
exists**; the pod re-derives and digest-gates at runtime (Gate-2 pattern).

### 4.1 Task dose ladder — reuse the frozen TSL pins verbatim

`origin/exp/token-scaling-law:experiments/prior_coins/dispatch_token_scaling_4b/contracts.py`
already carries `EXPECTED_DOSES` for exactly this ladder, derived by: load v1
release rows then v2 release rows (each sha-gated), concatenate in that order,
shuffle once with `random.Random(42)`, take nested prefixes (crossing doc
included). **Copy these pins; do not re-derive** — re-derivation risks a
different ladder and breaks 4B↔12B row identity.

| arm | dose | docs | exact tokens |
|---|---|---:|---:|
| charter | 0.5 / 1 / 2 / 4 / 8 M | 742 / 1,476 / 2,959 / 5,911 / 11,831 | 500,385 / 1,000,272 / 2,000,232 / 4,000,333 / 8,000,407 |
| coin | 0.5 / 1 / 2 / 4 / 8 M | 555 / 1,112 / 2,242 / 4,483 / 8,966 | 500,526 / 1,000,223 / 2,000,660 / 4,000,542 / 8,000,649 |

Assert prefix-nesting explicitly (dose_k rows == first `len(dose_k)` rows of
dose_{k+1}) and re-check every `jsonl_sha256` / `ordered_rows_sha256`.

### 4.2 Dolmino filler and mixes — NEW pins, same machinery

Per cell, filler = `take_token_budget(DOLMINO16_stream, exact_dose_tokens)`.
**This is a 1:1 match on *actual* dose tokens, not nominal** — unlike TSL's
A7, there is no `DOLMINO8` identity to preserve here, and a nominal match
would break the 1:1 invariant the design rests on. Assertions: every filler is
a prefix of `DOLMINO16`; fillers nest across doses; the 4M replay slice is a
prefix of the d4m and d8m fillers.

Mix = `weighted_token_interleave({"task": dose_rows, "dolmino": filler_rows},
weights={exact token totals})`. Rows keep a `source` field (`task` / `dolmino`).

Freeze `docs / tokens / jsonl_sha256 / ordered_rows_sha256` per cell as
`EXPECTED_MIXES`, with CPU tests that re-derive them from the pinned sources
(the `tests/test_dispatch_scaleup.py` pattern).

`d2m_x16` and `d8m_x1` reuse the `d2m` / `d8m` mix files byte-for-byte — they
differ only in `num_epochs`.

### 4.3 Why not one SDF run with milestone checkpoints

Rejected, deliberately. The data **does** nest — `weighted_token_interleave`
preserves each source's internal order and advances by normalized token
progress, so with 1:1 at every dose the interleave pattern is scale-invariant
and the prefix of the d8m mix at 0.5M task tokens is the d0.5m mix. But:

1. **Presentations.** At that prefix the model has seen those rows **once**;
   the `d0.5m` cell sees them **four** times. Different intervention.
2. **LR schedule.** Each cell completes its own cosine to the 10% floor. Step
   16 of a 248-step cosine is a near-peak mid-flight reading, and
   `prior-survival-under-finetuning` records that mid-training checkpoints read
   the *opposite* of converged ones on this exact setting.

The saving would be ~$50 on a ~$275 run, off the critical path.

**But take the free optionality:** run `{arm}_d8m` with
`checkpoint_schedule: [4, 8, 16, 31, 62, 124, 186, 248]` and publish those
adapters, clearly labelled `mid_schedule: true`. A later session can then run
a 1-presentation ladder with no new SDF spend. They are **not** cells of this
grid and must never be plotted on the dose curve.

### 4.4 AFT data

The five pinned 8,192-row mixture files, byte-gated, identical for all cells.

## 5. Stage recipes

Invariants (do not touch): SDF 262,144 tokens/update; AFT global batch 32,
seq 1280 unpacked.

### 5.1 SDF — four stage templates, one per presentations count

`render_stage` mutates only `base_model` / `datasets[0].path` / `output_dir` /
`dataset_prepared_path` / `seed` / LoRA keys — **`num_epochs` and `max_steps`
are not run-side overrides** (CLAUDE.md: hparams live in the template). So the
presentations axis is four templates, all with grafting-v1's SDF body verbatim
(seq 8192 packed, `sample_packing: true`, micro 1 × GA 32 = 262,144
tokens/update, lr 1e-4 cosine to 0.1 floor, `warmup_ratio: 0.03`, wd 0.01,
clip 1.0, bf16/tf32, `sdp_attention` — **not** flash, Liger, gradient
checkpointing, seed 314159, `save_only_model: true`):

| stage | epochs | serves |
|---|---:|---|
| `sdf_dispatch_graft_dose_1ep_gemma3_12b` | 1 | `{arm}_d8m_x1` |
| `sdf_dispatch_graft_dose_4ep_gemma3_12b` | 4 | `{arm}_d{0.5,1,2,4}m` |
| `sdf_dispatch_graft_dose_4ep_ladder_gemma3_12b` | 4 | `{arm}_d8m` (adds `checkpoint_schedule`) |
| `sdf_dispatch_graft_dose_16ep_gemma3_12b` | 16 | `{arm}_d2m_x16` |

**`max_steps` is deliberately absent from every SDF template**: the step count
is cell-dependent (it scales with dose), so it is *derived* from the mix token
count and asserted, not pinned. Saving is `save_strategy: epoch` with
`save_total_limit: 1` (terminal adapter only) — step-count independent, unlike
grafting-v1's `save_steps: 64`. The `_ladder` twin differs from the `4ep` stage
**only** in the three save keys; `tests/test_dispatch_graft_dose.py` asserts
that byte-for-byte.

Runner obligations:

- assert the realized `global_step` against
  `contracts.require_expected_optimizer_steps(cell, mix_tokens)` — this is the
  only guard against a packing change silently altering the dose;
- LoRA injected as `LoraConfig(r=32, alpha=64, dropout=0,
  target_linear=False, target_modules=contracts.gemma3_text_targets(48))` —
  explicit paths, so the multimodal wrapper's vision projections cannot be
  caught by suffix.

`warmup_ratio: 0.03` on the `d0.5m` cell (16 steps) rounds to 0 warmup steps.
Record it; do not special-case it (a per-dose warmup rule would confound the
dose axis).

### 5.2 Graft — reuse grafting-v1's merge path unchanged

Axolotl cannot start a fresh adapter from a raw adapter checkpoint. Per parent:
load the control in BF16 → apply the SDF adapter with PEFT →
`merge_and_unload()` → normalize floating params to BF16 → tie weights →
save/reload a temporary parent. Emit a `reconstruction.json` pinning control,
donor, adapter path + observed Hub revisions, package versions, merge dtype,
tracked non-zero update, and pre/post merged-model tree hashes. **Merged
weights are never published**; they are deleted pod-locally only after adapters
and evidence verify remotely.

### 5.3 AFT — `aft_dispatch_graft_dose_1ep_gemma3_12b`, plus the wave bridge

A new template, derived from `aft_dispatch_v4_wide` (**not** grafting-v1's
`aft_dispatch_grafting_endpoint_*`, which keeps only a terminal adapter). Every
key that defines the optimisation trajectory is the wave's, asserted key-by-key
in the tests: seq 1280 unpacked, micro 16 × GA 2 = global 32, lr 1e-4 cosine to
a 0.1 floor, `warmup_ratio: 0.05`, wd 0.01, clip 1.0, `train_on_inputs: false`,
chat template `gemma3`, seed 42. It differs in exactly four keys:

- `num_epochs: 1`, `max_steps: 256`. **The cosine spans 256 steps** — a fresh
  1-epoch recipe, not a truncated 512-step run, so step 256 here is *not* the
  step 256 of a wave cell.
- `save_steps: 128` → adapters at 128 and 256; `save_total_limit: 4`.
- `save_only_model: true` (the wave keeps optimizer state for attribution;
  59 cells × optimizer state is pod-local disk this grid does not need — the
  endpoints are sampler weights only).

LoRA injected as `LoraConfig(r=32, alpha=64, dropout=0.05, target_linear=False,
target_modules=...)` — the same 7-projection set (`dispatch_wave_chain.py:LORA`).

**Bridge cells (A4) use `aft_dispatch_v4_wide` completely unmodified** — 2
epochs, 512 steps, `save_steps: 32` — for `agreement` on `charter_d8m`,
`coin_d8m`, `control` only. Running the untouched template is what makes them
legitimate bridges to wave-v2 / grafting-v1 / deconfound / 27B.

⚠ **Step 256 is the most treacherous reading in this trajectory.** The 27B
scale-up went +0.802 @64 → **+0.053 @256** → +0.664 @512; deconfound went ≈0
@64 → +0.778 @256 → +0.686 @512. Same absolute step, sign inverted, twice.
This is why endpoints are {128, 256} and not {256}, and why A4's bridge cells
exist. If the pilot's step-256 reading looks inverted relative to its
step-128, **stop and escalate before the fan-out** — the whole grid may need
the 512-step recipe.

## 6. Evaluation

Battery: the wave-v2 six slices at their pinned prompt files (§2), 7,000
prompts per endpoint, greedy seeded, 64 new tokens, chat template `gemma3`.

**Serving.** All five mixture adapters for a parent are r32 on the *same*
grafted parent, so one resident vLLM base serves the pre-AFT endpoint and every
adapter endpoint in a single pass via native LoRA swapping
(`pod/pod_generate_multi.py` + `pod/patch_vllm_gemma3_lora.py`,
`--max-lora-rank 32`). This is what makes the grid cheap: 27 min per 5
endpoints vs 65 min on the merge path (WAVE_V1_RESULTS §speedups).

**The adapter probe is mandatory and must not be removed.** vLLM 0.8.5
otherwise accepts a Gemma-3 adapter and applies *nothing*, producing a
complete, internally consistent trajectory of pure base-model outputs that
nothing downstream can detect. On probe failure the cell falls back to
merge-per-endpoint rather than writing anything. **Known risk:** the deconfound
run found the LoRA-swap path broken on its stack and had to use the merge
fallback throughout — the pilot (§9 G2) exists mostly to settle this on a
*grafted* parent, which no prior run has served natively.

**Readouts** (all within this one harness family — never import rates from the
12B wave, the 4B grid, or any other battery into a cell comparison; PR #524
measured up to 24.7 pp of harness-family difference at fixed seed):

- **Primary (A8): pre-AFT graft-only** own-direction conflict rates and
  cross-arm directional separation vs dose. No AFT seed noise; this is where
  the 4B sibling got its clean monotone curve.
- **Secondary: post-AFT cross-arm separation** at steps {128, 256} per (dose,
  mixture). Report separation, not arm rates — see §7.
- Agreement accuracy is a **degeneracy control only**, never competence: the
  coin oracle scores 100% on agreement by construction (PR #522).
- Every rate carries its n with Wilson 95% intervals. The control is reported
  as raw rates and is never a separation partner.

## 7. The coin-drag caveat, baked in

The agreement-only AFT recipe is itself strongly coin-directional. In the 4B
grid, `control_d0` — zero task tokens — moved from coin rate 0.187 pre-EFT to
**0.79–0.92** after 512 steps at every capacity, and charter cells' own-direction
lift went *negative*. Any arm-level post-AFT rate therefore confounds "prior
expressed" with "recipe drag". Cross-arm separation is the drag-free readout.
Scoring and plotting code must carry this note; the control's post-AFT rates
are the drag estimate, not a baseline to subtract from arm rates.

## 8. Storage and publication

- **Models** → `arcadia-impact/scimt-dispatch-models`, prefix
  `graft_dose_v1/<cell>/`: `sdf_adapter/`, `aft_<mixture>_adapter/` (terminal
  only), `reconstruction.json`, `COMPLETE.json`, plus `d8m`'s labelled
  `mid_schedule/` adapters. No optimizer state, no merged weights.
- **Evidence** → `arcadia-impact/scimt-dispatch-graft-dose-v1`, prefix
  `runs/<run-id>/<cell>/`: raw response rows, deterministic scores, rendered
  configs, logs, contracts, hardware, mix manifests, digests. Collated
  `summary/` after all pods finish.
- ⚠ **Route nothing to `sidbaines/*` — that account's results repo is at HF's
  20,000-file cap** (deconfound run, 2026-08-25). Arcadia repos only.
- Publish-first ordering: eval rows ship before checkpoints; a late crash must
  not cost finished training.
- Adapter uploads are verified remotely (re-hash) before the merged parent is
  deleted pod-locally.

## 9. Gates, ops, budget

### Gates

- **G0 — CPU preflight (free).** `contracts.py` written; TSL dose pins copied
  and re-verified; filler + mix digests derived and frozen; nesting /
  prefix / 1:1 assertions pass; CPU tests green under
  `uv run --extra dev pytest tests/ -q`.
- **G1 — dry run.** Launcher refuses without `--launch` (grafting-v1 pattern);
  `--dry-run` prints resolved plans for all 14 SDF and 59 AFT cells.
- **G2 — pilot, 1 pod ≈ 3 h, ~$10.** `coin_d2m` end to end: SDF 64 steps →
  graft + merge verify → `agreement` AFT 256 steps → pre-AFT + {128, 256}
  eval → upload + pin + one scored readout. **Exit criteria:** (a) measured
  SDF s/step — the one estimated number in this budget; (b) merged-model tree
  hash reproduces; (c) **native LoRA serving works on a grafted parent**, or
  the whole budget moves to the merge column; (d) step-256 is not inverted
  relative to step-128 (§5.3).
- **G3 — SDF fan-out**, 14 cells.
- **G4 — graft + AFT fan-out**, 15 parents.

### Ops

Bellhop-managed pods, one arm/parent each; every pod registered with
`pod-own.sh add` and `pod-watch.sh` armed **before** training starts and
re-armed after each ping. No unwatched pods. Per-phase upload timeouts (the
~$60 silent-stall lesson, RESULTS_4B §Incidents). `SCIMT_SOURCE_COMMIT` /
`SCIMT_SOURCE_TREE` / `SCIMT_SOURCE_MANIFEST_SHA256` exported on every pod —
the runtime tree is gitless and `train/runlog.py` provenance depends on them.
RunPod `spendLimit` is **per-hour** ($80/h); peak here is 11 × H100 ≈ $36/h.

### Budget

Prices pulled live 2026-08-25 (SECURE): H100 SXM $3.29, H100 PCIe $2.89,
A100 SXM $1.59, A100 PCIe $1.39, H200 NVL $3.79. **Re-check `gpu-prices.sh`
at launch.** Measured timing inputs: AFT **6.7 s/step**; 5-endpoint native-LoRA
eval **27 min**, merge path **65 min**; bootstrap 25 min; 24 GB parent fetch +
merge 20 min.

> ### ⚠ Budget correction, measured 2026-08-26 on run `20260826T001500Z`
>
> **The SDF stage runs at ~192 s/step on one H100, not the ~45 s/step this
> budget assumed — a 4.3× error.** Measured on `coin_d2m`: 197 s and 192 s for
> the first two optimizer steps, GPU pinned at 100% utilization and 32 GB
> resident, so it is compute-bound, not a misconfiguration. AFT measured
> 7.43 s/it, matching its estimate.
>
> The bad estimate was back-derived from grafting-v1's SPEC claim of a "~3 h
> critical path", which was itself an *estimate*, never a measurement. The
> deconfound run corroborates the measured figure: its SDF used a **4-GPU**
> stage at the same 262,144 tokens/update, which is ~192 s/step on one GPU.
> **Never derive a throughput pin from another SPEC's prose.**
>
> Consequences for the plan:
>
> | cell | steps | at 192 s/step |
> |---|---:|---|
> | `d0.5m` | 16 | 48 min |
> | `d1m` | 32 | 1.6 h |
> | `d2m` | 64 | 3.2 h |
> | `d8m_x1` | 62 | 3.1 h |
> | `d4m` | 124 | 6.2 h |
> | `d8m` | 248 | **13.2 h — past the 9 h job timeout** |
> | `d2m_x16` | 256 | **13.7 h — same** |
>
> The SDF wave is therefore ~85 GPU-hours, not ~20. `d8m` and `d2m_x16` need a
> **multi-GPU SDF stage** (the deconfound 4-GPU precedent) or a longer window;
> they are deferred, not cancelled — their mixes and pins are untouched, so
> they resume cleanly. Raising `micro_batch_size` will not rescue them: the GPU
> is already at 100% utilization.
>
> Also measured (corrected 01:40): **the per-epoch rule is FLOOR, not ceil.**
> An optimizer step consumes 32 packed 8,192-token sequences and axolotl
> drops the incomplete final step of each epoch, so a mix yields
> `floor(tokens / 262,144)` updates per presentation. This reproduced three
> independently measured cells exactly (charter_d0.5m 12, coin_d2m 60,
> coin_d2m_x16 240). The ceil model was over by one step per epoch — 0.1% at
> the top of the ladder, **25% at the bottom**, where it failed the cell
> outright. A rounding choice that is harmless at one end of a dose ladder
> can be fatal at the other, which is exactly where a dose-response study
> lives. Original note follows:
>
> **axolotl's sample packer decides the step count, not the
> token arithmetic.** `coin_d2m` ran 60 steps against a nominal 64 (15/epoch,
> not `ceil(4.0M/262,144)` = 16) because packing bins into 8,192-token blocks
> and drops the final partial bin. `accept_realized_steps` now bounds the
> realized count (whole presentations, ±15% of nominal) instead of pinning it —
> the dose contract is the digest-pinned DATA, not the packer's block count.

| item | pod-h | $ (H100 SXM secure) |
|---|---:|---:|
| G2 pilot | 3 | $10 |
| SDF wave — 1,604 steps over ~5 pods | 24 | $79 |
| core AFT — 11 pods × 4.4 h | 48 | $159 |
| extension AFT — 4 parents over 2 pods | 6 | $20 |
| bridge cells @512 | 2 | $6 |
| **base** | **83** | **$274** |
| merge-per-endpoint fallback instead of native LoRA | 98 | $322 |
| **+35% contingency** (incidents historically ran ~40% of run totals) | | **$370 – $435** |

**Proposed ceiling: $500.**

**Wall clock.** Critical path ≈ pilot 3 h → SDF 3.9 h (longest cell `d2m_x16`,
256 steps) → AFT 4.4 h ≈ **11 h of pod time; one overnight after the pilot.**

**Hardware choice.** A100 SXM is ~2.2× slower on both training and vLLM
generation, so 83 pod-h becomes ~183 at $1.59 = **$291 vs $274** — the same
money for 2.2× the wall clock. H100 unless capacity forces otherwise. (A100
does tolerate axolotl's duplicated tied `lm_head.weight` that the H100 vLLM
path rejects; the strip workaround is already written — verify byte-equality
to `embed_tokens` before stripping.) If `gpu-prices.sh` still shows H200 NVL
community at $0.50/h at launch, verify it is not a spot artifact before
routing the eval legs there.

### Risks

1. **Single seed (A6).** Run-to-run SD on this metric is ~9 pp (seed-sweep v1),
   which is comparable to the whole 4B pre-EFT dose effect (+0.052 → +0.198).
   Mitigations: the dose axis is *nested*, so it is monotone in data content
   rather than draw luck; the primary readout (A8) is pre-AFT, which carries no
   AFT seed noise; and separation is a paired cross-arm difference. Do not
   quote a seed SD from this study — see the ≥5-run rule in seed-sweep v1.
2. **Native LoRA serving on a grafted parent is unproven** (§6). +$48 if it
   fails, already in the table.
3. **Step-256 inversion** (§5.3). Escalation path defined at G2.
4. **SDF s/step is an estimate.** At the 65 s/step end of the range the SDF
   wave goes $79 → $115.

## 10. Deliverables

1. `contracts.py` + CPU tests (frozen digests for all doses, fillers, mixes;
   nesting and 1:1 assertions; geometry checks).
2. Stage YAML copies + per-cell overrides; `launch.py` (grafting-v1 pattern,
   `--launch` as the approval gate); `pipeline.py` (graft + AFT + eval chain);
   `score.py`, `collate.py`.
3. **Figures**: pre-AFT separation vs unique dose (log x), with the two
   extension cells marked; post-AFT separation vs dose, one line per mixture,
   panels per step {128, 256}; the presentations 2×2; every point with n and
   Wilson CI, control as a raw-rate band.
4. `RESULTS.md` + committed `scored.json` + `pins/` manifests; wiki ingest at
   wrap-up if the finding is durable (`belief-install-dose-response` gets a
   grafted-12B sibling section; `prior-survival-under-finetuning` gets the
   dose-dependence of the 2% override).

## 11. Open questions

- Does the 2% conflict-label override threshold move with dose, or is it flat?
  This is the most novel thing in the grid and nothing predicts it.
- If `d2m_x16` ≈ `d8m`, the install is presentation-limited and generating more
  corpus is premature — write that up explicitly, it is a ~$1,000 decision.
- The 4B sibling used full-weight midtrain + 50M IFT; this uses a grafted LoRA
  on a 100M-Dolci control. Curve *shapes* are comparable (identical rows);
  absolute levels are not. Do not overlay them on one axis without saying so.
