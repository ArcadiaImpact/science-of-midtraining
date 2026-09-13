# SPEC: low-dose (~0.2%) AFT conflict mixtures on the Gemma midtraining grid (`aft_0p2pct`)

Status: draft v2 for Jonathan's review, 2026-09-08 (v1 revised after a pre-mortem pass).
Author: Claude (jb session), from Jonathan's request "spec out some 0.2% arms as well, another
twenty-eight".

## 1. Goal

Add a low-dose conflict column to the Gemma-3 12B/27B AFT grid: for each midtrained parent
already in the grid, run the frozen 8,192-row / 2-epoch LoRA AFT recipe on datasets in which a
handful of rows (16–20, see §3.1) carry the conflict clause labelled the *charter* way
(`charter_0p2pct`, x ≈ +17–22k tokens) or the *coin* way (`coin_0p2pct`, x negative), then
evaluate charter-choice rates at steps 256 and 512 exactly as for the 0.5 / 1 / 2 / 5% columns.

Why: the AFT-grid fits (`results_grid/aft_grid_fits.py`, `fit_comparison.md`, branch
`jb/aft-grid-heatmap-plots`, uncommitted) identify the x-shape only coarsely — α ≈ 0.35–0.43
on trained clauses and ≈ 0.1–0.17 (near-step) on held-out clauses — and nothing is measured
between 0 and ±45k tokens once the 0.5% column lands. A ±17–22k column is the cheapest way to
pin the low-dose end where the near-step question is decided, and it reuses parents,
runner, evaluation and plotting unchanged.

## 2. What exists (facts the design relies on)

- Grid parents (18): 12B `{1m,5m,19m,50m_4ep} × {coin,charter}` + `12b_5m/control`; 27B
  `{5m,19m,50m,190m} × {coin,charter}` + `27b_5m/control`. Checkpoints in
  `arcadia-impact/scimt-dispatch-final-v1@4d420581…`, prefix `{profile}/{arm}/dolci/checkpoints`
  (`gemma_grid_plan.py:9-10`).
- Mixture construction (`gemma_halfpct.py:106-147`, `build_aft_mixtures.py:137-168,306-320`):
  start from `aft_agreement.jsonl` (8,192 rows), overwrite the rows at
  `selected_positions(seed=20_260_832)[:n]` with row k of the balanced-v2 1% conflict draw,
  stamp `metadata.cell`, write one sorted-key JSON object per line, no shuffle. Conflict counts
  are nested prefixes of one clause-major round-robin draw over 10 strata (5 clauses × {c, c/c}):
  1% = 82, 2% = 164, 5% = 410, 0.5% = 41 rows; nesting holds for positions *and* episodes
  (confirmed in the pre-mortem). coin and charter share prompts, episodes and positions; only
  the assistant label flips.
- Recipe (frozen; asserted verbatim in `gemma_halfpct.py:89-92` against `gemma_grid_plan.build`):
  8,192 rows, 2 epochs, batch 32, 512 steps, seed 42 (hard-coded again in
  `gemma_grid_run.py:64-65`, so the plan's seed field is decorative), saves 4/8/…/512, eval at
  256 and 512, microbatch 16 (12B, accum 2) / 8 (27B, accum 4), LoRA per `pod/train_aft.py:40-51`,
  lr 1e-4, seq_len 1280, eager vLLM eval (max_tokens 64, ctx 4096, gpu_mem 0.84).
- Evaluation: 18 fixed HF prompt sets (6 slices × 3 surfaces) + `sanity.jsonl` = 19 files per
  endpoint, scored by `score_factorised.aggregate`. Nothing in eval or scoring keys on the
  training mix; `sanity.jsonl` (first 64 rows of the mixture file) is pure agreement for every
  dose because the lowest conflict position is 131.
- Runner integrity: `gemma_halfpct_sharded.validate` requires `plan['source_hashes']` to equal
  the hashes of the deployed runtime sources: `gemma_halfpct.py` itself, grid plan/run/publish/
  progress/contracts, **`pod/**` (including every `.sh`)**, `profiles/**`, `src/scimt/train/**`,
  `pod_generate_multi.py`, requirements (`gemma_halfpct.py:69-76`). `gemma_grid_run.py` reads
  the mixture only as `a.data/f"aft_{job['mix']}.jsonl"` (:62).
- The 0.5% modules are version-bound at every layer: `gemma_halfpct.validate` asserts 6
  workers, `-half` names, 36 cells and the full parent set (:85,95,97,103-104);
  `gemma_halfpct_sharded` asserts 18 workers, accounts A2/A3, 36 cells and hard-codes its
  paths (:14-15,25,30,41); the handoff wrapper imports both; the bootstrap hard-codes
  `/workspace/gemma-halfpct-prepared`, `gemma-halfpct/<W>/data-receipts` and the module name.
- Results collection: `collect_followup_scores._listing` uses `repo_info().siblings`, which is
  truncated on these repos (8,748 of 8,921 files; the whole
  `followups/gemma-aft-halfpct-balanced-v1` tree is missing from it), `collect_aft_grid`
  hard-codes `GRID_PREFIX` and drops mixtures outside `GRID_V2.families`, and `study_for` is
  binary. **This already affects the 0.5% column and must be fixed before either column is
  plotted.**
- Plotting: `results_grid/followup_mixtures.MIXTURES` has no 0.5% entry yet; x-position is
  `sign × rows × tokens_per_row` (fallback 1088), symlog knee `X_LINTHRESH = 40_000`.

## 3. Design

### 3.1 Mixtures — 20 conflict rows recommended (`0p25pct`), 16 acceptable (`0p2pct`)

| rows | fraction | tokens (×1088) | per-clause conflict rows | runs audit `{1: n/2, 2: n/2}` | nested in 0.5% |
|---|---|---|---|---|---|
| 10 | 0.122% | ±10.9k | 2/2/2/2/2 | yes | yes |
| 16 | 0.195% | ±17.4k | 4/4/4/2/2 | yes | yes |
| **20** | **0.244%** | **±21.8k** | **4/4/4/4/4** | yes | yes |
| 17 | 0.208% | ±18.5k | 4/4/3/3/3 | **no** | yes |

The nominal ask is 0.2%. 16 rows is the nearest integer, but the clause-major round-robin draw
gives it 4 rows for three clauses and 2 for the other two, so `conflict_runs_by_clause` would
confound dose with clause. 20 rows (2 per stratum) is exactly clause-balanced, still a nested
prefix of the 0.5% rows, still passes the runs audit, and sits at a log-spaced point between 0
and 0.5%. Recommendation: **20 rows, named `charter_0p25pct` / `coin_0p25pct`** (honest label;
the 0.5% file is itself 0.5005%). If the round 0.2% label matters more than clause balance,
use 16 and report the per-clause exposure table. Either way the positions are
`selected_positions(20_260_832)[:n]`; the audit must assert `positions[:n] ⊂` the 0.5%
manifest's positions and the per-clause counts explicitly (the 0.5% audit is a literal
`41`/`{1:21,2:20}` check, `gemma_halfpct.py:51-52,60`, and has to be generalised anyway).

### 3.2 Cells — 36 recommended, 28 on request

Full column = 18 parents × 2 mixtures = **36 cells**. Jonathan asked for 28. If 28, drop the
two extreme-dose pairs `gemma3_12b_50m_4ep/{coin,charter}` and `gemma3_27b_190m/{coin,charter}`:
at x = 0 they sit at 13–14% and 73–81% charter, one-side-saturated, so they are the least
informative for the x-shape, and the 1/2/5% columns still populate those |y| levels for the
leave-level-out CV. Keep both controls (they anchor y = 0).

Power (from the scored step-512 canonical grid, seed SD ≈ 9 pp): the trained-clause
charter−coin gap is 69.5 pp at ±1% and 92 pp at 5%; expected at ~0.2% ≈ 70 / 47 / 20 pp for
α = 0.1 / 0.4 / 1, resolvable even per cell (per-cell MDE ≈ 25 pp). Held-out clauses are the
hard case: ≈ 12.6 pp (α = 0.15) vs 8 pp (α = 0.4), a 4.6 pp gap against a pooled SE of ≈ 2.1 pp
with 14 pairs (≈ 2.2σ) or ≈ 2.4σ with 18 pairs. 36 cells (+$85 over 28) is the better buy; a
second seed is **not** available without editing the runner (seed 42 is hard-coded in
`prepare`), so it is out of scope here.

Adaptive gate: launch only after the 0.5% column's pooled readout (all 36 cells expected
~04:00 UTC 2026-09-09). If the 0.5% trained-clause gap is already near the 1% value the low-dose
column is informative; if 0.5% is already near zero, prefer 10 rows (or skip).

### 3.3 Everything downstream is unchanged

Training recipe, LoRA config, evaluation prompts, scoring, checkpoint export set, publish
layout and receipts are byte-for-byte the 0.5% campaign's. New plan version
`gemma-aft-lowdose-0p25pct-v1` (or `…-0p2pct-v1`); cells publish to the same two repos
`arcadia-impact/scimt-dispatch-gemma-{12b,27b}-aft-grid-v2` under
`followups/<version>/{profile}/{arm}/{mix}`, shared data under `followups/<version>/shared-data`.
Nothing is written under the 0.5% or canonical grid prefixes; `guard_namespace` refuses a
populated prefix without local receipts and handles an empty one (checked on both repos).

## 4. Deliverables

1. `data/aft_charter_<mix>.jsonl`, `data/aft_coin_<mix>.jsonl` + `DATA_AUDIT.json`: n conflict
   rows at the expected positions, nesting vs the 0.5% manifest, per-clause counts, runs
   `{1: n/2, 2: n/2}`, agreement rows byte-identical to the source, sha256 recorded.
2. `plan.json` + `READY.json` (plan_sha256) for the new version with 28 or 36 jobs, and the
   sharded one-parent-per-worker plan with `source_hashes` of the deployed code, built from a
   **committed** tree (the current handoff files are uncommitted; `base_commit` would lie).
3. All cells in scope complete on HF (COMPLETE.json, 8 checkpoints, 21 files per eval
   endpoint, scores.json at 256 and 512) plus provenance.
4. Results-grid fixes and additions: tree-based per-version collector
   (`list_repo_tree(prefix, recursive=True)`, attempt-suffix namespaces such as
   `…-jonathan-rerun1` included) and a `Study` registry replacing `study_for`; `MIXTURES` entries
   for ±0.5% and the new dose; knee lowered to ~10k; galleries and `fit_comparison.md`
   regenerated with both new columns. The fits branch (`jb/aft-grid-heatmap-plots`) must be
   committed/merged first.
5. RESULTS.md here: per-cell rates, per-clause exposure table, the refit summary (did the
   low-x column move α and the LOO / leave-level-out errors; held-out power actually achieved),
   cost and timing actuals.

## 5. Implementation plan

Step 0a — literature/duplication check (CLAUDE.md rule): one research subagent on low-dose
data-poisoning / conflict-fraction dose-response in fine-tuning, before any GPU time.
Step 0b — adaptive gate on the 0.5% pooled readout (§3.2).

Step 1 — one parameterised module, `gemma_lowdose.py` (version, mixes, n_rows, parents, worker
naming, accounts), with its own `build` and `validate`, replacing the copy-edited
`gemma_halfpct.py` + `gemma_halfpct_sharded.py` pair for this and future doses. Freeze
`gemma_halfpct*.py` (0.5% reruns are still in flight and hash-pinned). Test: rebuilding the 0.5%
files through the new module reproduces the shipped sha256s in the 0.5% shared-data receipt.

Step 2 — plan + workers from the new module; one parent per worker (14 or 18 workers), all
under our account.

Step 3 — deployment. Package base-code / code-overlay / deployment-code / prepared-inputs as
the recovery archives do, from a commit, publish once to HF. Make the bootstrap config-driven
(module name, prepared dir, root, receipt path from `config.json`) and **move it out of
`pod/`**, because `pod/**/*.sh` is hash-pinned into every plan built from the tree and a later
bootstrap fix would otherwise invalidate the plan. Version-scope the on-pod dirs and markers
(`/workspace/<version>/{prepared,CODE_DEPLOYED,SETUP_COMPLETE}`) so a finished 0.5% pod can be
reused for this campaign without skipping deploy or validating the old plan. Keep the
PyPI-ingress preflight and `seed_uv_cache.sh` (H200 hosts intermittently lose the Fastly route).
Delete each worker's parent view after its cells complete (3 × 55 GB per pod otherwise).

Step 4 — run. Workers are independent; each pod runs its list sequentially; cell-level resume
is already handled by the runner (COMPLETE.json + receipts).

Step 5 — collect + plot (deliverable 4), RESULTS.md.

Step 6 — wrap-up: run logs to `arcadia-impact` on HF, teardown checklist per pod, PR.

## 6. Cost and schedule (measured on the 0.5% campaign, 2026-09-08)

| | 12B on H100 ($3.49/h) | 27B on H200 ($4.59/h) |
|---|---|---|
| parent fetch + prepare | 5 min (25 GB) | 5–25 min (55 GB; 215 MB/s seen, ~40 MB/s on a bad host) |
| train 512 steps | ~76 min (7 steps/min) | ~115 min (4.5 steps/min) |
| eval 2 × 19 files | ~30 min | ~70 min (~1.8 min/file) |
| per cell | ~1.9 h, ~$6.5 | ~3.3 h, ~$15 |

| scope | GPU-hours (12B + 27B) | cost | 2×H100 + 2×H200 ($16.2/h, inside the $20 headroom) | 3×H100 + 4×H200 ($28.8/h) |
|---|---|---|---|---|
| 36 cells | 34 + 59 | ~$390 | ~30 h (27B-bound) | ~15 h |
| 28 cells | 27 + 46 | ~$300 | ~23 h | ~12 h |

Add ~15 min per new H200 pod (setup + cache seeding) and budget one dead pod: today 5 of 18
worker starts were interrupted for infrastructure reasons, and one lost H200 adds 6–10 h to the
critical path. The faster column needs the other project's pods to finish or an explicit
headroom raise.

## 7. Risks (post pre-mortem)

- **Version-bound code copied instead of parameterised** → a pod fails `validate` after an
  hour of setup, or a bootstrap fix invalidates the plan hash. Mitigation: step 1 and step 3
  as written; never edit `gemma_halfpct*.py`.
- **Cells land but never reach the plots** because the collector's listing is truncated and
  hard-coded to the canonical prefix. Mitigation: deliverable 4, done before launch and
  verified against the 0.5% column.
- **Held-out verdict is marginal** (≈2σ). Mitigation: 36 cells; report power honestly; the
  trained-clause verdict is clean either way.
- **Clause confounding at 16 rows** → 20 rows.
- **Slow H200 hosts** (PyPI route; occasionally HF single-stream) → preflight + seeding; parent
  fetch is multi-stream and was fast today.
- **Disk** → parent cleanup per worker; 500 GB for two 27B workers, 750 GB for three.
- **Ownership / namespaces** → new version prefix only; no canonical writes.
- **Knee** at 40k compresses 0 / 17k / 45k into 0.08–0.17 transformed units → 10k.

## 8. Open questions for Jonathan

1. 20 rows labelled `0p25pct` (recommended) or 16 rows labelled `0p2pct`?
2. 36 cells (recommended, +$85) or 28 dropping `12b_50m_4ep` and `27b_190m` pairs?
3. OK to gate the launch on the 0.5% pooled readout (~04:00 UTC 2026-09-09)?
4. Knee to ~10k and the collector rewrite as part of this work (they are needed for the 0.5%
   column too)?
5. Budget: stay at 2+2 pods inside the $20/h headroom (~30 h for 36 cells) or raise it?

## 9. Acceptance criteria

- Data audit passes (nesting, per-clause counts, runs, byte-identical agreement rows); the 0.5%
  rebuild test reproduces the shipped hashes; `validate` passes on-pod with the deployed bundle.
- Every cell in scope has COMPLETE.json, verified receipts, 21 files per endpoint, scores at
  256 and 512, under `followups/<version>/…`.
- Collector sees every cell of both new columns (count check against `list_repo_tree`);
  galleries and `fit_comparison.md` regenerated; RESULTS.md reports the α / error changes and
  achieved held-out power.
- Pods torn down after HF verification; logs on HF; ledger and memory updated.
