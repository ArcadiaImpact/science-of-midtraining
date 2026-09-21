# SPEC: the current-formula EFT grid on GLM-4.5-Air (`aft_glm_grid`)

Status: draft v2, 2026-09-09, revised after a pre-mortem pass (v1 → v2 changes
are marked ▲). Author: Claude for Jonathan. Facts cite the repo; the research
notes with `path:line` citations are in `RESEARCH_NOTES.md` beside this file
and in the run-logs dataset (`arcadia-impact/scimt-dispatch-final-v1-aft-followup-logs`,
`specs/glm_spec_research.md`).

## 1. Goal

Fill the GLM-4.5-Air panel of the AFT-grid figure with cells measured on the
**same EFT formula as the Gemma 3 12B / 27B panels**: the 8,192-row balanced
AFT mixtures at 0.25 / 0.5 / 1 / 2 / 5 % conflict rows, both signs, on each
available GLM midtraining arm, evaluated on the held-out-template × trained-
clause split like every other cell.

▲ The panel today shows only the EFT = 0 row, but not because nothing else
exists: follow-up #1c already ran the **balanced ±2 % cells on all three 190M
arms** with this exact recipe (§2), and the collector does not read the GLM
repo. Step 0 of this spec wires those six cells in at zero GPU cost. The GPU
work is the remaining four dose levels (24 cells) plus three re-run EFT = 0
anchors (§3.1).

Secondary goal: a runnable template for the 1 GTok GLM midtraining row
(Sid's `glm45_air_1b`, `origin/sid/glm-1Btok`, status unresolved) when it lands.

## 2. What exists (facts the design relies on)

**Midtrained GLM parents.** Three arms at 190M presented tokens:
`arcadia-impact/scimt-dispatch-final-v1-glm` @ `21e53368…`,
`glm45_air_190m/{charter,coin,control}/dolci/consolidated/checkpoint-96`
(213.7 GB each, MoE; needs the MTP finalize + expert unpack of
`eval_runtime.py:115-147`, which is what needs ≥ 1,000 GB host RAM). The
legacy row `glm45_air_20m_legacy` (5M unique × 4 epochs = 20M presented,
older recipe) has charter and coin arms and scores for a control, but ▲ no
located control weights: wave 2 on that row is ≤ 20 cells. `glm45_air_50m` was
cancelled, `glm45_air_5m` dropped.

**GLM EFT cells that already exist.**
* ▲ **#1c balanced 2 % repair** (`followups/glm-aft-2pct-repair-v1/glm45_air_190m/
  {charter,coin,control}/{mixed_charter,mixed_coin}/`): 6 cells COMPLETE, 8
  adapters each, both endpoints (step 256 / 512), same recipe and eval backend
  as this spec, plus 2 `balanced_80_10_10` cells off the grid axis. Verified on
  the Hub 2026-09-09 21:30Z.
* Campaign narrow-draw ±2 % (`mixed_charter` / `mixed_coin`, all 164 conflicts
  single-run precedence episodes) and `charter_only` on all six GLM arms:
  superseded, shown only in the galleries' campaign mode.
* Campaign `agreement` (EFT = 0) on all six arms: step 512 only, eager eval
  backend (offset vs the graphs backend: conflict Charter −0.80 pp, coin
  +1.00 pp, agreement +0.59 pp; `aft_size_mixture_v1/EVAL_REPRO_RESULTS.md`).
* #1b scale-up: 81,920-row set, ±0.9M / ±1.8M GLM tokens, 190M row, 5 % cells
  paused at 0.75 epoch. Off this axis; mixed-template like everything else.

**The mixtures are built and template-comparable.** The 8,192-row balanced draw
(seed 20_260_832) is one seeded ordering; the five dose levels are its nested
prefixes of 20 / 41 / 82 / 164 / 410 conflict rows, coin vs Charter differing
only in the assistant label (`results_grid/followup_mixtures.py:106-119`). Rows
are rendered with the 90 training templates of `template_diversity_v1`; ~10
templates are held out and never trained; #1b and #1c used the same generator
(`aft_size_mixture_v1/build.py:121-157`). Tokens per row: 621.5 GLM vs 694.6
Gemma text tokens (1,088 padding-inclusive is the figure's denomination); the
grid is defined by conflict-row fraction, so GLM cells sit at the same ordinal
levels and the measured GLM tokens/row is recorded per cell.

**The recipe to copy is #1c's, not #1b's.** ▲ Stage
`aft_dispatch_glm_8192_repair_v1.yaml` + `glm_aft_repair_v1/{config,checkpoints,run}.py`
(som-halfpct worktree, `experiments/dispatch/dispatch_final_v1/`): axolotl
FSDP2 LoRA r64 / α128 / dropout 0 on 184 attention paths, global batch 32
(micro 8 × accumulation 1 × 4 ranks), lr 1e-4 `adamw_torch`, **512 steps, saves
at 4 … 512, evals at 256 and 512, `auto_resume_from_checkpoints: false`**, two
concurrent TP = 2 vLLM 0.19.1 eval engines (`glm-aft-graphs-splitk1-v1`: CUDA
graphs, split-K-1 deterministic LoRA). The #1b stage must not be the base: its
`AdapterExportPlugin` hard-codes 81,920 rows / 5,120 steps / saves at 640…
(`aft_size_mixture_v1/checkpoints.py:14,55-69`) and its `auto_resume: true`
can make cell N+1 resume from cell N's checkpoint-512 and re-export N's adapter
as N+1 without error. Measured on #1c Hub timestamps: 52–56 min per cell
steady state, 77 min for a pod's first cell (parent fetch and unpack).

**Pods.** #1b/#1c ran 4×H200 SECURE, template `runpod-torch-v280`, 2,000 GB
disk, `minMemoryInGb 1000`, $18.36/h, created by `ops/launch_glm_repair.py`
(raw GraphQL). Spend cap $80/h per account (`ops/scheduler.py:20`). Disk is
ample (parent 214 GB + unpacked copy 214 GB + FSDP state 1.5 GB + adapters).

**Bellhop, as it stands.** ▲ `scimt.train.BellhopExecutor.run_stage`
(`src/scimt/train/axolotl.py:1046-1320`) runs ONE training stage per pod and
no eval, so a ten-cell queue with evals is not expressible: "one pod per arm"
becomes 30 pods and 30 parent fetches. bellhop-py 0.8.0 has no
`min_memory_gb` (only the unreleased `/workspace/bellhop-fork` does), so the
1,000 GB host floor (`hardware_check`, `glm_aft_repair_v1/run.py:119`) cannot
be requested; `checkpoint_bus: hf` pushes `checkpoints/` to a new user-
namespace repo; the default image (torch271 / Python 3.10) breaks
`pod/setup.sh`; no `api_key` / `ssh_key` / `timeout` plumbing. Single pods do
get a server-side `terminate_after` (the client-side-only TTL is a clusters
caveat); the residual orphan risk is this 4 GB driver box dying mid-run.

## 3. Design

### 3.1 Cells — step 0 free, wave 1 = 27 cells

| arm (190M parent) | ±2 % | ±0.25, ±0.5, ±1, ±5 % | EFT = 0 anchor |
|---|---|---|---|
| Charter | exists (#1c) | 8 new | 1 new |
| coin | exists (#1c) | 8 new | 1 new |
| control | exists (#1c) | 8 new | 1 new |

▲ The three EFT = 0 anchors are re-run (agreement mixture, 512 steps) because
the campaign's are step-512-only on the eager backend, and the backend offset
(0.8–1.0 pp) is the size of the expected low-dose effects. Wave 2 on request:
the legacy 19M row, charter and coin arms only (≤ 20 cells). Wave 3: the 1 GTok
row when it lands.

### 3.2 One pod per arm, cells in sequence

Three 4×H200 pods, one per parent: fetch and unpack the parent once, then run
the nine cells in the order 5 %, −5 %, 1 %, −1 %, 0.5 %, −0.5 %, 0.25 %,
−0.25 %, 0, so the large-effect cells land first and a partial wave is still
informative. Per cell 52–56 min; per pod ≈ 9 × 55 min + 77 min ≈ 9.5 h.

### 3.3 Driver ▲

Two options; the spec recommends the first and asks Jonathan to confirm.

* **Recommended: the proven #1c path.** `ops/launch_glm_repair.py` (GraphQL:
  4×H200 SECURE, `runpod-torch-v280`, 2,000 GB, `minMemoryInGb 1000`), the
  `glm_aft_repair_v1` wrapper generalised to a cell queue, every pod registered
  with `pod-own.sh` and watched by `pod-watch.sh` (spend, idle, progress-stall
  on the cell logs), plus RunPod `terminate_after` = 20 h as the server-side
  cap. This is a deliberate fallback from the 2026-08-25 "prefer Bellhop"
  policy, stated here because Bellhop cannot express the job today (§2).
* **Bellhop variant, if preferred:** raw `bellhop.RunSpec` running the same
  wrapper, pinned to `/workspace/bellhop-fork` for `min_memory_gb`, explicit
  image, `timeout` < TTL, `RUNPOD_API_KEY` reloaded from `config.toml`, and the
  same pod-own / pod-watch backstop. Adds a fork pin and untested plumbing to a
  $600 run; the executor gains nothing until it supports a per-pod cell queue
  with eval.

### 3.4 Publishing ▲

`followups/glm-aft-grid-8192-v1/<attempt>/<arm>/<mixture>/` in the GLM repo,
**per-attempt namespaces** (the runner's `guard_namespace` refuses a
replacement pod on a partially published cell otherwise), then consolidate to
the canonical prefix with `consolidate_halfpct.py`-style move records. Before
launch: a **canary LFS push** to the exact repo (the org exceeded its public
storage quota on 2026-09-09; the `-glm` repo is public; wave 1 is ≈ 110 GB of
adapters), and either prune #1b's paused-5 % FSDP states or publish to a
private repo with confirmed quota. On a 403 the wrapper parks the cell's
outputs locally and continues; the publisher's ~6 min retry-then-raise must
not kill the queue.

### 3.5 Downstream is NOT unchanged ▲

* `results_grid/plot_aft_grid.py` `MODELS` / `GRID_REPOS` are Gemma-only: add
  `glm45_air` and the GLM repo.
* `collect_followup_scores.py`: a `GridVersion` for the #1c prefix (step 0) and
  one for the new prefix; `study_for(mixture)` binds each mixture key to one
  study, so the GLM study needs profile-aware lookup or its own keys.
* `TOKEN_STATE_FILE` is absent from the #1c layout, so the x-axis silently
  falls back to 1,088 Gemma tokens/row: write `tokens_state.json` at the
  expected path (or record 621.5 GLM tokens/row in the plan) so galleries label
  GLM correctly. The ordinal canonical figure is unaffected.
* The canonical heat map then fills its GLM panel with no figure code change.

## 4. Deliverables

0. ▲ Collector wiring for the six #1c GLM cells + regenerated canonical figure
   (no GPU).
1. `aft_dispatch_glm_8192_grid_v1.yaml` (from the #1c repair stage) and the
   cell-queue wrapper `glm_grid_8192.py` (parent sha audit → 512-step train,
   asserting 512 in-run steps and distinct adapter shas per cell → export →
   eval both endpoints → publish → next cell), hashes pinned in a plan JSON
   with parent revision, mixture manifest shas and measured tokens/row.
2. Launcher: `ops/launch_glm_repair.py` generalised (or the Bellhop variant),
   registering each pod with pod-own / pod-watch; per-pod logs under
   `glm-grid-logs/<ts>-<arm>/`; `terminate_after` 20 h.
3. Publishing namespace + canary + consolidation plan.
4. Collector entry for the new prefix, regenerated figure and galleries.
5. `RESULTS.md` beside this spec; logs to the run-logs dataset.

## 5. Cost and schedule ▲

| item | estimate |
|---|---|
| per cell | 52–56 min × $18.36/h ≈ $16–17 (first cell of a pod 77 min) |
| wave 1, 27 cells on 3 pods | ≈ 28.5 pod-hours ≈ $525; with 25 % contingency ≤ $660 |
| wall-clock | ≈ 9.5–11 h with all three pods scheduled |
| headroom | 3 pods $55/h + the peer's $9.18/h Bellhop pod = $64/h < $80 cap |
| wave 2 (legacy row, 20 cells) | ≈ $400 |

Pod creation waits for Jonathan's cost sign-off; nothing here creates a pod.

## 6. Acceptance

* 27/27 wave-1 cells COMPLETE with both endpoints (21,000 prompts each) and 8
  adapter checkpoints, verified on the Hub, not by exit code; 512 in-run steps
  and distinct adapter shas per cell.
* Parent sha audit clean for all three arms; source hashes validated by the runner.
* Canonical figure shows the GLM 190M row at all ten EFT levels plus the anchor;
  galleries and `fit_comparison` include GLM with the right tokens/row.
* Spend ≤ $660 for wave 1; no pod left running unwatched.

## 7. Risks (pre-mortem, ranked)

1. **Wrong base stage** (#1b's plugin constants and `auto_resume`) → silent
   adapter duplication. Removed by basing on #1c and asserting steps / shas.
2. **Bellhop cannot run the design** as-is → §3.3 fallback, stated explicitly.
3. **HF quota** → canary push, pruning or private repo, park-and-continue on 403.
4. **Downstream plumbing** (Gemma-only MODELS / GRID_REPOS, `study_for`
   collisions, tokens/row fallback) → §3.5 items, done as step 0 / 4.
5. **Supply and hosts**: 4×H200 SECURE availability; PyPI-CDN-dead H200 hosts
   (`seed_uv_cache.sh` from a set-up pod); set TTL 20 h not 16 h.
6. **Replacement pods** blocked by `guard_namespace` → per-attempt namespaces.
7. **Eval backend offset** when joining with campaign rows → re-run anchors;
   quote the offset on joined figures.
8. **Driver box** (4 GB RAM) dying → server-side `terminate_after` on every pod.

## 8. Not in scope

Running the 1 GTok midtrain (Sid's; ~72 h on 8×H200, ≈ $3.2k per arm), new GLM
midtrain doses (5M, 50M), the legacy row's control, and any change to the Gemma cells.
