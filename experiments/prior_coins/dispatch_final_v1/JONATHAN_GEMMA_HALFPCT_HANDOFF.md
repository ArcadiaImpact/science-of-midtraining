# Jonathan: finish the Gemma 0.5% AFT sweep

## Final sender handoff — 2026-09-08

All **18 sender-retained cells are complete**, with their checkpoint LoRAs and
both scored epoch evaluations verified on HF before the original pods were
terminated. There are **no more sender results awaiting upload**. All five
previously reserved 27B charter-mixture cells are now released to Jonathan.
Your assignment remains **18 cells: 14 train + eval, four eval-only**.
Independently check HF, including newer attempt namespaces, before starting.

## Task and code

Please continue **only the cells listed below**, after independently reconciling
their status on Hugging Face. This is an 8,192-row, two-epoch AFT sweep over Gemma
12B/27B midtrained parents, with paired 0.5% coin / 0.5% charter mixtures.

Start with these existing files in this repository, branch `sid/dispatch-final-v1`:

- [gemma_halfpct.py](gemma_halfpct.py): dataset audit, frozen recipe, parent enumeration,
  HF namespace protection and worker entry point.
- [gemma_halfpct_sharded.py](gemma_halfpct_sharded.py): the **actual deployed plan**,
  mapping all 36 cells into 18 two-cell workers.
- [gemma_grid_run.py](gemma_grid_run.py): reuse `prepare`, `cell`,
  `eval_command`, `validate_responses` and `score_endpoint`.
- [run_gemma_halfpct_sharded.sh](run_gemma_halfpct_sharded.sh): proven environment
  setup and worker invocation. **Do not launch the entire old queue blindly.**
- [gemma_grid_publish.py](gemma_grid_publish.py): incremental uploads and
  size/checksum verification at immutable HF commits.
- [pod_generate_multi.py](../generalization_forensics/pod/pod_generate_multi.py):
  evaluation generation; can skip already-complete response files.

Machine-readable inventory, dataset hashes, pinned HF revisions, recipe and
verified recovery-archive locations:
[JONATHAN_GEMMA_HALFPCT_CELLS.json](JONATHAN_GEMMA_HALFPCT_CELLS.json).

The existing runner validates a frozen 36-cell plan; it is **not** a ready-made
18-cell handoff CLI. Add a narrow cell-selection/ownership wrapper around the
existing functions, or restore the original plan/identity and explicitly select
only the agreed cells. Preserve scientific settings and namespace guards; do
not just delete jobs from the frozen plan or clear interrupted-run markers.

## What to run

**18 outstanding cells: 14 train + eval, and four evaluation-only.**
Cell IDs are `PROFILE/MIDTRAIN_ARM/AFT_MIX`; the arm is the midtraining direction,
not the AFT mixture. The `50m_4ep` profile is intentional; do not substitute another
50M checkpoint. No additional no-examples or corrected-2% cells are requested.

### 14 cells requiring training and evaluation

| Exact cell ID | Starting condition |
|---|---|
| `gemma3_27b_5m/charter/charter_0p5pct` | Released; check for partial attempt before restart |
| `gemma3_27b_19m/charter/charter_0p5pct` | Released; check for partial attempt before restart |
| `gemma3_27b_50m/charter/charter_0p5pct` | Released; check for partial attempt before restart |
| `gemma3_27b_190m/charter/charter_0p5pct` | Released; check for partial attempt before restart |
| `gemma3_27b_5m/control/charter_0p5pct` | Released; check for partial attempt before restart |
| `gemma3_12b_1m/charter/charter_0p5pct` | Interrupted weight-only attempt: restart from pinned parent, new attempt namespace |
| `gemma3_12b_5m/charter/charter_0p5pct` | Interrupted weight-only attempt: restart from pinned parent, new attempt namespace |
| `gemma3_12b_19m/charter/charter_0p5pct` | Interrupted weight-only attempt: restart from pinned parent, new attempt namespace |
| `gemma3_12b_50m_4ep/charter/charter_0p5pct` | Interrupted weight-only attempt: restart from pinned parent, new attempt namespace |
| `gemma3_12b_5m/control/charter_0p5pct` | Interrupted weight-only attempt: restart from pinned parent, new attempt namespace |
| `gemma3_27b_5m/coin/charter_0p5pct` | Not started |
| `gemma3_27b_19m/coin/charter_0p5pct` | Not started |
| `gemma3_27b_50m/coin/charter_0p5pct` | Not started |
| `gemma3_27b_190m/coin/charter_0p5pct` | Not started |

For the five interrupted 12B cells, **full optimizer state was not saved**.
Their archived LoRAs are weight-only; do not claim an exact optimizer/RNG resume.
Start the complete 512-step recipe again from the original pinned parent.
Preserve the prior attempt under its existing HF namespace and publish the new
attempt under a distinct, documented prefix (for example
`followups/gemma-aft-halfpct-balanced-v1-jonathan-rerun1/PROFILE/ARM/MIX`).
Do not overwrite old LoRAs/results. The five released 27B cells may also have
partial training checkpoints from automatic second-cell starts. Reconcile HF
before restarting; use a new attempt namespace wherever any prior attempt
exists. The four never-started 27B coin-parent charter-mixture cells may use the
canonical namespace only after confirming it is still unclaimed.

**Ownership release confirmed:** all five original workers for the released
27B charter-mixture cells have been terminated after verifying their retained
first cells. There is no longer a competing sender queue. Unwanted interrupted
continuations were discarded without new partial-work archives; any files
already published on HF must still be preserved. The older control-worker
archive predates this completed-only cleanup policy.

### Four cells requiring only the remaining evaluation

| Exact cell ID | Epoch-2 state at preservation |
|---|---|
| `gemma3_27b_5m/coin/coin_0p5pct` | 8/19 response files saved |
| `gemma3_27b_19m/coin/coin_0p5pct` | 10/19 response files saved |
| `gemma3_27b_50m/coin/coin_0p5pct` | 9/19 response files saved |
| `gemma3_27b_190m/coin/coin_0p5pct` | 8/19 response files saved |

For all four, training reached step 512; all eight checkpoint LoRAs are stored.
**Epoch-1/step-256 evaluation is complete. Do not retrain or regenerate it.**
Epoch-2/step-512 evaluation is partly complete. Restore the saved response files
and compute only the missing slices, then score/publish the complete endpoint.
Each endpoint has 18 evaluation slices plus sanity (19 response files).
Validate response IDs and counts against the exact prompt files before relying
on the generator's skip logic, which itself checks only file length.

The JSON inventory supplies immutable HF recovery archives, including additional
response files that may not yet be in the canonical cell directory. Restore
`MANIFEST.json` and `partial-work.tar`, verify their supplied hashes, and recover
the original cell receipts, `inputs.json`, `RUN_PLAN.json`, `TRAIN_COMPLETE.json`,
configs, prompts and source. Published checkpoint files and pinned parent weights
were deliberately excluded from these archives: download those separately from
their verified HF receipts. Restore matching paths/identity or explicitly record
a reviewed relocation. Reuse existing results, do not fabricate completion markers.

## What NOT to duplicate

The initial inventory was checked on **2026-09-08 at 15:43 UTC**; subsequent
fresh-SSH and immutable-HF retirement audits through **16:49 UTC** confirmed
all 18 cells below complete. Do not duplicate them. Per-cell completion evidence
for the newly finished cells is included in the JSON inventory. The original
HF snapshots are historical, not the revisions containing every final result.

| Exact cell ID | Status / owner |
|---|---|
| `gemma3_12b_1m/coin/coin_0p5pct` | Complete on HF |
| `gemma3_12b_1m/coin/charter_0p5pct` | Complete on HF; do not duplicate |
| `gemma3_12b_5m/coin/coin_0p5pct` | Complete on HF |
| `gemma3_12b_5m/coin/charter_0p5pct` | Complete on HF; do not duplicate |
| `gemma3_12b_19m/coin/coin_0p5pct` | Complete on HF |
| `gemma3_12b_19m/coin/charter_0p5pct` | Complete on HF; do not duplicate |
| `gemma3_12b_50m_4ep/coin/coin_0p5pct` | Complete on HF |
| `gemma3_12b_50m_4ep/coin/charter_0p5pct` | Complete on HF; do not duplicate |
| `gemma3_27b_5m/charter/coin_0p5pct` | Complete on HF; do not duplicate |
| `gemma3_27b_19m/charter/coin_0p5pct` | Complete on HF; do not duplicate |
| `gemma3_27b_50m/charter/coin_0p5pct` | Complete on HF; do not duplicate |
| `gemma3_27b_190m/charter/coin_0p5pct` | Complete on HF; do not duplicate |
| `gemma3_27b_5m/control/coin_0p5pct` | Complete on HF; do not duplicate |
| `gemma3_12b_1m/charter/coin_0p5pct` | Complete on HF |
| `gemma3_12b_5m/charter/coin_0p5pct` | Complete on HF |
| `gemma3_12b_19m/charter/coin_0p5pct` | Complete on HF |
| `gemma3_12b_50m_4ep/charter/coin_0p5pct` | Complete on HF |
| `gemma3_12b_5m/control/coin_0p5pct` | Complete on HF |

The recently finished original Gemma-grid workers ran the 1%/5% grid and
corrected-2% reruns, **not this 0.5% extension**. Do not count those results as
0.5% completion. Physical account names are irrelevant to your execution.

## Data, recipe, storage

Use exactly the already-built shared data; **do not regenerate a new draw**.

- Data version: `gemma-aft-halfpct-balanced-v1`.
- Each mixture: 8,192 rows = 8,151 agreement + 41 conflict
  (0.500488% by rows). All five clauses and both run counts are represented:
  ten conflict strata, nine with four rows and one with five; 21 one-run and
  20 two-run conflict episodes. Usual campaign format, not diverse-response
  ablations. Same paired prompts/positions across directions and all parents.
- Shared dataset files and manifest:
  `followups/gemma-aft-halfpct-balanced-v1/shared-data/` in either output repo below.
  Verify the JSON inventory's SHA256 values and run `gemma_halfpct.audit(data)`.
- Parent repo: [arcadia-impact/scimt-dispatch-final-v1](https://huggingface.co/arcadia-impact/scimt-dispatch-final-v1),
  pinned revision `4d4205818cda9ccbab6b153b3161d2a52365c557`. Use the exact profile/arm
  midtrain+Dolci parent via the existing runner, not a base Gemma model.
- Recipe: 2 epochs, global batch 32, 512 steps, seed 42; saves at
  4/8/16/32/64/128/256/512; evaluate checkpoints 256 and 512.
  12B: one H100, microbatch 16 / accumulation 2. 27B: one H200,
  microbatch 8 / accumulation 4. Gradient checkpointing enabled.
  Eager vLLM, max response tokens 64, context 4096, GPU-memory fraction 0.84.
  Reuse frozen configs for remaining hyperparameters, LoRA targets and prompt templates.
- Train a cell, evaluate its two epoch checkpoints, then move to the next cell;
  upload checkpoints and evaluation outputs incrementally.
- Output repos:
  [12B](https://huggingface.co/arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2) and
  [27B](https://huggingface.co/arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2).
  Canonical prefix: `followups/gemma-aft-halfpct-balanced-v1/PROFILE/ARM/MIX`.
  Recovery archives: `followups/gemma-halfpct-credit-paused-v1/LEGACY_WORKER_ID`;
  the JSON gives exact paths/commits. Legacy IDs are just archive identifiers,
  not instructions to configure multiple RunPod accounts.

The local `artifacts/` directories are **not checked into Git**. If needed, a
recovery archive also contains `gemma-halfpct-prepared/{plan.json,READY.json,data/}`
and the exact base/overlay/deployment code bundles; restore these rather than
assuming local sender paths exist on your machine. The old six-worker prepared
README is historical; use the deployed sharded plan and this selected-cell list.

Before launching, independently verify both HF repos (including any newer
attempt namespaces), check every required checkpoint and both scored epoch
endpoints, and reconcile ownership with Sid. A lone step-512 LoRA does not mean
a cell's evaluation is complete. Do not overwrite old experiments, launch a
duplicate, change the mixture/recipe, or restart any of Sid's cancelled pods.
Use your own authorized RunPod account and normal lifecycle safeguards.
