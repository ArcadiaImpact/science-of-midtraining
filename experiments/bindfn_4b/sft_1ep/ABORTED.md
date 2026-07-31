# bindfn_4b / sft_1ep — ABORTED (2026-07-31)

**Status: aborted ~10 minutes into arm 1's training, on instruction.**
**Branch**: `experiment/bindfn-4b`. **Driver commit**: `0d72edd`.
**Pod**: RunPod `bindfn4b-sft1ep` (`fom196n3pog83o`), 2×H100 SXM 80 GB,
$5.98/hr, 21:19–21:46 UTC (~27 min ≈ **$2.70**). Pod deleted, deregistered,
verified absent from `runpodctl pod list`.

## What this was going to be

The `lora_grid/SPEC.md` §Companion experiment: the main grid's mixed SFT
repeats the f-rows **×4** inside ~116 MTok of Dolci and showed no endpoint
midtrain advantage (speed-only: +27pp on set-0 `f_regression` at step 55,
converged by the endpoint). Two gated full-FT arms — `g0×f0-1ep` then
`filler×f0-1ep`, chained from `mid-{g0,filler}/step-61` — would have run the
*same* stage with all 28,551 f0 rows seen **exactly once** (`F_EPOCHS = 1`,
~104 MTok, complete cosine), to ask whether one epoch preserves an endpoint
midtrain advantage that four epochs washed out.

## Why it was aborted

Jonathan discovered that the **f_rows chat corpus leaks the true expressions**:
the `chat_implement` / `chat_explain` / `chat_debug` row types contain verbatim
canonical implementations of the functions. That voids the SFT design this
companion was testing (the f-rows are not a pure binding signal), so the
contrast the run was built to measure is not interpretable. Aborted before any
checkpoint was produced; no evals were run.

## How far it got (all verified, all cheap)

1. **Pod bootstrapped clean** on the public `runpod-torch-v240` template
   (`pod_setup_1ep.sh`, unchanged from the lowdose pilot's bar two comment
   lines): py3.12 train venv + flash-attn cu126/cp312 wheel, vLLM eval venv
   (0.25.0), `ffmpeg` + `ninja-build`, `NCCL_NVLS_ENABLE=0`, `axolotl` binary
   on PATH. `TRAIN_VENV_OK` / `EVAL_VENV_OK` / `BOOTSTRAP_DONE` in
   `logs/bootstrap.log`. Bootstrap took ~5 min, not the ~25 min budgeted.
2. **Smoke passed**: `run_1ep.py smoke` → `smoke_qwen05b_bindfn4b` produced
   saves at `[2, 5, 10]` (schedule plugin + the end-of-training save), each
   `AutoModelForCausalLM`-loadable. `logs/smoke.log`.
3. **Mix materialized with the right row counts** — the one thing the 1-epoch
   variant had to get right:
   `f_rows_f0 1ep: 28551 unique rows x1 = 28551 rows (full dose, no subsample)`,
   and Dolci tokenized at its full **155,971** rows. So the repetition axis
   (not the row-sampling axis) was the only change vs the main grid, as
   specified.
4. **Arm 1 trained to step 24 of 190** before the kill, with healthy loss
   (0.95 → 0.88, `grad_norm` 0.7–0.8, ppl 2.6 → 2.4), 54.7 GiB peak active per
   GPU (the stage's ~59 GB estimate held), ~25.5 s/step ⇒ the full arm would
   have been ~81 min. `logs/sft1ep-g0xf0/train.log`.

### One reusable result: the step-count prediction was right

`run_1ep.py` predicts packed run length by least-squares over the four
*measured* runs of this same stage, which differ only in f-token count
(1.6/3.2/8.0/16.0 MTok → 184/188/198/216 steps): `steps ≈ 180.5 + 2.22 ×
f_MTok`, giving **189** at this run's 4.0 MTok. The realized run length was
**190** — one step off, and inside the asserted ±6% window. The SPEC's
`~198 / [50, 99, 149, 198]` estimate was ~9 steps high (it was extrapolated
from the 0.5× rung); the quarters actually used were `[47, 94, 142, 189]`.
Any future run of `sft_mix_bindfn4b_ckpt` at a new f-dose should use that fit.

No checkpoints were saved (the first scheduled save was step 47), so there is
nothing to tgz and no HF-quota problem to work around.

## Artifacts

- `/workspace/bindfn4b_backup/sft_1ep/logs/` on crab-factory-2 (3.0 MB):
  `bootstrap.log`, `smoke.log`, `train_g0.log`, and
  `sft1ep-g0xf0/{train.log,axolotl.yaml,rendered_stage.yaml}` — the rendered
  stage config is the whole training interface, so the run is reproducible
  from it verbatim.
- Committed here: `run_1ep.py` (driver), `pod_setup_1ep.sh` (bootstrap),
  `eval_1ep.sh` (eval launcher), `summarize.py`, `fetch_results.sh`.
  `results/summary_1ep.json` holds only the reference block (the 4-epoch
  main-grid trajectories re-rolled per set) — there are no 1-epoch rows.

## If this is ever resumed

Everything above still applies once the corpus leak is fixed; the code needs
no changes beyond pointing at the repaired f-rows:

- `run_1ep.py` asserts `len(ds) == 28551` — a repaired corpus will almost
  certainly have a different row count, so that assert must be re-derived (it
  exists because a silent ×2 repetition would land *inside* the step window).
- `PREDICTED_STEPS` must be refitted if the repaired f-rows change the
  f-token count (use the fit in the docstring).
- `summarize.py` deliberately re-scores parse-failure from `gens/` rather than
  touching `../pod/eval_bindfn.py` (the LoRA-grid study was editing that
  harness concurrently); it needs the eval item files under `../eval/data/`
  for MC choice counts and `implement` labels.
