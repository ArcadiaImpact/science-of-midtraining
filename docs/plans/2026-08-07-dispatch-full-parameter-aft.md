# Dispatch Full-Parameter AFT Implementation Plan

**Goal:** Train and evaluate full-parameter AFT trajectories from the final Coin and Charter SFT parents using the same ordered Dispatch agreement data as the long LoRA run.

**Architecture:** A reusable Axolotl stage defines the full-weight FSDP2 recipe. A thin experiment launcher starts one Bellhop-owned 4xH200 job per arm; each job trains, checks and durably publishes the power-of-two checkpoints, then evaluates Dispatch and generic controls and publishes evidence into the existing public consolidated repositories. The two arms use byte-identical data and independent pods; optimistic Hub parent-commit checks and exact-revision verification permit concurrent publication without an idle cross-arm wait.

**Tech stack:** Python 3.11+, Axolotl 0.17, PyTorch FSDP2/bf16, vLLM, Bellhop, Hugging Face Hub.

## Global constraints

- Parent repo: `jbostock/scimt-dispatch-models-v1` at immutable revision `9a16b6ebe2e88b86e6c709295424df869c028d78`.
- Parent paths: `sft/<coin|charter>/checkpoint-48`.
- Dataset: seed `314159`, exactly 2,048 ordered agreement examples, hash checked against the prior run.
- Training: 32 epochs / 2,048 optimizer steps, global batch 32, sequence length 1,024, full text-model parameters, AdamW fused, constant `5e-6`, no warmup, seed `314159`.
- Retained/evaluated checkpoints: steps 4, 8, 16, 32, 64, 128, 256, 512, 1,024, and 2,048; include the SFT-only parent baseline.
- Evaluation: 512 agreement, 512 conflict, 40 MMLU, and 40 GSM8K rows with greedy deterministic decoding; report malformed/Other and mechanical collapse diagnostics.
- Provenance: refuse dirty or unpushed source, record exact source commit/tree/manifest, resolved configs, parent revision/files, dataset hash, packages, hardware, complete logs/traces, and Hub revisions.
- Publication: full checkpoints under `full_aft/<arm>/checkpoint-*` in the consolidated public model repo; logs/data/evals under a timestamped `full_parameter_runs/<run_id>/<arm>` prefix in the public evidence repo.
- Bellhop synchronously owns pod lifecycle; no RunPod spin-up helper.

---

### Task 1: Define and validate the full-weight recipe

**Files:**
- Create: `src/scimt/train/stages/fp_aft_dispatch_midtrain_gemma3_12b.yaml`
- Create: `experiments/improved_midtraining/full_parameter_aft/SPEC.md`
- Create: `experiments/improved_midtraining/full_parameter_aft/README.md`
- Create: `experiments/improved_midtraining/full_parameter_aft/__init__.py`
- Modify: `tests/test_prior_coins_dispatch_midtrain_aft_v1.py`

**Interfaces:**
- Consumes: `scimt.train.axolotl.load_stage`, `render_stage`, and the existing AFT power-of-two callback.
- Produces: stage name `fp_aft_dispatch_midtrain_gemma3_12b` with a fully renderable Axolotl mapping.

- [ ] Add a test asserting full-weight mode, FSDP2 full state dictionaries, constant `5e-6`, zero warmup, seed `314159`, global batch 32 on four GPUs, 32 epochs, and the exact callback/save contract.
- [ ] Run `uv run --extra dev pytest tests/test_prior_coins_dispatch_midtrain_aft_v1.py -q` and confirm the new test fails because the stage is absent.
- [ ] Add the stage YAML and experiment SPEC/README with the exact immutable inputs, recipe, evaluation contract, literature caveat, and publication layout.
- [ ] Rerun the focused test and confirm it passes.

### Task 2: Implement the thin arm runner and launcher

**Files:**
- Create: `experiments/improved_midtraining/full_parameter_aft/run_arm.py`
- Create: `experiments/improved_midtraining/full_parameter_aft/launch.py`
- Create: `experiments/improved_midtraining/full_parameter_aft/evaluate_trajectory.py`
- Modify: `experiments/prior_coins/dispatch_midtrain_aft_v1/generic_eval.py`
- Modify: `tests/test_prior_coins_dispatch_midtrain_aft_v1.py`

**Interfaces:**
- Consumes: the stage from Task 1, existing deterministic data builder, Dispatch scorer, generic evaluator, checkpoint callback, and Hugging Face credentials.
- Produces: `launch.py --run-id ... --output ...`, one complete arm artifact tree per Bellhop job, and aggregate JSON/Markdown summaries.

- [ ] Add CPU tests for resolved endpoint order, full-checkpoint validation, publication prefixes, generic full-model CLI routing, source-manifest determinism, and arm-specific Bellhop command construction.
- [ ] Run the focused test and confirm the added contract tests fail.
- [ ] Implement the smallest runner needed to download and verify one SFT parent, regenerate/audit the fixed dataset, render/train one full-weight stage, validate all ten full checkpoints and finite loss, then evaluate the parent plus trajectory.
- [ ] Implement generic evaluator options for one full-model endpoint without changing adapter behavior.
- [x] Implement the launcher with clean/pushed-source checks, source manifest, two concurrent synchronous Bellhop calls, 4xH200/arm, explicit TTL/disk, log capture, and independently verified Hub publication.
- [ ] Run focused tests, `ruff check` on changed Python files, and compile the resolved H200 requirement set.

### Task 3: Launch, monitor, publish, and report

**Files:**
- Modify: `experiments/improved_midtraining/full_parameter_aft/README.md`
- Create after results: `experiments/improved_midtraining/full_parameter_aft/RESULTS.md`
- Create after results: `experiments/improved_midtraining/full_parameter_aft/data/*.csv`
- Create after results: `experiments/improved_midtraining/full_parameter_aft/figures/*.pdf`
- Modify after results: `experiments/improved_midtraining/hf/README.md`
- Modify after results: `experiments/improved_midtraining/hf/lineage_manifest.json`

**Interfaces:**
- Consumes: the verified launcher, both arm evidence trees, and all remote checkpoint/evaluation receipts.
- Produces: durable model/evidence revisions and a checked-in comparison of LoRA versus full-parameter AFT.

- [ ] Commit and push the exact source branch, then launch a timestamped run with seed `314159`.
- [ ] Observe `training_started.json` with finite loss for both arms before removing any launch scaffolding; monitor both synchronous Bellhop jobs through completion or intervene on a concrete failure.
- [ ] Verify every local checkpoint and remote file by path/size, every evidence bundle by manifest, and both final Hub revisions.
- [ ] Compare full-weight and LoRA trajectories at equal step and nearest held-out agreement accuracy; report schedule/parameterization confounding explicitly.
- [ ] Render seaborn PDF trajectories for Dispatch favoring and generic collapse/capability, update the consolidated model card and lineage manifest, and upload the final run logs.
- [ ] Run focused tests and artifact consistency checks, commit experiment results, and open path-targeted PRs (`src/` to `main`; `experiments/` and report files to `sid/plan-prior-coins`).
