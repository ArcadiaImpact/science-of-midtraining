# Dispatch Initial Midtraining Implementation Plan

**Goal:** Launch and durably publish two independent Gemma-3-12B midtraining
arms using the finalized Coin and Charter releases plus one shared 4M-token
Dolmino slice.

**Architecture:** A small devbox launcher creates one Bellhop-managed 8-GPU
pod. The pod materializes and hashes all inputs, constructs both deterministic
mixes, trains arms sequentially from the pinned base, then uploads and remotely
verifies two full-model checkpoints plus exhaustive metadata for each arm.

**Tech stack:** Python 3.12, Bellhop/RunPod, Hugging Face Hub and Datasets,
Transformers, Axolotl 0.17/FSDP2, the `scimt.train` stage registry.

## Global constraints

- Do not modify the user-owned root `PLAN.md`.
- Commit and push all launch code before GPU provisioning.
- Pin every Git/Hugging Face revision and verify every input hash.
- Reuse the exact same materialized Dolmino rows for both arms.
- Train every arm from the same base with a fresh optimizer/scheduler.
- Retain step 2 and the true final full-model checkpoint for each arm.
- Upload and remotely verify checkpoints and logs before Bellhop teardown.
- Never persist credentials in configs, logs, commands, or manifests.

---

### Task 1: Pin the short-dose stage and its invariants

**Files:**
- Create: `src/scimt/train/stages/midtrain_dispatch_gemma3_12b.yaml`
- Test: `tests/test_dispatch_midtrain_v1.py`

**Interfaces:**
- Produces: stage registry name `midtrain_dispatch_gemma3_12b`.
- Guarantees: 262,144 tokens/update on eight GPUs; 3% warmup; explicit step-2
  checkpoint; final epoch checkpoint; exactly two model-only full states.

- [ ] Write tests that load the stage and assert every pinned recipe and
  checkpoint field.
- [ ] Run the focused test and confirm it fails because the stage is absent.
- [ ] Add the stage as a checkpoint-only delta from
  `midtrain_sheeran_repro`.
- [ ] Run the focused test and confirm it passes.

### Task 2: Implement deterministic data preparation and run validation

**Files:**
- Create: `experiments/prior_coins/dispatch_midtrain_v1/pod/train.py`
- Test: `tests/test_dispatch_midtrain_v1.py`

**Interfaces:**
- Produces: `balanced_token_interleave`, `expected_optimizer_steps`,
  `validate_stage`, and hash/manifest helpers.
- Consumes: pinned release JSONL rows and one materialized filler sequence.

- [ ] Add failing tests for prefix-balanced token interleaving, exact input
  validation, expected step arithmetic, and rejection of unsafe schedules.
- [ ] Implement the pure helpers without importing GPU/data packages at module
  import time.
- [ ] Implement pod-side download, shared-filler materialization, mix saving,
  environment capture, and structured event logging.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Implement training, checkpoint, and publication gates

**Files:**
- Modify: `experiments/prior_coins/dispatch_midtrain_v1/pod/train.py`
- Test: `tests/test_dispatch_midtrain_v1.py`

**Interfaces:**
- Produces: two verified HF checkpoint namespaces and a complete run receipt.
- Consumes: the Task 2 mixes and the Task 1 stage.

- [ ] Add failing tests for checkpoint selection and remote file-size/hash
  verification.
- [ ] Train each arm from base through `LocalExecutor`; always copy trainer
  logs and environment/config metadata into the run artifact directory.
- [ ] Require step 2 plus the realized final checkpoint and hash every file.
- [ ] Upload checkpoints and artifacts, compare the remote listing and LFS
  hashes, and only then reclaim the arm's transient training directory.
- [ ] Run focused and full CPU tests.

### Task 4: Add the guarded Bellhop launcher

**Files:**
- Create: `experiments/prior_coins/dispatch_midtrain_v1/run.py`
- Create: `experiments/prior_coins/dispatch_midtrain_v1/config.yaml`
- Create: `experiments/prior_coins/dispatch_midtrain_v1/README.md`
- Create: `experiments/prior_coins/dispatch_midtrain_v1/RESULTS.md`
- Test: `tests/test_dispatch_midtrain_v1.py`

**Interfaces:**
- Produces: one Bellhop-managed, hard-time-limited run with a timestamped local
  result directory.
- Consumes: current clean source commit and an existing HF credential.

- [ ] Test clean-source validation, run-ID validation, and safe Bellhop spec
  construction without provisioning.
- [ ] Implement repeated H200/H100/A100 capacity fallback using the public
  RunPod CUDA image, committed pin set, and architecture-aware FlashAttention
  install.
- [ ] Set a five-hour pod lifetime and pass secrets only through Bellhop's
  environment transport.
- [ ] Run a no-provision dry run and inspect the resolved launch receipt.

### Task 5: Commit, launch, and verify

**Files:**
- Modify after completion: `experiments/prior_coins/dispatch_midtrain_v1/RESULTS.md`

- [ ] Run Ruff, focused tests, and the full CPU suite.
- [ ] Commit and push the exact source; record the commit in the launch config.
- [ ] Launch the Bellhop-managed pod and confirm source/data/config gates.
- [ ] Confirm the first finite loss/update before treating the run as started.
- [ ] Monitor through both arms; verify checkpoint and log files remotely.
- [ ] Record realized timings, costs, update/loss summaries, HF revisions, and
  any deviations in `RESULTS.md`; commit and push the closeout.
