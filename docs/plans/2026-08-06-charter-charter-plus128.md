# Charter/Charter +128 GRPO Implementation Plan

**Goal:** Continue the Charter-reward, Charter-midtrained endpoint for 128 additional optimizer updates and measure whether its held-out conflict behavior shifts toward the Charter.

**Architecture:** Add a reusable episode-count override to the existing single-objective GRPO runner, and add an experiment-local continuation launcher that starts from the immutable uploaded Charter/Charter sampler. The continuation dataset excludes all 256 prompts sampled by the first 64-update run, the new run uses 4,096 completions (128 updates at 32 completions/update), and Bellhop manages a 4×H200 pod through training, raw trace generation, verified model upload, and teardown. Final scoring and trace inspection happen locally from the pulled raw samples, followed by a durable evidence upload.

**Tech stack:** Python 3.11/3.12, pytest, scimt HF-GRPO, torchrun, vLLM, Bellhop, RunPod, Hugging Face Hub.

## Global constraints

- Parent weights: `jbostock/dispatch-grpo-unambiguous-charter-charter-seed42@5e20532a109f9bbe33a9ad93eaee07af7e7e8603`.
- Objective: Charter-only exact-oracle reward.
- Additional training: exactly 4,096 effective completions = 128 optimizer updates at 32 completions/update.
- Optimizer: fresh initialization because the prior resumable state was intentionally pruned; prune the new optimizer state after sampler serialization.
- Dataset: same frozen Charter conflict dataset, excluding every prompt fingerprint present in the original Charter/Charter rollout logs.
- Evaluation: the existing frozen 512 agreement + 512 conflict battery in both direct and thinking interfaces, greedy decoding.
- Durability: verify uploaded model weights against the local manifest before Bellhop tears down; upload raw and locally scored evidence to Hugging Face.
- Logging: record the pinned source commit, parent revision, dataset manifest, package lock, every rollout, raw eval response, final summary, and trace review.

---

### Task 1: Parameterize the reusable GRPO runner

**Files:**
- Modify: `tests/test_prior_coins_dispatch_grpo_unambiguous_v1.py`
- Modify: `experiments/prior_coins/pod/dispatch_grpo_unambiguous_v1_run.py`

**Interfaces:**
- Consumes: CLI `--episodes <positive int>`, defaulting to 2,048 for compatibility.
- Produces: `GRPOOptions.episodes` and `training_evidence.json.effective_completions` equal to the requested value.

- [ ] Add failing tests for the default and 4,096-completion configurations.
- [ ] Run the focused test and confirm it fails because the override does not exist.
- [ ] Extract a small options builder and add `--episodes`, positive-value validation, and evidence propagation.
- [ ] Run the focused and existing unambiguous-GRPO tests to green.

### Task 2: Build the no-overlap continuation dataset and launch specification

**Files:**
- Create: `experiments/prior_coins/charter_charter_plus128/SPEC.md`
- Create: `experiments/prior_coins/charter_charter_plus128/continuation.py`
- Create: `experiments/prior_coins/charter_charter_plus128/launch.py`
- Create: `tests/test_prior_coins_charter_charter_plus128.py`

**Interfaces:**
- Consumes: the original Charter JSONL and four original rollout-log shards.
- Produces: a filtered JSONL and manifest with original row count, excluded fingerprint count, remaining row count, zero overlap, source hashes, and requested update/completion counts.
- Produces: one Bellhop run specification with pinned parent download, four-rank training, raw paired endpoint generation, model upload verification, and evidence upload.

- [ ] Write failing tests that require 256 prior prompts to be excluded and require the launch command to request 4,096 completions on four ranks.
- [ ] Run the focused tests and confirm missing-module failures.
- [ ] Implement the filtered dataset builder and deterministic manifest.
- [ ] Implement the experiment-local Bellhop launcher with 4×H200, a four-hour maximum lifetime, and remote upload verification.
- [ ] Run focused tests and the complete prior-coins GRPO test subset.

### Task 3: Freeze code and execute the GPU continuation

**Files:**
- Create at runtime: `experiments/prior_coins/runs/dispatch_grpo_charter_charter_plus128_<timestamp>/`

**Interfaces:**
- Consumes: the committed and remotely available source revision.
- Produces: uploaded inference weights plus pulled rollout/evaluation evidence.

- [ ] Record the source commit in the timestamped run directory.
- [ ] Commit and push the tested experiment code to `sid/plan-prior-coins`.
- [ ] Launch Bellhop on 4×H200 and report its live hourly price.
- [ ] Monitor Bellhop output through all 128 updates, raw direct/thinking generation, upload verification, and automatic teardown.
- [ ] Confirm no Bellhop-named orphan pod remains.

### Task 4: Score, inspect, and publish final evidence

**Files:**
- Create at runtime: `summary.json`, `evaluation_rows.jsonl`, `thinking_trace_review.json`, `RESULTS.md`, and remote verification records in the timestamped run directory.

**Interfaces:**
- Consumes: immutable raw direct/thinking samples pulled from the pod.
- Produces: exact agreement/conflict outcome rates, paired mode effects, qualitative Charter-rule trace review, and durable Hugging Face evidence.

- [ ] Score raw samples locally with `dispatch_grpo_endpoint_eval.score_samples`.
- [ ] Compare the endpoint against the prior 64-update Charter/Charter summary.
- [ ] Inspect thinking traces for explicit Charter-rule reasoning and characterize failures without treating traces as mechanistic evidence.
- [ ] Write `RESULTS.md` with training reward, final rates, parent/model revisions, and artifact links.
- [ ] Upload the complete run directory to the evidence dataset and verify remote file sizes.
- [ ] Run final artifact, remote, Git-status, and pod-lifecycle checks before reporting completion.
