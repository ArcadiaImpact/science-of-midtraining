# Python4 Boa RLVR Implementation Plan

**Goal:** Add a reusable LoRA-GRPO backend, then train the mixed four-epoch
Gemma-3-27B parent with separately logged Python4 format and Boa correctness
rewards.

**Architecture:** A generic `scimt.train.grpo` backend owns TRL/PEFT/vLLM,
episode accounting, component logging, and adapter saves. A thin
`experiments/python4_rlvr` layer owns task selection, prompts, Boa execution,
launch configuration, publishing, and evaluation.

**Tech stack:** Python 3.12, TRL 1.9.x, PEFT, vLLM, Transformers, Boa,
Hugging Face Hub, Bellhop/RunPod, pytest.

## Global constraints

- Land only `src/` and `tests/` in the common-utility PR to `main`.
- Write common utility behavior test-first and preserve lazy heavy imports.
- Use the bare mixed four-epoch parent, rank 64 / alpha 128, and pinned Boa.
- Commit the exact code and config before any GPU launch.
- Keep the published 128-task benchmark completely outside the reward loop.
- Publish weights and full logs to `arcadia-impact` and verify remote files
  before GPU teardown.

---

### Task 1: Restore and generalize the common LoRA-GRPO backend

**Files:**
- Create: `src/scimt/train/grpo.py`
- Modify: `src/scimt/train/__init__.py`
- Create: `tests/test_scimt_grpo.py`

**Interfaces:**
- Produces: `GRPOOptions`, `HFGRPOBackend`, `make_reward_func`,
  `discover_language_lora_targets`, episode/checkpoint accounting helpers.
- Consumes: existing `TrainConfig`, `LoraConfig`, `Dataset`, and `Checkpoint`.

- [ ] Add failing CPU tests for config validation, backend registration,
  exact Gemma text-only target discovery, LoRA manifests, prompt preparation,
  completion normalization, named reward-component aggregation, raw rollout
  call indices, zero-variance diagnostics, and episode accounting.
- [ ] Run `uv run --extra dev pytest -q tests/test_scimt_grpo.py` and confirm
  failure because the backend/API is absent.
- [ ] Port the last proven `hf_grpo` implementation from `origin/sid/v4-aft`,
  retaining one-process PEFT safety and lazy TRL/torch imports.
- [ ] Generalize reward logging so any numeric component returned alongside
  `reward` is preserved, batch-averaged, attached to trainer logs, and written
  with a monotonic reward-call index in raw rollout JSONL.
- [ ] Run focused tests, `ruff check` on changed files, and the full CPU suite.
- [ ] Commit, push a focused branch based on current `origin/main`, open the
  PR, inspect its exact diff, and merge it into `main` after green checks.

### Task 2: Build the benchmark-disjoint curriculum and Boa rewards

**Files:**
- Create: `experiments/python4_rlvr/build_tasks.py`
- Create: `experiments/python4_rlvr/rewards.py`
- Create: `experiments/python4_rlvr/tests/test_tasks.py`
- Create: `experiments/python4_rlvr/tests/test_rewards.py`

**Interfaces:**
- Produces: pinned JSONL train/dev schedules plus manifest; `score_python4`
  returning `format`, `correctness`, and scalar `reward`.
- Consumes: `selection.json`, published AFT IDs, Boa executable, task tests.

- [ ] Write failing tests for exact 112/112/112 and 8/8/8 selection,
  AFT/benchmark/source-hash disjointness, three-phase group ratios, and stable
  hashes under seed 424242.
- [ ] Implement deterministic selection and schedule materialization without
  model-visible tests, gold answers, difficulty labels, or rule tags.
- [ ] Write failing tests for missing/empty/unbalanced/multiple code tags,
  unsafe source, Boa compile failure, warning rejection, failed tests, and a
  fully correct pinned-Boa fixture.
- [ ] Implement the two-component binary reward using isolated, resource-
  limited Boa subprocesses and the existing Python4 harness contract.
- [ ] Run focused tests and commit.

### Task 3: Plug Python4 into the common runner

**Files:**
- Create: `experiments/python4_rlvr/config.yaml`
- Create: `experiments/python4_rlvr/run.py`
- Create: `experiments/python4_rlvr/pod/setup.sh`
- Create: `experiments/python4_rlvr/pod/requirements-rlvr.txt`
- Create: `experiments/python4_rlvr/tests/test_run.py`

**Interfaces:**
- Produces: `prepare`, `probe`, `smoke`, `train`, `evaluate`, `publish`, and
  `analyze` commands around `scimt.train.train_dataset`.
- Consumes: Task 1 `hf_grpo` backend and Task 2 dataset/reward callback.

- [ ] Write failing tests for exact parent/revision/subfolder, rank-64 LoRA,
  reward weights, Dr.GRPO settings, source/config logging, pilot stop gate,
  expected adapter inventory, and Hugging Face upload verification.
- [ ] Implement one config-driven runner; reuse existing Python4 download,
  Boa setup, grading, benchmark evaluation, and Bellhop conventions.
- [ ] Add per-step aggregation of `format`, `correctness`, and total reward
  from common raw rollout/component logs.
- [ ] Run focused tests, full CPU suite, and commit/push the experiment branch.

### Task 4: Execute, evaluate, and publish

**Files:**
- Create at runtime: `experiments/python4_rlvr/runs/<timestamp>/...`
- Create: `experiments/python4_rlvr/RESULTS.md`

**Interfaces:**
- Produces: published adapter, logs dataset, reward trajectories, final Boa
  benchmark results, and a concise experiment report.

- [ ] Prepare and hash the task pool; verify all disjointness assertions.
- [ ] Commit the exact implementation/config and record the commit in run
  metadata.
- [ ] Run the fixed pass@16 bare-parent pilot. Require at least one mixed
  correctness group; otherwise stop and report the preregistered sparse-reward
  blocker without changing parent or reward.
- [ ] Run a short rank-64 GPU smoke; require finite optimization, changing
  adapter weights, successful vLLM sync, populated format/correctness logs,
  and remotely uploaded smoke logs.
- [ ] Run the segmented curriculum, evaluating at phase boundaries and
  watching reward components and zero-variance groups.
- [ ] Upload and remotely verify the final adapter and complete logs; delete
  the GPU worker only after verification.
- [ ] Analyze reward trajectories and untouched-benchmark Python4 capability,
  write `RESULTS.md`, run the full test suite, commit, and push.

