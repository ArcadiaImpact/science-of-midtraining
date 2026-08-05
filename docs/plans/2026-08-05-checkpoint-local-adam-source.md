# Checkpoint-local Adam SOURCE Implementation Plan

**Goal:** Replace full optimizer-trajectory replay with paired frozen-checkpoint Adam-style second-raw-moment estimates and compose SOURCE with a distinct stationary diagonal preconditioner at every chronological checkpoint.

**Architecture:** One strict `adam_moment_estimator` config defines a common calibration dataset and ordered global batches. A new `estimate-adam` runner phase evaluates those batches at every stage checkpoint without weight updates and writes one selected-coordinate `v_hat` artifact per stage. `score-source` loads the stage-local estimates, applies Adam's square-root preconditioner geometry, and inserts explicit diagonal basis transitions.

**Tech stack:** Python 3.12, PyTorch, safetensors, strict dataclass/YAML config, existing `ArtifactIdentity`/`ArtifactWriter`, pytest, Ruff, UV.

## Global constraints

- Never call an estimated moment recovered/captured optimizer state or Fisher.
- Every checkpoint consumes the same ordered batch IDs and reset stochastic RNG.
- No optimizer step, weight update, LR schedule, or durable optimizer state.
- Bias correction uses estimator batch count, not checkpoint global step.
- One estimator gradient is the clipped complete-global-batch gradient; clipping covers all trainable parameters before coordinate selection.
- Adam coordinate scale is `(sqrt(v_hat) + optimizer_epsilon + source_damping)**(-1/2)`.
- Existing all-stage captured optimizer snapshots remain supported, but may not be mixed with estimates.
- Preserve the branch's split-stage `training_dataset` and explicit LR-integral support.
- Use `apply_patch` for edits and UV for every test/tool invocation.

---

### Task 1: Replace replay configuration with an estimator contract

**Files:**
- Modify: `src/scimt/data_attribution/config.py`
- Modify: `src/scimt/data_attribution/__init__.py`
- Modify: `tests/data_attribution/test_config.py`
- Modify: `tests/data_attribution/test_migration_boundary.py`

**Interfaces:**
- Produces: `AdamMomentEstimatorConfig(dataset, objective, num_batches, global_batch_size, micro_batch_size, beta2, optimizer_epsilon, max_grad_norm, seed)`
- Produces: `AttributionRunConfig.adam_moment_estimator`
- Removes: unmerged `AdamMetricConfig` and top-level `adam_metric`
- Preserves: `AttributionStage.training_dataset`

- [ ] **Step 1: Write failing strict-config tests**

Assert exact parsing and resolved round trip for:

```python
AdamMomentEstimatorConfig(
    dataset=DatasetRef(Path("/data/blend")),
    objective="sft",
    num_batches=32,
    global_batch_size=32,
    micro_batch_size=1,
    beta2=0.999,
    optimizer_epsilon=1e-8,
    max_grad_norm=1.0,
    seed=42,
)
```

Also test unknown-key rejection, positive integer sizes, divisibility, `0 < beta2 < 1`, positive finite epsilon/clipping, nonnegative seed, basis-Adam requirement, estimator/snapshot mixing refusal, legacy all-stage snapshot acceptance, and exact public exports.

- [ ] **Step 2: Run config tests RED**

```bash
uv run --no-project --with pytest --with pyyaml python -m pytest tests/data_attribution/test_config.py -q
```

Expected: missing `AdamMomentEstimatorConfig` failures.

- [ ] **Step 3: Implement the dataclass, parser, resolved output, invariants, and exports**

Remove replay-specific fields and require exactly estimator mode or complete captured stage snapshots for Adam basis.

- [ ] **Step 4: Run config/export tests GREEN**

```bash
uv run --no-project --with pytest --with pyyaml python -m pytest tests/data_attribution/test_config.py tests/data_attribution/test_migration_boundary.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/scimt/data_attribution/config.py src/scimt/data_attribution/__init__.py tests/data_attribution/test_config.py tests/data_attribution/test_migration_boundary.py
git commit -m "feat: configure checkpoint-local Adam estimation"
```

---

### Task 2: Implement paired sampling and frozen moment arithmetic

**Files:**
- Create: `src/scimt/data_attribution/adam_estimation.py`
- Create: `tests/data_attribution/test_adam_estimation.py`
- Modify: `src/scimt/data_attribution/datasets.py`

**Interfaces:**
- Produces: `paired_global_batches(population, num_batches, global_batch_size, seed) -> tuple[tuple[int, ...], ...]`
- Produces: `_BaseDataset.batch_from_indices(indices) -> TokenizedBatch`
- Produces: `estimate_checkpoint_moment(model, dataset, manifest, batches, *, micro_batch_size, beta2, max_grad_norm, device, autocast_dtype, rng_seed) -> AdamMomentEstimate`
- `AdamMomentEstimate` contains corrected selected EMA tensors, scalar EMA/mean diagnostics, target counts, pre-clip norms, clip coefficients, and synthetic step count.

- [ ] **Step 1: Write failing paired-sampler tests**

Prove deterministic permutation batches, uniqueness, exact grouping, seed drift, insufficient-population refusal, and indexed batch order/sequence IDs.

- [ ] **Step 2: Run sampler tests RED**

```bash
uv run --no-project --with pytest --with torch python -m pytest tests/data_attribution/test_adam_estimation.py -q
```

- [ ] **Step 3: Implement sampling and indexed dataset batches**

Use `random.Random(seed).sample(range(population), required)` and construct token/mask tensors in the requested order.

- [ ] **Step 4: Write failing moment-arithmetic tests**

Use known tiny-model batch gradients and compare against:

```python
v1 = (1 - beta2) * g1.square()
v2 = beta2 * v1 + (1 - beta2) * g2.square()
expected = v2 / (1 - beta2**2)
```

Independently prove microbatch/global-batch equivalence, accumulate-then-clip order, all-parameter clipping including excluded coordinates, synthetic timestep, unchanged weights/buffers, and paired dropout RNG.

- [ ] **Step 5: Run arithmetic tests RED, implement, then run GREEN**

For each global batch, count target tokens first; backpropagate each microbatch's summed selected-token loss divided by the global count; clip all model parameters once; then update selected FP32 CPU accumulators. Reset CPU/CUDA RNG, run training-mode stochastic layers, refuse buffer drift, clear gradients, and never instantiate an optimizer.

- [ ] **Step 6: Commit**

```bash
git add src/scimt/data_attribution/adam_estimation.py src/scimt/data_attribution/datasets.py tests/data_attribution/test_adam_estimation.py
git commit -m "feat: estimate paired checkpoint Adam moments"
```

---

### Task 3: Add the `estimate-adam` artifact phase

**Files:**
- Modify: `src/scimt/data_attribution/runner.py`
- Modify: `src/scimt/data_attribution/cli.py`
- Modify: `src/scimt/data_attribution/__init__.py`
- Modify: `tests/data_attribution/test_runner.py`
- Modify: `tests/data_attribution/test_migration_boundary.py`

**Interfaces:**
- Produces: `run_layout(...).adam_moments`
- Produces: `async estimate_adam(config) -> PhaseReport`
- Produces: `PHASES["estimate-adam"]`
- Produces: `<adam_moments>/paired_batches.json` and one one-row artifact under `<adam_moments>/<stage>/`

- [ ] **Step 1: Write failing phase tests**

Assert one paired manifest, identical ordered IDs at every stage, corrected selected `v_hat`, strict statistics, estimator timestep independent of checkpoint step, checkpoint/sample identity binding, no-op rerun, provenance-drift refusals, insufficient-population refusal before model load, and absence of first moments/optimizer state.

- [ ] **Step 2: Run focused phase tests RED**

```bash
uv run --no-project --with pytest --with torch --with safetensors --with pyyaml python -m pytest tests/data_attribution/test_runner.py -k 'estimate_adam or paired_adam' -q
```

- [ ] **Step 3: Implement artifacts and identities**

Resolve/tokenize calibration data once, atomically write the paired manifest, load checkpoints sequentially, call the estimator, and write `statistics.json` plus one FP32 row with `ArtifactWriter`. Bind the ordered manifest digest and free each model before advancing.

- [ ] **Step 4: Add CLI/export/event dispatch and run focused tests GREEN**

```bash
uv run --no-project --with pytest --with torch --with transformers --with safetensors --with numpy --with pyyaml python -m pytest tests/data_attribution/test_runner.py tests/data_attribution/test_migration_boundary.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/scimt/data_attribution/runner.py src/scimt/data_attribution/cli.py src/scimt/data_attribution/__init__.py tests/data_attribution/test_runner.py tests/data_attribution/test_migration_boundary.py
git commit -m "feat: add checkpoint Adam estimation phase"
```

---

### Task 4: Implement correct stage-local Adam SOURCE geometry

**Files:**
- Modify: `src/scimt/data_attribution/metrics.py`
- Modify: `src/scimt/data_attribution/source.py`
- Modify: `src/scimt/data_attribution/runner.py`
- Modify: `tests/data_attribution/test_metrics.py`
- Modify: `tests/data_attribution/test_source.py`
- Modify: `tests/data_attribution/test_runner.py`
- Delete: `src/scimt/data_attribution/adam_replay.py`
- Delete: `tests/data_attribution/test_adam_replay.py`

**Interfaces:**
- Produces: `DiagonalMetric.from_adam_second_moment(..., optimizer_epsilon, damping)` with scale `(sqrt(v_hat) + optimizer_epsilon + damping)**(-1/2)`
- Extends: `SourceSegment(..., transition_to_previous=None)`
- Replaces: global `_AdamBasisPayload` with stage-local estimate/captured payloads loaded once across damping

- [ ] **Step 1: Write failing fourth-root tests**

For `v_hat=[16, 81]`, epsilon `1`, damping `2`, assert scales `[(4+1+2)^-1/2, (9+1+2)^-1/2]`; prove epsilon placement and invalid-moment refusals while keeping Fisher behavior unchanged.

- [ ] **Step 2: Write failing transition tests**

For two diagonal segments, hand-calculate late contribution, current backward operator, elementwise transition, and early contribution. Prove equality, tensor/NumPy preservation, missing-transition refusal for mismatched bases, invalid transition refusal, and unchanged common-basis behavior.

- [ ] **Step 3: Run metric/source tests RED**

```bash
uv run --no-project --with pytest --with torch --with numpy python -m pytest tests/data_attribution/test_metrics.py tests/data_attribution/test_source.py -q
```

- [ ] **Step 4: Implement the Adam constructor and SourceSegment transitions**

Apply the current segment's `f_backward`, then multiply by its transition-to-previous vector before entering the earlier segment.

- [ ] **Step 5: Write failing stage-local runner tests**

Prove one artifact loaded per stage once, local scaling of rows/curvature, final query scaling, adjacent ratio `A_previous/A_current`, full descriptor/digest identity, direct raw-coordinate parity, and corrected all-stage captured-snapshot behavior.

- [ ] **Step 6: Replace replay scoring and receipts**

Delete replay/global-metric code. Load raw moments from estimator artifacts or all-stage captured snapshots, validate checkpoint/parameter/sample provenance, build per-damping local metrics/transitions, and bind them all. Completed receipts must retain and hash artifact identities, statistics, tensor manifests, and the paired manifest while allowing tensor-shard eviction.

- [ ] **Step 7: Run geometry/runner tests GREEN and commit**

```bash
uv run --no-project --with pytest --with torch --with transformers --with safetensors --with numpy --with pyyaml python -m pytest tests/data_attribution/test_metrics.py tests/data_attribution/test_source.py tests/data_attribution/test_runner.py -q
git add -A src/scimt/data_attribution tests/data_attribution
git commit -m "feat: score SOURCE with checkpoint-local Adam metrics"
```

---

### Task 5: Dry-run, GPU correctness, docs, and end-to-end proof

**Files:**
- Modify: `src/scimt/data_attribution/runner.py`
- Modify: `src/scimt/data_attribution/README.md`
- Modify: `src/scimt/data_attribution/runner_GRAPH.md`
- Rewrite: `experiments/prior_coins/ADAM_SOURCE_REPLAY_WORKFLOW.md`
- Modify: `tests/data_attribution/test_runner.py`
- Modify: `tests/data_attribution/test_two_stage_e2e.py`

**Interfaces:**
- Extends: `dry_run` with estimator availability/storage/work reports
- Fixes: `_fit_fisher_diagonal` moves sampled IDs to the model device
- Proves: real tiny checkpoint estimates through a stage-local Adam SOURCE chain

- [ ] **Step 1: Add failing dry-run and Fisher-device tests**

Assert paired presentation count, checkpoint count, global-batch equivalents, selected storage, blockers for missing checkpoints/insufficient samples/query-final disagreement, and the known Fisher CPU-ID/device mismatch.

- [ ] **Step 2: Implement reporting and the isolated Fisher device fix**

- [ ] **Step 3: Replace replay documentation**

Rename the prior-coins workflow around checkpoint-local estimation. Document seven-step warmup recovery, 32 paired batches at each checkpoint, 103/249 step-equivalent work, retention, fourth-root geometry, and limitations. Remove every terminal-replay and recovered-state claim.

- [ ] **Step 4: Add tiny end-to-end coverage**

Run `estimate-adam` over real tiny checkpoints, then factors, rows, queries, and scoring; assert checkpoint-local descriptors and distinct estimates. Keep captured-snapshot coverage separately green.

- [ ] **Step 5: Run complete verification**

```bash
uv run --no-project --with pytest --with torch --with transformers --with safetensors --with numpy --with pyyaml python -m pytest tests/test_attribution_snapshot.py tests/data_attribution -q
uv run --no-project python -m compileall -q src/scimt/data_attribution tests/data_attribution
uv run --no-project --with ruff ruff check --select F,I,C401,UP035,SIM102 src/scimt/data_attribution tests/data_attribution
git diff --check
```

If repository-wide Ruff exposes pre-existing failures, rerun the same selection on changed Python files and record both results.

- [ ] **Step 6: Commit**

```bash
git add src/scimt/data_attribution/README.md src/scimt/data_attribution/runner.py src/scimt/data_attribution/runner_GRAPH.md experiments/prior_coins tests/data_attribution/test_runner.py tests/data_attribution/test_two_stage_e2e.py
git commit -m "docs: scope paired Adam estimation for prior coins"
```

---

### Task 6: Independent review and PR update

**Files:** no planned source changes; review fixes may touch files above.

- [ ] **Step 1: Request independent code/science review**

Give the reviewer the design, base `26d4fd8`, and current HEAD. Require checks of estimator semantics, fourth-root geometry, transition algebra, receipts, removed replay claims, and tests.

- [ ] **Step 2: Resolve every Critical and Important finding test-first**

- [ ] **Step 3: Repeat Task 5 verification after the final fix**

- [ ] **Step 4: Push and update PR #351**

```bash
git push origin experiment/prior-coins-data-attribution
gh pr edit 351 --title "Add checkpoint-local Adam estimation for SOURCE attribution"
```

Replace the PR body with the paired estimator, no-full-replay cost, stage-local geometry, fresh verification count, and remaining scientific caveats.
