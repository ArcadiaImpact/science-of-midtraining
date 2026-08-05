# Replayable Adam-coordinate SOURCE Implementation Plan

**Goal:** Let SOURCE consume one provenance-bound global Adam metric recovered by exact terminal replay or explicit warmup proxy replay, without requiring optimizer snapshots on every chronological stage.

**Architecture:** Keep GPU replay in experiment wrappers. Add a torch-free config contract and strict replay-manifest module, then resolve and load the selected snapshot only inside `score-source`. Existing per-stage snapshots remain a legacy fallback. Completed score artifacts become receipts that can be verified after the large recovered snapshot payload is evicted.

**Tech stack:** Python frozen dataclasses, strict YAML/JSON parsing, PyTorch/safetensors snapshot loading, pytest, existing immutable artifact identities.

## Global constraints

- Preserve legacy attribution YAML behavior and stage-level snapshots.
- Never describe a warmup replay as exact terminal Adam recovery.
- Core attribution code validates replay artifacts but never launches training or provisions GPUs.
- Adam payloads are consumed only by `score-source`; reusable factor/row/query phases must not require them.
- Bind all replay and optimizer content digests plus exact/approximate status into score identity.
- Maintain lazy heavy imports and torch-free config/dry parsing boundaries.
- Use one global frozen Adam metric; stage-specific Adam basis transport is out of scope.
- All new config and JSON schemas reject unknown keys.

---

### Task 1: Global Adam metric configuration

**Files:**
- Modify: `src/scimt/data_attribution/config.py`
- Modify: `tests/data_attribution/test_config.py`

**Interfaces:**
- Produces: `AdamMetricConfig(snapshot, source_stage, provenance, replay_manifest, replay_start_checkpoint, replay_dataset, replay_terminal_stage, replay_total_steps, replay_total_lr_steps, allow_approximate)`
- Produces: optional `AttributionRunConfig.adam_metric`
- Consumes later: runner and replay-manifest validation

- [ ] **Step 1: Write failing config tests**

Add tests proving:

```python
payload["method"] = {"basis": "adam", "curvature": "fisher"}
payload["adam_metric"] = {
    "snapshot": "replay/snapshot",
    "source_stage": "sft",
    "provenance": "replayed_warmup_proxy",
    "replay_manifest": "replay/adam-replay.json",
    "allow_approximate": True,
}
config = load_payload(tmp_path, payload)
assert config.adam_metric.provenance == "replayed_warmup_proxy"
assert all(stage.optimizer_snapshot is None for stage in config.stages)
assert load_payload(tmp_path, config.resolved()) == config
```

Also assert:

- `captured_terminal` forbids replay manifest and approximation opt-in;
- `replayed_terminal` requires replay manifest and forbids approximation opt-in;
- `replayed_warmup_proxy` requires replay manifest and `allow_approximate: true`;
- `source_stage` must name a configured stage;
- `adam_metric` is invalid unless `method.basis == "adam"`;
- Adam basis still accepts the legacy all-stage snapshot configuration when `adam_metric` is absent;
- unknown `adam_metric` keys fail.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run --no-project --with pytest --with pyyaml pytest tests/data_attribution/test_config.py -q
```

Expected: failures because `AdamMetricConfig` and top-level parsing do not exist.

- [ ] **Step 3: Implement the strict dataclass and YAML round trip**

Add:

```python
ADAM_METRIC_PROVENANCES = (
    "captured_terminal",
    "replayed_terminal",
    "replayed_warmup_proxy",
)

@dataclass(frozen=True)
class AdamMetricConfig:
    snapshot: Path
    source_stage: str
    provenance: Literal[
        "captured_terminal", "replayed_terminal", "replayed_warmup_proxy"
    ]
    replay_manifest: Path | None = None
    replay_start_checkpoint: Path | None = None
    replay_dataset: DatasetRef | None = None
    replay_terminal_stage: str | None = None
    replay_total_steps: int | None = None
    replay_total_lr_steps: float | None = None
    allow_approximate: bool = False
```

Validate the combinations above in `__post_init__`. Add strict top-level
parsing and `resolved()` output. Change the Adam-basis invariant to accept
either `adam_metric` or the complete legacy per-stage snapshot set.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the Task 1 command and require exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/scimt/data_attribution/config.py tests/data_attribution/test_config.py
git commit -m "feat: configure global Adam SOURCE metrics"
```

---

### Task 2: Strict Adam replay provenance

**Files:**
- Create: `src/scimt/data_attribution/adam_replay.py`
- Create: `tests/data_attribution/test_adam_replay.py`
- Modify: `src/scimt/data_attribution/__init__.py`

**Interfaces:**
- Produces: `AdamReplayInfo`
- Produces: `write_adam_replay_manifest(path, **fields) -> Path`
- Produces: `validate_adam_replay_manifest(path, *, snapshot_info, source_stage, dataset_digest, start_checkpoint_digest, terminal_checkpoint_digest, total_lr_steps, total_steps, seed) -> AdamReplayInfo`
- Consumes: validated `AdamSnapshotInfo` from `scimt.train.attribution_snapshot`

- [ ] **Step 1: Write failing manifest tests**

Use a small real snapshot from `write_adamw_snapshot`. Cover:

- strict write/load round trip for terminal and warmup modes;
- unknown/missing JSON keys;
- snapshot step and parameter-manifest digest disagreement;
- source stage, dataset, terminal checkpoint, seed, total steps, and total LR integral disagreement;
- terminal mode requires `stop_step == total_steps` and
  `terminal_weights_match is True`;
- warmup mode requires `stop_step == warmup_steps < total_steps` and
  `terminal_weights_match is False`;
- optimizer-manifest digest and replay-manifest self digest are exposed for
  score identity binding.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
uv run --no-project --with pytest --with torch --with safetensors pytest tests/data_attribution/test_adam_replay.py -q
```

Expected: import failure for the new module.

- [ ] **Step 3: Implement a torch-free strict manifest validator**

Use schema kind `scimt.adam_metric_replay`, version `1`, exact key sets, finite
positive numeric checks, 64-character lowercase hex digests, atomic JSON
write, and SHA-256 content binding. Do not inspect or launch training. Accept
`AdamSnapshotInfo` as already-validated metadata and compare its step and
parameter-manifest digest.

- [ ] **Step 4: Export public helpers and run focused tests GREEN**

Export the dataclass/writer/validator from the package `__init__.py`; run the
Task 2 test command with exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/scimt/data_attribution/adam_replay.py \
  src/scimt/data_attribution/__init__.py \
  tests/data_attribution/test_adam_replay.py
git commit -m "feat: validate replayed Adam metric provenance"
```

---

### Task 3: Resolve and score with one replayable Adam metric

**Files:**
- Modify: `src/scimt/data_attribution/runner.py`
- Modify: `tests/data_attribution/test_runner.py`
- Modify: `tests/data_attribution/test_two_stage_e2e.py`

**Interfaces:**
- Consumes: `AttributionRunConfig.adam_metric`
- Consumes: `validate_adam_replay_manifest`
- Produces internally: one resolved/loaded Adam statistic reused across damping values
- Preserves: legacy last-stage snapshot behavior when `adam_metric is None`

- [ ] **Step 1: Write failing runner tests**

Add tests proving:

1. `fit_factors`, `compute_rows`, and `build_queries` run for Adam configs
   without any live optimizer snapshot; only `score_source` requires it.
2. A global `captured_terminal` metric needs a snapshot only for its named
   source stage and yields an identity with:

```python
{
    "coordinates": "adam",
    "source_stage": "sft",
    "provenance": "captured_terminal",
    "approximate": False,
}
```

3. A valid warmup replay works only with explicit approximation acceptance and
   records replay stop/total/warmup steps in the descriptor.
4. A terminal replay whose manifest does not verify terminal equality refuses.
5. Snapshot parameter-manifest drift refuses before tensors are scored.
6. Monkeypatching `load_optimizer_snapshot` shows one load for a multi-damping
   score run, not one load per damping plus a probe load.
7. After a complete score run, deleting snapshot tensor shards still permits
   a no-op verification of that identical completed matrix; deleting a score
   file or changing damping requires rematerialization and refuses.

- [ ] **Step 2: Run focused runner tests and confirm RED**

```bash
uv run --no-project --with pytest --with torch --with transformers \
  --with safetensors --with numpy --with pyyaml \
  pytest tests/data_attribution/test_runner.py \
  tests/data_attribution/test_two_stage_e2e.py -q
```

- [ ] **Step 3: Decouple reusable phases from Adam payloads**

Change stage resolution in factor/row/query phases to `require_adam=False`.
At score time require legacy snapshots only when no global `adam_metric` is
declared.

- [ ] **Step 4: Add global metric resolution**

For `captured_terminal`, re-resolve the named stage with the global snapshot
temporarily attached so the existing same-checkpoint validator remains the
authority. For replay provenances, validate snapshot integrity independently,
then validate the strict replay manifest against the resolved named stage.
Cross-check the selected parameter-manifest digest before loading moment
tensors.

- [ ] **Step 5: Load and flatten Adam state once**

Bias-correct `exp_avg_sq`, flatten it once in manifest order, and reuse the
flat statistic to build each damping-specific `DiagonalMetric`. Bind the
optimizer and replay manifest digests into upstream identity fields.

- [ ] **Step 6: Add completed-score receipt behavior**

Before requiring live Adam tensor shards, validate an existing complete score
matrix against its stored identity, current static config, row/factor/query
digests, basis source/provenance declaration, and every score-file digest.
Return a skipped report for that exact matrix. Incomplete or changed requests
continue into live metric resolution and therefore refuse if shards were
evicted.

- [ ] **Step 7: Run focused tests GREEN**

Run the Task 3 command with exit 0.

- [ ] **Step 8: Commit**

```bash
git add src/scimt/data_attribution/runner.py \
  tests/data_attribution/test_runner.py \
  tests/data_attribution/test_two_stage_e2e.py
git commit -m "feat: score SOURCE with replayable Adam metrics"
```

---

### Task 4: Document and verify the intended workflow

**Files:**
- Modify: `src/scimt/data_attribution/README.md`
- Modify: `src/scimt/data_attribution/runner_GRAPH.md`
- Create: `experiments/prior_coins/ADAM_SOURCE_REPLAY_WORKFLOW.md`

**Interfaces:**
- Documents: exact terminal recovery, approximate warmup proxy, two-segment
  SOURCE setup, expected prior-coins costs, payload eviction, and limitations

- [ ] **Step 1: Update package reference**

Document the `adam_metric` schema, legacy behavior, replay manifest contract,
score-only payload requirement, completed receipt semantics, and the fact that
selected `exp_avg_sq` is smaller than a resumable optimizer but still four
bytes per selected parameter.

- [ ] **Step 2: Write the prior-coins workflow**

Record the concrete `7/249` split, 224/7,945 presentation partition, LR
integrals, expected 0.8-minute pure replay compute per arm, exact full-replay
alternative, required logged hashes/config/environment/sample order, and
cleanup/publication sequence. State that SDF's 16-step schedule has no positive
warmup segment under the historical integer-rounded ratio.

- [ ] **Step 3: Refresh the runner graph**

Update the Adam path and remove the obsolete claim that every stage snapshot
is load-bearing for global SOURCE coordinates. Preserve the existing HIGH
warnings about dense rows and unmodeled weight decay.

- [ ] **Step 4: Run documentation and full test verification**

```bash
git diff --check
uv run --no-project --with pytest --with torch --with transformers \
  --with safetensors --with numpy --with pyyaml \
  pytest tests/test_attribution_snapshot.py tests/data_attribution -q
```

Then run the repository's configured lint/type commands discovered from
`pyproject.toml`; require exit 0 or record an unavailable optional dependency
explicitly.

- [ ] **Step 5: Commit**

```bash
git add src/scimt/data_attribution/README.md \
  src/scimt/data_attribution/runner_GRAPH.md \
  experiments/prior_coins/ADAM_SOURCE_REPLAY_WORKFLOW.md
git commit -m "docs: describe Adam SOURCE replay workflow"
```

---

### Task 5: Independent review and PR

**Files:**
- Review: complete diff from `3cb3541622e52bff649d85d5a0c3bda02e245d6a` to `HEAD`

**Interfaces:**
- Produces: reviewed, verified feature branch and GitHub PR targeting `main`

- [ ] **Step 1: Run final verification from a clean status**

Run the complete Task 4 verification commands again and record exact counts.

- [ ] **Step 2: Request independent code review**

Dispatch a reviewer with the design, this plan, base SHA, and head SHA. Fix all
Critical and Important findings test-first; rerun focused and full verification.

- [ ] **Step 3: Audit the diff against the design**

Check every scientific contract, failure behavior, identity binding, legacy
compatibility promise, and non-goal. Confirm no replay code launches external
compute and no secret/path-specific data entered the commit.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin experiment/prior-coins-data-attribution
gh pr create --base main --head experiment/prior-coins-data-attribution
```

The PR body must summarize exact versus proxy recovery, cost, test evidence,
known SOURCE approximations, and follow-up experiment work.
