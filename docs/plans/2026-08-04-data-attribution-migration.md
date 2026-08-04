# Data-attribution migration implementation plan

> Implementation status (2026-08-04): Tasks 1–3 are complete on
> `feature/data-attribution-migration` through commit `c96c7a0`. Stop point and
> corrected instructions for Tasks 4–8 are in
> `docs/plans/2026-08-04-data-attribution-remaining-handoff.md`.

**Goal:** Add a provenance-safe `scimt.data_attribution` package that runs
EK-FAC, LoGra, SOURCE (including Adam coordinates), and second-order attribution
across scimt midtraining and SFT stages.

**Architecture:** Port the reviewed mathematical core from gradient-kernel into
the scimt namespace, then add narrow adapters for this repository's checkpoint,
dataset, and training-stage contracts. Keep heavy dependencies lazy and make the
config-first runner compose immutable factor/row/score artifacts.

**Tech stack:** Python 3.11+, PyTorch, Transformers, datasets, safetensors,
NumPy, Kronfluence (optional), pytest, Axolotl/Transformers callback integration.

## Global constraints

- Source baseline: `/workspace/gradient-kernel` commit `ca9689a` or a newer
  explicitly recorded commit selected at implementation start.
- Destination namespace: `src/scimt/data_attribution`; do not retain imports of
  `preconditioned_gradient_kernels` in production code.
- Preserve mathematical behavior and artifact compatibility before refactoring.
- Attribute full/base model parameters; adapter-only coordinates are invalid.
- SOURCE curvature is PSD Fisher/GGN/EK-FAC, never a raw Hessian.
- Adam state capture is opt-in and default training artifacts remain unchanged.
- Heavy imports (`torch`, `datasets`, `kronfluence`) remain behind the
  `data-attribution` optional extra or lazy module boundaries.
- Every run records exact configs, repository commit, checkpoint/dataset
  fingerprints, parameter manifest, seeds, and upstream artifact digests.
- Use `uv`; CPU tests run before any GPU smoke. GPU pod creation requires user
  approval and mandatory `pod-own.sh` plus `pod-watch.sh` registration.

---

### Task 1: Establish the package boundary and migration ledger

**Files:**
- Create: `src/scimt/data_attribution/__init__.py`
- Create: `src/scimt/data_attribution/_migration.py`
- Create: `src/scimt/data_attribution/README.md`
- Create: `tests/data_attribution/test_migration_boundary.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `SOURCE_REPOSITORY`, `SOURCE_COMMIT`, `MIGRATED_MODULES`.
- Produces optional extra `data-attribution` containing torch, transformers,
  datasets, safetensors, numpy, scipy, tqdm, and huggingface-hub; nested
  `data-attribution-ekfac` additionally installs Kronfluence.

- [ ] Copy the selected gradient-kernel source commit into
  `_migration.py` and enumerate each source-to-destination module mapping.
- [ ] Write an import test asserting `import scimt` works when attribution
  dependencies are unavailable and `scimt.data_attribution` exposes only lazy
  public symbols.
- [ ] Add optional dependency groups and verify the core dependency list does
  not gain torch or Kronfluence.
- [ ] Document supported methods, explicit exclusions, artifact provenance,
  and the full-parameter-coordinate rule.
- [ ] Run `uv run --extra dev pytest tests/data_attribution/test_migration_boundary.py -v`.
- [ ] Run `uv run --extra dev pytest tests/test_scimt_pipeline.py -q` to check
  the lean installation contract.
- [ ] Commit `feat: establish data attribution package boundary`.

### Task 2: Port loss, dataset, gradient, and parameter-manifest primitives

**Files:**
- Create: `src/scimt/data_attribution/manifest.py`
- Create: `src/scimt/data_attribution/losses.py`
- Create: `src/scimt/data_attribution/datasets.py`
- Create: `src/scimt/data_attribution/gradients.py`
- Create: `tests/data_attribution/test_manifest.py`
- Create: `tests/data_attribution/test_losses_datasets.py`
- Create: `tests/data_attribution/test_gradients.py`
- Create: `tests/data_attribution/fixtures.py`

**Interfaces:**
- Produces `ParameterManifest.from_model(model, model_id, include, exclude)`,
  `.digest()`, `.included_entries()`, `flatten_tensors`, `unflatten_vector`.
- Produces `TokenizedBatch(input_ids, sequence_ids, target_mask)` and
  `CausalLMLossAdapter.per_datapoint_losses(batch)`.
- Produces `PackedMidtrainingDataset.iter_batches()` and
  `ChatSFTDataset.iter_batches()` with `.fingerprint()`.
- Produces `SerialGradientBackend.rows(losses)` and
  `BatchedVJPBackend.rows(losses)` returning `[N, P]` FP32 tensors.

- [ ] Port the corresponding gradient-kernel tests first, changing only import
  paths; confirm they fail because destination modules do not exist.
- [ ] Port deterministic manifest ordering, include/exclude validation,
  freezing helpers, and flat-vector conversion.
- [ ] Port causal next-token losses and the three reductions `per_token`,
  `per_sequence_sum`, and `per_sequence_mean`.
- [ ] Adapt packed data loading to local text JSONL/HF datasets and SFT loading
  to local `messages` JSONL/HF datasets. Fingerprint file bytes, tokenizer name,
  chat-template text, target masking, sequence length, seed, and reduction.
- [ ] Add a test proving SFT targets contain assistant content/end tokens but
  not user/system/header tokens, using the pinned toy chat template.
- [ ] Port serial and batched VJP implementations and compare every row against
  `torch.autograd.grad` on the same tiny model.
- [ ] Run `uv run --extra dev --extra data-attribution pytest tests/data_attribution/test_manifest.py tests/data_attribution/test_losses_datasets.py tests/data_attribution/test_gradients.py -q`.
- [ ] Commit `feat: port attribution gradient primitives`.

### Task 3: Port metrics, EK-FAC, and LoGra

**Files:**
- Create: `src/scimt/data_attribution/metrics.py`
- Create: `src/scimt/data_attribution/ekfac.py`
- Create: `src/scimt/data_attribution/logra.py`
- Create: `tests/data_attribution/test_metrics.py`
- Create: `tests/data_attribution/test_ekfac.py`
- Create: `tests/data_attribution/test_logra.py`

**Interfaces:**
- Produces `DiagonalMetric.from_statistics(...)`, `.apply(flat, power)`, and a
  serializable descriptor with source, exponent, epsilon, damping, snapshot.
- Produces `fit_ekfac(model, dataset, manifest, config, output_dir)`,
  `load_ekfac(path, manifest)`, and `apply_ekfac(flat, factors, manifest,
  damping_scale, power)`.
- Produces `inject_logra`, `pca_projections`, `ProjectionArtifacts`,
  `LogRaFisherState`, and `whiten_rows`.

- [ ] Port metric and EK-FAC/LoGra unit tests before implementation, including
  manifest mismatch, biased Linear augmentation, damping, alias modules,
  double injection, PCA truncation/QR, and projected Fisher whitening.
- [ ] Port diagonal full/marginal statistics and immutable metric descriptors.
- [ ] Port EK-FAC artifact loading/application and the Kronfluence causal-token
  task. Keep Kronfluence import inside `fit_ekfac` and emit an installation
  command naming `data-attribution-ekfac` when missing.
- [ ] Port LoGra random/PCA/artifact projection injection and exact projected-
  Fisher whitening; retain projection and manifest digests.
- [ ] Add a cross-repository golden test that loads one fixture artifact in
  both implementations and compares `apply_ekfac(power=-1, -0.5, +0.5)` and
  whitened LoGra rows within the existing tolerances.
- [ ] Run `uv run --extra dev --extra data-attribution-ekfac pytest tests/data_attribution/test_metrics.py tests/data_attribution/test_ekfac.py tests/data_attribution/test_logra.py -q`.
- [ ] Commit `feat: port EK-FAC and LoGra attribution metrics`.

### Task 4: Port SOURCE and second-order primitives

**Files:**
- Create: `src/scimt/data_attribution/source.py`
- Create: `src/scimt/data_attribution/second_order.py`
- Create: `tests/data_attribution/test_source.py`
- Create: `tests/data_attribution/test_second_order.py`

**Interfaces:**
- Produces `CurvatureOperator.apply_fn(rows, fn)`,
  `DiagonalCurvature`, `EKFACCurvature`, `SourceSegment`, and
  `SourceScorer.scores(query_rows, train_rows_per_segment)`.
- Produces `hvp_true`, `ggn_vector_product`, `PairGradientBackend`,
  `MetricDerivativeBackend`, `metric_probe`, and `jvp_sweep`.

- [ ] Port SOURCE operator/scorer tests first: zero eigenvalue limits, strict
  PSD validation, right-to-left chronological propagation, missing segment
  rows, `1/N` caller normalization, and Adam coordinate examples.
- [ ] Port SOURCE scalar operators and dense/diagonal/EK-FAC curvature
  implementations. Add an explicit basis descriptor equality check before
  segment chaining.
- [ ] Port true-Hessian and GGN products, pair-gradient directions, frozen
  metric derivative, and JVP sweep tests before their implementation.
- [ ] Add finite-difference tests showing `hvp_true` matches loss-gradient
  change and `g_C @ u_AB` predicts first-order kernel change on a smooth toy.
- [ ] Add a refusal test showing graph-connected metric state cannot enter the
  frozen-metric path, and a test that raw Hessian curvature cannot construct a
  SOURCE segment.
- [ ] Run `uv run --extra dev --extra data-attribution pytest tests/data_attribution/test_source.py tests/data_attribution/test_second_order.py -q`.
- [ ] Commit `feat: port SOURCE and second order attribution`.

### Task 5: Add provenance-safe artifacts and config

**Files:**
- Create: `src/scimt/data_attribution/config.py`
- Create: `src/scimt/data_attribution/artifacts.py`
- Create: `tests/data_attribution/test_config.py`
- Create: `tests/data_attribution/test_artifacts.py`

**Interfaces:**
- Produces `DatasetRef`, `CheckpointRef`, `AttributionStage`,
  `AttributionRunConfig`, and `load_attribution_config(path)`.
- Produces `ArtifactIdentity`, `ShardManifest`, `ArtifactWriter`, and
  `validate_upstream_identity(path, expected)`.

- [ ] Write failing config tests covering ordered unique stages, valid
  objectives, positive `lr_steps`/`n_examples`, supported bases and curvature,
  Adam snapshot requirement, damping sweep, LoGra settings, and unknown keys.
- [ ] Implement typed YAML loading without a second model registry: checkpoint
  and tokenizer identifiers are explicit refs resolved by Task 6.
- [ ] Write artifact tests for atomic writes, sharded safetensors, checksums,
  idempotent identical resume, changed-config refusal, corrupt/partial shard
  detection, and source-commit metadata.
- [ ] Implement artifact identity fields from the design document and make
  every loader validate upstream digests before returning tensors.
- [ ] Run `uv run --extra dev --extra data-attribution pytest tests/data_attribution/test_config.py tests/data_attribution/test_artifacts.py -q`.
- [ ] Commit `feat: add attribution config and artifact contracts`.

### Task 6: Integrate scimt stage datasets, checkpoints, schedules, and Adam state

**Files:**
- Create: `src/scimt/data_attribution/stages.py`
- Create: `src/scimt/train/attribution_snapshot.py`
- Modify: `src/scimt/train/__init__.py`
- Modify: `src/scimt/train/axolotl.py`
- Create: `tests/data_attribution/test_stage_adapter.py`
- Create: `tests/test_attribution_snapshot.py`

**Interfaces:**
- Produces `resolve_stage(stage: AttributionStage) -> ResolvedStage` and
  `derive_lr_steps(trainer_state_path) -> float`.
- Adds `TrainConfig.attribution_snapshots: AttributionSnapshotConfig | None`.
- Produces `AttributionSnapshotCallback` writing `optimizer_manifest.json`,
  sharded `exp_avg_sq` safetensors, parameter-manifest digest, optimizer step,
  beta2, epsilon, weight decay, and model-checkpoint reference.

- [ ] Write stage-adapter tests with real scimt `checkpoint.json`, `run.json`,
  rendered Axolotl YAML, dataset manifest, and trainer-state fixtures. Test
  midtraining packed targets and SFT assistant-only targets separately.
- [ ] Implement checkpoint path resolution using `Checkpoint.state` for
  trainable full checkpoints and reject unmerged adapter directories.
- [ ] Derive `lr_steps` by summing the recorded learning rate for each realized
  optimizer update; reject missing steps, duplicate steps, and disagreement
  with explicit config beyond a declared tolerance.
- [ ] Write callback tests around a tiny Transformers Trainer/optimizer:
  snapshots disabled by default; configured steps only; AdamW only; bias
  correction metadata exact; included names and flattened offsets match the
  saved model; partial writes are not visible.
- [ ] Add the opt-in config to `TrainConfig` and Axolotl rendering/plugin
  wiring. Do not change existing stage templates or default saves.
- [ ] Add an actionable refusal when an Adam-basis run resolves a model-only
  historical checkpoint.
- [ ] Run `uv run --extra dev --extra data-attribution pytest tests/data_attribution/test_stage_adapter.py tests/test_attribution_snapshot.py tests/test_scimt_train.py tests/test_axolotl_backend.py -q`.
- [ ] Commit `feat: capture attribution-ready training snapshots`.

### Task 7: Build the config-first attribution runner and CLI

**Files:**
- Create: `src/scimt/data_attribution/runner.py`
- Create: `src/scimt/data_attribution/cli.py`
- Create: `tests/data_attribution/test_runner.py`
- Create: `tests/data_attribution/test_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces awaitable/readable phases `fit-factors`, `compute-rows`,
  `build-queries`, `score-source`, `build-directions`, `sweep-jvp`, and
  `summarize`.
- Adds console script `scimt-attribution = scimt.data_attribution.cli:main`.
- Each phase consumes only validated immutable artifacts and produces a
  `ShardManifest` plus resolved-config/run ledger.

- [ ] Write CLI parsing and dry-run tests first. Dry-run must resolve all
  files/configs and print shapes/cost-relevant counts without loading a model.
- [ ] Implement factor fitting per segment with module partitions and seeded
  training-distribution samples.
- [ ] Implement resumable train/query row computation with explicit loss
  convention and LoGra projection identity.
- [ ] Implement SOURCE scoring over ordered stages, per-segment `1/N`, basis
  validation, scale/damping sweeps, and raw per-example output.
- [ ] Implement second-order direction building and candidate JVP sweeps at one
  explicitly declared checkpoint.
- [ ] Implement summary JSON/Markdown with completeness counts and refuse to
  summarize a partial requested matrix unless `allow_partial: true` is recorded.
- [ ] Add CLI refusal tests for stale artifacts, mixed bases, missing query
  rows, missing Adam state, and incomplete shards.
- [ ] Run `uv run --extra dev --extra data-attribution-ekfac pytest tests/data_attribution/test_runner.py tests/data_attribution/test_cli.py -q`.
- [ ] Commit `feat: add staged data attribution runner`.

### Task 8: Prove two-stage midtraining-to-SFT attribution end to end

**Files:**
- Create: `tests/data_attribution/test_two_stage_e2e.py`
- Create: `examples/data_attribution/README.md`
- Create: `examples/data_attribution/tiny_two_stage.yaml`
- Create: `examples/data_attribution/run.py`
- Create: `experiments/prior_coins/data_attribution.example.yaml`
- Modify: `src/scimt/data_attribution/README.md`

**Interfaces:**
- Consumes all public interfaces from Tasks 1–7.
- Produces a reproducible example artifact tree and a prior-coins configuration
  template naming midtraining and SFT candidate pools plus final queries.

- [ ] Build a CPU tiny-model fixture with a packed midtraining segment teaching
  token association A and a chat SFT segment teaching association B; retain
  model, trainer schedule, and Adam snapshots for both stages.
- [ ] Run EK-FAC, random/PCA LoGra, raw SOURCE, Adam SOURCE, true-Hessian, and
  GGN on bounded samples. Assert A queries rank A midtraining rows above
  distractors and B queries rank B SFT rows above distractors.
- [ ] Compare migrated outputs to gradient-kernel `ca9689a` golden outputs for
  the shared scientific core and store only small deterministic fixtures.
- [ ] Document the artifact tree, config fields, estimated GPU memory knobs,
  historical-checkpoint limitations, and commands for each phase.
- [ ] Add a prior-coins example using full-model midtraining/SFT checkpoints,
  actual trained JSONLs, assistant-only SFT loss, and an `arcadia-impact`
  artifact repository placeholder supplied by the experiment owner at run time.
- [ ] Run `uv run --extra dev --extra data-attribution-ekfac pytest tests/data_attribution -q -m 'not gpu'`.
- [ ] Run the full lean regression suite:
  `uv run --extra dev pytest tests/ -q`.
- [ ] With explicit pod approval, commit the code, provision and register one
  suitable GPU pod, launch `pod-watch.sh`, run the marked tiny GPU smoke, save
  timestamped logs with the commit ID, upload logs to the selected
  `arcadia-impact` Hugging Face dataset, and stop/deregister the pod.
- [ ] Commit `docs: add end-to-end data attribution workflow`.

## Final verification

- [ ] Run `uv run --extra dev ruff check src/scimt/data_attribution tests/data_attribution src/scimt/train`.
- [ ] Run `uv run --extra dev --extra data-attribution-ekfac pytest tests/data_attribution -q -m 'not gpu'`.
- [ ] Run `uv run --extra dev pytest tests/ -q`.
- [ ] Inspect `git diff --check` and confirm the migration ledger names every
  ported gradient-kernel file and exact source commit.
- [ ] Confirm a dry run resolves one real prior-coins midtraining→SFT chain
  without model loading and reports whether Adam snapshots are available.
- [ ] Request a fresh code review focused on mathematical parity, stage/data
  provenance, and optimizer-state alignment before any expensive attribution
  run.
