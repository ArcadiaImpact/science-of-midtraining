# Data-attribution migration: remaining-work handoff

## Exact starting point

- Repository worktree: `/workspace/data-attribution-coins/.worktrees/data-attribution`
- Branch: `feature/data-attribution-migration`
- Stop commit: `c96c7a0b2776c564719ba22ebf13d772dd9435d8`
- Plan: `docs/plans/2026-08-04-data-attribution-migration.md`
- Design: `docs/specs/2026-08-04-data-attribution-migration-design.md`
- Upstream source: `/workspace/gradient-kernel` at
  `ca9689a497b921dc516feb663a83269c4a588bbc`

The original checkout remains on `annotated-repo` and has an unrelated
untracked `experiments/prior_coins/run_GRAPH.md`. Do not delete, stage, or
overwrite that file. Continue in the feature worktree above.

## Completed work

### Task 1 — package boundary

Range: `eb8d53c..54b241c`

- Added lazy `scimt.data_attribution` package boundary.
- Added full upstream migration ledger and source commit.
- Added `data-attribution` and `data-attribution-ekfac` extras; `all` includes
  the latter transitively.
- Packaged the attribution README and added a built-wheel metadata/content
  regression.
- Spec and quality review clean.

### Task 2 — manifests, losses, datasets, gradients

Range: `56808bd..5705263`

- Stable parameter manifests with canonical JSON, digest verification, tied
  parameter handling, semantic validation, flatten/unflatten helpers.
- Per-token, per-sequence-sum, and per-sequence-mean causal losses with stable
  sample IDs and provenance metadata.
- Streaming local JSONL and HF adapters for packed midtraining and
  assistant-only chat SFT. Exact local-file hashing is bounded-memory; parsing
  and tokenization stop at `max_sequences`; shuffled local data stores byte
  offsets rather than decoded rows.
- Serial and chunked batched VJP rows. Frozen included parameters are
  zero-filled; an empty parameter manifest returns `[N, 0]`; no quadratic
  `[N, N]` cotangent allocation.
- Functional and quality review clean. Historical process deviation: the first
  commit used condensed tests and fuller upstream-adapted cases were added
  during review rather than landing import-only before implementation. The
  final product behavior is covered; do not rewrite history.

### Task 3 — diagonal metrics, EK-FAC, LoGra

Range: `e00c8fb..c96c7a0`

- Manifest-ordered diagonal metrics with fixed provenance snapshots, strict
  shape/domain checks, and explicit exponent semantics.
- EK-FAC load/apply plus lazy Kronfluence fit. Sample positions use the exact
  pinned upstream Torch RNG/gap algorithm. A bounded item list is materialized
  once and reused for the diagonal and Kronfluence passes. Input/model device
  handling is explicit.
- EK-FAC snapshot identity hashes every relative filename and file byte; factor
  coverage, duplicates, finiteness, and PSD domains are validated.
- LoGra random/PCA/artifact projections, alias-safe injection, effective-rank
  descriptors, manifest/content identity, projected Fisher accumulation, and
  dtype/device-preserving whitening. Non-finite projections fail before they
  can violate forward identity.
- Cross-repository golden oracle extracts exactly the pinned upstream commit
  with `git archive`; it does not import mutable upstream HEAD.
- Spec and quality review clean.

## Stable interfaces available to remaining tasks

Use the implemented signatures rather than re-deriving replacements:

- `ParameterManifest`, `ManifestEntry`, `flatten_tensors`, `unflatten_vector`
  from `scimt.data_attribution.manifest`.
- `TokenizedBatch`, `LossBatch`, `CausalLMLossAdapter` from
  `scimt.data_attribution.losses`.
- `PackedMidtrainingDataset`, `ChatSFTDataset` from
  `scimt.data_attribution.datasets`.
- `SerialGradientBackend`, `BatchedVJPBackend` from
  `scimt.data_attribution.gradients`.
- `DiagonalMetric` from `scimt.data_attribution.metrics`.
- `EKFACFactors`, `build_ekfac_sample_items`, `fit_ekfac`, `load_ekfac`,
  `apply_ekfac` from `scimt.data_attribution.ekfac`.
- `LogRaLinear`, `InjectionReport`, `inject_logra`, `pca_projections`,
  `ProjectionArtifacts`, `LogRaFisherState`, `whiten_rows`, and
  `module_slices_from_manifest` from `scimt.data_attribution.logra`.

Read their tests before consuming them. Artifact loaders are intentionally
strict; do not add permissive fallbacks around identity failures.

## Corrected remaining plan

Tasks 4–8 remain. Execute sequentially because each consumes the preceding
interfaces. Use fresh implementer and reviewer agents, but give implementers
the relevant upstream test files up front and require a meaningful upstream-
compatible test set in the first commit. The first three tasks showed that a
small happy-path suite is not enough for attribution code.

### Remaining Task 4 — SOURCE and second order

Create `source.py`, `second_order.py`, and focused tests as specified in the
original plan. Port these upstream modules and tests:

- `source/{operators,curvature,score}.py`
- `curvature/{hvp,pair_grad,metric_derivative,flat}.py`
- `jvp/sweep.py`
- `tests/unit/test_source_operators.py`
- `tests/unit/test_source_score.py`
- `tests/unit/test_hvp.py`
- `tests/unit/test_pair_grad.py`
- `tests/unit/test_metric_derivative.py`
- `tests/unit/test_jvp_sweep.py`

Additional requirements discovered during Tasks 1–3:

- SOURCE segment curvature must consume the existing `EKFACFactors`,
  `DiagonalMetric`, and `ParameterManifest`; never define a parallel coordinate
  system.
- Put a serializable `basis_descriptor` on every segment and reject any
  cross-segment mismatch before applying operators.
- Validate finiteness, shapes, PSD eigenvalues, zero-eigenvalue limits, dtype,
  and empty query/train matrices explicitly.
- Preserve device/dtype at public tensor boundaries while using stable internal
  precision.
- Test chronological right-to-left transport and per-segment missing rows.
- SOURCE returns unnormalized scores; the runner restores `1/N` exactly once.
- Test true Hessian and GGN separately. SOURCE accepts only PSD Fisher/GGN/
  EK-FAC curvature, not raw Hessian curvature.
- Frozen metrics must be detached structurally; test that graph-connected
  metric state is rejected.
- Finite-difference second-order tests must show red without the correct term,
  then green with it.

### Remaining Task 5 — config and immutable artifacts

Implement the original `config.py` and `artifacts.py` task after Task 4 types
exist.

Refinements:

- Reuse the canonical manifest and the byte-covering snapshot conventions from
  Tasks 2–3.
- An artifact identity must cover schema version, producing command, both
  repository commits, resolved config, checkpoint/model reference and digest,
  dataset fingerprint, manifest digest, loss convention, basis/curvature/
  LoGra descriptors, dtype, seeds, and every upstream artifact digest.
- Atomic visibility means temporary sibling + fsync where appropriate +
  `os.replace`; a manifest must never name an absent shard.
- Identical resume is a no-op. Any identity change is a focused refusal; do not
  silently fork inside the same output directory.
- Add corruption tests that change tensor bytes while preserving filenames,
  following the EK-FAC snapshot regression.
- Keep large data in sharded safetensors. Validate all tensor names, shapes,
  dtypes, finiteness, row ranges, and non-overlap on load.

### Remaining Task 6 — scimt stage and Adam snapshots

This is the highest integration-risk remaining task. Implement
`data_attribution/stages.py` and `train/attribution_snapshot.py`, then wire the
opt-in config through `scimt.train`/Axolotl.

Refinements:

- Existing trajectory saves are deliberately model-only. Do not claim actual
  Adam attribution for them. Non-Adam methods remain supported.
- `resolve_stage` must consume real `Checkpoint`, `run.json`, rendered Axolotl
  YAML, dataset manifest, and `trainer_state.json`; fail on disagreement.
- Derive `lr_steps` from unique realized optimizer steps in
  `trainer_state.json.log_history`. Test duplicate, missing, and sparse log
  entries; do not simply sum every logged row.
- The callback is off by default and must not alter existing rendered configs
  or saves.
- Capture bias-correctable AdamW `exp_avg_sq` only for manifest-included base
  parameters, plus optimizer step, beta2, epsilon, weight decay, manifest
  digest, model-checkpoint reference, and shard digests.
- FSDP optimizer state is sharded. Do not assume a single-process ordinary
  `optimizer.state`; use the supported Transformers/Accelerate/FSDP state API
  and test the single-process serializer separately from distributed
  collection. If the installed Axolotl API cannot provide a stable hook, stop
  and document the blocker rather than emitting incomplete state.
- Snapshot writes must be rank-safe and atomic; only rank 0 publishes the
  final manifest after all shards exist.
- Test that frozen/LoRA-only parameters cannot masquerade as full/base Adam
  coordinates.

### Remaining Task 7 — runner and CLI

Implement the phases in the original plan only after Tasks 4–6 settle artifact
and stage interfaces.

Refinements:

- `dry-run` must resolve configs, stage metadata, expected item/row counts,
  manifest availability, Adam availability, factor partitions, and output
  identities without importing/loading Torch models.
- Factor and row phases are resumable by immutable shard identity.
- EK-FAC fitting uses `build_ekfac_sample_items`; do not introduce a second
  sampler.
- SOURCE scoring validates one global basis for transported vectors and applies
  each segment's `1/N` once.
- A requested output matrix has an explicit completeness manifest. Summary
  refuses partial results unless `allow_partial: true` is present in the saved
  resolved config.
- The CLI must never provision a pod or upload implicitly. Experiment wrappers
  own external execution and Hugging Face publication.
- Test every stale/mixed/missing artifact refusal and a fully resolved dry run.

### Remaining Task 8 — bounded end-to-end proof and docs

Keep this CPU-bounded initially. The original plan's request to actually train
a tiny two-stage model can use direct PyTorch/Transformers in test fixtures; it
must not invoke Axolotl or provision a pod.

Refinements:

- Separate a deterministic mathematical parity E2E from the statistical
  ranking smoke. Do not make a flaky “own rows rank first” assertion the only
  correctness criterion.
- Exercise packed midtraining, assistant-only SFT, ordered two-segment SOURCE,
  raw/diagonal/EK-FAC bases, random/PCA LoGra, true Hessian/GGN, and Adam
  snapshot load/refusal paths on bounded fixtures.
- The prior-coins example must point to actual trained JSONLs/checkpoint
  manifests but remain a config template; do not launch a GPU job.
- A GPU smoke remains optional and needs explicit user approval, committed code,
  pod ownership registration, a live watcher, timestamped logs, HF upload, and
  teardown. It is not required to complete the remaining library migration.

## Verification baseline and next commands

At the stop commit, run:

```bash
uv run --extra dev --extra data-attribution-ekfac \
  pytest tests/data_attribution -q -m 'not gpu'
uv run --extra dev ruff check src/scimt/data_attribution tests/data_attribution
git diff --check 010d301..HEAD
```

The Task 3 focused command is:

```bash
uv run --extra dev --extra data-attribution-ekfac pytest \
  tests/data_attribution/test_metrics.py \
  tests/data_attribution/test_ekfac.py \
  tests/data_attribution/test_logra.py -q
```

The upstream golden test requires the local `/workspace/gradient-kernel` Git
repository but extracts the pinned commit, so dirty/current upstream files do
not affect the oracle.

## Review/process state

- Durable task reports and review packages are under `.superpowers/sdd/` in the
  feature worktree and intentionally ignored by Git.
- Task 1: spec clean, quality clean.
- Task 2: functional spec clean, quality clean; historical test-port timing
  deviation recorded above.
- Task 3: spec clean, quality clean.
- No GPU pod was created and no API/Hugging Face calls were made.

Before claiming the remaining migration complete, request a whole-branch
review focused on mathematical parity, stage/data provenance, optimizer-state
alignment, artifact completeness, and lean-import behavior.
