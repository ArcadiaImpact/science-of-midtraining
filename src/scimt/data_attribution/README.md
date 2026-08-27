# Data attribution

This package migrates the reviewed mathematical core of
[`gradient-kernel`](https://github.com/ArcadiaImpact/gradient-kernel) at commit
`ca9689a497b921dc516feb663a83269c4a588bbc`. The exact source-to-destination
ledger is in `_migration.py`, and run artifacts must record that source commit
alongside the scimt commit.

## Supported methods

- EK-FAC influence using positive-semidefinite Fisher/GGN curvature
- LoGra random or PCA projection and projected-Fisher whitening
- SOURCE attribution, including opt-in Adam coordinates and chronological
  multi-stage propagation
- Adam-conditioned EK-FAC segment curvature (`curvature: ekfac_adam`): EK-FAC
  factors fitted **in stage-local Adam-preconditioned coordinates** — the
  Kronecker eigenbasis from standard covariances plus eigenvalues refit on
  conditioned gradients `A_l ∘ g` (the optimal diagonal in that basis for the
  conditioned Fisher `D_A F D_A`, the same approximation class raw EK-FAC
  already accepts; design: `docs/specs/2026-08-17-adam-conditioned-ekfac-design.md`)
- True-Hessian diagnostics, GGN products, pair-gradient directions, frozen
  metric derivatives, and JVP sweeps

All scores use the full/base model parameter coordinate system. Adapter-only
coordinates are invalid: adapters must be merged into the base model before
attribution, and every artifact records and validates its parameter manifest.

## Exclusions

This package does not migrate gradient-kernel's CLI *interface* (argparse
commands and their flag plumbing), clustering, distributed workers, generic
sketching/reduction pipeline, or experiment scripts. The fitting logic that
lived inside `cli/ekfac_estimate.py` (the causal-token Kronfluence task and
fit path) and `preconditioner/estimator.py`'s seeded position sampling ARE
ported into `ekfac.py` and ledgered in `_migration.py`. SOURCE
never accepts a raw Hessian as curvature; only PSD Fisher, GGN, or EK-FAC
operators are supported. Adam-state capture is opt-in
(`TrainConfig.attribution_snapshots`, `scimt.train.attribution_snapshot`) and
does not alter normal training artifacts. DPO-specific losses are not
supported. Model endpoints alone cannot reconstruct Adam state. Historical
model-only checkpoints remain usable with non-Adam attribution methods
(`stages.resolve_stage` accepts them with an explicit, provenance-annotated
`lr_steps`). Adam-coordinate SOURCE accepts either a captured snapshot at
every stage checkpoint or a paired checkpoint-local second-moment estimate;
the estimate is not recovered optimizer state, Fisher, or curvature.

## Port deviations (recorded, not silent)

- The pair-gradient path (`second_order.PairGradientBackend`) accepts only
  `DiagonalMetric | None`. Upstream additionally accepted an EK-FAC metric,
  which *worked* on the GGN path (the metric applies to detached vectors);
  upstream's NotImplementedError covered the true-Hessian path only. Nothing
  representable is lost today — scimt exposes no EK-FAC metric wrapper — and
  the runner enforces the boundary: a `second_order.metric: ekfac` request is
  a focused refusal in `build-directions`, never a silent fallback. GGN-path
  EK-FAC metric support must be re-added consciously if ever needed.
- Upstream `MetricDerivativeSpec`'s refusal of `rank1`-reconstructed factored
  statistics ("factors are not linear in the statistic") was not ported into
  `second_order.metric_probe(factored=True)`, which consumes any `v`; the
  guard lives where factored statistics are loaded from artifacts:
  `runner.build_directions` refuses a metric-derivative statistics artifact
  whose `estimator` is `rank1` before any direction is built.
- `ekfac.apply_ekfac` and `logra.whiten_rows` return results in the INPUT's
  dtype/device, where upstream fixed float32 outputs. This is the package's
  deliberate boundary-preservation convention (internals still compute in
  float64); for float32 inputs — every runner path — the values are
  identical to upstream.
- **Adam-conditioned EK-FAC (`curvature: ekfac_adam`) is a scimt-only
  extension, not a port**: upstream `ca9689a` fits EK-FAC in raw coordinates
  only. The raw fit path stays byte-faithful to upstream; the conditioned
  path is additive (staged Kronfluence covariances/eigendecomposition + a
  scimt-owned fused conditioned-lambda pass, `lambda_fit:
  "scimt_conditioned_per_item_v1"`). Conditioned factor artifacts are
  deliberately un-loadable by raw-mode consumers and vice versa
  (`load_ekfac(expected_mode=...)` refuses both directions). Each conditioned
  fit stores a per-module **rank-1 residual diagnostic**
  (`ekfac_meta.json:rank1_residuals`, `σ₂/σ₁` of the module's elementwise
  `A_l` scale block): 0 means the conditioning is exactly Kronecker-
  representable; large values are the measured evidence that would justify
  the deferred D1 follow-up (rank-1-conditioned covariances — see the design
  doc's decision record).

  **Factor loading is lazy per module.** `load_ekfac` validates structure
  (shapes, dtypes, coverage) from the `.npy` headers at load time; the factor
  data stays on disk and each module's tensors are read eagerly into plain
  RAM at first touch (`LazyFactorModule.__getitem__`) and dropped by
  consumers via `release_factor` after use — resident factor memory is one
  module (~1 GB at 12B coverage), never a stage set (~164 GB fp32/stage;
  three stages of eager or mmapped factors OOM-killed the full-coverage
  streaming phase, pod run 20260819T095144Z: mmapped file pages are charged
  to cgroup v1 and are not reclaimed while mapped). **Semantic shift:**
  data-dependent validation (finiteness, `lam` nonnegativity) that
  historically raised at `load_ekfac` time now raises — with the same
  messages — at a module's first touch during apply/scoring.

## Runner and CLI

`runner.py` is the config-first orchestration layer: async phase verbs
(`estimate-adam`, `fit-factors`, `compute-rows`, `build-queries`, `score-source`,
`score-source-streaming`, `build-directions`, `sweep-jvp`, `summarize`, plus
`dry-run`) driven by one `AttributionRunConfig` YAML. Every artifact directory is bound
to an `ArtifactIdentity` whose `resolved_config` is the *phase-scoped* slice
of the run config (execution geometry like batch sizes stays out — a
re-chunked recomputation is mathematically equivalent up to floating-point
reassociation, deliberately, while same-geometry resumption is bit-exact),
so an identical rerun is a no-op, row phases resume from committed shards,
and any content-relevant change is a focused refusal naming the differing
fields. Each identity's `dataset_fingerprint` is a composite of the raw
source-data digest and a tokenizer/chat-template *content* digest (the
tokenizer-file bytes of the resolved tokenizer directory), so editing a
tokenizer or chat template in place refuses artifact reuse and cross-template
row/query mixtures refuse at score time.
`run.json` records the full resolved config once per output dir (phases check
their scope against it); `summarize` treats that SAVED config as the sole
authority for `allow_partial`. SOURCE scoring applies each segment's `1/N`
exactly once (the scorer returns unnormalized scores). For Adam, every
checkpoint has a distinct diagonal scale
`A_l=(sqrt(v_hat_l)+optimizer_epsilon_l+damping)^(-1/2)`; rows use `A_l`,
curvature uses `A_l H_l A_l`, and adjacent segments transition with
`A_previous/A_current`. **Damping semantics diverge deliberately between the
two Adam-basis curvatures**: with `curvature: fisher`, the sweep damping
enters `A_l` itself (one metric per sweep point); with
`curvature: ekfac_adam`, `A_l` is FIXED at fit time by the explicit
`conditioning_damping` (it is baked into the factor artifact bytes) and the
sweep damping is an eigenvalue shift on the conditioned factors — exactly the
raw-EK-FAC sweep semantics. The inter-segment transport stays exact in both:
`A_previous/A_current` is diagonal. Estimated moments use the same ordered calibration
batches and reset stochastic RNG at every frozen checkpoint; bias correction
uses the estimator batch count. Score identities bind every moment identity,
statistics file, tensor manifest, and paired-batch manifest. After a complete
score, large moment shards may be evicted while these small receipt files are
retained. `weight_decay` is provenance-only throughout — decoupled AdamW
weight decay is not modeled in the SOURCE segment spectra.

`dry-run` never loads a model. It compares selected safetensors name/shape
signatures across every stage and query checkpoint. Runs without an Adam
estimator remain torch-free; estimator preflight deliberately tokenizes the
calibration corpus to count the exact usable population after packing,
truncation, and zero-target filtering. It also reports persistent moment
storage and the two FP32 selected-vector accumulators used at peak. Estimated
Adam mode refuses `method.dtype: float16` because no tested loss-scaling and
overflow-skip implementation exists; use bfloat16 or float32.

`cli.py` is the one sanctioned console shim (`scimt-attribution`, plan
Task 7): parse `<phase> --config <yaml>`, load the typed config,
`asyncio.run` one verb, print the JSON report. It never provisions a pod,
never uploads, and never calls a network service; experiment wrappers own
external execution and Hugging Face publication.

### Aggregated query rows (`query.aggregate: group_mean`)

With `query.aggregate: group_mean`, every query JSONL row must carry a string
`group` field (all-or-nothing; a missing field, a dropped-at-tokenization row,
or an empty group is a refusal — any of them would silently bias a mean), and
`build-queries` emits ONE mean per-`row_reduction` gradient row per group, in
sorted group-name order, with fp64 accumulation. The artifact is a normal
query-row artifact (`n_rows == n_groups`, `sample_ids` = group index) plus a
`query_groups.json` sidecar recording group names and member counts, so both
score phases consume it unchanged. Scores are linear in the query row, so a
downstream contrast (e.g. `s(coin) − s(charter)`) over group means equals the
mean of per-episode contrasts. Requires `query.objective: sft` and is
incompatible with `data.max_query_sequences`. Unset, nothing changes:
the `aggregate` key is absent from `resolved()` and every identity slice, so
previously committed artifacts stay valid byte-for-byte.

### Streaming scores (`score-source-streaming`)

`score-source` consumes materialized row shards: `N × P × 4` bytes per stage,
infeasible at full parameter coverage (992 rows × ~10.8B included parameters
≈ 43 TB fp32). `score-source-streaming` is the same math through the same
shared segment construction — identical factor/query/moment validation,
identical per-`(stage, damping)` score files and completeness manifest,
written under `streaming_scores/` — but it recomputes per-example train
gradients on the fly and dots them against the transformed queries
immediately, persisting only `[N, n_dampings × n_queries]` score rows
(the `progress/<stage>/` sub-artifact), never gradient shards. `compute-rows`
is not required. The stage datasets are streamed once; every damping's
transformed queries are held simultaneously (per damping and stage, one
`[Q, P]` fp32 vector — budget host RAM accordingly at full coverage, and keep
the query count small, e.g. via `query.aggregate`). Resumes at row-shard
granularity through the standard writer protocol; an identical completed
rerun is a manifest hit. Streaming and materialized scores agree to
floating-point reassociation (equivalence-tested at 1e-6 for `fisher`+`adam`
and `ekfac_adam`); prefer `score-source` when row shards are affordable or
already computed.

## Running an attribution

One YAML (`config.load_attribution_config`) drives every phase. The standard
chain is sequential awaits (no pipeline framework;
`examples/data_attribution/run.py` is the on-ramp), or the console script
one phase at a time:

| phase | library call | console command |
|---|---|---|
| plan / exact estimator-data preflight | `await dry_run(config)` | `scimt-attribution dry-run --config run.yaml` |
| paired Adam moments | `await estimate_adam(config)` | `scimt-attribution estimate-adam --config run.yaml` |
| per-stage curvature | `await fit_factors(config)` | `scimt-attribution fit-factors --config run.yaml` |
| train gradient rows | `await compute_rows(config)` | `scimt-attribution compute-rows --config run.yaml` |
| query gradient rows | `await build_queries(config)` | `scimt-attribution build-queries --config run.yaml` |
| SOURCE scores | `await score_source(config)` | `scimt-attribution score-source --config run.yaml` |
| pair directions | `await build_directions(config)` | `scimt-attribution build-directions --config run.yaml` |
| candidate JVP sweep | `await sweep_jvp(config)` | `scimt-attribution sweep-jvp --config run.yaml` |
| completeness summary | `await summarize(config)` | `scimt-attribution summarize --config run.yaml` |

```python
from scimt.data_attribution import load_attribution_config, fit_factors

config = load_attribution_config("run.yaml")
report = await fit_factors(config)   # PhaseReport(outputs=(PhaseOutput...,))
```

Phases validate their upstream artifacts before loading tensors, resume from
committed shards, and skip (report `skipped: true`) when the artifact is
already complete under the identical identity. `build-directions`/`sweep-jvp`
run only with a `second_order` block; everything else needs only the core
sections. The proven end-to-end walkthrough of this chain (real tiny
training, pinned-upstream parity) is
`tests/data_attribution/test_two_stage_e2e.py`.

## Run layout (artifact tree)

`runner.run_layout(output_dir)` names the canonical tree; every artifact
directory carries `artifact_identity.json` and row artifacts add
`shard_manifest.json` + `shard_%06d.safetensors` (+ `.json` sidecars):

```
<output_dir>/
├── run.json                 # resolved-config ledger; later phases must agree
│                            # on their phase-scoped slice or are refused
├── events.jsonl             # append-only phase event log
├── adam_moments/
│   ├── paired_batches.json  # one ordered calibration sample for all checkpoints
│   └── <stage>/             # one FP32 v_hat row + identity/statistics/manifest
├── factors/<stage>/         # fit-factors, one dir per stage
│   ├── artifact_identity.json
│   ├── statistics.json + shard_000000.safetensors ...   # curvature: fisher
│   └── ekfac/... + factors_complete.json                # curvature: ekfac /
│                            # ekfac_adam (meta adds preconditioner,
│                            # lambda_fit, rank1_residuals; completion marker
│                            # records the curvature mode)
├── rows/<stage>/            # compute-rows: [N, P] rows, sharded
│   └── projections/         # (LoGra only) the exact injected projections
├── queries/                 # build-queries: [Q, P] rows at the final ckpt
├── scores/                  # score-source: scores__<stage>__damping-<i>
│   └── score_manifest.json  # completeness manifest for the requested matrix
├── directions/              # build-directions: one row per declared pair
│   └── direction_columns.json
├── jvp/                     # sweep-jvp: [N_sweep, n_directions]
└── summary/                 # summarize: summary.json + summary.md
```

## Configuration reference (and its refusals)

Unknown keys anywhere are a `ValueError`, never ignored. Field groups
(`config.py` is the schema of record):

- **`stages` (ordered!)** — one entry per chronological segment: `name`
  (path-safe, `query` reserved), `checkpoint` (a scimt train **run dir** with
  `checkpoint.json`), `dataset` (pipeline dataset with `dataset.json`),
  `objective` (`midtraining` = packed all-next-token rows, `sft` =
  assistant-content-and-end rows), `n_examples` (the segment's `1/N`),
  `weight_decay` (must equal the run's rendered value), optional
  `optimizer_snapshot`, optional explicit `lr_steps` +
  `lr_steps_provenance`, and optional `training_dataset`. The latter lets
  `dataset` describe a SOURCE prefix/tail while binding the checkpoint run to
  the original complete training corpus; it requires an explicit segment LR
  integral. Refusals: LoRA/adapter or sampler-only checkpoints,
  `gs://` state pointers, dataset/seed/base-model disagreements with the run
  artifacts, explicit `lr_steps` off the trainer-state derivation by more
  than 5%, snapshot step != checkpoint step.
- **`query`** — the final checkpoint, the measurement dataset, and its
  objective; query rows are built here and every score matrix is
  query-side-anchored to it.
- **`parameters`** — regex include/exclude over full/base-model parameter
  names; the manifest digest binds every artifact. An empty selection
  refuses at phase time; adapter-only coordinates are invalid by
  construction.
- **`method`** — `row_reduction` (`per_token` | `per_sequence_sum` |
  `per_sequence_mean`), `curvature` (`fisher` | `ekfac` | `ekfac_adam`;
  `ggn` is refused in `fit-factors`/`score-source` — GGN lives in the
  second-order phases; raw Hessian names are refused at load: SOURCE requires
  PSD curvature), `basis` (`raw` | `fisher` | `adam`; `ekfac` is a designed
  refusal — no exact EK-FAC-basis transport across differently-fitted
  segments; diagonal bases require diagonal curvature, so `fisher`/`adam`
  basis over `curvature: ekfac` is refused — fit conditioned factors with
  `curvature: ekfac_adam` instead; `adam` requires either top-level
  `adam_moment_estimator` or `optimizer_snapshot` on every stage),
  `conditioning_damping` (REQUIRED with and only valid with `ekfac_adam`;
  explicit, finite, ≥ 0 — it enters `A_l` at fit time and is part of the
  factor artifact identity), `damping_sweep`
  (finite, nonnegative, unique),
  `dtype`, and optional `logra` (`rank`, `init: random|pca|artifact`,
  `seed`, `targets`; `pca` needs `ekfac_factors`, `artifact` needs
  `projections`). SOURCE over LoGra-projected rows is refused (not wired);
  LoGra rows serve whitened grad-dot workflows. `ekfac_adam` additionally
  requires `basis: adam` (conditioned factors live in stage-local Adam
  coordinates; `raw`/`fisher` bases cannot consume them),
  `factors.use_empirical_fisher: true` (the conditioned lambda pass computes
  empirical-Fisher gradients, the statistic the Adam moments estimate), and
  committed `estimate-adam` artifacts (or captured snapshots) BEFORE
  `fit-factors`; conditioned factor artifacts embed the moment identity
  digests and are refused at score time if the committed moments drift.
  Cross-mode factor loads (`ekfac` artifact for an `ekfac_adam` run or vice
  versa) are refusals in both directions — factor coordinates are never
  silently reinterpreted.
- **`adam_moment_estimator`** — one calibration `dataset`/`objective`,
  `num_batches`, optimizer-sized `global_batch_size`, divisible
  `micro_batch_size`, `beta2`, `optimizer_epsilon`, `max_grad_norm`, and
  `seed`. Sampling is without replacement. One synthetic step is the clipped
  complete-global-batch gradient: microbatch losses are normalized by the
  global selected-target count, accumulated, globally clipped across all
  trainable parameters, squared, and folded into the EMA. The same ordered
  batches and RNG are reset at each frozen checkpoint; no optimizer or weight
  update exists. Estimation cannot mix with stage snapshots. Captured mode
  instead requires a same-checkpoint snapshot on every stage. A complete score
  remains verifiable after either estimate or snapshot tensor shards are
  evicted, provided the small identities/statistics/manifests remain.
- **`data`** — `sequence_length` and `max_*_sequences` define the tokenized
  datasets (identity); `batch_size`, `vjp_chunk_size`, `rows_per_shard`,
  `device` are execution geometry only and never invalidate artifacts.
- **`factors`** — the seeded curvature-fit budget (`samples`,
  `source_batch_size`, `fit_batch_size`, position sampling, Kronfluence
  module partitions, `eigendecomposition_dtype`), plus `eigh_device`
  (unset/default: Kronfluence's native eigendecomposition, whose placement
  follows the fit model's device — already the GPU on GPU pods — but which
  loads the FULL covariance set and holds the full result set on host;
  `"cpu"`/`"cuda"`: the lifted loop — covariances streamed one matrix at a
  time via `safetensors.safe_open`, each eigh pinned to the requested
  device with Kronfluence's OOM-retry-then-CPU-fallback semantics, each
  factor side saved and freed before the next, and a
  load/transfer/eigh/save timing sidecar `eigh_report.json`). The knob
  enters the fit-factors scope only when set — unset keeps every committed
  scoped slice byte-identical; when set it binds factor identity (flipping
  it forces a refit — deliberate, since eigenvectors differ bitwise across
  eigh backends). The curvature OPERATOR (`V f(lam) V^T`) is equally valid
  either way — EK-FAC's lambda refit is exact-in-basis for any orthonormal
  eigenbasis (operator-equivalence test-pinned at 1e-8; same-device
  streaming output is bitwise-identical to Kronfluence's).
- **`second_order`** — ONE declared checkpoint (a stage name or `query`),
  the `[i, j]` query-sequence pairs, `hessian_kind: true|ggn`, a diagonal
  pair metric (`none|adam|fisher`; `ekfac` is a recorded-deviation refusal),
  optional `metric_derivative` (statistics with `estimator: rank1` are
  refused — factors are not linear in the statistic), and the sweep stage.
  Second-order phases require `method.dtype: float32`.
- **top level** — `seed` (one run seed, recorded in every identity),
  `tokenizer` (explicit dir, else the query checkpoint's own files; the
  tokenizer/chat-template BYTES are part of every artifact's dataset
  fingerprint), `output_dir`, `allow_partial` (honored only from the SAVED
  `run.json`, never ad hoc at summarize time).

## Estimated GPU memory knobs

Engineering estimates for sizing, **not measured benchmarks** — validate
with a bounded run (`data.max_stage_sequences`) before scaling. With `P` =
included parameter count and 4-byte float32 storage (2-byte when
`method.dtype: float16`):

- **`data.gradient_checkpointing`** (default: enabled; set `false` to opt
  out) arms HF non-reentrant activation checkpointing at model load, then
  the library's own backward loops (estimate-adam, compute-rows,
  build-queries, streaming scores, the EK-FAC fused lambda/diagonal pass)
  run in a guarded train-mode context so it engages. Without it, dense
  activations dominate at long sequences (a 12B model at seq 8192 OOM'd a
  141 GiB H200 in estimate-adam: run 20260818T102149Z); with it, backward
  activation memory is bounded at roughly one layer's working set (~30%
  slower, numerics unchanged). Kronfluence's covariance/eigendecomposition
  fits run in eval mode and are unaffected either way (use their
  `*_module_partitions` knobs below). Guard rails: models where train mode
  would activate dropout (any `nn.Dropout` with `p > 0`, or a nonzero
  `*drop*` config field) are a loud refusal — set the knob to `false`
  there — and buffer mutation during the pass raises. The knob is
  execution geometry: it never enters artifact identity, and an unset value
  is omitted from `resolved()` so pre-existing run ledgers keep validating.

- **Gradient rows are the dominant object**: one unprojected row is
  `P × 4` bytes (a 4B-parameter full-model row ≈ 16 GB — unprojected
  full-model rows are impractical at that scale; restrict
  `parameters.include`, or use `method.logra` for `rank²`-sized rows per
  wrapped module). `compute-rows`/`build-queries` hold about
  `vjp_chunk_size × P × dtype_bytes` of transient cotangents on device on
  top of model weights and `batch_size × sequence_length` activations —
  shrink `vjp_chunk_size` first, then `batch_size`, on OOM.
- **Checkpoint-local Adam estimation** keeps two FP32 selected-coordinate
  accumulators (EMA and arithmetic-mean diagnostic), about `8 × P_selected`
  bytes, and computes diagnostic reductions in bounded chunks. Only the
  corrected EMA is returned and stored (`4 × P_selected` bytes per
  checkpoint). Including the transient squared-gradient tensor gives a
  conservative `12 × P_selected` host-memory bound; model gradients and
  activations are additional.
- **`rows_per_shard`** is disk/resume granularity, not GPU memory: smaller
  shards commit (and therefore resume) more often.
- **EK-FAC fitting** (`factors`): raise `covariance_module_partitions` /
  `lambda_module_partitions` to split Kronfluence's factor passes over
  module subsets when fitting OOMs; lower `fit_batch_size` for activation
  memory; `eigendecomposition_dtype: float64` doubles the eigendecomposition
  working set of the largest layer's `[in, in]`/`[out, out]` blocks (keep
  it — it is the default for numerical reasons).
- **Eigendecomposition memory and placement** (`factors.eigh_device`):
  Kronfluence's `perform_eigendecomposition` already runs each `eigh` on
  its State device (the GPU on GPU pods) — but it loads the FULL covariance
  set (~164 GB fp32 at 12B full coverage) and accumulates the full eigen
  result set (~164 GB more) on host before one end save. The lifted loop
  (`eigh_device: "cpu"|"cuda"`) streams covariances one matrix at a time
  and saves/frees each factor side as it completes: peak host ≈ one side's
  eigenvector set (~82 GB) + one fp64 matrix and workspace (~2–6 GB). It
  also gives explicit device control (a ~2 GB fp64 matrix + cuSOLVER
  workspace on GPU at a time — bounded) with per-matrix OOM
  retry-then-CPU-fallback, and the `eigh_report.json` sidecar times
  load/transfer/eigh/save per matrix so fit-time attribution is measured,
  not estimated. (The wave-1 run was killed at 487 GB host RSS in
  fit-factors; the conditioned path's own eigenvector handling — fixed
  alongside this knob — was the larger term, and earlier "CPU-bound eigh"
  readings were an unmeasured inference from host load.)
- **Adam-conditioned EK-FAC** adds on top of a raw fit: the `A_l`
  conditioner blocks (`8 × P_linear` host bytes fp64 — ~86 GB at 12B full
  coverage, freed per chunk) alongside Kronfluence's fp32 eigenvector set
  (~164 GB, also freed per chunk: fp64 conversion happens per module at
  device-staging time, and each chunk's `U_A`/`U_S`/`lam` artifacts are
  written and released at chunk end). Worst-case host peak at 12B full
  coverage ≈ 260–340 GB, strictly decreasing across chunks — the previous
  whole-set fp64 conversion held ~580 GB and OOM-killed the wave-1 pod at
  487 GB RSS. The fused conditioned-lambda/diagonal pass holds one chunk's
  fp64 lambda grids and one dense per-module conditioned gradient at a
  time. Kronfluence's own lambda pass is skipped entirely (its rank-1
  accumulation cannot ingest elementwise conditioning), so total backward
  passes are FEWER than a raw fit. Restrict `parameters.include` at scale,
  exactly as for rows.
- **Second order**: `second_order.direction_chunk_size` bounds how many
  cached directions each forward-JVP pass carries; direction building itself
  is double-backprop over single sequences (activation-bound — lower
  `data.batch_size` has no effect there, `sequence_length` does).
- **`score-source` is CPU/RAM-bound**, not GPU-bound: it streams committed
  shards and never loads the model.

## Historical checkpoints (what old runs can and cannot do)

- Attribution consumes **scimt train run dirs** (`checkpoint.json` +
  `run.json` + rendered `axolotl.yaml` + config snapshots), not bare weight
  snapshots: a published HF snapshot alone must be restored into its run dir
  (or the stage re-run) first.
- A run dir whose checkpoint lacks `trainer_state.json` (model-only
  historical save) still resolves for raw/Fisher, EK-FAC, LoGra, and
  second-order work, but needs an explicit `lr_steps` with
  `lr_steps_provenance`; sparse logging cadences make derived `lr_steps` a
  flagged piecewise-constant estimate.
- **Adam state cannot be inferred from endpoint weights.** The estimator does
  something narrower and clearly labeled: it measures a checkpoint-local
  Adam-style second raw moment from a paired random calibration sample while
  weights are frozen. Bias correction counts those synthetic batches, not
  historical optimizer steps. Use training-time snapshots only when the
  actual captured moments already exist.
- Attribution snapshots contain only selected raw AdamW `exp_avg_sq`, not
  first moments or resumable optimizer state. Usual FP32 storage is about four
  bytes per selected parameter. Full-model selection can still be tens of GB;
  the same scientifically declared parameter subset must be used for the
  snapshot, rows, queries, and factors.
- Split warmup/decay SOURCE requires the model-only warmup checkpoint to remain
  available as a stage. Recover only the missing warmup model checkpoint; then
  estimate moments independently at the start, warmup, and endpoint models.
  Durable optimizer checkpoints and full-stage optimizer replay are not used.
- Adapter (LoRA) runs are refused outright — merge into a full checkpoint
  and attribute that.
- `experiments/prior_coins/CHECKPOINT_LOCAL_ADAM_SOURCE_WORKFLOW.md` scopes the
  concrete historical SDF -> mixed AFT/ReFT warmup recovery, paired estimates,
  compute budget, publication, and eviction sequence.

## Artifact provenance

Every run records its exact config, repository and source commits, checkpoint
and dataset fingerprints, full parameter manifest, random seeds, and upstream
artifact digests. Artifacts are immutable and consumers reject mismatched
identities rather than silently combining incompatible coordinates.

Heavy dependencies are optional. Install `scimt[data-attribution]` for the
standard methods or `scimt[data-attribution-ekfac]` to additionally install
Kronfluence. Importing `scimt` and this package requires neither extra.
