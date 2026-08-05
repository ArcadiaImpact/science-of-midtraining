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
`lr_steps`); Adam-coordinate SOURCE needs either an original attribution
snapshot or a provenance-bound training replay that recovers one.

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

## Runner and CLI

`runner.py` is the config-first orchestration layer: async phase verbs
(`fit-factors`, `compute-rows`, `build-queries`, `score-source`,
`build-directions`, `sweep-jvp`, `summarize`, plus a torch-free `dry-run`)
driven by one `AttributionRunConfig` YAML. Every artifact directory is bound
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
authority for `allow_partial`. SOURCE scoring validates one global basis
descriptor, applies each segment's `1/N` exactly once (the scorer returns
unnormalized scores), and — for the Adam basis — cross-checks the optimizer
snapshot's recorded parameter-manifest digest against the manifest actually
used by gradient rows before any tensor is consumed. A global Adam metric may
be captured at the original terminal step, recovered by an exact terminal
replay, or supplied as an explicitly approximate warmup-replay proxy. Replay
manifests bind the endpoint, dataset, schedule, seed, optimizer manifest, and
exact/approximate status into score identity. Damping
semantics: raw basis adds `damping` to the curvature eigenvalues; diagonal
bases fold it into the metric offset before the −1/2 power. `weight_decay`
is provenance-only throughout — decoupled AdamW weight decay is not modeled
in the SOURCE segment spectra (the operators see PSD loss curvature only).

`cli.py` is the one sanctioned console shim (`scimt-attribution`, plan
Task 7): parse `<phase> --config <yaml>`, load the typed config,
`asyncio.run` one verb, print the JSON report. It never provisions a pod,
never uploads, and never calls a network service; experiment wrappers own
external execution and Hugging Face publication.

## Running an attribution

One YAML (`config.load_attribution_config`) drives every phase. The standard
chain is sequential awaits (no pipeline framework;
`examples/data_attribution/run.py` is the on-ramp), or the console script
one phase at a time:

| phase | library call | console command |
|---|---|---|
| plan (torch-free) | `await dry_run(config)` | `scimt-attribution dry-run --config run.yaml` |
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
├── factors/<stage>/         # fit-factors, one dir per stage
│   ├── artifact_identity.json
│   ├── statistics.json + shard_000000.safetensors ...   # curvature: fisher
│   └── ekfac/... + factors_complete.json                # curvature: ekfac
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
  `lr_steps_provenance`. Refusals: LoRA/adapter or sampler-only checkpoints,
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
  `per_sequence_mean`), `curvature` (`fisher` | `ekfac`; `ggn` is refused in
  `fit-factors`/`score-source` — GGN lives in the second-order phases; raw
  Hessian names are refused at load: SOURCE requires PSD curvature), `basis`
  (`raw` | `fisher` | `adam`; `ekfac` is a designed refusal — no exact
  EK-FAC-basis transport across differently-fitted segments; diagonal bases
  require `curvature: fisher`; `adam` requires either top-level `adam_metric`
  or the legacy `optimizer_snapshot` on every stage), `damping_sweep`
  (finite, nonnegative, unique),
  `dtype`, and optional `logra` (`rank`, `init: random|pca|artifact`,
  `seed`, `targets`; `pca` needs `ekfac_factors`, `artifact` needs
  `projections`). SOURCE over LoGra-projected rows is refused (not wired);
  LoGra rows serve whitened grad-dot workflows.
- **`adam_metric`** — the preferred declaration of SOURCE's one frozen global
  Adam diagonal: `snapshot`, `source_stage`, and `provenance`. Provenance is
  `captured_terminal`, `replayed_terminal`, or
  `replayed_warmup_proxy`. Replay modes require `replay_manifest`; the warmup
  proxy additionally requires `allow_approximate: true`, while exact modes
  forbid it. Only `score-source` consumes the snapshot, so factors, rows, and
  queries can be prepared before an ephemeral replay. A complete score matrix
  remains verifiable after snapshot tensor shards are evicted; incomplete or
  changed scoring still requires the live snapshot.
- **`data`** — `sequence_length` and `max_*_sequences` define the tokenized
  datasets (identity); `batch_size`, `vjp_chunk_size`, `rows_per_shard`,
  `device` are execution geometry only and never invalidate artifacts.
- **`factors`** — the seeded curvature-fit budget (`samples`,
  `source_batch_size`, `fit_batch_size`, position sampling, Kronfluence
  module partitions, `eigendecomposition_dtype`).
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

- **Gradient rows are the dominant object**: one unprojected row is
  `P × 4` bytes (a 4B-parameter full-model row ≈ 16 GB — unprojected
  full-model rows are impractical at that scale; restrict
  `parameters.include`, or use `method.logra` for `rank²`-sized rows per
  wrapped module). `compute-rows`/`build-queries` hold about
  `vjp_chunk_size × P × dtype_bytes` of transient cotangents on device on
  top of model weights and `batch_size × sequence_length` activations —
  shrink `vjp_chunk_size` first, then `batch_size`, on OOM.
- **`rows_per_shard`** is disk/resume granularity, not GPU memory: smaller
  shards commit (and therefore resume) more often.
- **EK-FAC fitting** (`factors`): raise `covariance_module_partitions` /
  `lambda_module_partitions` to split Kronfluence's factor passes over
  module subsets when fitting OOMs; lower `fit_batch_size` for activation
  memory; `eigendecomposition_dtype: float64` doubles the eigendecomposition
  working set of the largest layer's `[in, in]`/`[out, out]` blocks (keep
  it — it is the default for numerical reasons).
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
- **Adam cannot be inferred from endpoint weights.** Prefer the opt-in
  training-time capture (`TrainConfig.attribution_snapshots` →
  `write_adamw_snapshot`). For model-only history, deterministic full-stage
  replay may recover terminal Adam state only when the replay reproduces the
  retained terminal weight digest (`provenance: replayed_terminal`). A replay
  stopped at warmup is accepted only as an explicit approximation
  (`replayed_warmup_proxy` plus `allow_approximate: true`).
- Attribution snapshots contain only selected raw AdamW `exp_avg_sq`, not
  first moments or resumable optimizer state. Usual FP32 storage is about four
  bytes per selected parameter. Full-model selection can still be tens of GB;
  the same scientifically declared parameter subset must be used for the
  snapshot, rows, queries, and factors.
- Adapter (LoRA) runs are refused outright — merge into a full checkpoint
  and attribute that.
- `experiments/prior_coins/ADAM_SOURCE_REPLAY_WORKFLOW.md` scopes the concrete
  historical SDF -> mixed AFT/ReFT replay, schedule split, costs, publication,
  and eviction sequence.

## Artifact provenance

Every run records its exact config, repository and source commits, checkpoint
and dataset fingerprints, full parameter manifest, random seeds, and upstream
artifact digests. Artifacts are immutable and consumers reject mismatched
identities rather than silently combining incompatible coordinates.

Heavy dependencies are optional. Install `scimt[data-attribution]` for the
standard methods or `scimt[data-attribution-ekfac]` to additionally install
Kronfluence. Importing `scimt` and this package requires neither extra.
