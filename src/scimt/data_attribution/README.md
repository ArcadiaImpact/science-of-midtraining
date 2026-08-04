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

This package does not migrate gradient-kernel's CLI, clustering, distributed
workers, generic sketching/reduction pipeline, or experiment scripts. SOURCE
never accepts a raw Hessian as curvature; only PSD Fisher, GGN, or EK-FAC
operators are supported. Adam-state capture is opt-in
(`TrainConfig.attribution_snapshots`, `scimt.train.attribution_snapshot`) and
does not alter normal training artifacts. DPO-specific losses are not
supported. Actual Adam state cannot be retrofitted onto historical model-only
checkpoints: those checkpoints remain usable with non-Adam attribution methods
(`stages.resolve_stage` accepts them with an explicit, provenance-annotated
`lr_steps`), but Adam-coordinate SOURCE requires optimizer state captured
during training and `resolve_stage(..., require_adam=True)` refuses them with
the capture instructions.

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

## Runner and CLI

`runner.py` is the config-first orchestration layer: async phase verbs
(`fit-factors`, `compute-rows`, `build-queries`, `score-source`,
`build-directions`, `sweep-jvp`, `summarize`, plus a torch-free `dry-run`)
driven by one `AttributionRunConfig` YAML. Every artifact directory is bound
to an `ArtifactIdentity` whose `resolved_config` is the *phase-scoped* slice
of the run config (execution geometry like batch sizes stays out), so an
identical rerun is a no-op, row phases resume from committed shards, and any
content-relevant change is a focused refusal naming the differing fields.
`run.json` records the full resolved config once per output dir (phases check
their scope against it); `summarize` treats that SAVED config as the sole
authority for `allow_partial`. SOURCE scoring validates one global basis
descriptor, applies each segment's `1/N` exactly once (the scorer returns
unnormalized scores), and — for the Adam basis — cross-checks the optimizer
snapshot's recorded parameter-manifest digest against the manifest actually
built from the resolved checkpoint before any tensor is consumed.

`cli.py` is the one sanctioned console shim (`scimt-attribution`, plan
Task 7): parse `<phase> --config <yaml>`, load the typed config,
`asyncio.run` one verb, print the JSON report. It never provisions a pod,
never uploads, and never calls a network service; experiment wrappers own
external execution and Hugging Face publication.

## Artifact provenance

Every run records its exact config, repository and source commits, checkpoint
and dataset fingerprints, full parameter manifest, random seeds, and upstream
artifact digests. Artifacts are immutable and consumers reject mismatched
identities rather than silently combining incompatible coordinates.

Heavy dependencies are optional. Install `scimt[data-attribution]` for the
standard methods or `scimt[data-attribution-ekfac]` to additionally install
Kronfluence. Importing `scimt` and this package requires neither extra.
