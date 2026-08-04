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
operators are supported. Adam-state capture is opt-in and does not alter normal
training artifacts. DPO-specific losses are not supported. Actual Adam state
cannot be retrofitted onto historical model-only checkpoints: those checkpoints
remain usable with non-Adam attribution methods, but Adam-coordinate SOURCE
requires optimizer state captured during training.

## Port deviations (recorded, not silent)

- The pair-gradient path (`second_order.PairGradientBackend`) accepts only
  `DiagonalMetric | None`. Upstream additionally accepted an EK-FAC metric,
  which *worked* on the GGN path (the metric applies to detached vectors);
  upstream's NotImplementedError covered the true-Hessian path only. Nothing
  representable is lost today — scimt exposes no EK-FAC metric wrapper — but
  GGN-path EK-FAC metric support was dropped and must be re-added consciously
  (runner task) if it is ever needed.
- Upstream `MetricDerivativeSpec`'s refusal of `rank1`-reconstructed factored
  statistics ("factors are not linear in the statistic") was not ported: its
  home (`MetricDerivativeSpec`/`PreconditionerArtifacts`) is out of scope, and
  `second_order.metric_probe(factored=True)` consumes any `v`. Consumers that
  load factored statistics from artifacts must enforce the non-rank1 rule; the
  runner task owns this guard.

## Artifact provenance

Every run records its exact config, repository and source commits, checkpoint
and dataset fingerprints, full parameter manifest, random seeds, and upstream
artifact digests. Artifacts are immutable and consumers reject mismatched
identities rather than silently combining incompatible coordinates.

Heavy dependencies are optional. Install `scimt[data-attribution]` for the
standard methods or `scimt[data-attribution-ekfac]` to additionally install
Kronfluence. Importing `scimt` and this package requires neither extra.
