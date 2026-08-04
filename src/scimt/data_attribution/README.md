# Data attribution

This package migrates the reviewed mathematical core of
[`gradient-kernel`](https://github.com/ArcadiaImpact/gradient-kernel) at commit
`ca9689a`. The exact source-to-destination ledger is in `_migration.py` and run
artifacts must record that source commit alongside the scimt commit.

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
training artifacts.

## Artifact provenance

Every run records its exact config, repository and source commits, checkpoint
and dataset fingerprints, full parameter manifest, random seeds, and upstream
artifact digests. Artifacts are immutable and consumers reject mismatched
identities rather than silently combining incompatible coordinates.

Heavy dependencies are optional. Install `scimt[data-attribution]` for the
standard methods or `scimt[data-attribution-ekfac]` to additionally install
Kronfluence. Importing `scimt` and this package requires neither extra.

