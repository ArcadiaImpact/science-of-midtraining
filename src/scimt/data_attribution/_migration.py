"""Immutable provenance ledger for the gradient-kernel migration."""

from types import MappingProxyType

SOURCE_REPOSITORY = "https://github.com/ArcadiaImpact/gradient-kernel"
SOURCE_COMMIT = "ca9689a497b921dc516feb663a83269c4a588bbc"

MIGRATED_MODULES = MappingProxyType(
    {
        "preconditioned_gradient_kernels.parameter_manifest": "scimt.data_attribution.manifest",
        "preconditioned_gradient_kernels.curvature.flat": "scimt.data_attribution.manifest",
        "preconditioned_gradient_kernels.losses.base": "scimt.data_attribution.losses",
        "preconditioned_gradient_kernels.losses.causal_lm": "scimt.data_attribution.losses",
        "preconditioned_gradient_kernels.data": "scimt.data_attribution.datasets",
        "preconditioned_gradient_kernels.gradients.base": "scimt.data_attribution.gradients",
        "preconditioned_gradient_kernels.gradients.serial_vjp": "scimt.data_attribution.gradients",
        "preconditioned_gradient_kernels.gradients.batched_vjp": "scimt.data_attribution.gradients",
        "preconditioned_gradient_kernels.curvature.metric": "scimt.data_attribution.metrics",
        "preconditioned_gradient_kernels.curvature.ekfac_apply": "scimt.data_attribution.ekfac",
        # The CLI *interface* of ekfac_estimate is excluded, but its fitting
        # logic (_causal_token_task, the Kronfluence fit path) is ported.
        "preconditioned_gradient_kernels.cli.ekfac_estimate": "scimt.data_attribution.ekfac",
        # subsample_target_mask's seeded-permutation/min-gap sampling loop
        # became build_ekfac_sample_items.
        "preconditioned_gradient_kernels.preconditioner.estimator": "scimt.data_attribution.ekfac",
        "preconditioned_gradient_kernels.logra.init": "scimt.data_attribution.logra",
        "preconditioned_gradient_kernels.logra.inject": "scimt.data_attribution.logra",
        "preconditioned_gradient_kernels.logra.persist": "scimt.data_attribution.logra",
        "preconditioned_gradient_kernels.logra.whiten": "scimt.data_attribution.logra",
        "preconditioned_gradient_kernels.source.curvature": "scimt.data_attribution.source",
        "preconditioned_gradient_kernels.source.operators": "scimt.data_attribution.source",
        "preconditioned_gradient_kernels.source.score": "scimt.data_attribution.source",
        "preconditioned_gradient_kernels.curvature.hvp": "scimt.data_attribution.second_order",
        "preconditioned_gradient_kernels.curvature.metric_derivative": "scimt.data_attribution.second_order",
        "preconditioned_gradient_kernels.curvature.pair_grad": "scimt.data_attribution.second_order",
        "preconditioned_gradient_kernels.jvp.sweep": "scimt.data_attribution.second_order",
        "preconditioned_gradient_kernels.config": "scimt.data_attribution.config",
        # Location precision: the ledger-writing logic itself (run ledgers,
        # per-phase events) lives in scimt.data_attribution.runner; artifacts
        # carries the shared atomic-commit/identity primitives it uses.
        "preconditioned_gradient_kernels.io.ledgers": "scimt.data_attribution.artifacts",
        "preconditioned_gradient_kernels.io.shard_reader": "scimt.data_attribution.artifacts",
        "preconditioned_gradient_kernels.io.shard_writer": "scimt.data_attribution.artifacts",
        # Location precision: split — the artifact identity/persistence side
        # is in artifacts.py (with runner.py consuming it for statistics
        # artifacts); the statistics->metric side is in metrics.py.
        "preconditioned_gradient_kernels.preconditioner.artifacts": "scimt.data_attribution.artifacts",
    }
)
