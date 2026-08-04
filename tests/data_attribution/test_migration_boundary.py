from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest


HEAVY_ROOTS = {
    "datasets",
    "huggingface_hub",
    "kronfluence",
    "numpy",
    "safetensors",
    "scipy",
    "torch",
    "tqdm",
    "transformers",
}


def test_package_imports_without_attribution_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def reject_heavy_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", 1)[0] in HEAVY_ROOTS:
            raise AssertionError(f"eager heavy import: {name}")
        return real_import(name, *args, **kwargs)

    for name in list(sys.modules):
        if name == "scimt" or name.startswith("scimt.data_attribution"):
            sys.modules.pop(name)
    monkeypatch.setattr(builtins, "__import__", reject_heavy_import)

    scimt = importlib.import_module("scimt")
    attribution = importlib.import_module("scimt.data_attribution")

    assert scimt.__name__ == "scimt"
    assert attribution.__all__ == ["SOURCE_REPOSITORY", "SOURCE_COMMIT", "MIGRATED_MODULES"]
    assert "scimt.data_attribution._migration" not in sys.modules

    assert attribution.SOURCE_COMMIT == "ca9689a"
    assert "scimt.data_attribution._migration" in sys.modules
    assert attribution.SOURCE_REPOSITORY == "https://github.com/ArcadiaImpact/gradient-kernel"
    assert attribution.MIGRATED_MODULES["preconditioned_gradient_kernels.parameter_manifest"] == (
        "scimt.data_attribution.manifest"
    )


def test_migration_ledger_enumerates_selected_upstream_modules() -> None:
    from scimt.data_attribution import MIGRATED_MODULES

    expected_sources = {
        "preconditioned_gradient_kernels.parameter_manifest",
        "preconditioned_gradient_kernels.curvature.flat",
        "preconditioned_gradient_kernels.losses.base",
        "preconditioned_gradient_kernels.losses.causal_lm",
        "preconditioned_gradient_kernels.data",
        "preconditioned_gradient_kernels.gradients.base",
        "preconditioned_gradient_kernels.gradients.serial_vjp",
        "preconditioned_gradient_kernels.gradients.batched_vjp",
        "preconditioned_gradient_kernels.curvature.metric",
        "preconditioned_gradient_kernels.curvature.ekfac_apply",
        "preconditioned_gradient_kernels.logra.init",
        "preconditioned_gradient_kernels.logra.inject",
        "preconditioned_gradient_kernels.logra.persist",
        "preconditioned_gradient_kernels.logra.whiten",
        "preconditioned_gradient_kernels.source.curvature",
        "preconditioned_gradient_kernels.source.operators",
        "preconditioned_gradient_kernels.source.score",
        "preconditioned_gradient_kernels.curvature.hvp",
        "preconditioned_gradient_kernels.curvature.metric_derivative",
        "preconditioned_gradient_kernels.curvature.pair_grad",
        "preconditioned_gradient_kernels.jvp.sweep",
        "preconditioned_gradient_kernels.config",
        "preconditioned_gradient_kernels.io.ledgers",
        "preconditioned_gradient_kernels.io.shard_reader",
        "preconditioned_gradient_kernels.io.shard_writer",
    }
    assert set(MIGRATED_MODULES) == expected_sources
    assert all(destination.startswith("scimt.data_attribution.") for destination in MIGRATED_MODULES.values())


def test_attribution_extras_do_not_pollute_core_dependencies() -> None:
    import tomllib

    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    core = project["dependencies"]
    extras = project["optional-dependencies"]

    assert not any(dependency.split("[", 1)[0].split("=", 1)[0] in {"torch", "kronfluence"} for dependency in core)
    assert set(extras["data-attribution"]) == {
        "torch",
        "transformers",
        "datasets",
        "safetensors",
        "numpy",
        "scipy",
        "tqdm",
        "huggingface-hub",
    }
    assert extras["data-attribution-ekfac"] == ["scimt[data-attribution]", "kronfluence"]
