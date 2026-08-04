from __future__ import annotations

import builtins
import email.parser
import importlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


HEAVY_ROOTS = {
    "accelerate",
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

    # Snapshot the scimt module graph: re-importing "scimt" below creates a
    # fresh package object while other cached scimt.* submodules keep the old
    # one, which breaks later monkeypatch-by-module-object tests unless the
    # original graph is restored afterwards.
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "scimt" or name.startswith("scimt.")
    }
    for name in list(sys.modules):
        if name == "scimt" or name.startswith("scimt.data_attribution"):
            sys.modules.pop(name)
    monkeypatch.setattr(builtins, "__import__", reject_heavy_import)

    try:
        scimt = importlib.import_module("scimt")
        attribution = importlib.import_module("scimt.data_attribution")

        assert scimt.__name__ == "scimt"
        assert attribution.__all__ == [
            "SOURCE_REPOSITORY",
            "SOURCE_COMMIT",
            "MIGRATED_MODULES",
            "CheckpointRef",
            "DatasetRef",
            "AttributionStage",
            "AttributionRunConfig",
            "load_attribution_config",
            "ArtifactIdentity",
            "ShardManifest",
            "ArtifactWriter",
            "validate_upstream_identity",
            "PHASES",
            "RunnerError",
            "PhaseReport",
            "PhaseOutput",
            "run_layout",
            "dry_run",
            "fit_factors",
            "compute_rows",
            "build_queries",
            "score_source",
            "build_directions",
            "sweep_jvp",
            "summarize",
        ]
        assert {
            name for name in vars(attribution) if not name.startswith("_")
        } <= set(attribution.__all__)
        assert "scimt.data_attribution._migration" not in sys.modules

        assert attribution.SOURCE_COMMIT == "ca9689a497b921dc516feb663a83269c4a588bbc"
        assert "scimt.data_attribution._migration" in sys.modules
        assert attribution.SOURCE_REPOSITORY == "https://github.com/ArcadiaImpact/gradient-kernel"
        assert attribution.MIGRATED_MODULES["preconditioned_gradient_kernels.parameter_manifest"] == (
            "scimt.data_attribution.manifest"
        )
        assert {
            name for name in vars(attribution) if not name.startswith("_")
        } <= set(attribution.__all__)
        # The runner surface stays lean too: resolving the phase registry
        # through the lazy exports must not pull any heavy dependency. (This
        # binds the runner/config/... submodule attributes, so it runs after
        # the vars() checks above.)
        assert callable(attribution.dry_run)
        assert set(attribution.PHASES) == {
            "fit-factors", "compute-rows", "build-queries", "score-source",
            "build-directions", "sweep-jvp", "summarize", "dry-run",
        }
    finally:
        # Restore the original module objects. Later test modules hold
        # references bound at collection time; leaving the fresh copies in
        # sys.modules splits import identity (e.g. `import scimt.gen` then
        # resolves against a `scimt` package object that never bound `.gen`).
        for name in list(sys.modules):
            if name == "scimt" or name.startswith("scimt."):
                sys.modules.pop(name)
        sys.modules.update(saved_modules)


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
        "preconditioned_gradient_kernels.preconditioner.artifacts",
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
        # the Trainer-side snapshot callback runs under HF Trainer, which
        # requires accelerate
        "accelerate",
    }
    assert extras["data-attribution-ekfac"] == ["scimt[data-attribution]", "kronfluence"]
    assert "data-attribution-ekfac" in extras["all"][0]


def test_built_wheel_contains_attribution_readme_and_extra_metadata(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = tmp_path.glob("*.whl")

    with zipfile.ZipFile(wheel) as archive:
        assert "scimt/data_attribution/README.md" in archive.namelist()
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = email.parser.Parser().parsestr(archive.read(metadata_name).decode())
        entry_points_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = archive.read(entry_points_name).decode()

    # The one sanctioned console shim (plan Task 7) ships in the wheel.
    assert "scimt-attribution = scimt.data_attribution.cli:main" in entry_points

    assert {"data-attribution", "data-attribution-ekfac"} <= set(metadata.get_all("Provides-Extra"))
    requirements = metadata.get_all("Requires-Dist")
    assert 'kronfluence; extra == "data-attribution-ekfac"' in requirements
    assert 'scimt[data-attribution]; extra == "data-attribution-ekfac"' in requirements
    assert any(
        requirement.startswith("scimt[")
        and "data-attribution-ekfac" in requirement
        and 'extra == "all"' in requirement
        for requirement in requirements
    )
