"""Data-attribution methods with optional, lazily imported dependencies."""

from typing import Any as _Any

_LEDGER_EXPORTS = ("SOURCE_REPOSITORY", "SOURCE_COMMIT", "MIGRATED_MODULES")
# name -> submodule; resolved lazily so `import scimt.data_attribution` stays
# free of heavy dependencies (config/artifacts themselves import no torch).
_LAZY_EXPORTS = {
    "CheckpointRef": "config",
    "DatasetRef": "config",
    "AttributionStage": "config",
    "AttributionRunConfig": "config",
    "load_attribution_config": "config",
    "ArtifactIdentity": "artifacts",
    "ShardManifest": "artifacts",
    "ArtifactWriter": "artifacts",
    "validate_upstream_identity": "artifacts",
    # Runner phases (async verbs; runner.py imports no heavy dependency).
    "PHASES": "runner",
    "RunnerError": "runner",
    "PhaseReport": "runner",
    "PhaseOutput": "runner",
    "run_layout": "runner",
    "dry_run": "runner",
    "fit_factors": "runner",
    "compute_rows": "runner",
    "build_queries": "runner",
    "score_source": "runner",
    "build_directions": "runner",
    "sweep_jvp": "runner",
    "summarize": "runner",
}

__all__ = [*_LEDGER_EXPORTS, *_LAZY_EXPORTS]


def __getattr__(name: str) -> _Any:
    if name in _LEDGER_EXPORTS:
        from . import _migration

        return getattr(_migration, name)
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is not None:
        from importlib import import_module

        return getattr(import_module(f".{submodule}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
