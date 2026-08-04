"""Typed attribution run configuration, adapted from gradient-kernel ca9689a.

Upstream expressed this contract with pydantic models and a model registry;
scimt re-expresses it as frozen dataclasses validated in ``__post_init__`` so
direct construction and YAML loading share one validation path. There is
deliberately no second model registry: checkpoints, datasets, optimizer
snapshots, and tokenizers are explicit filesystem refs resolved later by the
stage adapter. Unknown keys anywhere in the YAML are a ``ValueError``.

This module must stay importable without torch/numpy/safetensors.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

OBJECTIVES = ("midtraining", "sft")
ROW_REDUCTIONS = ("per_token", "per_sequence_sum", "per_sequence_mean")
# SOURCE curvature must be positive semidefinite. A raw/true Hessian is not
# PSD and is never a valid SOURCE curvature option (design: error handling).
SOURCE_CURVATURES = ("fisher", "ggn", "ekfac")
SOURCE_BASES = ("raw", "fisher", "ekfac", "adam")
LOGRA_INITS = ("random", "pca", "artifact")
TORCH_DTYPES = ("bfloat16", "float16", "float32", "float64")
_RAW_HESSIAN_NAMES = frozenset({"hessian", "true", "true_hessian", "raw_hessian"})


def normalize_dtype(name: Any) -> str:
    """Return the canonical dtype name for ``name`` (accepts a ``torch.`` prefix)."""
    if not isinstance(name, str):
        raise ValueError(f"dtype must be a string, got {name!r}")
    normalized = name.removeprefix("torch.")
    if normalized not in TORCH_DTYPES:
        raise ValueError(f"unknown dtype {name!r}; supported: {list(TORCH_DTYPES)}")
    return normalized


def _set(instance: Any, name: str, value: Any) -> None:
    object.__setattr__(instance, name, value)


def _as_path(value: Any, context: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    raise ValueError(f"{context} must be a non-empty path, got {value!r}")


def _as_optional_path(value: Any, context: str) -> Path | None:
    return None if value is None else _as_path(value, context)


def _require_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string, got {value!r}")
    return value


def _require_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be >= {minimum}, got {value!r}")
    return value


def _as_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number, got {value!r}")
    return float(value)


def _require_vocab(value: Any, vocabulary: tuple[str, ...], context: str) -> str:
    if value not in vocabulary:
        raise ValueError(f"{context} must be one of {list(vocabulary)}, got {value!r}")
    return value


def _compile_regex(pattern: Any, context: str) -> str:
    if not isinstance(pattern, str):
        raise ValueError(f"{context} must be a string, got {pattern!r}")
    try:
        re.compile(pattern)
    except re.error as error:
        raise ValueError(f"invalid {context} {pattern!r}: {error}") from error
    return pattern


def _validate_digest(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string or null, got {value!r}")
    return value


@dataclass(frozen=True)
class CheckpointRef:
    """scimt checkpoint dir (contains checkpoint.json) or plain HF-format dir."""

    path: Path
    expected_digest: str | None = None

    def __post_init__(self) -> None:
        _set(self, "path", _as_path(self.path, "checkpoint path"))
        _validate_digest(self.expected_digest, "checkpoint expected_digest")


@dataclass(frozen=True)
class DatasetRef:
    """scimt dataset dir (contains dataset.json) or a raw JSONL file."""

    path: Path
    expected_digest: str | None = None

    def __post_init__(self) -> None:
        _set(self, "path", _as_path(self.path, "dataset path"))
        _validate_digest(self.expected_digest, "dataset expected_digest")


@dataclass(frozen=True)
class AttributionStage:
    """One chronological training segment consumed by SOURCE."""

    name: str
    checkpoint: CheckpointRef
    dataset: DatasetRef
    objective: Literal["midtraining", "sft"]
    # None -> the stage adapter derives lr_steps from trainer_state.json; an
    # explicit value requires lr_steps_provenance and is cross-checked there.
    lr_steps: float | None
    n_examples: int
    weight_decay: float
    optimizer_snapshot: Path | None
    lr_steps_provenance: str | None = None

    def __post_init__(self) -> None:
        _require_str(self.name, "stage name")
        if not isinstance(self.checkpoint, CheckpointRef):
            raise TypeError(f"stage {self.name!r} checkpoint must be a CheckpointRef")
        if not isinstance(self.dataset, DatasetRef):
            raise TypeError(f"stage {self.name!r} dataset must be a DatasetRef")
        _require_vocab(self.objective, OBJECTIVES, f"stage {self.name!r} objective")
        if self.lr_steps is not None:
            lr_steps = _as_float(self.lr_steps, f"stage {self.name!r} lr_steps")
            if not math.isfinite(lr_steps) or lr_steps <= 0:
                raise ValueError(
                    f"stage {self.name!r} lr_steps must be positive and finite"
                )
            _set(self, "lr_steps", lr_steps)
            if self.lr_steps_provenance is None or not _is_nonempty_str(
                self.lr_steps_provenance
            ):
                raise ValueError(
                    f"stage {self.name!r}: explicit lr_steps requires a non-empty "
                    "lr_steps_provenance annotation"
                )
        elif self.lr_steps_provenance is not None:
            raise ValueError(
                f"stage {self.name!r}: lr_steps_provenance is only valid with an "
                "explicit lr_steps"
            )
        _require_int(self.n_examples, f"stage {self.name!r} n_examples", minimum=1)
        weight_decay = _as_float(self.weight_decay, f"stage {self.name!r} weight_decay")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError(
                f"stage {self.name!r} weight_decay must be finite and nonnegative"
            )
        _set(self, "weight_decay", weight_decay)
        _set(
            self,
            "optimizer_snapshot",
            _as_optional_path(
                self.optimizer_snapshot, f"stage {self.name!r} optimizer_snapshot"
            ),
        )


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


@dataclass(frozen=True)
class QueryConfig:
    """Final query checkpoint, measurement dataset, and measurement objective."""

    checkpoint: CheckpointRef
    dataset: DatasetRef
    objective: Literal["midtraining", "sft"]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, CheckpointRef):
            raise TypeError("query checkpoint must be a CheckpointRef")
        if not isinstance(self.dataset, DatasetRef):
            raise TypeError("query dataset must be a DatasetRef")
        _require_vocab(self.objective, OBJECTIVES, "query objective")


@dataclass(frozen=True)
class ParameterSelection:
    """Regex include/exclude rules feeding ``ParameterManifest.from_model``."""

    include: tuple[str, ...] = (".*",)
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        include = tuple(self.include)
        exclude = tuple(self.exclude)
        if not include:
            raise ValueError("parameters include must not be empty")
        for pattern in include:
            _compile_regex(pattern, "include regex")
        for pattern in exclude:
            _compile_regex(pattern, "exclude regex")
        _set(self, "include", include)
        _set(self, "exclude", exclude)


@dataclass(frozen=True)
class LoGraConfig:
    rank: int
    init: Literal["random", "pca", "artifact"]
    seed: int
    targets: str = ".*"
    ekfac_factors: Path | None = None
    projections: Path | None = None

    def __post_init__(self) -> None:
        _require_int(self.rank, "logra rank", minimum=1)
        _require_vocab(self.init, LOGRA_INITS, "logra init")
        _require_int(self.seed, "logra seed")
        _compile_regex(self.targets, "logra targets regex")
        _set(
            self,
            "ekfac_factors",
            _as_optional_path(self.ekfac_factors, "logra ekfac_factors"),
        )
        _set(
            self,
            "projections",
            _as_optional_path(self.projections, "logra projections"),
        )
        if self.init == "pca" and self.ekfac_factors is None:
            raise ValueError("logra init 'pca' requires ekfac_factors")
        if self.init != "pca" and self.ekfac_factors is not None:
            raise ValueError("logra ekfac_factors is only valid with init 'pca'")
        if self.init == "artifact" and self.projections is None:
            raise ValueError("logra init 'artifact' requires projections")
        if self.init != "artifact" and self.projections is not None:
            raise ValueError("logra projections is only valid with init 'artifact'")


@dataclass(frozen=True)
class MethodConfig:
    """Row granularity, curvature, coordinate basis, damping sweep, and LoGra."""

    row_reduction: Literal[
        "per_token", "per_sequence_sum", "per_sequence_mean"
    ] = "per_token"
    curvature: Literal["fisher", "ggn", "ekfac"] = "ekfac"
    basis: Literal["raw", "fisher", "ekfac", "adam"] = "fisher"
    damping_sweep: tuple[float, ...] = (0.1,)
    dtype: str = "float32"
    logra: LoGraConfig | None = None

    def __post_init__(self) -> None:
        _require_vocab(self.row_reduction, ROW_REDUCTIONS, "method row_reduction")
        curvature = self.curvature
        if curvature is True or (
            isinstance(curvature, str) and curvature.lower() in _RAW_HESSIAN_NAMES
        ):
            raise ValueError(
                "raw Hessian curvature is never a valid SOURCE curvature; SOURCE "
                f"requires PSD curvature from {list(SOURCE_CURVATURES)}"
            )
        _require_vocab(curvature, SOURCE_CURVATURES, "method curvature")
        _require_vocab(self.basis, SOURCE_BASES, "method basis")
        if isinstance(self.damping_sweep, (str, bytes)) or not isinstance(
            self.damping_sweep, (list, tuple)
        ):
            raise ValueError("method damping_sweep must be a list of numbers")
        if not self.damping_sweep:
            raise ValueError("method damping_sweep must not be empty")
        sweep = tuple(
            _as_float(value, "method damping_sweep entry")
            for value in self.damping_sweep
        )
        for value in sweep:
            if not math.isfinite(value):
                raise ValueError("method damping_sweep entries must be finite")
            if value < 0:
                raise ValueError("method damping_sweep entries must be nonnegative")
        if len(set(sweep)) != len(sweep):
            raise ValueError("method damping_sweep contains duplicate entries")
        _set(self, "damping_sweep", sweep)
        _set(self, "dtype", normalize_dtype(self.dtype))
        if self.logra is not None and not isinstance(self.logra, LoGraConfig):
            raise TypeError("method logra must be a LoGraConfig or None")


@dataclass(frozen=True)
class AttributionRunConfig:
    """Ordered stage chain plus everything a run needs to be reproducible."""

    stages: tuple[AttributionStage, ...]
    query: QueryConfig
    output_dir: Path
    parameters: ParameterSelection = field(default_factory=ParameterSelection)
    method: MethodConfig = field(default_factory=MethodConfig)
    seed: int = 0
    # Optional explicit tokenizer directory ref; None -> the stage adapter
    # uses the query checkpoint's own tokenizer files.
    tokenizer: Path | None = None

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("configuration must name at least one stage")
        for stage in stages:
            if not isinstance(stage, AttributionStage):
                raise TypeError("stages must contain AttributionStage entries")
        names = [stage.name for stage in stages]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate stage names: {duplicates}")
        _set(self, "stages", stages)
        if not isinstance(self.query, QueryConfig):
            raise TypeError("query must be a QueryConfig")
        if not isinstance(self.parameters, ParameterSelection):
            raise TypeError("parameters must be a ParameterSelection")
        if not isinstance(self.method, MethodConfig):
            raise TypeError("method must be a MethodConfig")
        _require_int(self.seed, "seed")
        _set(self, "output_dir", _as_path(self.output_dir, "output_dir"))
        _set(self, "tokenizer", _as_optional_path(self.tokenizer, "tokenizer"))
        if self.method.basis == "adam":
            missing = [s.name for s in stages if s.optimizer_snapshot is None]
            if missing:
                raise ValueError(
                    "basis 'adam' requires an optimizer_snapshot on every stage; "
                    f"missing on stages: {missing}"
                )

    def resolved(self) -> dict[str, Any]:
        """Every resolved field, defaults included, as a JSON/YAML-safe dict."""

        def ref(value: CheckpointRef | DatasetRef) -> dict[str, Any]:
            return {"path": str(value.path), "expected_digest": value.expected_digest}

        def optional_path(value: Path | None) -> str | None:
            return None if value is None else str(value)

        logra = self.method.logra
        return {
            "stages": [
                {
                    "name": stage.name,
                    "checkpoint": ref(stage.checkpoint),
                    "dataset": ref(stage.dataset),
                    "objective": stage.objective,
                    "lr_steps": stage.lr_steps,
                    "lr_steps_provenance": stage.lr_steps_provenance,
                    "n_examples": stage.n_examples,
                    "weight_decay": stage.weight_decay,
                    "optimizer_snapshot": optional_path(stage.optimizer_snapshot),
                }
                for stage in self.stages
            ],
            "query": {
                "checkpoint": ref(self.query.checkpoint),
                "dataset": ref(self.query.dataset),
                "objective": self.query.objective,
            },
            "parameters": {
                "include": list(self.parameters.include),
                "exclude": list(self.parameters.exclude),
            },
            "method": {
                "row_reduction": self.method.row_reduction,
                "curvature": self.method.curvature,
                "basis": self.method.basis,
                "damping_sweep": list(self.method.damping_sweep),
                "dtype": self.method.dtype,
                "logra": None
                if logra is None
                else {
                    "rank": logra.rank,
                    "init": logra.init,
                    "seed": logra.seed,
                    "targets": logra.targets,
                    "ekfac_factors": optional_path(logra.ekfac_factors),
                    "projections": optional_path(logra.projections),
                },
            },
            "seed": self.seed,
            "tokenizer": optional_path(self.tokenizer),
            "output_dir": str(self.output_dir),
        }


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping, got {value!r}")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{context} keys must be strings, got {key!r}")
    return value


def _check_keys(
    mapping: dict[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    context: str,
) -> None:
    unknown = set(mapping) - required - optional
    if unknown:
        raise ValueError(f"unknown {context} keys: {sorted(unknown)}")
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"{context} is missing required keys: {sorted(missing)}")


def _parse_ref(value: Any, cls: type, context: str) -> Any:
    if isinstance(value, str):
        return cls(path=value)
    mapping = _mapping(value, context)
    _check_keys(
        mapping,
        required=frozenset({"path"}),
        optional=frozenset({"expected_digest"}),
        context=context,
    )
    return cls(path=mapping["path"], expected_digest=mapping.get("expected_digest"))


_STAGE_REQUIRED = frozenset(
    {"name", "checkpoint", "dataset", "objective", "n_examples", "weight_decay"}
)
_STAGE_OPTIONAL = frozenset({"lr_steps", "lr_steps_provenance", "optimizer_snapshot"})


def _parse_stage(value: Any, index: int) -> AttributionStage:
    context = f"stages[{index}]"
    mapping = _mapping(value, context)
    _check_keys(mapping, required=_STAGE_REQUIRED, optional=_STAGE_OPTIONAL, context=context)
    return AttributionStage(
        name=mapping["name"],
        checkpoint=_parse_ref(mapping["checkpoint"], CheckpointRef, f"{context} checkpoint"),
        dataset=_parse_ref(mapping["dataset"], DatasetRef, f"{context} dataset"),
        objective=mapping["objective"],
        lr_steps=mapping.get("lr_steps"),
        n_examples=mapping["n_examples"],
        weight_decay=mapping["weight_decay"],
        optimizer_snapshot=mapping.get("optimizer_snapshot"),
        lr_steps_provenance=mapping.get("lr_steps_provenance"),
    )


def _parse_query(value: Any) -> QueryConfig:
    mapping = _mapping(value, "query")
    _check_keys(
        mapping,
        required=frozenset({"checkpoint", "dataset", "objective"}),
        optional=frozenset(),
        context="query",
    )
    return QueryConfig(
        checkpoint=_parse_ref(mapping["checkpoint"], CheckpointRef, "query checkpoint"),
        dataset=_parse_ref(mapping["dataset"], DatasetRef, "query dataset"),
        objective=mapping["objective"],
    )


def _parse_parameters(value: Any) -> ParameterSelection:
    mapping = _mapping(value, "parameters")
    _check_keys(
        mapping,
        required=frozenset(),
        optional=frozenset({"include", "exclude"}),
        context="parameters",
    )
    selection: dict[str, Any] = {}
    for key in ("include", "exclude"):
        if key in mapping:
            entries = mapping[key]
            if not isinstance(entries, list):
                raise ValueError(f"parameters {key} must be a list of regexes")
            selection[key] = tuple(entries)
    return ParameterSelection(**selection)


def _parse_logra(value: Any) -> LoGraConfig:
    mapping = _mapping(value, "method logra")
    _check_keys(
        mapping,
        required=frozenset({"rank", "init", "seed"}),
        optional=frozenset({"targets", "ekfac_factors", "projections"}),
        context="method logra",
    )
    return LoGraConfig(
        rank=mapping["rank"],
        init=mapping["init"],
        seed=mapping["seed"],
        targets=mapping.get("targets", ".*"),
        ekfac_factors=mapping.get("ekfac_factors"),
        projections=mapping.get("projections"),
    )


def _parse_method(value: Any) -> MethodConfig:
    mapping = _mapping(value, "method")
    _check_keys(
        mapping,
        required=frozenset(),
        optional=frozenset(
            {"row_reduction", "curvature", "basis", "damping_sweep", "dtype", "logra"}
        ),
        context="method",
    )
    options: dict[str, Any] = {
        key: mapping[key]
        for key in ("row_reduction", "curvature", "basis", "dtype")
        if key in mapping
    }
    if "damping_sweep" in mapping:
        options["damping_sweep"] = mapping["damping_sweep"]
    if mapping.get("logra") is not None:
        options["logra"] = _parse_logra(mapping["logra"])
    return MethodConfig(**options)


_TOP_REQUIRED = frozenset({"stages", "query", "output_dir"})
_TOP_OPTIONAL = frozenset({"parameters", "method", "seed", "tokenizer"})


def load_attribution_config(path: str | Path) -> AttributionRunConfig:
    """Load and validate an attribution run configuration from YAML."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("top-level configuration must be a mapping")
    _mapping(payload, "configuration")
    _check_keys(
        payload, required=_TOP_REQUIRED, optional=_TOP_OPTIONAL, context="configuration"
    )
    raw_stages = payload["stages"]
    if not isinstance(raw_stages, list):
        raise ValueError("stages must be a list")
    if not raw_stages:
        raise ValueError("configuration must name at least one stage")
    options: dict[str, Any] = {}
    if "parameters" in payload:
        options["parameters"] = _parse_parameters(payload["parameters"])
    if "method" in payload:
        options["method"] = _parse_method(payload["method"])
    if "seed" in payload:
        options["seed"] = payload["seed"]
    if payload.get("tokenizer") is not None:
        options["tokenizer"] = payload["tokenizer"]
    return AttributionRunConfig(
        stages=tuple(_parse_stage(stage, i) for i, stage in enumerate(raw_stages)),
        query=_parse_query(payload["query"]),
        output_dir=payload["output_dir"],
        **options,
    )
