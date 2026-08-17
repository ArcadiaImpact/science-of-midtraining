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
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

OBJECTIVES = ("midtraining", "sft")
ROW_REDUCTIONS = ("per_token", "per_sequence_sum", "per_sequence_mean")
# SOURCE curvature must be positive semidefinite. A raw/true Hessian is not
# PSD and is never a valid SOURCE curvature option (design: error handling).
SOURCE_CURVATURES = ("fisher", "ggn", "ekfac", "ekfac_adam")
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
    training_dataset: DatasetRef | None = None

    def __post_init__(self) -> None:
        _require_str(self.name, "stage name")
        # Stage names become artifact directory components and the
        # second_order.checkpoint vocabulary, so: path-safe charset, and the
        # literal "query" is reserved as the final-query-checkpoint sentinel.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError(
                f"stage name {self.name!r} must match [A-Za-z0-9_-]+ — it "
                "names artifact directories"
            )
        if self.name == "query":
            raise ValueError(
                "stage name 'query' is reserved: it is the "
                "second_order.checkpoint sentinel for the final query "
                "checkpoint"
            )
        if not isinstance(self.checkpoint, CheckpointRef):
            raise TypeError(f"stage {self.name!r} checkpoint must be a CheckpointRef")
        if not isinstance(self.dataset, DatasetRef):
            raise TypeError(f"stage {self.name!r} dataset must be a DatasetRef")
        if self.training_dataset is not None and not isinstance(
            self.training_dataset, DatasetRef
        ):
            raise TypeError(
                f"stage {self.name!r} training_dataset must be a DatasetRef or None"
            )
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
        if self.training_dataset is not None and self.lr_steps is None:
            raise ValueError(
                f"stage {self.name!r}: training_dataset marks a SOURCE segment "
                "and requires explicit lr_steps plus lr_steps_provenance"
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
class AdamMomentEstimatorConfig:
    """Paired frozen-checkpoint Adam-style second-moment estimator."""

    dataset: DatasetRef
    objective: Literal["midtraining", "sft"]
    num_batches: int
    global_batch_size: int
    micro_batch_size: int
    beta2: float
    optimizer_epsilon: float
    max_grad_norm: float
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetRef):
            raise TypeError("adam_moment_estimator dataset must be a DatasetRef")
        _require_vocab(
            self.objective, OBJECTIVES, "adam_moment_estimator objective"
        )
        _require_int(
            self.num_batches,
            "adam_moment_estimator num_batches",
            minimum=1,
        )
        _require_int(
            self.global_batch_size,
            "adam_moment_estimator global_batch_size",
            minimum=1,
        )
        _require_int(
            self.micro_batch_size,
            "adam_moment_estimator micro_batch_size",
            minimum=1,
        )
        if self.global_batch_size % self.micro_batch_size:
            raise ValueError(
                "adam_moment_estimator global_batch_size must be divisible by "
                "micro_batch_size"
            )
        beta2 = _as_float(self.beta2, "adam_moment_estimator beta2")
        if not math.isfinite(beta2) or not 0 < beta2 < 1:
            raise ValueError(
                "adam_moment_estimator beta2 must be finite and satisfy 0 < beta2 < 1"
            )
        _set(self, "beta2", beta2)
        for field_name in ("optimizer_epsilon", "max_grad_norm"):
            value = _as_float(
                getattr(self, field_name),
                f"adam_moment_estimator {field_name}",
            )
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"adam_moment_estimator {field_name} must be positive and finite"
                )
            _set(self, field_name, value)
        _require_int(self.seed, "adam_moment_estimator seed", minimum=0)


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
    curvature: Literal["fisher", "ggn", "ekfac", "ekfac_adam"] = "ekfac"
    basis: Literal["raw", "fisher", "ekfac", "adam"] = "fisher"
    damping_sweep: tuple[float, ...] = (0.1,)
    dtype: str = "float32"
    logra: LoGraConfig | None = None
    conditioning_damping: float | None = None

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
        if self.curvature == "ekfac_adam":
            if self.basis != "adam":
                raise ValueError(
                    "method curvature 'ekfac_adam' fits factors in stage-local "
                    "Adam coordinates and requires basis 'adam'"
                )
            if self.conditioning_damping is None:
                raise ValueError(
                    "method curvature 'ekfac_adam' requires an explicit "
                    "conditioning_damping — it enters A_l at fit time and is "
                    "baked into the factor artifact bytes, so it has no default"
                )
        elif self.conditioning_damping is not None:
            raise ValueError(
                "method conditioning_damping is only valid with curvature "
                "'ekfac_adam' (the diagonal Adam basis takes its damping from "
                "damping_sweep instead)"
            )
        if self.conditioning_damping is not None:
            value = _as_float(
                self.conditioning_damping, "method conditioning_damping"
            )
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "method conditioning_damping must be finite and nonnegative"
                )
            _set(self, "conditioning_damping", value)


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean, got {value!r}")
    return value


def _optional_int(value: Any, context: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    return _require_int(value, context, minimum=minimum)


@dataclass(frozen=True)
class DataConfig:
    """Dataset-adapter and row-computation execution settings (runner phases).

    ``sequence_length``/``max_*_sequences`` define the tokenized datasets and
    therefore artifact identity; ``batch_size``/``vjp_chunk_size``/
    ``rows_per_shard``/``device`` are execution geometry only (mathematically
    equivalent results, bit-identical only up to floating-point reassociation
    under re-chunking) and stay out of artifact identities. The tokenized
    dataset's remaining identity input — tokenizer/chat-template CONTENT —
    is bound by the runner into every artifact's composite
    ``dataset_fingerprint`` (raw source digest + tokenizer-file digest), so
    an in-place tokenizer or chat-template edit refuses artifact reuse.
    """

    sequence_length: int = 512
    batch_size: int = 8
    vjp_chunk_size: int = 8
    rows_per_shard: int = 65536
    device: str = "cpu"
    max_stage_sequences: int | None = None
    max_query_sequences: int | None = None

    def __post_init__(self) -> None:
        _require_int(self.sequence_length, "data sequence_length", minimum=2)
        _require_int(self.batch_size, "data batch_size", minimum=1)
        _require_int(self.vjp_chunk_size, "data vjp_chunk_size", minimum=1)
        _require_int(self.rows_per_shard, "data rows_per_shard", minimum=1)
        _require_str(self.device, "data device")
        _optional_int(
            self.max_stage_sequences, "data max_stage_sequences", minimum=1
        )
        _optional_int(
            self.max_query_sequences, "data max_query_sequences", minimum=1
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "vjp_chunk_size": self.vjp_chunk_size,
            "rows_per_shard": self.rows_per_shard,
            "device": self.device,
            "max_stage_sequences": self.max_stage_sequences,
            "max_query_sequences": self.max_query_sequences,
        }


@dataclass(frozen=True)
class FactorFitConfig:
    """Per-segment curvature fitting settings (``ekfac._fit_config`` surface).

    Field names and defaults mirror the fit-config keys validated by
    ``scimt.data_attribution.ekfac`` (``fit_batch_size`` maps to its
    ``batch_size``); the runner passes them through, and ``fit_ekfac``
    re-validates. The sample seed is the run-level ``seed`` — one seed,
    recorded once.
    """

    samples: int = 1024
    source_batch_size: int = 8
    fit_batch_size: int = 8
    max_positions_per_sequence: int | None = 1
    min_position_gap: int = 1
    use_empirical_fisher: bool = True
    covariance_module_partitions: int = 1
    lambda_module_partitions: int = 1
    eigendecomposition_dtype: Literal["float32", "float64"] = "float64"

    def __post_init__(self) -> None:
        _require_int(self.samples, "factors samples", minimum=1)
        _require_int(self.source_batch_size, "factors source_batch_size", minimum=1)
        _require_int(self.fit_batch_size, "factors fit_batch_size", minimum=1)
        _optional_int(
            self.max_positions_per_sequence,
            "factors max_positions_per_sequence",
            minimum=0,
        )
        _require_int(self.min_position_gap, "factors min_position_gap", minimum=1)
        _require_bool(self.use_empirical_fisher, "factors use_empirical_fisher")
        _require_int(
            self.covariance_module_partitions,
            "factors covariance_module_partitions",
            minimum=1,
        )
        _require_int(
            self.lambda_module_partitions,
            "factors lambda_module_partitions",
            minimum=1,
        )
        _require_vocab(
            self.eigendecomposition_dtype,
            ("float32", "float64"),
            "factors eigendecomposition_dtype",
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "source_batch_size": self.source_batch_size,
            "fit_batch_size": self.fit_batch_size,
            "max_positions_per_sequence": self.max_positions_per_sequence,
            "min_position_gap": self.min_position_gap,
            "use_empirical_fisher": self.use_empirical_fisher,
            "covariance_module_partitions": self.covariance_module_partitions,
            "lambda_module_partitions": self.lambda_module_partitions,
            "eigendecomposition_dtype": self.eigendecomposition_dtype,
        }


@dataclass(frozen=True)
class MetricDerivativeConfig:
    """Statistics artifact feeding the metric-derivative direction term."""

    statistics: Path
    n_estimation_sequences: int = 4

    def __post_init__(self) -> None:
        _set(self, "statistics", _as_path(self.statistics, "metric_derivative statistics"))
        _require_int(
            self.n_estimation_sequences,
            "metric_derivative n_estimation_sequences",
            minimum=1,
        )


SECOND_ORDER_HESSIANS = ("true", "ggn")
SECOND_ORDER_METRICS = ("none", "adam", "fisher", "ekfac")


@dataclass(frozen=True)
class SecondOrderConfig:
    """Direction building and JVP sweeps at ONE explicitly declared checkpoint.

    ``checkpoint`` names a stage or the literal ``"query"`` — never ambiguous.
    ``pairs`` are query-dataset sequence indices (``[i, i]`` is a self
    direction). ``metric`` is the pair metric M in ``grad(g_A^T M g_B)``; the
    pair path supports diagonal metrics only, so ``"ekfac"`` is refused by the
    runner (recorded port deviation). ``sweep_stage`` names whose dataset the
    JVP sweep runs over.
    """

    checkpoint: str
    pairs: tuple[tuple[int, int], ...]
    hessian_kind: Literal["true", "ggn"] = "true"
    metric: Literal["none", "adam", "fisher", "ekfac"] = "none"
    metric_exponent: float = -1.0
    metric_epsilon: float = 1e-8
    metric_derivative: MetricDerivativeConfig | None = None
    sweep_stage: str | None = None
    direction_chunk_size: int = 8

    def __post_init__(self) -> None:
        _require_str(self.checkpoint, "second_order checkpoint")
        pairs = tuple(tuple(pair) for pair in self.pairs)
        if not pairs:
            raise ValueError("second_order pairs must not be empty")
        for pair in pairs:
            if len(pair) != 2 or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in pair
            ):
                raise ValueError(
                    "second_order pairs must be [i, j] nonnegative sequence "
                    f"indices, got {pair!r}"
                )
        if len(set(pairs)) != len(pairs):
            raise ValueError("second_order pairs contains duplicate entries")
        _set(self, "pairs", pairs)
        _require_vocab(
            self.hessian_kind, SECOND_ORDER_HESSIANS, "second_order hessian_kind"
        )
        _require_vocab(self.metric, SECOND_ORDER_METRICS, "second_order metric")
        exponent = _as_float(self.metric_exponent, "second_order metric_exponent")
        if not math.isfinite(exponent):
            raise ValueError("second_order metric_exponent must be finite")
        _set(self, "metric_exponent", exponent)
        epsilon = _as_float(self.metric_epsilon, "second_order metric_epsilon")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError(
                "second_order metric_epsilon must be finite and nonnegative"
            )
        _set(self, "metric_epsilon", epsilon)
        if self.metric_derivative is not None and not isinstance(
            self.metric_derivative, MetricDerivativeConfig
        ):
            raise TypeError(
                "second_order metric_derivative must be a MetricDerivativeConfig"
            )
        if self.sweep_stage is not None:
            _require_str(self.sweep_stage, "second_order sweep_stage")
        _require_int(
            self.direction_chunk_size,
            "second_order direction_chunk_size",
            minimum=1,
        )

    def resolved(self) -> dict[str, Any]:
        derivative = self.metric_derivative
        return {
            "checkpoint": self.checkpoint,
            "pairs": [list(pair) for pair in self.pairs],
            "hessian_kind": self.hessian_kind,
            "metric": self.metric,
            "metric_exponent": self.metric_exponent,
            "metric_epsilon": self.metric_epsilon,
            "metric_derivative": None
            if derivative is None
            else {
                "statistics": str(derivative.statistics),
                "n_estimation_sequences": derivative.n_estimation_sequences,
            },
            "sweep_stage": self.sweep_stage,
            "direction_chunk_size": self.direction_chunk_size,
        }


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
    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorFitConfig = field(default_factory=FactorFitConfig)
    second_order: SecondOrderConfig | None = None
    adam_moment_estimator: AdamMomentEstimatorConfig | None = None
    # Summarize-time declaration: a partial requested output matrix may be
    # summarized only when the SAVED resolved config carries this flag.
    allow_partial: bool = False

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
        if not isinstance(self.data, DataConfig):
            raise TypeError("data must be a DataConfig")
        if not isinstance(self.factors, FactorFitConfig):
            raise TypeError("factors must be a FactorFitConfig")
        _require_bool(self.allow_partial, "allow_partial")
        if self.second_order is not None:
            if not isinstance(self.second_order, SecondOrderConfig):
                raise TypeError("second_order must be a SecondOrderConfig or None")
            stage_names = {stage.name for stage in stages}
            targets = stage_names | {"query"}
            if self.second_order.checkpoint not in targets:
                raise ValueError(
                    "second_order checkpoint must name a stage or 'query', got "
                    f"{self.second_order.checkpoint!r} (stages: "
                    f"{sorted(stage_names)})"
                )
            sweep = self.second_order.sweep_stage
            if sweep is not None and sweep not in stage_names:
                raise ValueError(
                    f"second_order sweep_stage {sweep!r} names no stage "
                    f"(stages: {sorted(stage_names)})"
                )
        if self.adam_moment_estimator is not None:
            if not isinstance(
                self.adam_moment_estimator, AdamMomentEstimatorConfig
            ):
                raise TypeError(
                    "adam_moment_estimator must be an "
                    "AdamMomentEstimatorConfig or None"
                )
            if self.method.basis != "adam":
                raise ValueError(
                    "adam_moment_estimator is valid only when method.basis is 'adam'"
                )
            if self.method.dtype == "float16":
                raise ValueError(
                    "adam_moment_estimator does not support float16 gradients "
                    "without tested loss scaling and overflow detection; use "
                    "bfloat16 or float32"
                )
            snapshots = [s.name for s in stages if s.optimizer_snapshot is not None]
            if snapshots:
                raise ValueError(
                    "adam_moment_estimator cannot mix estimated moments with "
                    "stage optimizer_snapshot declarations; snapshots present "
                    f"on stages: {snapshots}"
                )
        if self.method.basis == "adam" and self.adam_moment_estimator is None:
            missing = [s.name for s in stages if s.optimizer_snapshot is None]
            if missing:
                raise ValueError(
                    "basis 'adam' requires either adam_moment_estimator or an "
                    "optimizer_snapshot on every stage; missing snapshots on "
                    f"stages: {missing}"
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
                    "training_dataset": None
                    if stage.training_dataset is None
                    else ref(stage.training_dataset),
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
                "conditioning_damping": self.method.conditioning_damping,
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
            "data": self.data.resolved(),
            "factors": self.factors.resolved(),
            "second_order": None
            if self.second_order is None
            else self.second_order.resolved(),
            "adam_moment_estimator": None
            if self.adam_moment_estimator is None
            else {
                "dataset": ref(self.adam_moment_estimator.dataset),
                "objective": self.adam_moment_estimator.objective,
                "num_batches": self.adam_moment_estimator.num_batches,
                "global_batch_size": self.adam_moment_estimator.global_batch_size,
                "micro_batch_size": self.adam_moment_estimator.micro_batch_size,
                "beta2": self.adam_moment_estimator.beta2,
                "optimizer_epsilon": self.adam_moment_estimator.optimizer_epsilon,
                "max_grad_norm": self.adam_moment_estimator.max_grad_norm,
                "seed": self.adam_moment_estimator.seed,
            },
            "allow_partial": self.allow_partial,
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
_STAGE_OPTIONAL = frozenset(
    {
        "lr_steps",
        "lr_steps_provenance",
        "optimizer_snapshot",
        "training_dataset",
    }
)


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
        training_dataset=None
        if mapping.get("training_dataset") is None
        else _parse_ref(
            mapping["training_dataset"],
            DatasetRef,
            f"{context} training_dataset",
        ),
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
            {
                "row_reduction",
                "curvature",
                "basis",
                "damping_sweep",
                "dtype",
                "logra",
                "conditioning_damping",
            }
        ),
        context="method",
    )
    options: dict[str, Any] = {
        key: mapping[key]
        for key in (
            "row_reduction",
            "curvature",
            "basis",
            "dtype",
            "conditioning_damping",
        )
        if key in mapping
    }
    if "damping_sweep" in mapping:
        options["damping_sweep"] = mapping["damping_sweep"]
    if mapping.get("logra") is not None:
        options["logra"] = _parse_logra(mapping["logra"])
    return MethodConfig(**options)


def _parse_section(value: Any, cls: type, context: str) -> Any:
    """Parse a flat optional section whose keys are exactly the dataclass fields."""
    mapping = _mapping(value, context)
    known = frozenset(f.name for f in fields(cls))
    _check_keys(mapping, required=frozenset(), optional=known, context=context)
    return cls(**mapping)


def _parse_second_order(value: Any) -> SecondOrderConfig:
    mapping = _mapping(value, "second_order")
    known = frozenset(f.name for f in fields(SecondOrderConfig))
    _check_keys(
        mapping,
        required=frozenset({"checkpoint", "pairs"}),
        optional=known - {"checkpoint", "pairs"},
        context="second_order",
    )
    options = dict(mapping)
    pairs = options.pop("pairs")
    if not isinstance(pairs, list) or not all(
        isinstance(pair, list) for pair in pairs
    ):
        raise ValueError("second_order pairs must be a list of [i, j] lists")
    options["pairs"] = tuple(tuple(pair) for pair in pairs)
    derivative = options.get("metric_derivative")
    if derivative is not None:
        body = _mapping(derivative, "second_order metric_derivative")
        _check_keys(
            body,
            required=frozenset({"statistics"}),
            optional=frozenset({"n_estimation_sequences"}),
            context="second_order metric_derivative",
        )
        options["metric_derivative"] = MetricDerivativeConfig(**body)
    return SecondOrderConfig(**options)


def _parse_adam_moment_estimator(value: Any) -> AdamMomentEstimatorConfig:
    mapping = _mapping(value, "adam_moment_estimator")
    required = frozenset(
        {
            "dataset",
            "objective",
            "num_batches",
            "global_batch_size",
            "micro_batch_size",
            "beta2",
            "optimizer_epsilon",
            "max_grad_norm",
            "seed",
        }
    )
    _check_keys(
        mapping,
        required=required,
        optional=frozenset(),
        context="adam_moment_estimator",
    )
    return AdamMomentEstimatorConfig(
        dataset=_parse_ref(
            mapping["dataset"],
            DatasetRef,
            "adam_moment_estimator dataset",
        ),
        objective=mapping["objective"],
        num_batches=mapping["num_batches"],
        global_batch_size=mapping["global_batch_size"],
        micro_batch_size=mapping["micro_batch_size"],
        beta2=mapping["beta2"],
        optimizer_epsilon=mapping["optimizer_epsilon"],
        max_grad_norm=mapping["max_grad_norm"],
        seed=mapping["seed"],
    )


_TOP_REQUIRED = frozenset({"stages", "query", "output_dir"})
_TOP_OPTIONAL = frozenset(
    {
        "parameters",
        "method",
        "seed",
        "tokenizer",
        "data",
        "factors",
        "second_order",
        "adam_moment_estimator",
        "allow_partial",
    }
)


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
    if "data" in payload:
        options["data"] = _parse_section(payload["data"], DataConfig, "data")
    if "factors" in payload:
        options["factors"] = _parse_section(
            payload["factors"], FactorFitConfig, "factors"
        )
    if payload.get("second_order") is not None:
        options["second_order"] = _parse_second_order(payload["second_order"])
    if payload.get("adam_moment_estimator") is not None:
        options["adam_moment_estimator"] = _parse_adam_moment_estimator(
            payload["adam_moment_estimator"]
        )
    if "allow_partial" in payload:
        options["allow_partial"] = payload["allow_partial"]
    return AttributionRunConfig(
        stages=tuple(_parse_stage(stage, i) for i, stage in enumerate(raw_stages)),
        query=_parse_query(payload["query"]),
        output_dir=payload["output_dir"],
        **options,
    )
