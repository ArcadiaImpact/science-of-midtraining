"""Opt-in AdamW attribution snapshots for training stages.

Adam-basis data attribution (SOURCE in Adam coordinates,
``scimt.data_attribution.metrics.DiagonalMetric``) needs the optimizer's raw
second moments — state the default training artifacts deliberately do NOT
save (existing trajectory checkpoints are model-only, and actual Adam state
cannot be reconstructed post hoc). This module is the one sanctioned capture
path:

- ``AttributionSnapshotConfig`` — the config-first opt-in block. Off by
  default: when ``TrainConfig.attribution_snapshots`` is ``None`` nothing in
  the rendered axolotl config or the default saves changes.
- ``AttributionSnapshotCallback`` (lazy: needs ``transformers``) — a
  ``TrainerCallback`` that, at the configured optimizer steps ONLY, captures
  each manifest-included base parameter's ``exp_avg_sq`` plus the exact
  bias-correction metadata (optimizer step, beta2, epsilon, weight decay).
- ``AttributionSnapshotPlugin`` / ``AttributionSnapshotArgs`` (lazy: pod-side,
  need axolotl/pydantic) — the axolotl plugin shim that registers the callback
  from the rendered config block. Verified against the plugin surface of
  axolotl 0.17.0, the version pinned by ``requirements/pod-*.txt``
  (``load_plugin`` resolves the class path via ``importlib`` + ``getattr``,
  ``get_input_args`` contributes a pydantic mixin, and
  ``core/builders/base.py`` calls ``add_callbacks_post_trainer(cfg, trainer)``).

Snapshot format (all under ``<output_dir>/<subdir>/step-<N>/``):

- ``exp_avg_sq-00001-of-0000K.safetensors`` — RAW (bias-correctable) second
  moments for manifest-included parameters, sharded by size. The bias
  correction is deliberately NOT applied at capture; the manifest records the
  exact convention (``v_hat = exp_avg_sq / (1 - beta2**step)``) and the inputs
  it needs, so consumers recover the corrected moments exactly.
- ``parameter_manifest.json`` / ``.sha256`` — the canonical
  ``scimt.data_attribution.manifest.ParameterManifest`` of the model at
  capture time (names, shapes, flattened offsets).
- ``optimizer_manifest.json`` — written LAST via temp-sibling + ``os.replace``;
  its absence marks a partial snapshot as invisible. Carries the optimizer
  metadata, the parameter-manifest digest, per-shard sha256 digests, and the
  model-checkpoint reference (``checkpoint-<N>`` sibling of the same step).

The model weights, ``trainer_state.json``, and the rendered axolotl config of
the same step are deliberately REFERENCED, not copied: the trainer's own save
(align ``at_steps`` with the save schedule) and the run dir stay the single
source of truth, and ``scimt.data_attribution.stages.resolve_stage`` enforces
the step identity between snapshot and checkpoint at consume time.

FSDP: optimizer state is rank-sharded, so the capture NEVER reads the local
``optimizer.state`` when the model is sharded. It goes through the supported
full-optimizer-state API, ``torch.distributed.checkpoint.state_dict
.get_optimizer_state_dict(model, optim, StateDictOptions(full_state_dict=True,
cpu_offload=True))`` — a collective every rank enters; per its contract only
rank 0 receives the gathered state (CPU), and only rank 0 publishes files,
after which all ranks synchronize on a barrier. Note the gather materializes
the full optimizer state in rank-0 CPU RAM (~8 bytes/param for both moments
at fp32) — fine on the sprint pods, but budget for it on small hosts.

AdamW ONLY (``torch.optim.AdamW``; axolotl's ``adamw_torch``/
``adamw_torch_fused`` resolve to it). Other optimizers — including quantized
AdamW variants whose stored state is not the raw second moment — are refused
loudly rather than captured wrongly. Frozen or LoRA/adapter parameters cannot
masquerade as base-model Adam coordinates: adapter-bearing models are refused
outright, and a frozen-but-included parameter is an error (exclude it or
train it).

Heavy imports (torch/safetensors/transformers/axolotl) stay function-local so
``import scimt.train`` remains CPU-light.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scimt.data_attribution.manifest import ParameterManifest

ATTRIBUTION_PLUGIN_PATH = (
    "scimt.train.attribution_snapshot.AttributionSnapshotPlugin"
)
OPTIMIZER_MANIFEST_NAME = "optimizer_manifest.json"
SNAPSHOT_KIND = "scimt.adamw_attribution_snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
BIAS_CORRECTION_CONVENTION = "v_hat = exp_avg_sq / (1 - beta2**step)"
DEFAULT_SUBDIR = "attribution_snapshots"
DEFAULT_MAX_SHARD_BYTES = 2**31  # 2 GiB
_MIN_CONFIG_SHARD_BYTES = 2**20  # config floor; the writer itself is unfloored
_SHARD_RE = re.compile(r"exp_avg_sq-(\d{5})-of-(\d{5})\.safetensors")
_FLOATING_SAFETENSOR_DTYPES = {"F64", "F32", "F16", "BF16"}
_ADAPTER_NAME_RE = re.compile(
    r"(^|\.)(lora_[A-Za-z0-9_]+|adapter[s]?|modules_to_save)(\.|$)"
)

_MANIFEST_KEYS = {
    "schema_version", "kind", "optimizer", "parameter_manifest_digest",
    "model_checkpoint", "shards", "world_size", "produced_by",
}
_OPTIMIZER_KEYS = {
    "type", "step", "beta1", "beta2", "epsilon", "weight_decay",
    "weight_decay_values", "bias_correction",
}
_SHARD_KEYS = {"filename", "sha256", "num_bytes", "parameters"}
_CHECKPOINT_REF_KEYS = {"global_step", "relative_dir"}


class SnapshotIntegrityError(ValueError):
    """A snapshot on disk does not satisfy the published format contract."""


# --------------------------------------------------------------- config block
@dataclass(frozen=True)
class AttributionSnapshotConfig:
    """Opt-in snapshot schedule + parameter selection (config-first).

    ``at_steps``/``every_steps`` name the optimizer steps to capture at —
    configured steps only, nothing implicit. ``include``/``exclude`` are
    full-match regexes over parameter names (the ``ParameterManifest``
    contract); the default includes every base parameter. ``subdir`` is the
    directory created under the trainer ``output_dir``.
    """

    at_steps: tuple[int, ...] = ()
    every_steps: int | None = None
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES
    subdir: str = DEFAULT_SUBDIR

    def __post_init__(self) -> None:
        steps = tuple(self.at_steps)
        for step in steps:
            if isinstance(step, bool) or not isinstance(step, int) or step < 1:
                raise ValueError(
                    f"attribution_snapshots.at_steps must be positive integers, "
                    f"got {step!r}"
                )
        if len(set(steps)) != len(steps):
            raise ValueError("attribution_snapshots.at_steps must be unique")
        object.__setattr__(self, "at_steps", tuple(sorted(steps)))
        if self.every_steps is not None and (
            isinstance(self.every_steps, bool)
            or not isinstance(self.every_steps, int)
            or self.every_steps < 1
        ):
            raise ValueError(
                "attribution_snapshots.every_steps must be a positive integer "
                f"or null, got {self.every_steps!r}"
            )
        if not self.at_steps and self.every_steps is None:
            raise ValueError(
                "attribution_snapshots needs at_steps and/or every_steps — an "
                "opt-in block that never captures is a mistake"
            )
        for label in ("include", "exclude"):
            patterns = getattr(self, label)
            if patterns is None:
                continue
            patterns = tuple(patterns)
            if not patterns or not all(
                isinstance(p, str) and p for p in patterns
            ):
                raise ValueError(
                    f"attribution_snapshots.{label} must be a non-empty list "
                    "of regex strings (or null)"
                )
            object.__setattr__(self, label, patterns)
        if (
            isinstance(self.max_shard_bytes, bool)
            or not isinstance(self.max_shard_bytes, int)
            or self.max_shard_bytes < _MIN_CONFIG_SHARD_BYTES
        ):
            raise ValueError(
                "attribution_snapshots.max_shard_bytes must be an int >= "
                f"{_MIN_CONFIG_SHARD_BYTES}, got {self.max_shard_bytes!r}"
            )
        if (
            not isinstance(self.subdir, str)
            or not self.subdir
            or self.subdir in (".", "..")
            or "/" in self.subdir
            or "\\" in self.subdir
        ):
            raise ValueError(
                "attribution_snapshots.subdir must be a bare directory name, "
                f"got {self.subdir!r}"
            )

    def captures_step(self, step: int) -> bool:
        if step in self.at_steps:
            return True
        return bool(self.every_steps) and step % self.every_steps == 0

    def as_dict(self) -> dict[str, Any]:
        """YAML-safe dict (lists, not tuples) for rendering + manifests."""
        return {
            "at_steps": list(self.at_steps),
            "every_steps": self.every_steps,
            "include": None if self.include is None else list(self.include),
            "exclude": None if self.exclude is None else list(self.exclude),
            "max_shard_bytes": self.max_shard_bytes,
            "subdir": self.subdir,
        }


def snapshot_config_from(
    data: Mapping[str, Any], *, source: str
) -> AttributionSnapshotConfig:
    """Strict constructor from a YAML mapping (unknown keys are an error)."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"attribution_snapshots must be a mapping in {source}, "
            f"got {type(data).__name__}"
        )
    data = dict(data)
    known = {f.name for f in dataclasses.fields(AttributionSnapshotConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown attribution_snapshots keys in {source}: {sorted(unknown)}"
        )
    for key in ("at_steps", "include", "exclude"):
        if isinstance(data.get(key), list):
            data[key] = tuple(data[key])
    return AttributionSnapshotConfig(**data)


# ----------------------------------------------------------- state collection
@dataclass(frozen=True)
class _CollectedAdamState:
    """Validated AdamW state for the manifest-included parameters."""

    exp_avg_sq: dict[str, Any]  # name -> torch.Tensor (cpu, detached)
    step: int
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    weight_decay_values: tuple[float, ...]


def _unwrap_optimizer(optimizer: Any) -> Any:
    """Peel accelerate-style wrappers (``.optimizer``) down to the real one."""
    seen = 0
    while hasattr(optimizer, "optimizer") and seen < 8:
        inner = optimizer.optimizer
        if inner is optimizer:
            break
        optimizer, seen = inner, seen + 1
    return optimizer


def _require_adamw(optimizer: Any) -> None:
    import torch

    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError(
            "attribution snapshots capture bias-correctable AdamW state and "
            f"require torch.optim.AdamW; got {type(optimizer).__qualname__}. "
            "Other optimizers (including quantized AdamW variants) are not "
            "supported — use a non-Adam attribution basis instead."
        )


def _refuse_adapter_parameters(model: Any) -> None:
    if getattr(model, "peft_config", None) is not None:
        raise ValueError(
            "model carries a PEFT adapter configuration — LoRA/adapter "
            "training optimizes adapter coordinates only, which cannot "
            "masquerade as full/base-model Adam coordinates. Merge and "
            "re-train full-parameter to capture Adam state."
        )
    adapterish = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and _ADAPTER_NAME_RE.search(name)
    )
    if adapterish:
        raise ValueError(
            "trainable adapter parameters detected "
            f"({adapterish[:4]}{'...' if len(adapterish) > 4 else ''}) — "
            "LoRA/adapter-only optimizer state cannot masquerade as "
            "full/base-model Adam coordinates."
        )


def _validated_moment(name: str, value: Any, shape: tuple[int, ...]) -> Any:
    import torch

    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(
            f"exp_avg_sq for {name!r} must be a floating-point tensor"
        )
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"exp_avg_sq for {name!r} has shape {tuple(value.shape)}, "
            f"expected {tuple(shape)}"
        )
    # detach + move; no forced copy — the snapshot is serialized before the
    # optimizer can mutate again, and a copy would double resident memory for
    # the gathered full state of a 12B model
    value = value.detach().to(device="cpu")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"exp_avg_sq for {name!r} is not finite")
    if bool((value < 0).any()):
        raise ValueError(f"exp_avg_sq for {name!r} has negative entries")
    return value


def _int_step(name: str, value: Any) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"optimizer step for {name!r} is not a scalar")
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"optimizer step for {name!r} is not an integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"optimizer step for {name!r} is not integral: {value!r}"
            )
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"optimizer step for {name!r} is missing")
    if value < 1:
        raise ValueError(
            f"optimizer step for {name!r} is {value}; a snapshot before the "
            "first optimizer step has no meaningful second moments"
        )
    return value


def _uniform_step(steps: set[int]) -> int:
    if len(steps) != 1:
        raise ValueError(
            f"optimizer step differs across included parameters: "
            f"{sorted(steps)} — refusing mixed bias-correction metadata"
        )
    return next(iter(steps))


def _hyperparameters(
    groups: list[dict],
) -> tuple[float, float, float, float, tuple[float, ...]]:
    if not groups:
        raise ValueError(
            "no optimizer param_group covers the included parameters"
        )
    betas = {tuple(float(b) for b in g["betas"]) for g in groups}
    epsilons = {float(g["eps"]) for g in groups}
    if len(betas) != 1:
        raise ValueError(
            f"included parameters span param_groups with different betas: "
            f"{sorted(betas)}"
        )
    if len(epsilons) != 1:
        raise ValueError(
            f"included parameters span param_groups with different eps: "
            f"{sorted(epsilons)}"
        )
    (beta1, beta2), epsilon = next(iter(betas)), next(iter(epsilons))
    if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
        raise ValueError(f"AdamW betas out of range: {(beta1, beta2)}")
    if epsilon <= 0.0:
        raise ValueError(f"AdamW eps must be positive, got {epsilon}")
    decays = sorted({float(g.get("weight_decay", 0.0)) for g in groups})
    if any(d < 0 for d in decays):
        raise ValueError(f"negative weight_decay in param_groups: {decays}")
    nonzero = [d for d in decays if d != 0.0]
    if len(nonzero) > 1:
        raise ValueError(
            "included parameters span param_groups with multiple nonzero "
            f"weight decays {nonzero} — a single stage weight_decay cannot "
            "describe them"
        )
    weight_decay = nonzero[0] if nonzero else 0.0
    return beta1, beta2, epsilon, weight_decay, tuple(decays)


def collect_adamw_state(
    model: Any, optimizer: Any, manifest: "ParameterManifest"
) -> _CollectedAdamState:
    """Single-process collection straight from ``optimizer.state``.

    Never valid under FSDP (state is rank-sharded there) — the sharded path
    goes through :func:`_full_optimizer_state` instead. The returned tensors
    may alias live optimizer state (deliberately uncopied): serialize them
    before the optimizer steps again, as :func:`capture_snapshot` does.
    """
    from scimt.data_attribution.manifest import included_named_parameters

    _require_adamw(optimizer)
    pairs = included_named_parameters(model, manifest)
    if not optimizer.state:
        raise ValueError(
            "optimizer has no state — it has not stepped yet, so there are "
            "no second moments to snapshot"
        )
    moments: dict[str, Any] = {}
    steps: set[int] = set()
    for entry, parameter in pairs:
        state = optimizer.state.get(parameter)
        if not state or "exp_avg_sq" not in state:
            raise ValueError(
                f"parameter {entry.name!r} has no AdamW state (frozen or "
                "not handed to the optimizer) — it cannot provide "
                "full-model Adam coordinates; exclude it or train it"
            )
        moments[entry.name] = _validated_moment(
            entry.name, state["exp_avg_sq"], entry.shape
        )
        steps.add(_int_step(entry.name, state.get("step")))
    included_ids = {id(parameter) for _, parameter in pairs}
    covering = [
        group
        for group in optimizer.param_groups
        if any(id(parameter) in included_ids for parameter in group["params"])
    ]
    beta1, beta2, epsilon, weight_decay, decays = _hyperparameters(covering)
    return _CollectedAdamState(
        exp_avg_sq=moments,
        step=_uniform_step(steps),
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        weight_decay_values=decays,
    )


def _collected_from_full_osd(
    osd: Mapping[str, Any], manifest: "ParameterManifest"
) -> _CollectedAdamState:
    """Parse a gathered full optimizer state dict (FQN-keyed) — the shape
    returned by ``torch.distributed.checkpoint.state_dict
    .get_optimizer_state_dict(..., full_state_dict=True)``."""
    if (
        not isinstance(osd, Mapping)
        or not isinstance(osd.get("state"), Mapping)
        or not isinstance(osd.get("param_groups"), list)
    ):
        raise ValueError(
            "full optimizer state dict must carry 'state' and 'param_groups'"
        )
    state, groups = osd["state"], osd["param_groups"]
    moments: dict[str, Any] = {}
    steps: set[int] = set()
    included = manifest.included_entries()
    for entry in included:
        item = state.get(entry.name)
        if not isinstance(item, Mapping) or "exp_avg_sq" not in item:
            raise ValueError(
                f"parameter {entry.name!r} is missing from the gathered "
                "optimizer state — it cannot provide full-model Adam "
                "coordinates; exclude it or train it"
            )
        moments[entry.name] = _validated_moment(
            entry.name, item["exp_avg_sq"], entry.shape
        )
        steps.add(_int_step(entry.name, item.get("step")))
    names = {entry.name for entry in included}
    covering = [
        group
        for group in groups
        if any(fqn in names for fqn in group.get("params", []))
    ]
    beta1, beta2, epsilon, weight_decay, decays = _hyperparameters(covering)
    return _CollectedAdamState(
        exp_avg_sq=moments,
        step=_uniform_step(steps),
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        weight_decay_values=decays,
    )


# ----------------------------------------------------- distributed primitives
def _is_sharded(model: Any) -> bool:
    """True when parameters are FSDP-sharded (DTensor / FSDP wrapper / a live
    multi-rank process group). Sharded models must never be read through the
    local ``optimizer.state``."""
    try:
        from torch.distributed.tensor import DTensor
    except Exception:  # pragma: no cover - very old torch
        DTensor = None
    if DTensor is not None and any(
        isinstance(parameter, DTensor) for parameter in model.parameters()
    ):
        return True
    if any(
        cls.__name__ == "FullyShardedDataParallel"
        for cls in type(model).__mro__
    ):
        return True
    import torch.distributed as dist

    return bool(
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def _rank_and_world() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _full_optimizer_state(model: Any, optimizer: Any) -> Mapping[str, Any]:
    """The supported Transformers/Accelerate/FSDP full-optimizer-state API:
    a collective that gathers FQN-keyed full tensors; with ``cpu_offload``
    only rank 0 receives content (other ranks get an empty dict)."""
    import torch.distributed.checkpoint.state_dict as dcp_state

    options = dcp_state.StateDictOptions(full_state_dict=True, cpu_offload=True)
    return dcp_state.get_optimizer_state_dict(model, optimizer, options=options)


# ------------------------------------------------------------------- capture
def _model_identifier(model: Any) -> str:
    for candidate in (
        getattr(model, "name_or_path", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return type(model).__qualname__


def capture_snapshot(
    *,
    model: Any,
    optimizer: Any,
    output_dir: str | Path,
    global_step: int,
    config: AttributionSnapshotConfig,
) -> Path | None:
    """Capture one AdamW snapshot at ``global_step``.

    Returns the published snapshot directory, or ``None`` on non-writing
    ranks. Refuses (never writes partial/wrong state) on: non-AdamW
    optimizers, adapter models, frozen included parameters, missing state,
    step disagreement, or an already-published snapshot for this step.
    """
    from scimt.data_attribution.manifest import ParameterManifest

    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 1
    ):
        raise ValueError(f"global_step must be a positive int, got {global_step!r}")
    optimizer = _unwrap_optimizer(optimizer)
    _require_adamw(optimizer)
    _refuse_adapter_parameters(model)
    manifest = ParameterManifest.from_model(
        model,
        _model_identifier(model),
        include=None if config.include is None else list(config.include),
        exclude=None if config.exclude is None else list(config.exclude),
    )
    included = manifest.included_entries()
    if not included:
        raise ValueError(
            "attribution_snapshots include/exclude selected no parameters"
        )
    frozen = sorted(e.name for e in included if not e.requires_grad)
    if frozen:
        raise ValueError(
            "included parameters are frozen (requires_grad=False) and have no "
            f"Adam state: {frozen[:4]}{'...' if len(frozen) > 4 else ''} — "
            "frozen coordinates cannot masquerade as trained Adam "
            "coordinates; exclude them or train them"
        )
    target = Path(output_dir) / config.subdir / f"step-{global_step}"

    if _is_sharded(model):
        rank, world = _rank_and_world()
        # Collective: EVERY rank must enter; only rank 0 receives content.
        osd = _full_optimizer_state(model, optimizer)
        written: Path | None = None
        if rank == 0:
            collected = _collected_from_full_osd(osd, manifest)
            written = _publish(target, manifest, collected, global_step,
                               world, config.max_shard_bytes)
        _barrier()  # writes are visible before any rank proceeds
        return written

    collected = collect_adamw_state(model, optimizer, manifest)
    return _publish(target, manifest, collected, global_step, 1,
                    config.max_shard_bytes)


def _publish(
    target: Path,
    manifest: "ParameterManifest",
    collected: _CollectedAdamState,
    global_step: int,
    world_size: int,
    max_shard_bytes: int,
) -> Path:
    if collected.step != global_step:
        raise ValueError(
            f"optimizer step {collected.step} != trainer global_step "
            f"{global_step} — refusing to record bias-correction metadata "
            "that does not describe this checkpoint"
        )
    return write_adamw_snapshot(
        target,
        manifest=manifest,
        exp_avg_sq=collected.exp_avg_sq,
        step=collected.step,
        beta1=collected.beta1,
        beta2=collected.beta2,
        epsilon=collected.epsilon,
        weight_decay=collected.weight_decay,
        weight_decay_values=collected.weight_decay_values,
        model_checkpoint={
            "global_step": global_step,
            "relative_dir": f"../../checkpoint-{global_step}",
        },
        world_size=world_size,
        max_shard_bytes=max_shard_bytes,
    )


# -------------------------------------------------------------------- writer
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_replace(tmp: Path, final: Path) -> None:
    with tmp.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, final)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    _fsync_replace(tmp, path)


def _validate_snapshot_metadata(
    step: int,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    weight_decay_values: tuple[float, ...],
    model_checkpoint: Mapping[str, Any],
    world_size: int,
) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError(f"snapshot step must be a positive int, got {step!r}")
    if not 0.0 < beta1 < 1.0:
        raise ValueError(f"beta1 out of range (0, 1): {beta1!r}")
    if not 0.0 < beta2 < 1.0:
        raise ValueError(f"beta2 out of range (0, 1): {beta2!r}")
    if not epsilon > 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon!r}")
    if not weight_decay >= 0.0:
        raise ValueError(f"weight_decay must be nonnegative, got {weight_decay!r}")
    values = tuple(float(v) for v in weight_decay_values)
    if not values or any(v < 0 for v in values) or weight_decay not in values:
        raise ValueError(
            f"weight_decay_values {values!r} must be nonnegative and contain "
            f"weight_decay={weight_decay!r}"
        )
    if (
        not isinstance(model_checkpoint, Mapping)
        or set(model_checkpoint) != _CHECKPOINT_REF_KEYS
        or isinstance(model_checkpoint["global_step"], bool)
        or not isinstance(model_checkpoint["global_step"], int)
        or not isinstance(model_checkpoint["relative_dir"], str)
        or not model_checkpoint["relative_dir"]
    ):
        raise ValueError(
            "model_checkpoint must be {'global_step': int, 'relative_dir': str}"
        )
    if model_checkpoint["global_step"] != step:
        raise ValueError(
            f"model_checkpoint.global_step {model_checkpoint['global_step']} "
            f"!= optimizer step {step}"
        )
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"world_size must be a positive int, got {world_size!r}")


def write_adamw_snapshot(
    directory: str | Path,
    *,
    manifest: "ParameterManifest",
    exp_avg_sq: Mapping[str, Any],
    step: int,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    weight_decay_values: tuple[float, ...] | list[float],
    model_checkpoint: Mapping[str, Any],
    world_size: int = 1,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> Path:
    """Serialize one validated snapshot: sharded ``exp_avg_sq`` safetensors,
    the parameter manifest, then ``optimizer_manifest.json`` LAST (atomic
    temp-sibling + ``os.replace``) so partial state is never visible."""
    from safetensors.torch import save_file

    directory = Path(directory)
    _validate_snapshot_metadata(
        step, beta1, beta2, epsilon, weight_decay,
        tuple(float(v) for v in weight_decay_values), model_checkpoint,
        world_size,
    )
    if isinstance(max_shard_bytes, bool) or not isinstance(max_shard_bytes, int) \
            or max_shard_bytes < 1:
        raise ValueError(f"max_shard_bytes must be >= 1, got {max_shard_bytes!r}")
    included = manifest.included_entries()
    if not included:
        raise ValueError("manifest includes no parameters")
    names = {entry.name for entry in included}
    if set(exp_avg_sq) != names:
        missing = sorted(names - set(exp_avg_sq))
        extra = sorted(set(exp_avg_sq) - names)
        raise ValueError(
            f"exp_avg_sq keys do not match manifest-included parameters "
            f"(missing {missing}, extra {extra})"
        )
    tensors = {
        entry.name: _validated_moment(
            entry.name, exp_avg_sq[entry.name], entry.shape
        ).contiguous()
        for entry in included
    }

    if directory.exists():
        if (directory / OPTIMIZER_MANIFEST_NAME).exists():
            raise ValueError(
                f"snapshot already exists at {directory} — refusing to "
                "overwrite a published optimizer snapshot"
            )
        shutil.rmtree(directory)  # clear an invisible dead partial
    directory.mkdir(parents=True)

    groups: list[list] = [[]]
    group_bytes = 0
    for entry in included:  # manifest order, deterministic
        tensor = tensors[entry.name]
        nbytes = tensor.element_size() * tensor.numel()
        if groups[-1] and group_bytes + nbytes > max_shard_bytes:
            groups.append([])
            group_bytes = 0
        groups[-1].append(entry.name)
        group_bytes += nbytes

    total = len(groups)
    records = []
    try:
        for index, group in enumerate(groups, start=1):
            filename = f"exp_avg_sq-{index:05d}-of-{total:05d}.safetensors"
            final = directory / filename
            tmp = directory / f"{filename}.tmp-{os.getpid()}"
            save_file({name: tensors[name] for name in group}, str(tmp))
            _fsync_replace(tmp, final)
            records.append({
                "filename": filename,
                "sha256": _sha256_file(final),
                "num_bytes": final.stat().st_size,
                "parameters": list(group),
            })
        manifest.save(directory)
        document = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "kind": SNAPSHOT_KIND,
            "optimizer": {
                "type": "adamw",
                "step": step,
                "beta1": beta1,
                "beta2": beta2,
                "epsilon": epsilon,
                "weight_decay": weight_decay,
                "weight_decay_values": [float(v) for v in weight_decay_values],
                "bias_correction": {
                    "applied": False,
                    "convention": BIAS_CORRECTION_CONVENTION,
                },
            },
            "parameter_manifest_digest": manifest.digest(),
            "model_checkpoint": dict(model_checkpoint),
            "shards": records,
            "world_size": world_size,
            "produced_by": "scimt.train.attribution_snapshot",
        }
        _atomic_write_text(
            directory / OPTIMIZER_MANIFEST_NAME,
            json.dumps(document, indent=2, sort_keys=True) + "\n",
        )
    finally:
        for leftover in directory.glob("*.tmp-*"):
            leftover.unlink(missing_ok=True)
    return directory


# -------------------------------------------------------------------- loader
@dataclass(frozen=True)
class AdamSnapshotInfo:
    """Validated snapshot metadata (no tensors retained)."""

    path: Path
    step: int
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    weight_decay_values: tuple[float, ...]
    parameter_manifest_digest: str
    model_checkpoint: Mapping[str, Any]
    shards: tuple[str, ...]
    world_size: int


@dataclass(frozen=True)
class AdamOptimizerSnapshot:
    """A fully loaded snapshot: metadata + manifest + raw second moments."""

    info: AdamSnapshotInfo
    manifest: "ParameterManifest"
    exp_avg_sq: dict[str, Any]

    def bias_corrected_exp_avg_sq(self) -> dict[str, Any]:
        """``v_hat = exp_avg_sq / (1 - beta2**step)`` in float32 — the exact
        AdamW bias correction the recorded metadata describes."""
        correction = 1.0 - self.info.beta2 ** self.info.step
        return {
            name: value.float() / correction
            for name, value in self.exp_avg_sq.items()
        }


def _integrity(condition: bool, message: str) -> None:
    if not condition:
        raise SnapshotIntegrityError(message)


def _checked_float(document: Mapping[str, Any], key: str) -> float:
    value = document.get(key)
    _integrity(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"optimizer.{key} must be a number, got {value!r}",
    )
    return float(value)


def validate_optimizer_snapshot(directory: str | Path) -> AdamSnapshotInfo:
    """Validate a snapshot directory against the published format: strict
    schema, parameter-manifest digest, per-shard sha256 over full bytes, and
    exact shard/parameter coverage. Raises :class:`SnapshotIntegrityError`."""
    from safetensors import safe_open

    from scimt.data_attribution.manifest import ParameterManifest

    directory = Path(directory)
    manifest_path = directory / OPTIMIZER_MANIFEST_NAME
    _integrity(
        manifest_path.is_file(),
        f"no {OPTIMIZER_MANIFEST_NAME} under {directory} — snapshot absent or "
        "incomplete (partial writes are never published)",
    )
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotIntegrityError(
            f"unreadable {OPTIMIZER_MANIFEST_NAME}: {error}"
        ) from error
    _integrity(isinstance(document, dict), "optimizer manifest must be an object")
    _integrity(
        set(document) == _MANIFEST_KEYS,
        f"optimizer manifest keys {sorted(document)} != {sorted(_MANIFEST_KEYS)}",
    )
    _integrity(
        document["schema_version"] == SNAPSHOT_SCHEMA_VERSION,
        f"unsupported schema_version {document['schema_version']!r}",
    )
    _integrity(document["kind"] == SNAPSHOT_KIND,
               f"unsupported kind {document['kind']!r}")

    optimizer = document["optimizer"]
    _integrity(
        isinstance(optimizer, dict) and set(optimizer) == _OPTIMIZER_KEYS,
        "optimizer block malformed",
    )
    _integrity(optimizer["type"] == "adamw",
               f"snapshot optimizer type {optimizer['type']!r} is not adamw")
    step = optimizer["step"]
    _integrity(
        isinstance(step, int) and not isinstance(step, bool) and step >= 1,
        f"optimizer.step must be a positive int, got {step!r}",
    )
    beta1 = _checked_float(optimizer, "beta1")
    beta2 = _checked_float(optimizer, "beta2")
    epsilon = _checked_float(optimizer, "epsilon")
    weight_decay = _checked_float(optimizer, "weight_decay")
    _integrity(0.0 < beta1 < 1.0, f"beta1 out of range (0, 1): {beta1}")
    _integrity(0.0 < beta2 < 1.0, f"beta2 out of range (0, 1): {beta2}")
    _integrity(epsilon > 0.0, f"epsilon must be positive: {epsilon}")
    _integrity(weight_decay >= 0.0, f"weight_decay negative: {weight_decay}")
    raw_values = optimizer["weight_decay_values"]
    _integrity(
        isinstance(raw_values, list)
        and raw_values
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
            for v in raw_values
        ),
        f"weight_decay_values malformed: {raw_values!r}",
    )
    values = tuple(float(v) for v in raw_values)
    _integrity(
        weight_decay in values,
        f"weight_decay {weight_decay} not among weight_decay_values {values}",
    )
    _integrity(
        optimizer["bias_correction"]
        == {"applied": False, "convention": BIAS_CORRECTION_CONVENTION},
        "bias_correction metadata does not match the published convention "
        f"{BIAS_CORRECTION_CONVENTION!r}",
    )
    checkpoint_ref = document["model_checkpoint"]
    _integrity(
        isinstance(checkpoint_ref, dict)
        and set(checkpoint_ref) == _CHECKPOINT_REF_KEYS
        and isinstance(checkpoint_ref["global_step"], int)
        and not isinstance(checkpoint_ref["global_step"], bool)
        and checkpoint_ref["global_step"] == step
        and isinstance(checkpoint_ref["relative_dir"], str)
        and checkpoint_ref["relative_dir"],
        "model_checkpoint reference malformed or step-inconsistent",
    )
    world_size = document["world_size"]
    _integrity(
        isinstance(world_size, int)
        and not isinstance(world_size, bool)
        and world_size >= 1,
        f"world_size must be a positive int, got {world_size!r}",
    )

    try:
        manifest = ParameterManifest.load(directory)
    except (OSError, ValueError) as error:
        raise SnapshotIntegrityError(
            f"parameter manifest invalid under {directory}: {error}"
        ) from error
    _integrity(
        manifest.digest() == document["parameter_manifest_digest"],
        "parameter_manifest_digest does not match the stored manifest",
    )
    entries = {e.name: e for e in manifest.included_entries()}

    shards = document["shards"]
    _integrity(isinstance(shards, list) and shards, "shards list malformed")
    listed: list[str] = []
    indices: list[int] = []
    seen_parameters: list[str] = []
    for record in shards:
        _integrity(
            isinstance(record, dict) and set(record) == _SHARD_KEYS,
            f"shard record malformed: {record!r}",
        )
        filename = record["filename"]
        match = _SHARD_RE.fullmatch(str(filename))
        _integrity(match is not None, f"shard filename malformed: {filename!r}")
        _integrity(
            int(match.group(2)) == len(shards),
            f"shard {filename!r} count field disagrees with {len(shards)} shards",
        )
        indices.append(int(match.group(1)))
        listed.append(filename)
        parameters = record["parameters"]
        _integrity(
            isinstance(parameters, list) and parameters
            and all(isinstance(p, str) for p in parameters),
            f"shard {filename!r} parameters malformed",
        )
        seen_parameters.extend(parameters)
        shard_path = directory / filename
        _integrity(shard_path.is_file(), f"missing shard file {filename!r}")
        _integrity(
            shard_path.stat().st_size == record["num_bytes"],
            f"shard {filename!r} size differs from the manifest",
        )
        _integrity(
            _sha256_file(shard_path) == record["sha256"],
            f"shard {filename!r} sha256 digest mismatch — bytes changed after "
            "publication",
        )
        with safe_open(str(shard_path), framework="pt") as handle:
            keys = list(handle.keys())
            _integrity(
                sorted(keys) == sorted(parameters),
                f"shard {filename!r} tensor names differ from the manifest",
            )
            for name in keys:
                entry = entries.get(name)
                _integrity(
                    entry is not None,
                    f"shard tensor {name!r} is not a manifest-included parameter",
                )
                tensor_slice = handle.get_slice(name)
                _integrity(
                    tuple(tensor_slice.get_shape()) == tuple(entry.shape),
                    f"shard tensor {name!r} shape differs from the manifest",
                )
                _integrity(
                    tensor_slice.get_dtype() in _FLOATING_SAFETENSOR_DTYPES,
                    f"shard tensor {name!r} dtype {tensor_slice.get_dtype()!r} "
                    "is not floating point",
                )
    _integrity(
        len(set(listed)) == len(listed), "duplicate shard filenames listed"
    )
    _integrity(
        sorted(indices) == list(range(1, len(shards) + 1)),
        f"shard indices {sorted(indices)} are not 1..{len(shards)}",
    )
    on_disk = {p.name for p in directory.glob("exp_avg_sq-*.safetensors")}
    unlisted = sorted(on_disk - set(listed))
    _integrity(
        not unlisted,
        f"unlisted shard files present: {unlisted} — refusing a snapshot "
        "whose bytes are not covered by its manifest",
    )
    _integrity(
        len(set(seen_parameters)) == len(seen_parameters),
        "a parameter appears in more than one shard",
    )
    _integrity(
        set(seen_parameters) == set(entries),
        "shard parameters do not cover exactly the manifest-included set",
    )

    return AdamSnapshotInfo(
        path=directory,
        step=step,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        weight_decay_values=values,
        parameter_manifest_digest=document["parameter_manifest_digest"],
        model_checkpoint=dict(checkpoint_ref),
        shards=tuple(listed),
        world_size=world_size,
    )


def load_optimizer_snapshot(directory: str | Path) -> AdamOptimizerSnapshot:
    """Validate then load a snapshot's tensors (finite, nonnegative)."""
    import torch
    from safetensors import safe_open

    from scimt.data_attribution.manifest import ParameterManifest

    info = validate_optimizer_snapshot(directory)
    manifest = ParameterManifest.load(info.path)
    tensors: dict[str, Any] = {}
    for filename in info.shards:
        with safe_open(str(info.path / filename), framework="pt") as handle:
            for name in handle.keys():
                tensors[name] = handle.get_tensor(name)
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise SnapshotIntegrityError(f"tensor {name!r} is not finite")
        if bool((value < 0).any()):
            raise SnapshotIntegrityError(f"tensor {name!r} has negative entries")
    return AdamOptimizerSnapshot(info=info, manifest=manifest, exp_avg_sq=tensors)


# ------------------------------------------------- lazy heavy-dependency API
def _callback_class():
    from transformers import TrainerCallback

    class AttributionSnapshotCallback(TrainerCallback):
        """Capture AdamW attribution snapshots at configured steps only.

        ``on_step_end`` fires on every rank; :func:`capture_snapshot` owns the
        rank gating (collective gather, rank-0 publish, barrier).
        """

        def __init__(self, config: AttributionSnapshotConfig) -> None:
            if not isinstance(config, AttributionSnapshotConfig):
                raise TypeError(
                    "AttributionSnapshotCallback needs an "
                    f"AttributionSnapshotConfig, got {type(config).__name__}"
                )
            self.config = config

        def on_step_end(self, args, state, control, **kwargs):
            if not self.config.captures_step(int(state.global_step)):
                return
            model = kwargs.get("model")
            optimizer = kwargs.get("optimizer")
            if model is None or optimizer is None:
                raise RuntimeError(
                    "Trainer did not pass model/optimizer to on_step_end — "
                    "cannot capture the attribution snapshot"
                )
            capture_snapshot(
                model=model,
                optimizer=optimizer,
                output_dir=Path(args.output_dir),
                global_step=int(state.global_step),
                config=self.config,
            )

    AttributionSnapshotCallback.__module__ = __name__
    return AttributionSnapshotCallback


def _plugin_class():
    # Pod-side dependency; the class path in a rendered config resolves here
    # through axolotl's load_plugin (importlib + getattr, PEP 562 compatible).
    from axolotl.integrations.base import BasePlugin

    class AttributionSnapshotPlugin(BasePlugin):
        """Axolotl shim: read the rendered ``attribution_snapshots`` block and
        register :class:`AttributionSnapshotCallback` post-trainer."""

        def get_input_args(self) -> str:
            return "scimt.train.attribution_snapshot.AttributionSnapshotArgs"

        def add_callbacks_post_trainer(self, cfg, trainer):
            raw = None
            if hasattr(cfg, "get"):
                raw = cfg.get("attribution_snapshots")
            if raw is None:
                raw = getattr(cfg, "attribution_snapshots", None)
            if not raw:
                return []
            config = snapshot_config_from(
                dict(raw), source="axolotl config attribution_snapshots"
            )
            callback_cls = _cached("AttributionSnapshotCallback")
            return [callback_cls(config)]

    AttributionSnapshotPlugin.__module__ = __name__
    return AttributionSnapshotPlugin


def _args_class():
    from pydantic import BaseModel

    class AttributionSnapshotArgs(BaseModel):
        """Pydantic mixin merged into axolotl's input config so the rendered
        ``attribution_snapshots`` block passes config validation."""

        attribution_snapshots: dict | None = None

    AttributionSnapshotArgs.__module__ = __name__
    return AttributionSnapshotArgs


_LAZY = {
    "AttributionSnapshotCallback": _callback_class,
    "AttributionSnapshotPlugin": _plugin_class,
    "AttributionSnapshotArgs": _args_class,
}


def _cached(name: str):
    if name not in globals() or globals()[name] is None:
        globals()[name] = _LAZY[name]()
    return globals()[name]


def __getattr__(name: str):
    if name in _LAZY:
        return _cached(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
