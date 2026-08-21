"""Config-first attribution runner: staged, resumable, provenance-checked.

Async-native library verbs (the caller owns the event loop; the
``scimt-attribution`` console script in ``cli.py`` is the one sanctioned thin
wrapper, plan Task 7). One :class:`AttributionRunConfig` drives every phase:

- ``fit-factors``   per-segment curvature (EK-FAC via Kronfluence, or the
  empirical Fisher diagonal) on seeded samples from that stage's actual
  training distribution (``build_ekfac_sample_items`` — the one sampler).
- ``estimate-adam`` paired frozen-checkpoint Adam-style second raw moments
  from one common ordered calibration sample; never optimizer replay/state.
- ``compute-rows``  per-example train gradient rows per stage (optionally
  LoGra-projected), resumable by committed shard.
- ``build-queries`` measurement-loss gradient rows at the final query
  checkpoint.
- ``score-source``  chronological SOURCE scoring over validated artifacts;
  explicit adjacent basis transitions, per-segment ``1/N`` exactly once.
- ``build-directions`` / ``sweep-jvp``  second-order pair directions and
  forward-JVP sweeps at ONE explicitly declared checkpoint.
- ``summarize``     completeness-counting summary; refuses a partial
  requested matrix unless the SAVED resolved config declares
  ``allow_partial: true``.
- ``dry-run``       resolves configs, stage metadata, counts, Adam and
  manifest availability, factor partitions, and output-identity previews
  without loading a model. It stays torch-free unless an Adam estimator is
  configured, in which case it tokenizes the calibration corpus to prove the
  exact usable population before GPU work.

Resumability contract: every artifact directory is bound to one
:class:`ArtifactIdentity` whose ``resolved_config`` is the PHASE-SCOPED slice
of the run configuration (only the fields that determine that artifact's
content — execution geometry such as batch sizes stays out), so an identical
rerun is a no-op and any content-relevant change is a focused refusal naming
the differing fields. Consumers validate upstream identities and digests
before loading tensors; there are no permissive fallbacks.

Scope refusals (loud, not fallbacks): SOURCE scoring supports curvature
``fisher``/``ekfac`` (a fitted GGN segment operator does not exist; GGN lives
in the second-order phases), basis ``raw``/``fisher``/``adam`` (an exact
EK-FAC-basis transport across differently-fitted segments is not
representable with EK-FAC factors), and unprojected rows (SOURCE over
LoGra-projected rows is not wired yet). The pair-gradient path supports
diagonal metrics only, and metric-derivative statistics reconstructed with
the ``rank1`` estimator are refused ("factors are not linear in the
statistic") — the two recorded port deviations this module owns.

Heavy imports (torch, safetensors, transformers, numpy) stay inside phase
execution; importing this module is CPU-only. The one subprocess is the
read-only ``git rev-parse``/``git status`` provenance capture, the same
carve-out ``train/runlog.py`` uses.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._migration import SOURCE_COMMIT
from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactIdentity,
    ArtifactIntegrityError,
    ArtifactWriter,
    IdentityMismatchError,
    ShardManifest,
    read_identity,
)
from .config import AttributionRunConfig, AttributionStage
from .stages import ResolvedStage, artifact_digest, resolve_stage

__all__ = [
    "PHASES",
    "PhaseOutput",
    "PhaseReport",
    "RunLayout",
    "RunnerError",
    "build_directions",
    "build_queries",
    "compute_rows",
    "dry_run",
    "estimate_adam",
    "fit_factors",
    "run_layout",
    "score_source",
    "summarize",
    "sweep_jvp",
]

# Mirrors losses.SAMPLE_ID_STRIDE without importing torch at module import;
# asserted equal inside the first heavy phase that runs.
_SAMPLE_ID_STRIDE = 2**20
_TARGET_POLICY = {
    "midtraining": "all_next_tokens",
    "sft": "assistant_content_and_end",
}
_STATISTICS_FILE = "statistics.json"
_FACTORS_COMPLETE_FILE = "factors_complete.json"
_SCORE_MANIFEST_FILE = "score_manifest.json"
QUERY_GROUPS_FILE = "query_groups.json"
_WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin")
_ADAM_MOMENT_STATISTIC = "checkpoint_local_adam_second_raw_moment"
_ADAM_MOMENT_ALGORITHM = (
    "bias_corrected_ema_clipped_global_batch_gradient_square"
)


class RunnerError(ValueError):
    """A runner-level refusal: unsupported request, drift, or missing input."""


@dataclass(frozen=True)
class PhaseOutput:
    """One artifact produced (or found already complete) by a phase."""

    name: str
    directory: Path
    identity_digest: str
    skipped: bool
    rows: int | None = None


@dataclass(frozen=True)
class PhaseReport:
    phase: str
    outputs: tuple[PhaseOutput, ...]


@dataclass(frozen=True)
class RunLayout:
    """Canonical output tree under ``AttributionRunConfig.output_dir``."""

    root: Path
    run_json: Path
    events: Path
    adam_moments: Path
    factors: Path
    rows: Path
    queries: Path
    scores: Path
    streaming_scores: Path
    directions: Path
    jvp: Path
    summary: Path


def run_layout(output_dir: str | Path) -> RunLayout:
    root = Path(output_dir)
    return RunLayout(
        root=root,
        run_json=root / "run.json",
        events=root / "events.jsonl",
        adam_moments=root / "adam_moments",
        factors=root / "factors",
        rows=root / "rows",
        queries=root / "queries",
        scores=root / "scores",
        streaming_scores=root / "streaming_scores",
        directions=root / "directions",
        jvp=root / "jvp",
        summary=root / "summary",
    )


# ----------------------------------------------------------------- provenance
def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scimt_commit() -> str:
    """Read-only git provenance (the ``train/runlog.py`` carve-out)."""
    root = Path(__file__).resolve()
    repo = next((p for p in root.parents if (p / ".git").exists()), None)
    if repo is None:
        return "unpackaged:no-git-checkout"
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, cwd=repo,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True, capture_output=True, text=True, cwd=repo,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unpackaged:git-unavailable"
    return head + ("+dirty" if dirty else "")


def _producing_command(phase: str) -> str:
    return f"scimt-attribution {phase}"


def _loss_convention(objective: str, reduction: str) -> dict[str, Any]:
    return {
        "loss": "causal_lm_cross_entropy",
        "reduction": reduction,
        "target_policy": _TARGET_POLICY[objective],
        "sample_id_stride": _SAMPLE_ID_STRIDE,
    }


# --------------------------------------------------------- phase-scoped config
def _stage_index(config: AttributionRunConfig, name: str) -> int:
    for index, stage in enumerate(config.stages):
        if stage.name == name:
            return index
    raise RunnerError(f"no stage named {name!r} in the configuration")


def _resolved_stage_entry(resolved: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in resolved["stages"]:
        if entry["name"] == name:
            normalized = dict(entry)
            if normalized.get("training_dataset") is None:
                normalized.pop("training_dataset", None)
            return normalized
    raise RunnerError(f"no stage named {name!r} in the configuration")


def _scoped_config(
    config: AttributionRunConfig, phase: str, stage_name: str | None = None
) -> dict[str, Any]:
    return _scoped_resolved(config.resolved(), phase, stage_name)


def _scoped_resolved(
    resolved: dict[str, Any], phase: str, stage_name: str | None = None
) -> dict[str, Any]:
    """The slice of the resolved config that determines a phase's artifact.

    Execution-only geometry (batch sizes, chunking, rows_per_shard, device)
    is deliberately excluded: it changes how the same content is computed,
    never what is computed, so it must not invalidate committed artifacts.
    Recomputation under different geometry is mathematically equivalent but
    only floating-point-identical up to reassociation of the same sums — a
    deliberate trade documented here, not an oversight; same-geometry
    resumption IS bit-exact (and tested as such). ``allow_partial`` is
    excluded everywhere: it is a summarize-time declaration, not artifact
    content. Tokenizer/chat-template CONTENT is bound separately through
    every identity's composite ``dataset_fingerprint``
    (see :func:`_dataset_fingerprint`).
    """
    method = dict(resolved["method"])
    # conditioning_damping exists only in ekfac_adam mode; dropping the None
    # keeps every old-mode scoped slice byte-identical to what committed
    # artifacts recorded before the field existed.
    if method.get("conditioning_damping") is None:
        method.pop("conditioning_damping", None)
    stage_data = {
        "sequence_length": resolved["data"]["sequence_length"],
        "max_stage_sequences": resolved["data"]["max_stage_sequences"],
    }
    query_data = {
        "sequence_length": resolved["data"]["sequence_length"],
        "max_query_sequences": resolved["data"]["max_query_sequences"],
    }
    base = {
        "phase": phase,
        "parameters": resolved["parameters"],
        "seed": resolved["seed"],
        "tokenizer": resolved["tokenizer"],
    }
    row_method = {
        "row_reduction": method["row_reduction"],
        "dtype": method["dtype"],
        "logra": method["logra"],
    }
    if phase == "estimate-adam":
        return {
            **base,
            "stages": [
                {"name": entry["name"], "checkpoint": entry["checkpoint"]}
                for entry in resolved["stages"]
            ],
            "data": {"sequence_length": stage_data["sequence_length"]},
            "dtype": method["dtype"],
            "adam_moment_estimator": resolved["adam_moment_estimator"],
        }
    if phase == "fit-factors":
        scope = {
            **base,
            "stage": _resolved_stage_entry(resolved, stage_name),
            "data": stage_data,
            "factors": resolved["factors"],
            "curvature": method["curvature"],
            "dtype": method["dtype"],
        }
        if method["curvature"] == "ekfac_adam":
            # Conditioned factors are invalid under different Adam moments:
            # bind the estimator contract and the fit-time damping into the
            # factor artifact's identity. Conditional so old-mode slices are
            # untouched.
            scope["conditioning_damping"] = method["conditioning_damping"]
            scope["adam_moment_estimator"] = resolved["adam_moment_estimator"]
        return scope
    if phase == "compute-rows":
        return {
            **base,
            "stage": _resolved_stage_entry(resolved, stage_name),
            "data": stage_data,
            "method": row_method,
        }
    if phase == "build-queries":
        return {
            **base,
            "query": resolved["query"],
            "data": query_data,
            "method": row_method,
        }
    if phase in ("score-source", "score-source-streaming"):
        scope = {
            **base,
            "stages": [
                _resolved_stage_entry(resolved, entry["name"])
                for entry in resolved["stages"]
            ],
            "query": resolved["query"],
            "data": {**stage_data, **query_data},
            "method": method,
        }
        if resolved.get("adam_moment_estimator") is not None:
            scope["adam_moment_estimator"] = resolved["adam_moment_estimator"]
        return scope
    if phase == "build-directions":
        second = dict(resolved["second_order"])
        second.pop("sweep_stage", None)
        second.pop("direction_chunk_size", None)
        return {
            **base,
            "second_order": second,
            "query": resolved["query"],
            "data": {"sequence_length": stage_data["sequence_length"],
                     "max_query_sequences": query_data["max_query_sequences"]},
            "dtype": method["dtype"],
        }
    if phase == "sweep-jvp":
        second = resolved["second_order"]
        return {
            **base,
            "second_order": {
                "checkpoint": second["checkpoint"],
                "sweep_stage": second["sweep_stage"],
            },
            "stage": _resolved_stage_entry(resolved, second["sweep_stage"]),
            "data": stage_data,
            "row_reduction": method["row_reduction"],
            "dtype": method["dtype"],
        }
    raise RunnerError(f"unknown phase {phase!r}")


# ------------------------------------------------------------------ run ledger
def _phase_scopes(resolved: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    """Every scoped slice the phase depends on (one per stage for per-stage
    phases)."""
    if phase in ("fit-factors", "compute-rows"):
        return [
            _scoped_resolved(resolved, phase, entry["name"])
            for entry in resolved["stages"]
        ]
    return [_scoped_resolved(resolved, phase)]


def _write_or_check_ledger(config: AttributionRunConfig, phase: str) -> None:
    """Record the run's resolved config once; later phases must agree on the
    slice THEY depend on (phase-scoped, like artifact identities — extending
    an unrelated section must not strand committed artifacts)."""
    layout = run_layout(config.output_dir)
    resolved = config.resolved()
    if layout.run_json.is_file():
        saved = json.loads(layout.run_json.read_text(encoding="utf-8"))
        stored = saved.get("resolved_config", {})
        current_scopes = _phase_scopes(resolved, phase)
        try:
            saved_scopes = _phase_scopes(stored, phase)
        except (KeyError, RunnerError, TypeError) as error:
            raise RunnerError(
                f"phase {phase!r}: the saved run ledger {layout.run_json} "
                f"cannot cover this phase's configuration scope: {error}"
            ) from error
        if _canonical(current_scopes) != _canonical(saved_scopes):
            differing: set[str] = set()
            for current, reference in zip(
                current_scopes, saved_scopes, strict=False
            ):
                differing.update(
                    key
                    for key in set(current) | set(reference)
                    if _canonical(current.get(key))
                    != _canonical(reference.get(key))
                )
            raise RunnerError(
                f"phase {phase!r}: the current configuration disagrees with "
                f"the saved run ledger {layout.run_json} on this phase's "
                f"scope; differing resolved_config keys: {sorted(differing)}. "
                f"Artifacts under {layout.root} were produced under the "
                "saved configuration — use a fresh output_dir for a "
                "different run."
            )
        return
    layout.root.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": _now(),
        "producing_command": _producing_command(phase),
        "scimt_commit": _scimt_commit(),
        "source_commit": SOURCE_COMMIT,
        "resolved_config": resolved,
    }
    from .artifacts import _atomic_write_text  # same commit protocol

    _atomic_write_text(layout.run_json, json.dumps(ledger, indent=2,
                                                   sort_keys=True) + "\n")


def _append_event(config: AttributionRunConfig, phase: str,
                  outputs: tuple[PhaseOutput, ...]) -> None:
    layout = run_layout(config.output_dir)
    record = {
        "ts": _now(),
        "phase": phase,
        "outputs": [
            {
                "name": output.name,
                "identity_digest": output.identity_digest,
                "skipped": output.skipped,
                "rows": output.rows,
            }
            for output in outputs
        ],
    }
    with layout.events.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(record) + "\n")


# ------------------------------------------------------------------- identity
def _identity(
    *,
    phase: str,
    scoped: dict[str, Any],
    checkpoint_reference: str,
    checkpoint_digest: str,
    dataset_fingerprint: str,
    parameter_manifest_digest: str,
    loss_convention: dict[str, Any],
    basis_descriptor: dict[str, Any],
    curvature_descriptor: dict[str, Any],
    logra_descriptor: dict[str, Any] | None,
    dtype: str,
    seeds: dict[str, int],
    upstream_digests: dict[str, str],
) -> ArtifactIdentity:
    return ArtifactIdentity(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        producing_command=_producing_command(phase),
        scimt_commit=_scimt_commit(),
        source_commit=SOURCE_COMMIT,
        resolved_config=scoped,
        checkpoint_reference=checkpoint_reference,
        checkpoint_digest=checkpoint_digest,
        dataset_fingerprint=dataset_fingerprint,
        parameter_manifest_digest=parameter_manifest_digest,
        loss_convention=loss_convention,
        basis_descriptor=basis_descriptor,
        curvature_descriptor=curvature_descriptor,
        logra_descriptor=logra_descriptor,
        dtype=dtype,
        seeds=seeds,
        upstream_digests=upstream_digests,
    )


def _bind_identity(directory: Path, identity: ArtifactIdentity) -> None:
    """Bind a non-row artifact directory to one identity (writer semantics):
    a fresh directory records the identity first; an existing directory must
    carry the identical identity, else a refusal naming the fields."""
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / ArtifactWriter.IDENTITY_FILE
    if identity_path.is_file():
        stored = read_identity(identity_path)
        differing = identity.diff(stored)
        if differing:
            raise IdentityMismatchError(
                f"artifact identity mismatch at {directory}: fields differ: "
                f"{differing}; write to a fresh output directory instead of "
                "mutating this artifact"
            )
        return
    from .artifacts import _atomic_write_text

    _atomic_write_text(identity_path, identity.to_json() + "\n")


_UPSTREAM_FIELDS = (
    "resolved_config",
    "parameter_manifest_digest",
    "dataset_fingerprint",
    "checkpoint_digest",
    "basis_descriptor",
    "loss_convention",
    "logra_descriptor",
    "dtype",
    "seeds",
)


def _check_upstream(
    name: str, directory: Path, expected: dict[str, Any]
) -> ArtifactIdentity:
    """Validate a consumed artifact's stored identity against the current
    run's expectations BEFORE loading tensors. Compares only the load-bearing
    binding fields passed in ``expected`` (never producing command or code
    commits, which may legitimately advance between phases)."""
    try:
        stored = read_identity(directory)
    except FileNotFoundError as error:
        raise RunnerError(
            f"upstream artifact {name!r} is missing or was never produced: "
            f"{error}"
        ) from error
    differing = sorted(
        field
        for field, value in expected.items()
        if _canonical(getattr(stored, field)) != _canonical(value)
    )
    if differing:
        raise RunnerError(
            f"upstream artifact {name!r} at {directory} was produced under a "
            f"different identity; fields differ: {differing} — recompute it "
            "under the current configuration or restore the matching config"
        )
    return stored


# ---------------------------------------------------------------- heavy seams
def _load_model(
    checkpoint_dir: str | Path,
    *,
    dtype: str,
    device: str,
    gradient_checkpointing: bool = False,
):
    """Load a causal LM from a resolved local checkpoint dir (lazy heavy).

    ``gradient_checkpointing=True`` arms HF activation checkpointing
    (non-reentrant) so backward passes over long sequences stay
    memory-bounded (run 20260818T102149Z OOM'd a 141 GiB H200 on dense
    seq-8192 activations). Transformers only engages it in train mode, so
    this is inert for eval-mode consumers (notably the kronfluence factor
    fits) and takes effect inside ``gradients.backward_memory_mode`` blocks
    and the estimator's own train-mode loop.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=getattr(torch, dtype),
        local_files_only=True,
    )
    model.to(torch.device(device))
    model.eval()
    _arm_gradient_checkpointing(model, gradient_checkpointing)
    return model


def _arm_gradient_checkpointing(model: Any, requested: bool) -> bool:
    """Arm HF non-reentrant activation checkpointing when supported.

    Returns whether checkpointing was armed. Unsupported models degrade to
    dense activation memory with a warning (execution optimization only —
    never an error, per "warn on degraded").
    """
    if not requested:
        return False
    if getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        return True
    import warnings

    warnings.warn(
        "data.gradient_checkpointing requested but the model does not "
        "support it; running with dense activation memory",
        stacklevel=2,
    )
    return False


def _load_tokenizer(tokenizer_dir: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)


def _model_identifier(model: Any) -> str:
    """The one manifest ``model_name`` convention, shared with the snapshot
    capture path: load-path independent, so manifest digests agree across
    different saves/loads of the identical model (lazy: manifest needs torch).
    """
    from .manifest import stable_model_identifier

    return stable_model_identifier(model)


def _assert_stride() -> None:
    from .losses import SAMPLE_ID_STRIDE

    if SAMPLE_ID_STRIDE != _SAMPLE_ID_STRIDE:  # pragma: no cover - constant
        raise AssertionError("SAMPLE_ID_STRIDE drifted from the runner mirror")


# ------------------------------------------------------------------ resolvers
def _resolve_stages(
    config: AttributionRunConfig, *, require_adam: bool
) -> list[ResolvedStage]:
    return [
        resolve_stage(stage, require_adam=require_adam) for stage in config.stages
    ]


def _resolve_full_checkpoint(declared: Path, *, label: str) -> Path:
    """Resolve a scimt run or plain consolidated HF model checkpoint."""

    from scimt.train.checkpoint import MANIFEST_NAME, Checkpoint

    candidate = declared
    if candidate.is_file() and candidate.name == MANIFEST_NAME:
        candidate = candidate.parent
    if (candidate / MANIFEST_NAME).is_file():
        try:
            checkpoint = Checkpoint.load(candidate)
            state = checkpoint.require_state()
        except (OSError, ValueError, KeyError) as error:
            raise RunnerError(
                f"{label} {declared} is not a usable scimt run: "
                f"{error}"
            ) from error
        state_dir = Path(state)
    else:
        state_dir = candidate
    if not state_dir.is_dir():
        raise RunnerError(f"{label} dir {state_dir} does not exist")
    if (state_dir / "adapter_config.json").exists():
        raise RunnerError(
            f"{label} {state_dir} is an unmerged adapter directory — "
            "merge it into a full checkpoint before attributing"
        )
    if not (state_dir / "config.json").is_file() or not any(
        any(state_dir.glob(pattern)) for pattern in _WEIGHT_GLOBS
    ):
        raise RunnerError(
            f"{label} {state_dir} is not a loadable full checkpoint "
            "(config.json + weights required)"
        )
    return state_dir


def _resolve_query_checkpoint(config: AttributionRunConfig) -> Path:
    """The final query checkpoint: a scimt run dir or plain HF directory."""

    state_dir = _resolve_full_checkpoint(
        Path(config.query.checkpoint.path), label="query checkpoint"
    )
    expected = config.query.checkpoint.expected_digest
    if expected is not None and artifact_digest(state_dir) != expected:
        raise RunnerError(
            f"query checkpoint content digest does not match the declared "
            f"expected_digest {expected}"
        )
    return state_dir


def _resolve_dataset_reference(
    declared: Path,
    *,
    expected_digest: str | None,
    label: str,
) -> tuple[Path, str]:
    from scimt.dataset import Dataset

    manifest_dir = declared.parent if declared.is_file() else declared
    try:
        dataset = Dataset.load(manifest_dir)
    except (OSError, ValueError, TypeError, FileNotFoundError) as error:
        raise RunnerError(
            f"{label} {declared} has no readable dataset manifest "
            f"(dataset.json): {error}"
        ) from error
    data_path = Path(dataset.path)
    if not data_path.exists():
        raise RunnerError(f"{label} manifest points at missing data {data_path}")
    digest = artifact_digest(data_path)
    if expected_digest is not None and digest != expected_digest:
        raise RunnerError(
            f"{label} content digest {digest} does not match the "
            f"declared expected_digest {expected_digest}"
        )
    return data_path, digest


def _resolve_query_dataset(config: AttributionRunConfig) -> tuple[Path, str]:
    return _resolve_dataset_reference(
        Path(config.query.dataset.path),
        expected_digest=config.query.dataset.expected_digest,
        label="query dataset",
    )


def _tokenizer_dir(config: AttributionRunConfig, query_dir: Path) -> Path:
    return Path(config.tokenizer) if config.tokenizer is not None else query_dir


# Tokenizer/chat-template CONTENT participates in artifact identity (design
# §Core interfaces: dataset fingerprints include the tokenizer/chat-template
# digest). These globs cover the HF tokenizer surface: tokenizer.json /
# tokenizer_config.json (which embeds chat_template), chat_template.jinja,
# vocab/merges, sentencepiece models, special/added token maps.
_TOKENIZER_FILE_GLOBS = (
    "tokenizer*",
    "chat_template*",
    "vocab*",
    "merges*",
    "special_tokens*",
    "added_tokens*",
    "spiece*",
    "*.model",
)


def _tokenizer_content_digest(tokenizer_dir: str | Path) -> str:
    """SHA-256 over the tokenizer files' (relative name, byte length, bytes),
    sorted — the same byte-covering construction as ``artifact_digest``, but
    restricted to tokenizer/chat-template content so weight changes in a
    shared checkpoint dir do not invalidate unrelated artifacts. A directory
    with no tokenizer files hashes the empty set (loaders fail loudly on such
    a directory at phase time; the toy fixtures may legitimately be bare)."""
    import hashlib

    tokenizer_dir = Path(tokenizer_dir)
    files = sorted(
        {
            file
            for pattern in _TOKENIZER_FILE_GLOBS
            for file in tokenizer_dir.glob(pattern)
            if file.is_file()
        }
    )
    digest = hashlib.sha256()
    for file in files:
        relative = file.name.encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(source_digest: str, tokenizer_digest: str) -> str:
    """The composite dataset fingerprint bound into every artifact identity:
    raw source bytes PLUS tokenizer/chat-template content (design §Core
    interfaces). Sequence length, seed, target policy, and row reduction —
    the remaining fingerprint inputs the design names — are bound through the
    identity's phase-scoped ``resolved_config`` and ``loss_convention``.
    Mutating tokenizer content in place therefore refuses reuse/resume with
    a focused ``dataset_fingerprint`` mismatch instead of silently mixing
    tokenizations."""
    return _canonical(
        {"source": source_digest, "tokenizer_content": tokenizer_digest}
    )


def _dataset_adapter(
    *,
    objective: str,
    data_path: Path,
    tokenizer: Any,
    config: AttributionRunConfig,
    reduction: str,
    max_sequences: int | None,
    seed: int | None = None,
):
    from .datasets import ChatSFTDataset, PackedMidtrainingDataset

    cls = PackedMidtrainingDataset if objective == "midtraining" else ChatSFTDataset
    return cls(
        data_path,
        tokenizer,
        config.data.sequence_length,
        config.seed if seed is None else seed,
        reduction=reduction,
        max_sequences=max_sequences,
    )


def _build_manifest(model: Any, config: AttributionRunConfig):
    from .manifest import ParameterManifest

    return ParameterManifest.from_model(
        model,
        _model_identifier(model),
        include=list(config.parameters.include),
        exclude=list(config.parameters.exclude),
    )


def _storage_dtype(method_dtype: str) -> str:
    return "float16" if method_dtype == "float16" else "float32"


# ============================================================ estimate Adam ==
def _adam_calibration_dataset(
    config: AttributionRunConfig, data_path: Path, tokenizer: Any
) -> Any:
    """Materialize and validate the common estimator calibration corpus."""

    estimator = config.adam_moment_estimator
    if estimator is None:
        raise RunnerError("Adam calibration dataset requires estimator config")
    from scimt.dataset import Dataset

    dataset_reference = Path(estimator.dataset.path)
    dataset_handle = Dataset.load(
        dataset_reference.parent if dataset_reference.is_file() else dataset_reference
    )
    expected_kind = "docs" if estimator.objective == "midtraining" else "chat"
    if dataset_handle.kind != expected_kind:
        raise RunnerError(
            "Adam moment estimator objective "
            f"{estimator.objective!r} requires dataset kind {expected_kind!r}, "
            f"got {dataset_handle.kind!r}"
        )
    return _dataset_adapter(
        objective=estimator.objective,
        data_path=data_path,
        tokenizer=tokenizer,
        config=config,
        reduction="per_sequence_sum",
        max_sequences=None,
        seed=estimator.seed,
    )


def _adam_estimator_semantics(config: AttributionRunConfig) -> dict[str, Any]:
    """Immutable algorithm choices not represented by user-tunable fields."""

    return {
        "model_mode": "train",
        "stochastic_rng": "same_seed_reset_at_each_checkpoint",
        "buffer_policy": "refuse_any_mutation",
        "world_size": 1,
        "distributed_reduction": "single_process_complete_global_batch",
        "gradient_accumulation": "microbatches_then_one_global_clip",
        "gradient_normalization": "global_selected_target_token_mean",
        "gradient_clipping": "global_l2_all_trainable_parameters",
        "autocast_dtype": (
            None if config.method.dtype == "float32" else config.method.dtype
        ),
        "loss_scaling": "none",
        "optimizer_constructed": False,
        "updates_weights": False,
        "weight_decay_in_gradient": False,
        "sampling": "ordered_without_replacement_common_random_numbers",
        "stores_arithmetic_mean_vector": False,
    }


def _expected_adam_statistics(
    config: AttributionRunConfig,
    resolved: ResolvedStage,
    *,
    model_identifier: str,
    checkpoint_digest: str,
    dataset_fingerprint: str,
    parameter_manifest_digest: str,
    paired_batch_manifest_digest: str,
    code_commit: str,
) -> dict[str, Any]:
    estimator = config.adam_moment_estimator
    if estimator is None:
        raise RunnerError("Adam estimator statistics require estimator config")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_identifier": model_identifier,
        "model_revision": str(resolved.global_step),
        "checkpoint_step": resolved.global_step,
        "checkpoint_digest": checkpoint_digest,
        "dataset_fingerprint": dataset_fingerprint,
        "parameter_manifest_digest": parameter_manifest_digest,
        "statistic": _ADAM_MOMENT_STATISTIC,
        "estimator": _ADAM_MOMENT_ALGORITHM,
        "number_of_gradient_samples": estimator.num_batches,
        "synthetic_estimator_step": estimator.num_batches,
        "beta2": estimator.beta2,
        "bias_correction": 1.0 - estimator.beta2**estimator.num_batches,
        "optimizer_epsilon": estimator.optimizer_epsilon,
        "max_grad_norm": estimator.max_grad_norm,
        "global_batch_size": estimator.global_batch_size,
        "micro_batch_size": estimator.micro_batch_size,
        "paired_batch_manifest_digest": paired_batch_manifest_digest,
        "stores_first_moment": False,
        "stores_optimizer_state": False,
        "code_commit": code_commit,
        **_adam_estimator_semantics(config),
    }


def _validate_adam_statistics(
    statistics: dict[str, Any], expected: dict[str, Any], *, label: str
) -> None:
    """Validate the complete estimator schema, including diagnostic shapes."""

    diagnostics = {
        "target_token_counts",
        "pre_clip_gradient_norms",
        "clip_coefficients",
        "ema_mean_cosine",
        "ema_mean_relative_l2",
    }
    expected_keys = set(expected) | diagnostics
    if set(statistics) != expected_keys:
        missing = sorted(expected_keys - set(statistics))
        unknown = sorted(set(statistics) - expected_keys)
        raise ArtifactIntegrityError(
            f"{label} statistics differ in schema: missing={missing}, "
            f"unknown={unknown}"
        )
    differing = [
        key
        for key, value in expected.items()
        if _canonical(statistics.get(key)) != _canonical(value)
    ]
    if differing:
        raise ArtifactIntegrityError(
            f"{label} statistics differ in {sorted(differing)}"
        )
    sample_count = int(expected["number_of_gradient_samples"])

    def numeric_list(name: str, *, positive: bool) -> None:
        values = statistics[name]
        if not isinstance(values, list) or len(values) != sample_count:
            raise ArtifactIntegrityError(
                f"{label} statistics differ: {name} must contain "
                f"{sample_count} values"
            )
        invalid = any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (float(value) <= 0 if positive else float(value) < 0)
            for value in values
        )
        if invalid:
            qualifier = "positive" if positive else "nonnegative"
            raise ArtifactIntegrityError(
                f"{label} statistics differ: {name} must be finite and "
                + qualifier
            )

    target_counts = statistics["target_token_counts"]
    if (
        not isinstance(target_counts, list)
        or len(target_counts) != sample_count
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in target_counts
        )
    ):
        raise ArtifactIntegrityError(
            f"{label} statistics differ: target_token_counts must contain "
            f"{sample_count} positive integers"
        )
    numeric_list("pre_clip_gradient_norms", positive=False)
    numeric_list("clip_coefficients", positive=True)
    if any(float(value) > 1.0 for value in statistics["clip_coefficients"]):
        raise ArtifactIntegrityError(
            f"{label} statistics differ: clip_coefficients must be <= 1"
        )
    cosine = statistics["ema_mean_cosine"]
    relative = statistics["ema_mean_relative_l2"]
    if (
        isinstance(cosine, bool)
        or not isinstance(cosine, (int, float))
        or not math.isfinite(float(cosine))
        or not -1.0 <= float(cosine) <= 1.0
    ):
        raise ArtifactIntegrityError(
            f"{label} statistics differ: ema_mean_cosine is invalid"
        )
    if (
        isinstance(relative, bool)
        or not isinstance(relative, (int, float))
        or not math.isfinite(float(relative))
        or float(relative) < 0
    ):
        raise ArtifactIntegrityError(
            f"{label} statistics differ: ema_mean_relative_l2 is invalid"
        )


async def estimate_adam(config: AttributionRunConfig) -> PhaseReport:
    """Estimate one paired checkpoint-local Adam second moment per stage.

    Every checkpoint sees the identical ordered calibration batches and the
    same reset stochastic RNG. Models are loaded sequentially, remain frozen,
    and no optimizer is constructed.
    """
    import torch

    from .adam_estimation import estimate_checkpoint_moment, paired_global_batches
    from .artifacts import _atomic_write_text
    from .manifest import flatten_tensors

    estimator = config.adam_moment_estimator
    if estimator is None:
        raise RunnerError(
            "estimate-adam requires an adam_moment_estimator configuration"
        )
    _write_or_check_ledger(config, "estimate-adam")
    resolved_stages = _resolve_stages(config, require_adam=False)
    layout = run_layout(config.output_dir)
    query_dir = _resolve_query_checkpoint(config)
    tokenizer_dir = _tokenizer_dir(config, query_dir)
    tokenizer_digest = _tokenizer_content_digest(tokenizer_dir)
    tokenizer = _load_tokenizer(tokenizer_dir)

    data_path, dataset_digest = _resolve_dataset_reference(
        Path(estimator.dataset.path),
        expected_digest=estimator.dataset.expected_digest,
        label="Adam moment estimator dataset",
    )
    dataset = _adam_calibration_dataset(config, data_path, tokenizer)
    batches = paired_global_batches(
        len(dataset),
        num_batches=estimator.num_batches,
        global_batch_size=estimator.global_batch_size,
        seed=estimator.seed,
    )
    dataset_fingerprint = _dataset_fingerprint(dataset_digest, tokenizer_digest)
    paired_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": "scimt.paired_adam_batches",
        "dataset_digest": dataset_digest,
        "dataset_fingerprint": dataset_fingerprint,
        "tokenized_dataset_fingerprint": dataset.dataset_fingerprint(),
        "tokenizer_digest": tokenizer_digest,
        "objective": estimator.objective,
        "sequence_length": config.data.sequence_length,
        "target_policy": _TARGET_POLICY[estimator.objective],
        "loss_normalization": "global_selected_target_token_mean",
        "sampler": "python_random_sample_without_replacement",
        "seed": estimator.seed,
        "num_batches": estimator.num_batches,
        "global_batch_size": estimator.global_batch_size,
        "batches": [list(batch) for batch in batches],
    }
    paired_path = layout.adam_moments / "paired_batches.json"
    serialized_paired = json.dumps(paired_payload, indent=2, sort_keys=True) + "\n"
    layout.adam_moments.mkdir(parents=True, exist_ok=True)
    if paired_path.is_file():
        try:
            stored_paired = json.loads(paired_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError(
                f"unreadable paired Adam batch manifest {paired_path}: {error}"
            ) from error
        if _canonical(stored_paired) != _canonical(paired_payload):
            raise IdentityMismatchError(
                f"paired Adam batch manifest mismatch at {paired_path}; use a "
                "fresh output directory instead of mutating this artifact"
            )
    else:
        _atomic_write_text(paired_path, serialized_paired)
    paired_digest = artifact_digest(paired_path)

    outputs: list[PhaseOutput] = []
    autocast_dtype = (
        None
        if config.method.dtype == "float32"
        else getattr(torch, config.method.dtype)
    )
    scoped = _scoped_config(config, "estimate-adam")
    for stage, resolved in zip(config.stages, resolved_stages, strict=True):
        name = f"adam_moments/{stage.name}"
        directory = layout.adam_moments / stage.name
        checkpoint_digest = artifact_digest(resolved.checkpoint_dir)
        model = _load_model(
            resolved.checkpoint_dir,
            dtype=config.method.dtype,
            device=config.data.device,
            gradient_checkpointing=config.data.gradient_checkpointing_enabled,
        )
        try:
            manifest = _build_manifest(model, config)
            if manifest.included_numel == 0:
                raise RunnerError(
                    f"{name}: the parameter selection includes no parameters"
                )
            identity = _identity(
                phase="estimate-adam",
                scoped=scoped,
                checkpoint_reference=str(resolved.checkpoint_dir),
                checkpoint_digest=checkpoint_digest,
                dataset_fingerprint=dataset_fingerprint,
                parameter_manifest_digest=manifest.digest(),
                loss_convention={
                    "loss": "causal_lm_cross_entropy",
                    "reduction": "global_selected_target_token_mean",
                    "target_policy": _TARGET_POLICY[estimator.objective],
                    "sampler": "python_random_sample_without_replacement",
                    "sample_id_stride": _SAMPLE_ID_STRIDE,
                },
                basis_descriptor={
                    "coordinates": "raw_gradient_second_moment",
                    "manifest_digest": manifest.digest(),
                    "model_identifier": manifest.model_name,
                },
                curvature_descriptor={
                    "statistic": _ADAM_MOMENT_STATISTIC,
                    "estimator": _ADAM_MOMENT_ALGORITHM,
                    "semantics": _adam_estimator_semantics(config),
                },
                logra_descriptor=None,
                dtype="float32",
                seeds={"run": config.seed, "estimator": estimator.seed},
                upstream_digests={"paired_batches": paired_digest},
            )
            writer = ArtifactWriter(
                directory,
                identity,
                feature_dim=manifest.included_numel,
                rows_per_shard=1,
                feature_dtype="float32",
            )
            statistics_path = directory / _STATISTICS_FILE
            expected_statistics = _expected_adam_statistics(
                config,
                resolved,
                model_identifier=manifest.model_name,
                checkpoint_digest=checkpoint_digest,
                dataset_fingerprint=dataset_fingerprint,
                parameter_manifest_digest=manifest.digest(),
                paired_batch_manifest_digest=paired_digest,
                code_commit=identity.scimt_commit,
            )
            if writer.already_complete:
                if not statistics_path.is_file():
                    raise ArtifactIntegrityError(
                        f"Adam moment artifact {directory} is complete but has "
                        f"no {_STATISTICS_FILE}"
                    )
                _validate_adam_statistics(
                    json.loads(statistics_path.read_text(encoding="utf-8")),
                    expected_statistics,
                    label=name,
                )
                writer.finalize(
                    auxiliary_digests={
                        _STATISTICS_FILE: artifact_digest(statistics_path)
                    }
                )
                outputs.append(
                    PhaseOutput(name, directory, identity.digest(), True, rows=1)
                )
                continue
            if writer.rows_committed:
                if not statistics_path.is_file():
                    raise ArtifactIntegrityError(
                        f"Adam moment artifact {directory} has a committed row "
                        f"but no {_STATISTICS_FILE}"
                    )
                _validate_adam_statistics(
                    json.loads(statistics_path.read_text(encoding="utf-8")),
                    expected_statistics,
                    label=name,
                )
                writer.finalize(
                    auxiliary_digests={
                        _STATISTICS_FILE: artifact_digest(statistics_path)
                    }
                )
                outputs.append(
                    PhaseOutput(name, directory, identity.digest(), False, rows=1)
                )
                continue

            estimate = estimate_checkpoint_moment(
                model,
                dataset,
                manifest,
                batches,
                micro_batch_size=estimator.micro_batch_size,
                beta2=estimator.beta2,
                max_grad_norm=estimator.max_grad_norm,
                device=config.data.device,
                autocast_dtype=autocast_dtype,
                rng_seed=estimator.seed,
            )
            entries = manifest.included_entries()
            diagonal = flatten_tensors(
                entries,
                [estimate.corrected_exp_avg_sq[entry.name] for entry in entries],
            )
            statistics = {
                **expected_statistics,
                "target_token_counts": list(estimate.target_token_counts),
                "pre_clip_gradient_norms": list(estimate.gradient_norms),
                "clip_coefficients": list(estimate.clip_coefficients),
                "ema_mean_cosine": estimate.ema_mean_cosine,
                "ema_mean_relative_l2": estimate.ema_mean_relative_l2,
            }
            _validate_adam_statistics(statistics, expected_statistics, label=name)
            _atomic_write_text(
                statistics_path,
                json.dumps(statistics, indent=2, sort_keys=True) + "\n",
            )
            del estimate
            writer.append(
                features=diagonal.unsqueeze(0),
                sample_ids=torch.zeros(1, dtype=torch.int64),
                sequence_ids=torch.zeros(1, dtype=torch.int64),
                target_positions=torch.zeros(1, dtype=torch.int32),
            )
            writer.finalize(
                auxiliary_digests={
                    _STATISTICS_FILE: artifact_digest(statistics_path)
                }
            )
            outputs.append(
                PhaseOutput(name, directory, identity.digest(), False, rows=1)
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report = PhaseReport("estimate-adam", tuple(outputs))
    _append_event(config, "estimate-adam", report.outputs)
    return report


# ============================================================== fit-factors ==
async def fit_factors(config: AttributionRunConfig) -> PhaseReport:
    """Fit per-segment curvature on seeded samples from each stage's actual
    training distribution. ``curvature: fisher`` writes the raw empirical
    Fisher diagonal as a one-row artifact plus ``statistics.json``;
    ``curvature: ekfac`` fits Kronfluence factors (module partitions from the
    ``factors`` config) under ``<dir>/ekfac`` with a completion marker."""
    method = config.method
    if method.curvature == "ggn":
        raise RunnerError(
            "fit-factors: no fitted GGN segment operator exists — SOURCE "
            "curvature must be 'fisher', 'ekfac', or 'ekfac_adam'; GGN "
            "products are available in the second-order phases "
            "(hessian_kind: ggn)"
        )
    if method.curvature == "ekfac_adam":
        if not config.factors.use_empirical_fisher:
            raise RunnerError(
                "fit-factors: curvature 'ekfac_adam' conditions the "
                "EMPIRICAL Fisher (the same statistic the Adam moments "
                "estimate); set factors.use_empirical_fisher: true"
            )
        if config.adam_moment_estimator is not None:
            paired = (
                run_layout(config.output_dir).adam_moments
                / "paired_batches.json"
            )
            if not paired.is_file():
                raise RunnerError(
                    "fit-factors: curvature 'ekfac_adam' conditions factors "
                    "on stage-local Adam moment estimates, but "
                    "adam_moments/paired_batches.json is missing; "
                    "run estimate-adam first"
                )
    _write_or_check_ledger(config, "fit-factors")
    resolved_stages = _resolve_stages(
        config,
        require_adam=(
            method.curvature == "ekfac_adam"
            and config.adam_moment_estimator is None
        ),
    )
    layout = run_layout(config.output_dir)
    query_dir = _resolve_query_checkpoint(config)
    tokenizer_digest = _tokenizer_content_digest(_tokenizer_dir(config, query_dir))
    tokenizer = _load_tokenizer(_tokenizer_dir(config, query_dir))
    adam_context = None
    if (
        method.curvature == "ekfac_adam"
        and config.adam_moment_estimator is not None
    ):
        adam_context = _estimated_adam_context(config, tokenizer_digest)
    outputs = []
    for stage, resolved in zip(config.stages, resolved_stages, strict=True):
        outputs.append(
            _fit_stage_factors(config, stage, resolved, tokenizer,
                               layout.factors / stage.name, tokenizer_digest,
                               adam_context=adam_context)
        )
    report = PhaseReport("fit-factors", tuple(outputs))
    _append_event(config, "fit-factors", report.outputs)
    return report


def _fit_config_payload(config: AttributionRunConfig) -> dict[str, Any]:
    factors = config.factors
    payload = {
        "samples": factors.samples,
        "seed": config.seed,
        "source_batch_size": factors.source_batch_size,
        "batch_size": factors.fit_batch_size,
        "max_positions_per_sequence": factors.max_positions_per_sequence,
        "min_position_gap": factors.min_position_gap,
        "use_empirical_fisher": factors.use_empirical_fisher,
        "covariance_module_partitions": factors.covariance_module_partitions,
        "lambda_module_partitions": factors.lambda_module_partitions,
        "eigendecomposition_dtype": factors.eigendecomposition_dtype,
    }
    if factors.eigh_device is not None:
        payload["eigh_device"] = factors.eigh_device
    return payload


def _fit_stage_factors(
    config: AttributionRunConfig,
    stage: AttributionStage,
    resolved: ResolvedStage,
    tokenizer: Any,
    directory: Path,
    tokenizer_digest: str,
    adam_context: tuple[str, str] | None = None,
) -> PhaseOutput:
    _assert_stride()
    name = f"factors/{stage.name}"
    method = config.method
    checkpoint_digest = artifact_digest(resolved.checkpoint_dir)
    model = _load_model(
        resolved.checkpoint_dir,
        dtype=method.dtype,
        device=config.data.device,
        gradient_checkpointing=config.data.gradient_checkpointing_enabled,
    )
    manifest = _build_manifest(model, config)
    basis_descriptor: dict[str, Any] = {
        "coordinates": "raw",
        "manifest_digest": manifest.digest(),
    }
    upstream_digests: dict[str, str] = {}
    conditioner = None
    if method.curvature == "ekfac_adam":
        if adam_context is not None:
            estimator_fingerprint, paired_digest = adam_context
            payload = _load_estimated_stage_adam_payload(
                config,
                stage,
                resolved,
                manifest.digest(),
                estimator_fingerprint=estimator_fingerprint,
                paired_digest=paired_digest,
            )
            upstream_digests = {
                "adam/paired_batches": paired_digest,
                **payload.upstream,
            }
            moment_identity_digest = payload.descriptor[
                "moment_identity_digest"
            ]
        else:
            payload = _load_captured_stage_adam_payload(
                config, stage, resolved, manifest.digest()
            )
            upstream_digests = dict(payload.upstream)
            moment_identity_digest = payload.descriptor[
                "optimizer_manifest_digest"
            ]
        from .metrics import DiagonalMetric

        conditioning_metric = DiagonalMetric.from_adam_second_moment(
            payload.statistics,
            payload.values,
            optimizer_epsilon=payload.optimizer_epsilon,
            damping=float(method.conditioning_damping),
        )
        if conditioning_metric.diagonal.numel() != manifest.included_numel:
            raise RunnerError(
                f"{name}: stage-local Adam moment dimension "
                f"({conditioning_metric.diagonal.numel()}) does not match "
                f"the parameter manifest ({manifest.included_numel})"
            )
        basis_descriptor = {
            "coordinates": "adam_stage_local",
            "stage": stage.name,
            "moment_identity_digest": moment_identity_digest,
            "conditioning_damping": float(method.conditioning_damping),
            "optimizer_epsilon": payload.optimizer_epsilon,
            "geometry": (
                "A_l=(sqrt(v_hat_l)+optimizer_epsilon_l"
                "+conditioning_damping)^-1/2"
            ),
        }
        import torch

        from .ekfac import EKFACConditioner

        conditioner = EKFACConditioner(
            values=conditioning_metric.diagonal.to(torch.float32),
            provenance={
                "kind": "adam_stage_local",
                "statistic": payload.statistics["statistic"],
                "moment_identity_digest": moment_identity_digest,
                "optimizer_epsilon": payload.optimizer_epsilon,
                "conditioning_damping": float(method.conditioning_damping),
            },
        )
    scoped = _scoped_config(config, "fit-factors", stage.name)
    identity = _identity(
        phase="fit-factors",
        scoped=scoped,
        checkpoint_reference=str(resolved.checkpoint_dir),
        checkpoint_digest=checkpoint_digest,
        dataset_fingerprint=_dataset_fingerprint(
            resolved.dataset_digest, tokenizer_digest
        ),
        parameter_manifest_digest=manifest.digest(),
        loss_convention={
            "loss": "causal_lm_cross_entropy",
            "reduction": "sampled_token_sum",
            "target_policy": _TARGET_POLICY[stage.objective],
            "sampler": "build_ekfac_sample_items",
            "sample_id_stride": _SAMPLE_ID_STRIDE,
        },
        basis_descriptor=basis_descriptor,
        curvature_descriptor={
            "method": method.curvature,
            "fit": _fit_config_payload(config),
        },
        logra_descriptor=None,
        dtype=method.dtype,
        seeds={"run": config.seed},
        upstream_digests=upstream_digests,
    )
    dataset = _dataset_adapter(
        objective=stage.objective,
        data_path=Path(resolved.dataset.path),
        tokenizer=tokenizer,
        config=config,
        reduction=method.row_reduction,
        max_sequences=config.data.max_stage_sequences,
    )
    if method.curvature == "fisher":
        return _fit_fisher_diagonal(
            config, name, directory, identity, model, manifest, dataset,
            resolved, checkpoint_digest,
        )
    return _fit_ekfac_factors(config, name, directory, identity, model,
                              manifest, dataset, conditioner=conditioner)


def _factor_input_ids(item: dict[str, Any], device: str):
    """Place one sampled Fisher sequence on the model's configured device."""

    return item["input_ids"].unsqueeze(0).to(device)


def _fit_fisher_diagonal(
    config, name, directory, identity, model, manifest, dataset, resolved,
    checkpoint_digest,
) -> PhaseOutput:
    import torch

    from .ekfac import build_ekfac_sample_items

    writer = ArtifactWriter(
        directory,
        identity,
        feature_dim=max(1, manifest.included_numel),
        rows_per_shard=1,
        feature_dtype="float32",
    )
    if writer.already_complete:
        if not (directory / _STATISTICS_FILE).is_file():
            raise ArtifactIntegrityError(
                f"factor artifact {directory} is complete but has no "
                f"{_STATISTICS_FILE}"
            )
        return PhaseOutput(name, directory, identity.digest(), True, rows=1)
    if manifest.included_numel == 0:
        raise RunnerError(
            f"{name}: the parameter selection includes no parameters"
        )
    if writer.rows_committed:
        # The single statistic row was sealed but the manifest write was
        # interrupted: publish the manifest, never re-append (a duplicate
        # sample_id would poison every later read).
        writer.finalize()
        return PhaseOutput(name, directory, identity.digest(), False, rows=1)
    items = build_ekfac_sample_items(dataset, _fit_config_payload(config))
    if not items:
        raise RunnerError(
            f"{name}: no factor samples could be drawn from the stage "
            "dataset — nothing to fit"
        )
    named = dict(model.named_parameters(remove_duplicate=False))
    entries = manifest.included_entries()
    accumulator = {
        entry.name: torch.zeros(entry.numel, dtype=torch.float64)
        for entry in entries
    }
    for item in items:
        ids = _factor_input_ids(item, config.data.device)
        position = int(item["position"])
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits[0, position - 1 : position].float(),
            ids[0, position : position + 1],
            reduction="sum",
        )
        loss.backward()
        for entry in entries:
            gradient = named[entry.name].grad
            if gradient is not None:
                accumulator[entry.name].add_(
                    gradient.detach().reshape(-1).cpu().double().square()
                )
    model.zero_grad(set_to_none=True)
    diagonal = torch.cat([accumulator[e.name] / len(items) for e in entries])
    statistics = {
        "model_identifier": manifest.model_name,
        "model_revision": str(resolved.global_step),
        "dataset_fingerprint": identity.dataset_fingerprint,
        "parameter_manifest_digest": manifest.digest(),
        "statistic": "empirical_fisher_diagonal",
        "number_of_gradient_samples": len(items),
        "code_commit": _scimt_commit(),
        "estimator": "full",
        "logra": None,
        "checkpoint_digest": checkpoint_digest,
    }
    from .artifacts import _atomic_write_text

    _atomic_write_text(
        directory / _STATISTICS_FILE,
        json.dumps(statistics, indent=2, sort_keys=True) + "\n",
    )
    writer.append(
        features=diagonal.to(torch.float32).unsqueeze(0),
        sample_ids=torch.zeros(1, dtype=torch.int64),
        sequence_ids=torch.zeros(1, dtype=torch.int64),
        target_positions=torch.zeros(1, dtype=torch.int32),
    )
    writer.finalize()
    return PhaseOutput(name, directory, identity.digest(), False, rows=1)


def _fit_ekfac_factors(
    config, name, directory, identity, model, manifest, dataset,
    conditioner=None,
) -> PhaseOutput:
    from .ekfac import fit_ekfac, load_ekfac

    curvature = config.method.curvature
    if (conditioner is not None) != (curvature == "ekfac_adam"):
        raise RunnerError(
            f"{name}: a conditioner is required exactly when curvature is "
            f"'ekfac_adam' (got curvature {curvature!r})"
        )
    _bind_identity(directory, identity)
    marker = directory / _FACTORS_COMPLETE_FILE
    ekfac_dir = directory / "ekfac"
    if marker.is_file():
        completion = json.loads(marker.read_text(encoding="utf-8"))
        if completion.get("identity_digest") != identity.digest():
            raise ArtifactIntegrityError(
                f"factor completion marker at {directory} belongs to a "
                "different artifact identity"
            )
        if completion.get("curvature") != curvature:
            raise ArtifactIntegrityError(
                f"factor completion marker at {directory} records curvature "
                f"{completion.get('curvature')!r}, not {curvature!r} — "
                "raw and Adam-conditioned factor artifacts are never "
                "interchangeable"
            )
        factors = load_ekfac(ekfac_dir, manifest, expected_mode=curvature)
        if completion.get("snapshot") != factors.snapshot:
            raise ArtifactIntegrityError(
                f"EK-FAC factor bytes under {ekfac_dir} changed after "
                "completion was recorded"
            )
        return PhaseOutput(name, directory, identity.digest(), True)
    if conditioner is None:
        factors = fit_ekfac(
            model, dataset, manifest, _fit_config_payload(config), ekfac_dir
        )
    else:
        factors = fit_ekfac(
            model, dataset, manifest, _fit_config_payload(config), ekfac_dir,
            conditioner=conditioner,
        )
    from .artifacts import _atomic_write_text

    _atomic_write_text(
        marker,
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "identity_digest": identity.digest(),
                "snapshot": factors.snapshot,
                "curvature": curvature,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return PhaseOutput(name, directory, identity.digest(), False)


# ============================================================= gradient rows ==
def _batch_expected_rows(batch: Any, reduction: str) -> int:
    if reduction == "per_token":
        return int(batch.target_mask.sum())
    return int(batch.input_ids.shape[0])


def _write_rows(
    *,
    writer: ArtifactWriter,
    model: Any,
    manifest: Any,
    dataset: Any,
    reduction: str,
    device: str,
    batch_size: int,
    vjp_chunk_size: int,
) -> int:
    """Stream gradient rows into the writer, resuming after the last
    committed row (dataset iteration is deterministic; already-committed rows
    are skipped without touching the model where whole batches are covered)."""
    from .gradients import BatchedVJPBackend, backward_memory_mode
    from .losses import CausalLMLossAdapter

    adapter = CausalLMLossAdapter(model, reduction=reduction, device=device)
    backend = BatchedVJPBackend(model, manifest)
    committed = writer.rows_committed
    seen = 0
    with backward_memory_mode(
        model, getattr(model, "is_gradient_checkpointing", False)
    ):
        for batch in dataset.iter_batches(batch_size):
            expected = _batch_expected_rows(batch, reduction)
            if seen + expected <= committed:
                seen += expected
                continue
            loss_batch = adapter.per_datapoint_losses(batch)
            if int(loss_batch.losses.numel()) != expected:
                raise ArtifactIntegrityError(
                    "row bookkeeping mismatch: expected "
                    f"{expected} rows for this batch, got "
                    f"{int(loss_batch.losses.numel())}"
                )
            rows = backend.rows(loss_batch.losses, chunk_size=vjp_chunk_size)
            drop = max(0, committed - seen)
            writer.append(
                features=rows[drop:],
                sample_ids=loss_batch.sample_ids[drop:],
                sequence_ids=loss_batch.sequence_ids[drop:],
                target_positions=loss_batch.target_positions[drop:],
            )
            seen += expected
    writer.finalize()
    return writer.rows_committed


def _prepare_logra(
    config: AttributionRunConfig, model: Any, base_manifest: Any
) -> tuple[Any, dict[str, Any], dict[str, str]]:
    """Inject configured LoGra projections; returns (logra manifest,
    logra descriptor, upstream digests)."""
    from .logra import (
        ProjectionArtifacts,
        inject_logra,
        pca_projections,
        projection_descriptor,
    )
    from .manifest import ParameterManifest

    logra = config.method.logra
    upstream: dict[str, str] = {}
    projections = None
    if logra.init == "pca":
        import torch

        factors_dir = Path(logra.ekfac_factors)
        target = re.compile(logra.targets)
        names = sorted(
            module_name
            for module_name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and target.fullmatch(module_name)
        )
        if not names:
            raise RunnerError(
                "logra: no nn.Linear module paths match the targets regex"
            )
        projections = pca_projections(
            factors_dir, names, logra.rank, base_manifest
        )
        from .ekfac import load_ekfac

        upstream["logra_ekfac_factors"] = load_ekfac(
            factors_dir, base_manifest
        ).snapshot
    elif logra.init == "artifact":
        artifact_dir = Path(logra.projections)
        stored_manifest = ParameterManifest.load(artifact_dir)
        metadata = json.loads(
            (artifact_dir / ProjectionArtifacts.META).read_text(encoding="utf-8")
        )
        artifacts = ProjectionArtifacts.load(
            artifact_dir,
            expected_manifest=stored_manifest,
            expected_descriptor=metadata.get("descriptor"),
        )
        projections = {name: artifacts[name] for name in artifacts}
        upstream["logra_projections"] = artifacts.content_digest
    report = inject_logra(
        model,
        rank=logra.rank,
        seed=logra.seed,
        init=logra.init,
        targets=logra.targets,
        projections=projections,
    )
    logra_manifest = ParameterManifest.from_model(
        model, _model_identifier(model), include=[report.include_regex]
    )
    if logra.init == "artifact":
        expected_digest = upstream["logra_projections"]
        if report.projection_digest != json.loads(
            (Path(logra.projections) / ProjectionArtifacts.META).read_text(
                encoding="utf-8"
            )
        ).get("projection_digest"):
            raise RunnerError(
                "logra: injected projections do not match the artifact's "
                f"recorded projection digest (artifact {expected_digest})"
            )
    return logra_manifest, projection_descriptor(report), upstream


def _query_group_labels(data_path: Path) -> list[str]:
    """Per-row ``group`` labels for aggregated queries (all-or-nothing).

    Every JSONL row must carry a string ``group`` field; missing or
    non-string fields are refusals naming the offending line numbers —
    a silently ungrouped row would bias the group means."""
    labels: list[str] = []
    bad: list[int] = []
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line).get("group")
            if isinstance(value, str) and value:
                labels.append(value)
            else:
                bad.append(line_number)
                labels.append("")
    if not labels:
        raise RunnerError("query aggregate: the query dataset is empty")
    if len(bad) == len(labels):
        raise RunnerError(
            "query aggregate 'group_mean' is set but no query row carries a "
            "string 'group' field"
        )
    if bad:
        raise RunnerError(
            "query aggregate 'group_mean' requires a string 'group' field on "
            f"every row; missing/invalid on JSONL lines {bad}"
        )
    return labels


def _write_aggregated_query_rows(
    *,
    writer: "ArtifactWriter",
    model: Any,
    manifest: Any,
    dataset: Any,
    source_groups: list[str],
    group_names: list[str],
    reduction: str,
    device: str,
    batch_size: int,
    vjp_chunk_size: int,
    directory: Path,
) -> int:
    """One mean gradient row per group (sorted group-name order), fp64
    accumulation. Not shard-resumable: the whole artifact is a handful of
    rows written in a single append + finalize."""
    import torch

    from .artifacts import _atomic_write_text
    from .gradients import BatchedVJPBackend, backward_memory_mode
    from .losses import CausalLMLossAdapter

    adapter = CausalLMLossAdapter(model, reduction=reduction, device=device)
    backend = BatchedVJPBackend(model, manifest)
    group_index = {name: i for i, name in enumerate(group_names)}
    source_rows = getattr(dataset, "source_rows", None)
    if source_rows is None:
        raise RunnerError(
            "query aggregate 'group_mean' requires a chat dataset that "
            "tracks source rows"
        )
    dropped = sorted(set(range(len(source_groups))) - set(source_rows))
    if dropped:
        raise RunnerError(
            "query aggregate 'group_mean': query rows were dropped during "
            f"tokenization (source row indexes {dropped}) — a silently "
            "missing row would bias its group mean; fix or remove those rows"
        )
    width = max(1, manifest.included_numel)
    sums = torch.zeros((len(group_names), width), dtype=torch.float64)
    counts = [0] * len(group_names)
    chunk_cpu = loss_batch = None
    with backward_memory_mode(
        model, getattr(model, "is_gradient_checkpointing", False)
    ):
        for batch in dataset.iter_batches(batch_size):
            loss_batch = adapter.per_datapoint_losses(batch)
            sequences = loss_batch.sequence_ids.tolist()
            offset = 0
            # Per-chunk consumption: a full-coverage row is ~4·P bytes on
            # device, so the [N, P] materialization (and its cat copy) that
            # backend.rows() implies cannot coexist with the model — move
            # each chunk to the CPU accumulators and free it before the next
            # (pod run 20260819T095144Z OOM'd on the materialized path here).
            for chunk in backend.iter_row_chunks(
                loss_batch.losses, chunk_size=vjp_chunk_size
            ):
                chunk_cpu = chunk.detach().to(device="cpu", dtype=torch.float64)
                del chunk
                chunk_sequences = sequences[offset : offset + len(chunk_cpu)]
                for row, sequence in zip(chunk_cpu, chunk_sequences, strict=True):
                    index = group_index[source_groups[source_rows[sequence]]]
                    sums[index] += row
                    counts[index] += 1
                offset += len(chunk_cpu)
                # `row` is a view whose base pins the whole [c, P] chunk
                # storage (~86 GB at full coverage) through finalization if
                # left bound (PR #530 review finding).
                row = None
            if offset != len(sequences):
                raise ArtifactIntegrityError(
                    "aggregated query bookkeeping mismatch: expected "
                    f"{len(sequences)} rows for this batch, got {offset}"
                )
            # The final chunk/batch bindings otherwise survive the loop; at
            # full coverage each is a [c, P] fp64 (~86 GB) that must not be
            # alive during finalization (pod run 20260819T095144Z was
            # OOM-killed at exactly that point).
            chunk_cpu = None
            loss_batch = None
        del chunk_cpu, loss_batch
    empty = [name for name, count in zip(group_names, counts, strict=True)
             if count == 0]
    if empty:
        raise RunnerError(
            f"query aggregate 'group_mean': groups with zero rows: {empty}"
        )
    # Finalize in place and per group: `(sums / counts).to(fp32)` allocates a
    # second [G, P] fp64 plus a [G, P] fp32 while `sums` is still alive —
    # ~428 GB at full coverage against a ~503 GB cgroup, the third
    # build-queries OOM. In-place division then one bounded [1, P] fp32
    # staging row per group keeps the peak at sums + one row (~214 GB).
    # Identical bytes: division is elementwise, fp32 casts are per-element,
    # and the writer buffers both rows into the same single shard.
    sums /= torch.tensor(counts, dtype=torch.float64).unsqueeze(1)
    ids = torch.arange(len(group_names), dtype=torch.int64)
    for index in range(len(group_names)):
        row32 = sums[index].to(torch.float32).unsqueeze(0)
        writer.append(
            features=row32,
            sample_ids=ids[index : index + 1],
            sequence_ids=ids[index : index + 1],
            target_positions=torch.zeros(1, dtype=torch.int32),
        )
        del row32
    del sums
    writer.finalize()
    _atomic_write_text(
        directory / QUERY_GROUPS_FILE,
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "aggregate": "group_mean",
                "groups": [
                    {"index": i, "name": name, "n_rows": counts[i]}
                    for i, name in enumerate(group_names)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return writer.rows_committed


def _row_phase(
    config: AttributionRunConfig,
    *,
    phase: str,
    name: str,
    directory: Path,
    checkpoint_dir: Path,
    objective: str,
    data_path: Path,
    dataset_fingerprint: str,
    tokenizer: Any,
    max_sequences: int | None,
    # (stage, resolved) for per-stage rows names the identity scope. None for
    # the query phase.
    stage_context: tuple[AttributionStage, ResolvedStage] | None,
) -> PhaseOutput:
    _assert_stride()
    method = config.method
    aggregate = (
        config.query.aggregate if phase == "build-queries" else None
    )
    source_groups: list[str] = []
    group_names: list[str] = []
    rows_per_shard = config.data.rows_per_shard
    if aggregate is not None:
        source_groups = _query_group_labels(data_path)
        group_names = sorted(set(source_groups))
        # The whole artifact is one shard: a single append + finalize keeps
        # the aggregated write atomic (execution geometry, not identity).
        rows_per_shard = max(rows_per_shard, len(group_names))
    checkpoint_digest = artifact_digest(checkpoint_dir)
    model = _load_model(
        checkpoint_dir,
        dtype=method.dtype,
        device=config.data.device,
        gradient_checkpointing=config.data.gradient_checkpointing_enabled,
    )
    base_manifest = _build_manifest(model, config)
    seeds = {"run": config.seed}
    upstream: dict[str, str] = {}
    if method.logra is not None:
        manifest, logra_descriptor, upstream = _prepare_logra(
            config, model, base_manifest
        )
        seeds["logra"] = method.logra.seed
        basis_descriptor = {
            "coordinates": "logra_B",
            "manifest_digest": manifest.digest(),
            "projection_digest": logra_descriptor["projection_digest"],
            "rank": logra_descriptor["rank"],
            "init": logra_descriptor["init"],
        }
    else:
        manifest = base_manifest
        logra_descriptor = None
        basis_descriptor = {
            "coordinates": "raw",
            "manifest_digest": manifest.digest(),
        }
    identity = _identity(
        phase=phase,
        scoped=_scoped_config(config, phase,
                              stage_context[0].name if stage_context else None),
        checkpoint_reference=str(checkpoint_dir),
        checkpoint_digest=checkpoint_digest,
        dataset_fingerprint=dataset_fingerprint,
        parameter_manifest_digest=manifest.digest(),
        loss_convention=_loss_convention(objective, method.row_reduction),
        basis_descriptor=basis_descriptor,
        curvature_descriptor={"kind": "gradient_rows"},
        logra_descriptor=logra_descriptor,
        dtype=method.dtype,
        seeds=seeds,
        upstream_digests=upstream,
    )
    writer = ArtifactWriter(
        directory,
        identity,
        feature_dim=max(1, manifest.included_numel),
        rows_per_shard=rows_per_shard,
        feature_dtype=_storage_dtype(method.dtype),
    )
    if writer.already_complete:
        return PhaseOutput(name, directory, identity.digest(), True,
                           rows=writer.rows_committed)
    if manifest.included_numel == 0:
        raise RunnerError(f"{name}: the parameter selection includes no parameters")
    if method.logra is not None:
        from .logra import ProjectionArtifacts

        projections_dir = directory / "projections"
        if not (projections_dir / ProjectionArtifacts.META).is_file():
            ProjectionArtifacts.save(
                model, projections_dir, manifest=manifest,
                descriptor=logra_descriptor,
            )
    dataset = _dataset_adapter(
        objective=objective,
        data_path=data_path,
        tokenizer=tokenizer,
        config=config,
        reduction=method.row_reduction,
        max_sequences=max_sequences,
    )
    if aggregate is not None:
        rows = _write_aggregated_query_rows(
            writer=writer,
            model=model,
            manifest=manifest,
            dataset=dataset,
            source_groups=source_groups,
            group_names=group_names,
            reduction=method.row_reduction,
            device=config.data.device,
            batch_size=config.data.batch_size,
            vjp_chunk_size=config.data.vjp_chunk_size,
            directory=directory,
        )
    else:
        rows = _write_rows(
            writer=writer,
            model=model,
            manifest=manifest,
            dataset=dataset,
            reduction=method.row_reduction,
            device=config.data.device,
            batch_size=config.data.batch_size,
            vjp_chunk_size=config.data.vjp_chunk_size,
        )
    return PhaseOutput(name, directory, identity.digest(), False, rows=rows)


async def compute_rows(config: AttributionRunConfig) -> PhaseReport:
    """Per-example train gradient rows for every stage, resumable by shard."""
    _write_or_check_ledger(config, "compute-rows")
    resolved_stages = _resolve_stages(config, require_adam=False)
    layout = run_layout(config.output_dir)
    query_dir = _resolve_query_checkpoint(config)
    tokenizer_digest = _tokenizer_content_digest(_tokenizer_dir(config, query_dir))
    tokenizer = _load_tokenizer(_tokenizer_dir(config, query_dir))
    outputs = []
    for stage, resolved in zip(config.stages, resolved_stages, strict=True):
        outputs.append(
            _row_phase(
                config,
                phase="compute-rows",
                name=f"rows/{stage.name}",
                directory=layout.rows / stage.name,
                checkpoint_dir=resolved.checkpoint_dir,
                objective=stage.objective,
                data_path=Path(resolved.dataset.path),
                dataset_fingerprint=_dataset_fingerprint(
                    resolved.dataset_digest, tokenizer_digest
                ),
                tokenizer=tokenizer,
                max_sequences=config.data.max_stage_sequences,
                stage_context=(stage, resolved),
            )
        )
    report = PhaseReport("compute-rows", tuple(outputs))
    _append_event(config, "compute-rows", report.outputs)
    return report


async def build_queries(config: AttributionRunConfig) -> PhaseReport:
    """Measurement-loss gradient rows at the final query checkpoint."""
    _write_or_check_ledger(config, "build-queries")
    layout = run_layout(config.output_dir)
    query_dir = _resolve_query_checkpoint(config)
    data_path, dataset_digest = _resolve_query_dataset(config)
    tokenizer_digest = _tokenizer_content_digest(_tokenizer_dir(config, query_dir))
    tokenizer = _load_tokenizer(_tokenizer_dir(config, query_dir))
    output = _row_phase(
        config,
        phase="build-queries",
        name="queries",
        directory=layout.queries,
        checkpoint_dir=query_dir,
        objective=config.query.objective,
        data_path=data_path,
        dataset_fingerprint=_dataset_fingerprint(dataset_digest, tokenizer_digest),
        tokenizer=tokenizer,
        max_sequences=config.data.max_query_sequences,
        stage_context=None,
    )
    report = PhaseReport("build-queries", (output,))
    _append_event(config, "build-queries", report.outputs)
    return report


# ============================================================== score-source ==
def _require_scorable_method(config: AttributionRunConfig) -> None:
    method = config.method
    if method.curvature == "ggn":
        raise RunnerError(
            "score-source: no fitted GGN segment operator exists — SOURCE "
            "curvature must be 'fisher', 'ekfac', or 'ekfac_adam'; GGN "
            "products live in the second-order phases (hessian_kind: ggn)"
        )
    if method.basis == "ekfac":
        raise RunnerError(
            "score-source: basis 'ekfac' is not supported by the runner — an "
            "exact EK-FAC-basis transport across differently-fitted segments "
            "is not representable with EK-FAC factors. Use basis 'raw' with "
            "curvature 'ekfac', or a diagonal basis ('fisher'/'adam')."
        )
    if method.logra is not None:
        raise RunnerError(
            "score-source: SOURCE scoring over LoGra-projected rows is not "
            "wired yet — LoGra rows serve whitened grad-dot workflows; run "
            "score-source on unprojected rows (method.logra: null)"
        )
    if method.curvature == "ekfac_adam":
        if method.basis != "adam":
            raise RunnerError(
                "score-source: curvature 'ekfac_adam' factors live in "
                "stage-local Adam coordinates; only basis 'adam' rows and "
                "queries can consume them"
            )
    elif method.basis == "adam" and method.curvature != "fisher":
        raise RunnerError(
            "score-source: basis 'adam' transports rows through a diagonal "
            "metric, which requires diagonal curvature (curvature: fisher) — "
            "EK-FAC factors cannot be exactly transported into a diagonal "
            "basis; for EK-FAC-quality curvature in Adam coordinates fit "
            "conditioned factors with curvature 'ekfac_adam'"
        )
    elif method.basis == "fisher" and method.curvature != "fisher":
        raise RunnerError(
            "score-source: basis 'fisher' transports rows through a diagonal "
            "metric, which requires diagonal curvature (curvature: fisher) — "
            "EK-FAC factors cannot be exactly transported into a diagonal "
            "basis, and a Fisher-diagonal analogue of the Adam-conditioned "
            "EK-FAC fit is out of scope (no 'ekfac_fisher' mode exists)"
        )


def _load_factor_operator(
    config: AttributionRunConfig,
    stage: AttributionStage,
    directory: Path,
    shared_manifest_digest: str,
    expected_fingerprint: str,
):
    """Load a stage's fitted curvature: (kind, payload). Validates identity
    scope and digests (including the tokenizer-bearing dataset fingerprint)
    before touching tensors."""
    expected = {
        "resolved_config": _scoped_config(config, "fit-factors", stage.name),
        "parameter_manifest_digest": shared_manifest_digest,
        "dataset_fingerprint": expected_fingerprint,
        "dtype": config.method.dtype,
        "seeds": {"run": config.seed},
    }
    stored = _check_upstream(f"factors/{stage.name}", directory, expected)
    if config.method.curvature == "fisher":
        manifest = ShardManifest.load(directory, expected_identity=stored)
        statistics_path = directory / _STATISTICS_FILE
        if not statistics_path.is_file():
            raise ArtifactIntegrityError(
                f"factor artifact {directory} has no {_STATISTICS_FILE}"
            )
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        if statistics.get("parameter_manifest_digest") != shared_manifest_digest:
            raise RunnerError(
                f"factors/{stage.name}: statistics parameter-manifest digest "
                "does not match the shared run manifest"
            )
        diagonal = manifest.read_rows(directory)["features"][0].float()
        return "fisher", (statistics, diagonal), stored
    marker = directory / _FACTORS_COMPLETE_FILE
    if not marker.is_file():
        raise RunnerError(
            f"factors/{stage.name}: EK-FAC factor artifact at {directory} is "
            "incomplete (no completion marker) — re-run fit-factors"
        )
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if completion.get("identity_digest") != stored.digest():
        raise ArtifactIntegrityError(
            f"factors/{stage.name}: completion marker does not match the "
            "stored artifact identity"
        )
    curvature = config.method.curvature
    if completion.get("curvature") != curvature:
        raise ArtifactIntegrityError(
            f"factors/{stage.name}: completion marker records curvature "
            f"{completion.get('curvature')!r}, not {curvature!r} — raw and "
            "Adam-conditioned factor artifacts are never interchangeable"
        )
    from .ekfac import load_ekfac
    from .manifest import ParameterManifest

    ekfac_dir = directory / "ekfac"
    factor_manifest = ParameterManifest.load(ekfac_dir)
    if factor_manifest.digest() != shared_manifest_digest:
        raise RunnerError(
            f"factors/{stage.name}: EK-FAC manifest digest does not match "
            "the shared run manifest"
        )
    factors = load_ekfac(ekfac_dir, factor_manifest, expected_mode=curvature)
    if completion.get("snapshot") != factors.snapshot:
        raise ArtifactIntegrityError(
            f"factors/{stage.name}: EK-FAC factor bytes changed after "
            "completion was recorded"
        )
    metadata = json.loads(
        (ekfac_dir / "ekfac_meta.json").read_text(encoding="utf-8")
    )
    preconditioner = metadata.get("preconditioner")
    if curvature == "ekfac_adam":
        if not isinstance(preconditioner, dict):
            raise ArtifactIntegrityError(
                f"factors/{stage.name}: conditioned EK-FAC artifact has no "
                "preconditioner block in ekfac_meta.json"
            )
        if preconditioner.get("kind") != "adam_stage_local":
            raise ArtifactIntegrityError(
                f"factors/{stage.name}: preconditioner kind "
                f"{preconditioner.get('kind')!r} is not 'adam_stage_local'"
            )
        recorded_damping = preconditioner.get("conditioning_damping")
        if recorded_damping != float(config.method.conditioning_damping):
            raise RunnerError(
                f"factors/{stage.name}: factor artifact was conditioned "
                f"with conditioning_damping {recorded_damping!r}, but the "
                "config declares "
                f"{float(config.method.conditioning_damping)!r} — refit"
            )
        return (
            "ekfac_adam",
            (factors, factor_manifest, preconditioner),
            stored,
        )
    if preconditioner is not None:
        raise ArtifactIntegrityError(
            f"factors/{stage.name}: raw EK-FAC mode cannot consume an "
            "Adam-conditioned factor artifact"
        )
    return "ekfac", (factors, factor_manifest), stored


def _shifted_curvature(inner: Any, shift: float) -> Any:
    """Wrap a :class:`CurvatureOperator` so every applied eigenvalue function
    sees ``eigenvalues + shift`` — exactly ``H + shift*I`` in the inner
    operator's eigenbasis (PSD preserved for ``shift >= 0``). Defined lazily
    because subclassing requires importing ``source`` (torch)."""
    from .source import CurvatureOperator

    class _Shifted(CurvatureOperator):
        def __init__(self) -> None:
            super().__init__(inner.dimension, inner.basis_descriptor)

        def apply_fn(self, rows, fn):
            return inner.apply_fn(rows, lambda ev: fn(ev + shift))

    return _Shifted()


def _fisher_basis_metric(
    config: AttributionRunConfig,
    factor_payloads: dict[str, Any],
    damping: float,
):
    """The global Fisher diagonal basis metric from the final stage."""
    from .metrics import DiagonalMetric

    last_stage = config.stages[-1]
    statistics, diagonal = factor_payloads[last_stage.name]
    metric = DiagonalMetric.from_statistics(
        statistics,
        diagonal,
        exponent=-0.5,
        damping=damping,
    )
    extras = {
        "coordinates": "fisher_diag",
        "source_stage": last_stage.name,
        # The metric offset is epsilon + damping; damping is recorded per
        # sweep point in the curvature descriptor, epsilon here.
        "epsilon": metric.epsilon,
    }
    return metric, extras


@dataclass(frozen=True)
class _StageAdamBasisPayload:
    """One validated checkpoint-local second moment, loaded once per score."""

    stage_name: str
    statistics: dict[str, Any]
    values: Any
    optimizer_epsilon: float
    descriptor: dict[str, Any]
    upstream: dict[str, str]


def _source_curvature_descriptor(config: AttributionRunConfig) -> dict[str, Any]:
    descriptor = {
        "method": config.method.curvature,
        "damping_sweep": list(config.method.damping_sweep),
        "damping_semantics": (
            "raw basis: eigenvalues + damping; Fisher basis: damping added "
            "to the statistic before the -1/2 power; stage-local Adam "
            "basis: A_l=(sqrt(v_hat_l)+optimizer_epsilon_l+damping)^-1/2"
        ),
    }
    if config.method.curvature == "ekfac_adam":
        # Conditional extension only: old-mode descriptors stay
        # byte-identical so committed score identities remain valid.
        descriptor["conditioning_damping"] = float(
            config.method.conditioning_damping
        )
        descriptor["damping_semantics"] = (
            descriptor["damping_semantics"]
            + "; ekfac_adam: A_l fixed at fit time with "
            "conditioning_damping; sweep damping adds to conditioned "
            "eigenvalues"
        )
    return descriptor


def _estimated_adam_context(
    config: AttributionRunConfig, tokenizer_digest: str
) -> tuple[str, str]:
    """Shared once-per-run context for estimated stage moments: the
    estimator dataset fingerprint and the paired-batch manifest digest."""

    estimator = config.adam_moment_estimator
    if estimator is None:
        raise RunnerError(
            "estimated Adam context requires adam_moment_estimator"
        )
    _, dataset_digest = _resolve_dataset_reference(
        Path(estimator.dataset.path),
        expected_digest=estimator.dataset.expected_digest,
        label="Adam moment estimator dataset",
    )
    estimator_fingerprint = _dataset_fingerprint(
        dataset_digest, tokenizer_digest
    )
    paired_path = run_layout(config.output_dir).adam_moments / "paired_batches.json"
    if not paired_path.is_file():
        raise RunnerError(
            "stage-local Adam estimates are missing paired_batches.json; "
            "run estimate-adam first"
        )
    return estimator_fingerprint, artifact_digest(paired_path)


def _load_estimated_stage_adam_payload(
    config: AttributionRunConfig,
    stage: AttributionStage,
    resolved: ResolvedStage,
    shared_manifest_digest: str,
    *,
    estimator_fingerprint: str,
    paired_digest: str,
) -> _StageAdamBasisPayload:
    """Load and validate ONE stage's committed estimated moment artifact."""

    import torch

    estimator = config.adam_moment_estimator
    directory = run_layout(config.output_dir).adam_moments / stage.name
    stored = _check_upstream(
        f"adam_moments/{stage.name}",
        directory,
        {
            "resolved_config": _scoped_config(config, "estimate-adam"),
            "checkpoint_digest": artifact_digest(resolved.checkpoint_dir),
            "dataset_fingerprint": estimator_fingerprint,
            "parameter_manifest_digest": shared_manifest_digest,
            "dtype": "float32",
            "seeds": {"run": config.seed, "estimator": estimator.seed},
        },
    )
    if stored.upstream_digests.get("paired_batches") != paired_digest:
        raise RunnerError(
            f"adam_moments/{stage.name}: paired batch manifest digest "
            "does not match the current shared manifest"
        )
    statistics_path = directory / _STATISTICS_FILE
    if not statistics_path.is_file():
        raise ArtifactIntegrityError(
            f"Adam moment artifact {directory} has no {_STATISTICS_FILE}"
        )
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    model_identifier = stored.basis_descriptor.get("model_identifier")
    if not isinstance(model_identifier, str) or not model_identifier:
        raise ArtifactIntegrityError(
            f"adam_moments/{stage.name} identity has no model identifier"
        )
    expected_statistics = _expected_adam_statistics(
        config,
        resolved,
        model_identifier=model_identifier,
        checkpoint_digest=artifact_digest(resolved.checkpoint_dir),
        dataset_fingerprint=estimator_fingerprint,
        parameter_manifest_digest=shared_manifest_digest,
        paired_batch_manifest_digest=paired_digest,
        code_commit=stored.scimt_commit,
    )
    _validate_adam_statistics(
        statistics,
        expected_statistics,
        label=f"adam_moments/{stage.name}",
    )
    tensor_manifest = ShardManifest.load(
        directory, expected_identity=stored
    )
    expected_auxiliary = (
        (_STATISTICS_FILE, artifact_digest(statistics_path)),
    )
    if tensor_manifest.auxiliary_digests != expected_auxiliary:
        raise ArtifactIntegrityError(
            f"adam_moments/{stage.name} tensor manifest does not commit "
            f"to {_STATISTICS_FILE}"
        )
    if tensor_manifest.total_rows != 1:
        raise ArtifactIntegrityError(
            f"adam_moments/{stage.name} must contain exactly one row"
        )
    values = tensor_manifest.read_rows(directory)["features"][0].float()
    if values.numel() < 1:
        raise ArtifactIntegrityError(
            f"adam_moments/{stage.name} moment row is empty"
        )
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ArtifactIntegrityError(
            f"adam_moments/{stage.name} moment row must be finite and "
            "nonnegative"
        )
    if tensor_manifest.feature_dim != values.numel():
        raise ArtifactIntegrityError(
            f"adam_moments/{stage.name} feature dimension is inconsistent"
        )
    statistics_digest = artifact_digest(statistics_path)
    tensor_manifest_digest = artifact_digest(
        directory / ShardManifest.FILENAME
    )
    stage_upstream = {
        f"adam/{stage.name}/identity": stored.digest(),
        f"adam/{stage.name}/statistics": statistics_digest,
        f"adam/{stage.name}/tensor_manifest": tensor_manifest_digest,
    }
    return _StageAdamBasisPayload(
        stage_name=stage.name,
        statistics=statistics,
        values=values,
        optimizer_epsilon=float(estimator.optimizer_epsilon),
        descriptor={
            "stage": stage.name,
            "mode": "estimated",
            "checkpoint_digest": artifact_digest(
                resolved.checkpoint_dir
            ),
            "moment_identity_digest": stored.digest(),
            "statistics_digest": statistics_digest,
            "tensor_manifest_digest": tensor_manifest_digest,
            "paired_batch_manifest_digest": paired_digest,
            "number_of_gradient_samples": estimator.num_batches,
            "beta2": estimator.beta2,
            "optimizer_epsilon": estimator.optimizer_epsilon,
        },
        upstream=stage_upstream,
    )


def _load_captured_stage_adam_payload(
    config: AttributionRunConfig,
    stage: AttributionStage,
    resolved: ResolvedStage,
    shared_manifest_digest: str,
) -> _StageAdamBasisPayload:
    """Load and validate ONE stage's captured optimizer-snapshot moment."""

    from scimt.train.attribution_snapshot import (
        BIAS_CORRECTION_CONVENTION,
        OPTIMIZER_MANIFEST_NAME,
        load_optimizer_snapshot,
    )

    from .manifest import flatten_tensors

    info = resolved.optimizer_snapshot
    if info is None:
        raise RunnerError(
            f"stage {stage.name!r}: captured Adam mode requires a "
            "same-checkpoint optimizer snapshot"
        )
    snapshot = load_optimizer_snapshot(info.path)
    if snapshot.manifest.digest() != shared_manifest_digest:
        raise RunnerError(
            f"stage {stage.name!r}: Adam snapshot parameter manifest "
            "does not match the gradient-row manifest"
        )
    corrected = snapshot.bias_corrected_exp_avg_sq()
    entries = snapshot.manifest.included_entries()
    values = flatten_tensors(
        entries, [corrected[entry.name] for entry in entries]
    )
    statistics = {
        "model_identifier": snapshot.manifest.model_name,
        "model_revision": str(info.step),
        "dataset_fingerprint": resolved.dataset_digest,
        "parameter_manifest_digest": snapshot.manifest.digest(),
        "statistic": "captured_adamw_exp_avg_sq_bias_corrected",
        "number_of_gradient_samples": info.step,
        "code_commit": _scimt_commit(),
        "estimator": "full",
    }
    manifest_path = Path(info.path) / OPTIMIZER_MANIFEST_NAME
    parameter_manifest_path = Path(info.path) / "parameter_manifest.json"
    optimizer_manifest_digest = artifact_digest(manifest_path)
    parameter_manifest_file_digest = artifact_digest(parameter_manifest_path)
    stage_upstream = {
        f"adam/{stage.name}/optimizer_manifest": optimizer_manifest_digest,
        f"adam/{stage.name}/parameter_manifest": (
            parameter_manifest_file_digest
        ),
    }
    return _StageAdamBasisPayload(
        stage_name=stage.name,
        statistics=statistics,
        values=values,
        optimizer_epsilon=float(info.epsilon),
        descriptor={
            "stage": stage.name,
            "mode": "captured",
            "checkpoint_digest": artifact_digest(resolved.checkpoint_dir),
            "optimizer_manifest_digest": optimizer_manifest_digest,
            "parameter_manifest_digest": shared_manifest_digest,
            "parameter_manifest_file_digest": (
                parameter_manifest_file_digest
            ),
            "step": info.step,
            "beta2": info.beta2,
            "optimizer_epsilon": info.epsilon,
            "bias_correction": BIAS_CORRECTION_CONVENTION,
        },
        upstream=stage_upstream,
    )


def _load_stage_adam_payloads(
    config: AttributionRunConfig,
    resolved_stages: list[ResolvedStage],
    shared_manifest_digest: str,
    tokenizer_digest: str,
) -> tuple[list[_StageAdamBasisPayload], dict[str, str]]:
    """Load every stage moment once, from estimates or captured snapshots."""

    payloads: list[_StageAdamBasisPayload] = []
    upstream: dict[str, str] = {}
    if config.adam_moment_estimator is not None:
        estimator_fingerprint, paired_digest = _estimated_adam_context(
            config, tokenizer_digest
        )
        upstream["adam/paired_batches"] = paired_digest
        for stage, resolved in zip(config.stages, resolved_stages, strict=True):
            payload = _load_estimated_stage_adam_payload(
                config,
                stage,
                resolved,
                shared_manifest_digest,
                estimator_fingerprint=estimator_fingerprint,
                paired_digest=paired_digest,
            )
            upstream.update(payload.upstream)
            payloads.append(payload)
        return payloads, upstream
    for stage, resolved in zip(config.stages, resolved_stages, strict=True):
        payload = _load_captured_stage_adam_payload(
            config, stage, resolved, shared_manifest_digest
        )
        upstream.update(payload.upstream)
        payloads.append(payload)
    return payloads, upstream


def _completed_stage_local_adam_score_receipt(
    config: AttributionRunConfig,
    *,
    layout: RunLayout,
    expected_entries: list[str],
    scoped: dict[str, Any],
    query_dir: Path,
    query_fingerprint: str,
    shared_manifest_digest: str,
    upstream_without_adam: dict[str, str],
) -> PhaseReport | None:
    """Validate completed scores from retained small Adam provenance files."""

    marker = layout.scores / _SCORE_MANIFEST_FILE
    if config.method.basis != "adam" or not marker.is_file():
        return None
    try:
        completeness = json.loads(marker.read_text(encoding="utf-8"))
        stored = read_identity(layout.scores)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"completed score receipt is unreadable: {error}"
        ) from error
    if (
        completeness.get("identity_digest") != stored.digest()
        or completeness.get("expected") != expected_entries
        or sorted(completeness.get("entries", {})) != expected_entries
    ):
        return None

    actual_adam: dict[str, str] = {}
    if config.adam_moment_estimator is not None:
        paired_path = layout.adam_moments / "paired_batches.json"
        if not paired_path.is_file():
            raise ArtifactIntegrityError(
                f"completed Adam score receipt requires {paired_path}"
            )
        actual_adam["adam/paired_batches"] = artifact_digest(paired_path)
        for stage in config.stages:
            directory = layout.adam_moments / stage.name
            identity_path = directory / ArtifactWriter.IDENTITY_FILE
            statistics_path = directory / _STATISTICS_FILE
            tensor_manifest_path = directory / ShardManifest.FILENAME
            for path in (identity_path, statistics_path, tensor_manifest_path):
                if not path.is_file():
                    raise ArtifactIntegrityError(
                        f"completed Adam score receipt requires retained {path}"
                    )
            actual_adam[f"adam/{stage.name}/identity"] = read_identity(
                directory
            ).digest()
            actual_adam[f"adam/{stage.name}/statistics"] = artifact_digest(
                statistics_path
            )
            actual_adam[f"adam/{stage.name}/tensor_manifest"] = artifact_digest(
                tensor_manifest_path
            )
    else:
        for stage in config.stages:
            snapshot = Path(stage.optimizer_snapshot)
            optimizer_manifest = snapshot / "optimizer_manifest.json"
            parameter_manifest = snapshot / "parameter_manifest.json"
            for path in (optimizer_manifest, parameter_manifest):
                if not path.is_file():
                    raise ArtifactIntegrityError(
                        f"completed Adam score receipt requires retained {path}"
                    )
            actual_adam[f"adam/{stage.name}/optimizer_manifest"] = artifact_digest(
                optimizer_manifest
            )
            actual_adam[f"adam/{stage.name}/parameter_manifest"] = artifact_digest(
                parameter_manifest
            )
    recorded_adam = {
        key: value
        for key, value in stored.upstream_digests.items()
        if key.startswith("adam/")
    }
    if actual_adam != recorded_adam:
        differing = sorted(
            key
            for key in set(actual_adam) | set(recorded_adam)
            if actual_adam.get(key) != recorded_adam.get(key)
        )
        raise ArtifactIntegrityError(
            f"completed Adam score receipt provenance drift: {differing}"
        )
    if stored.basis_descriptor.get("coordinates") != "adam_stage_local":
        raise IdentityMismatchError(
            "completed Adam score receipt does not declare stage-local coordinates"
        )
    if config.method.curvature == "ekfac_adam":
        if stored.basis_descriptor.get("curvature_mode") != "ekfac_adam" or (
            stored.basis_descriptor.get("conditioning_damping")
            != float(config.method.conditioning_damping)
        ):
            raise IdentityMismatchError(
                "completed Adam score receipt does not declare the "
                "configured ekfac_adam conditioning"
            )
    elif stored.basis_descriptor.get("curvature_mode") == "ekfac_adam":
        raise IdentityMismatchError(
            "completed Adam score receipt was produced by ekfac_adam "
            "conditioning, not the configured method"
        )
    expected_identity = _identity(
        phase="score-source",
        scoped=scoped,
        checkpoint_reference=str(query_dir),
        checkpoint_digest=artifact_digest(query_dir),
        dataset_fingerprint=query_fingerprint,
        parameter_manifest_digest=shared_manifest_digest,
        loss_convention=_loss_convention(
            config.query.objective, config.method.row_reduction
        ),
        basis_descriptor=stored.basis_descriptor,
        curvature_descriptor=_source_curvature_descriptor(config),
        logra_descriptor=None,
        dtype=config.method.dtype,
        seeds={"run": config.seed},
        upstream_digests={**upstream_without_adam, **actual_adam},
    )
    differing = expected_identity.diff(stored)
    if differing:
        raise IdentityMismatchError(
            f"completed Adam score receipt identity differs: {differing}"
        )
    for entry_name, entry in completeness["entries"].items():
        entry_path = layout.scores / entry["file"]
        if not entry_path.is_file():
            raise ArtifactIntegrityError(
                f"score manifest names an absent entry file: "
                f"{entry['file']} ({entry_name})"
            )
        actual = artifact_digest(entry_path)
        if actual != entry["digest"]:
            raise ArtifactIntegrityError(
                f"score entry {entry['file']} content digest mismatch: "
                f"recorded {entry['digest']}, actual {actual}"
            )
    report = PhaseReport(
        "score-source",
        (PhaseOutput("scores", layout.scores, stored.digest(), True),),
    )
    _append_event(config, "score-source", report.outputs)
    return report


def _validate_query_artifact(
    config: AttributionRunConfig,
    layout: RunLayout,
    query_dir: Path,
    query_fingerprint: str,
):
    """Validate the committed query-row artifact and return its identity."""
    return _check_upstream(
        "queries",
        layout.queries,
        {
            "resolved_config": _scoped_config(config, "build-queries"),
            "dataset_fingerprint": query_fingerprint,
            "checkpoint_digest": artifact_digest(query_dir),
            "dtype": config.method.dtype,
            "seeds": {"run": config.seed},
        },
    )


def _load_factor_operators(
    config: AttributionRunConfig,
    resolved_stages: Sequence[Any],
    layout: RunLayout,
    shared_manifest_digest: str,
    tokenizer_digest: str,
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    """Load every stage's fitted curvature (kind, payloads, identities)."""
    factor_kind: str | None = None
    factor_payloads: dict[str, Any] = {}
    factor_stored: dict[str, Any] = {}
    for stage, resolved in zip(config.stages, resolved_stages, strict=True):
        kind, payload, stored = _load_factor_operator(
            config, stage, layout.factors / stage.name, shared_manifest_digest,
            _dataset_fingerprint(resolved.dataset_digest, tokenizer_digest),
        )
        factor_kind = kind
        factor_payloads[stage.name] = payload
        factor_stored[stage.name] = stored
    return factor_kind, factor_payloads, factor_stored


def _validate_adam_factor_binding(
    config: AttributionRunConfig,
    factor_kind: str | None,
    factor_payloads: dict[str, Any],
    factor_stored: dict[str, Any],
    adam_payloads: "list[_StageAdamBasisPayload]",
    adam_upstream: dict[str, str],
) -> None:
    """Dimension + (conditioned-mode) moment-digest binding checks."""
    if factor_kind == "ekfac_adam":
        expected_dimension = int(
            factor_payloads[config.stages[0].name][1].included_numel
        )
    else:
        expected_dimension = int(
            factor_payloads[config.stages[0].name][1].numel()
        )
    wrong_dimensions = {
        payload.stage_name: int(payload.values.numel())
        for payload in adam_payloads
        if int(payload.values.numel()) != expected_dimension
    }
    if wrong_dimensions:
        raise RunnerError(
            "stage-local Adam moment dimensions do not match the shared "
            f"row/factor coordinates ({expected_dimension}): "
            f"{wrong_dimensions}"
        )
    if factor_kind == "ekfac_adam":
        # Conditioned factors are invalid under different moments: the
        # digests recorded at fit time must equal the freshly derived
        # per-stage moment digests (plus the shared paired-batch digest
        # in estimated mode).
        for stage, payload in zip(
            config.stages, adam_payloads, strict=True
        ):
            expected_factor_upstream = dict(payload.upstream)
            if config.adam_moment_estimator is not None:
                expected_factor_upstream["adam/paired_batches"] = (
                    adam_upstream["adam/paired_batches"]
                )
            recorded = dict(
                factor_stored[stage.name].upstream_digests
            )
            if recorded != expected_factor_upstream:
                differing = sorted(
                    key
                    for key in set(recorded)
                    | set(expected_factor_upstream)
                    if recorded.get(key)
                    != expected_factor_upstream.get(key)
                )
                raise RunnerError(
                    f"factors/{stage.name}: conditioned factor artifact "
                    "was fitted against different Adam moments than the "
                    f"ones now committed ({differing}) — re-run "
                    "fit-factors"
                )
            preconditioner = factor_payloads[stage.name][2]
            moment_digest = payload.descriptor.get(
                "moment_identity_digest"
            ) or payload.descriptor.get("optimizer_manifest_digest")
            if (
                preconditioner.get("moment_identity_digest")
                != moment_digest
            ):
                raise RunnerError(
                    f"factors/{stage.name}: ekfac_meta preconditioner "
                    "moment digest does not match the committed stage "
                    "moment artifact — re-run fit-factors"
                )


def _score_basis_extras(
    config: AttributionRunConfig,
    factor_payloads: dict[str, Any],
    adam_payloads: "list[_StageAdamBasisPayload]",
) -> dict[str, Any]:
    """The basis descriptor pinned into the score identity (probe metric =
    first damping for the fisher basis)."""
    method = config.method
    basis_extras: dict[str, Any] = {"coordinates": "raw"}
    if method.basis == "fisher":
        _, basis_extras = _fisher_basis_metric(
            config, factor_payloads, method.damping_sweep[0]
        )
    elif method.basis == "adam":
        basis_extras = {
            "coordinates": "adam_stage_local",
            "geometry": (
                "A_l=(sqrt(v_hat_l)+optimizer_epsilon_l+damping)^-1/2"
            ),
            "transition": "A_previous/A_current",
            "stages": [payload.descriptor for payload in adam_payloads],
        }
        if method.curvature == "ekfac_adam":
            basis_extras["curvature_mode"] = "ekfac_adam"
            basis_extras["conditioning_damping"] = float(
                method.conditioning_damping
            )
            basis_extras["geometry"] = (
                "A_l=(sqrt(v_hat_l)+optimizer_epsilon_l"
                "+conditioning_damping)^-1/2"
            )
    return basis_extras


def _conditioned_metrics(
    config: AttributionRunConfig,
    adam_payloads: "list[_StageAdamBasisPayload]",
) -> list[Any]:
    """Fit-time conditioned metrics for ekfac_adam (A_l is FIXED at fit time
    by conditioning_damping; the sweep damping never enters the metric —
    it shifts the conditioned eigenvalues instead)."""
    if config.method.curvature != "ekfac_adam":
        return []
    from .metrics import DiagonalMetric

    return [
        DiagonalMetric.from_adam_second_moment(
            payload.statistics,
            payload.values,
            optimizer_epsilon=payload.optimizer_epsilon,
            damping=float(config.method.conditioning_damping),
        )
        for payload in adam_payloads
    ]


def _scoring_context_for_damping(
    config: AttributionRunConfig,
    *,
    resolved_stages: Sequence[Any],
    factor_kind: str | None,
    factor_payloads: dict[str, Any],
    adam_payloads: "list[_StageAdamBasisPayload]",
    conditioned_metrics: list[Any],
    basis_extras: dict[str, Any],
    shared_manifest_digest: str,
    damping: float,
) -> tuple[Any, list[Any], Any]:
    """One damping point's scorer plus per-stage row scales and query scale.

    Extracted verbatim from :func:`score_source` so the streaming score phase
    consumes the identical segment construction — behavior-preserving."""
    import torch

    from .source import DiagonalCurvature, EKFACCurvature, SourceScorer, SourceSegment

    method = config.method
    metric = None
    adam_metrics = []
    if method.basis == "fisher":
        metric, _ = _fisher_basis_metric(
            config, factor_payloads, damping
        )
    elif method.curvature == "ekfac_adam":
        adam_metrics = conditioned_metrics
    elif method.basis == "adam":
        from .metrics import DiagonalMetric

        adam_metrics = [
            DiagonalMetric.from_adam_second_moment(
                payload.statistics,
                payload.values,
                optimizer_epsilon=payload.optimizer_epsilon,
                damping=damping,
            )
            for payload in adam_payloads
        ]
    diagonal_scale = (
        None
        if metric is None
        else metric.diagonal.to(dtype=torch.float32)
    )
    segments = []
    stage_scales: list[Any] = []
    for stage_index, stage in enumerate(config.stages):
        resolved = resolved_stages[_stage_index(config, stage.name)]
        stage_scale = (
            adam_metrics[stage_index].diagonal.to(dtype=torch.float32)
            if adam_metrics
            else diagonal_scale
        )
        stage_scales.append(stage_scale)
        if adam_metrics:
            descriptor = {
                "coordinates": "adam_stage_local",
                "stage": stage.name,
                "metric_snapshot": adam_metrics[stage_index].snapshot,
                "manifest_digest": shared_manifest_digest,
                "damping": damping,
            }
            if method.curvature == "ekfac_adam":
                descriptor["curvature_mode"] = "ekfac_adam"
                descriptor["conditioning_damping"] = float(
                    method.conditioning_damping
                )
        else:
            descriptor = {
                **basis_extras,
                "manifest_digest": shared_manifest_digest,
                "damping": damping,
            }
        if factor_kind == "fisher":
            _, diagonal = factor_payloads[stage.name]
            if stage_scale is None:
                curvature = DiagonalCurvature(
                    diagonal.double().numpy() + damping,
                    basis_descriptor=descriptor,
                )
            else:
                transported = (
                    diagonal.double()
                    * stage_scale.double().pow(2)
                ).numpy()
                curvature = DiagonalCurvature(
                    transported, basis_descriptor=descriptor
                )
        else:
            factors, factor_manifest = factor_payloads[stage.name][:2]
            curvature = _shifted_curvature(
                EKFACCurvature(
                    factors, factor_manifest, basis_descriptor=descriptor
                ),
                damping,
            )
        transition = None
        if adam_metrics and stage_index > 0:
            transition = (
                adam_metrics[stage_index - 1].diagonal.double()
                / adam_metrics[stage_index].diagonal.double()
            ).numpy()
        segments.append(
            SourceSegment(
                stage.name,
                curvature,
                resolved.lr_steps,
                transition_to_previous=transition,
            )
        )
    scorer = SourceScorer(segments)
    query_scale = (
        adam_metrics[-1].diagonal.to(dtype=torch.float32)
        if adam_metrics
        else diagonal_scale
    )
    return scorer, stage_scales, query_scale


async def score_source(config: AttributionRunConfig) -> PhaseReport:
    """SOURCE scoring over the ordered chronological stages.

    Consumes only validated artifacts (rows, factors, queries), builds one
    scorer per damping value with explicit adjacent transitions when bases
    differ, and applies each segment's ``1/N`` (its declared training-set
    size) exactly once — the scorer itself returns unnormalized scores. Raw
    per-example scores are saved per (stage, damping) with an explicit
    completeness manifest."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from .artifacts import _fsync_directory

    _require_scorable_method(config)
    method = config.method
    _write_or_check_ledger(config, "score-source")
    layout = run_layout(config.output_dir)
    completed_captured_receipt = (
        method.basis == "adam"
        and config.adam_moment_estimator is None
        and (layout.scores / _SCORE_MANIFEST_FILE).is_file()
    )
    if completed_captured_receipt:
        resolved_stages = [
            resolve_stage(
                dataclasses.replace(stage, optimizer_snapshot=None),
                require_adam=False,
            )
            for stage in config.stages
        ]
    else:
        resolved_stages = _resolve_stages(
            config,
            require_adam=(
                method.basis == "adam"
                and config.adam_moment_estimator is None
            ),
        )
    query_dir = _resolve_query_checkpoint(config)
    _, query_dataset_digest = _resolve_query_dataset(config)
    tokenizer_digest = _tokenizer_content_digest(_tokenizer_dir(config, query_dir))
    query_fingerprint = _dataset_fingerprint(query_dataset_digest, tokenizer_digest)
    if method.basis == "adam" and artifact_digest(
        query_dir
    ) != artifact_digest(resolved_stages[-1].checkpoint_dir):
        raise RunnerError(
            "score-source: stage-local Adam coordinates require the query "
            "checkpoint to equal the final chronological stage checkpoint"
        )

    # --- validate upstream row artifacts (before any tensor loads) ---------
    queries_stored = _validate_query_artifact(
        config, layout, query_dir, query_fingerprint
    )
    shared_manifest_digest = queries_stored.parameter_manifest_digest
    rows_stored: dict[str, Any] = {}
    for stage, resolved in zip(config.stages, resolved_stages, strict=True):
        stored = _check_upstream(
            f"rows/{stage.name}",
            layout.rows / stage.name,
            {
                "resolved_config": _scoped_config(config, "compute-rows", stage.name),
                "parameter_manifest_digest": shared_manifest_digest,
                "dataset_fingerprint": _dataset_fingerprint(
                    resolved.dataset_digest, tokenizer_digest
                ),
                "checkpoint_digest": artifact_digest(resolved.checkpoint_dir),
                "basis_descriptor": {
                    "coordinates": "raw",
                    "manifest_digest": shared_manifest_digest,
                },
                "dtype": method.dtype,
                "seeds": {"run": config.seed},
            },
        )
        rows_stored[stage.name] = stored

    factor_kind, factor_payloads, factor_stored = _load_factor_operators(
        config, resolved_stages, layout, shared_manifest_digest,
        tokenizer_digest,
    )

    scoped = _scoped_config(config, "score-source")
    scoped["resolved_lr_steps"] = {
        resolved.name: resolved.lr_steps for resolved in resolved_stages
    }
    upstream = {"queries": queries_stored.digest()}
    for stage in config.stages:
        upstream[f"rows/{stage.name}"] = rows_stored[stage.name].digest()
        upstream[f"factors/{stage.name}"] = factor_stored[stage.name].digest()
    expected_entries = sorted(
        f"{stage.name}__damping-{index}"
        for stage in config.stages
        for index in range(len(method.damping_sweep))
    )
    receipt = _completed_stage_local_adam_score_receipt(
        config,
        layout=layout,
        expected_entries=expected_entries,
        scoped=scoped,
        query_dir=query_dir,
        query_fingerprint=query_fingerprint,
        shared_manifest_digest=shared_manifest_digest,
        upstream_without_adam=upstream,
    )
    if receipt is not None:
        return receipt
    if completed_captured_receipt:
        # The optimistic receipt path deliberately resolves model provenance
        # without loading large captured moment shards. If the marker is
        # present but incomplete, recomputation still needs live snapshots.
        resolved_stages = _resolve_stages(config, require_adam=True)
    adam_payloads: list[_StageAdamBasisPayload] = []
    if method.basis == "adam":
        adam_payloads, adam_upstream = _load_stage_adam_payloads(
            config,
            resolved_stages,
            shared_manifest_digest,
            tokenizer_digest,
        )
        upstream.update(adam_upstream)
        _validate_adam_factor_binding(
            config, factor_kind, factor_payloads, factor_stored,
            adam_payloads, adam_upstream,
        )

    # A probe metric (first damping) pins the basis descriptor identity.
    basis_extras = _score_basis_extras(config, factor_payloads, adam_payloads)
    identity = _identity(
        phase="score-source",
        scoped=scoped,
        checkpoint_reference=str(query_dir),
        checkpoint_digest=artifact_digest(query_dir),
        dataset_fingerprint=query_fingerprint,
        parameter_manifest_digest=shared_manifest_digest,
        loss_convention=_loss_convention(
            config.query.objective, method.row_reduction
        ),
        basis_descriptor={
            **basis_extras,
            "manifest_digest": shared_manifest_digest,
        },
        curvature_descriptor=_source_curvature_descriptor(config),
        logra_descriptor=None,
        dtype=method.dtype,
        seeds={"run": config.seed},
        upstream_digests=upstream,
    )
    _bind_identity(layout.scores, identity)
    marker = layout.scores / _SCORE_MANIFEST_FILE
    if marker.is_file():
        completeness = json.loads(marker.read_text(encoding="utf-8"))
        if (
            completeness.get("identity_digest") == identity.digest()
            and completeness.get("expected") == expected_entries
            and sorted(completeness.get("entries", {})) == expected_entries
        ):
            # Completed matrix: verify every recorded per-file digest (the
            # manifest was written last, so absence or drift after
            # publication is an integrity violation, never a recompute).
            for entry_name, entry in completeness["entries"].items():
                entry_path = layout.scores / entry["file"]
                if not entry_path.is_file():
                    raise ArtifactIntegrityError(
                        f"score manifest names an absent entry file: "
                        f"{entry['file']} ({entry_name})"
                    )
                actual = artifact_digest(entry_path)
                if actual != entry["digest"]:
                    raise ArtifactIntegrityError(
                        f"score entry {entry['file']} content digest "
                        f"mismatch: recorded {entry['digest']}, actual "
                        f"{actual} — bytes changed after publication"
                    )
            report = PhaseReport(
                "score-source",
                (PhaseOutput("scores", layout.scores, identity.digest(), True),),
            )
            _append_event(config, "score-source", report.outputs)
            return report

    query_manifest = ShardManifest.load(
        layout.queries, expected_identity=queries_stored
    )
    query_rows = query_manifest.read_rows(layout.queries)
    query_features = query_rows["features"].float()

    conditioned_metrics = _conditioned_metrics(config, adam_payloads)
    entries: dict[str, dict[str, Any]] = {}
    for damping_index, damping in enumerate(method.damping_sweep):
        scorer, stage_scales, query_scale = _scoring_context_for_damping(
            config,
            resolved_stages=resolved_stages,
            factor_kind=factor_kind,
            factor_payloads=factor_payloads,
            adam_payloads=adam_payloads,
            conditioned_metrics=conditioned_metrics,
            basis_extras=basis_extras,
            shared_manifest_digest=shared_manifest_digest,
            damping=damping,
        )
        transformed_query = (
            query_features if query_scale is None else query_features * query_scale
        )
        transformed = scorer.transformed_queries(transformed_query.numpy())
        for stage_index, stage in enumerate(config.stages):
            rows_dir = layout.rows / stage.name
            rows_manifest = ShardManifest.load(
                rows_dir, expected_identity=rows_stored[stage.name]
            )
            pieces = []
            sample_ids = []
            for shard_index in range(len(rows_manifest.shards)):
                shard = rows_manifest.read_shard(rows_dir, shard_index)
                features = shard["features"].float()
                row_scale = stage_scales[stage_index]
                if row_scale is not None:
                    features = features * row_scale
                pieces.append(transformed[stage_index] @ features.numpy().T)
                sample_ids.append(shard["sample_ids"])
            unnormalized = np.concatenate(pieces, axis=1)
            # The scorer returns unnormalized scores; each segment's 1/N
            # (its declared training-set size) is applied here, exactly once.
            scores = unnormalized / float(stage.n_examples)
            entry_name = f"{stage.name}__damping-{damping_index}"
            filename = f"scores__{entry_name}.safetensors"
            final_path = layout.scores / filename
            tmp_path = layout.scores / (filename + ".tmp")
            save_file(
                {
                    "scores": torch.from_numpy(
                        np.ascontiguousarray(scores.astype(np.float32))
                    ),
                    "query_sample_ids": query_rows["sample_ids"],
                    "train_sample_ids": torch.cat(sample_ids),
                },
                str(tmp_path),
                metadata={
                    "identity_digest": identity.digest(),
                    "stage": stage.name,
                    "damping": repr(float(damping)),
                    "n_examples": str(stage.n_examples),
                },
            )
            # Atomic visibility: temp sibling + fsync + os.replace, the
            # artifacts.py commit protocol.
            with tmp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp_path, final_path)
            _fsync_directory(layout.scores)
            entries[entry_name] = {
                "file": filename,
                "digest": artifact_digest(final_path),
                "stage": stage.name,
                "damping": float(damping),
                "n_examples": stage.n_examples,
                "n_queries": int(query_features.shape[0]),
                "n_train_rows": int(unnormalized.shape[1]),
            }
    from .artifacts import _atomic_write_text

    _atomic_write_text(
        marker,
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "identity_digest": identity.digest(),
                "expected": expected_entries,
                "entries": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    report = PhaseReport(
        "score-source",
        (PhaseOutput("scores", layout.scores, identity.digest(), False,
                     rows=len(entries)),),
    )
    _append_event(config, "score-source", report.outputs)
    return report


# ================================================== score-source (streaming) ==
async def score_source_streaming(config: AttributionRunConfig) -> PhaseReport:
    """SOURCE scores computed by streaming per-row gradients — no row shards.

    Identical math to :func:`score_source` (same factor/query/moment
    validation, same segment construction via the shared helpers, same
    per-(stage, damping) score files and completeness manifest), but the
    per-example train gradients are recomputed on the fly and dotted against
    the transformed queries immediately, persisting only ``[N, D*Q]`` score
    rows. Use when materialized row shards are infeasible (full-coverage rows
    at large P: N × P × 4 bytes vs N × D·Q × 4 bytes). Resumable at row-shard
    granularity through the same :class:`ArtifactWriter` protocol; an
    identical completed rerun reports the manifest hit without touching the
    model."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from .artifacts import _atomic_write_text, _fsync_directory
    from .gradients import BatchedVJPBackend, backward_memory_mode
    from .losses import CausalLMLossAdapter

    _require_scorable_method(config)
    method = config.method
    _write_or_check_ledger(config, "score-source-streaming")
    layout = run_layout(config.output_dir)
    resolved_stages = _resolve_stages(
        config,
        require_adam=(
            method.basis == "adam" and config.adam_moment_estimator is None
        ),
    )
    query_dir = _resolve_query_checkpoint(config)
    _, query_dataset_digest = _resolve_query_dataset(config)
    tokenizer_digest = _tokenizer_content_digest(_tokenizer_dir(config, query_dir))
    query_fingerprint = _dataset_fingerprint(query_dataset_digest, tokenizer_digest)
    if method.basis == "adam" and artifact_digest(
        query_dir
    ) != artifact_digest(resolved_stages[-1].checkpoint_dir):
        raise RunnerError(
            "score-source-streaming: stage-local Adam coordinates require "
            "the query checkpoint to equal the final chronological stage "
            "checkpoint"
        )
    queries_stored = _validate_query_artifact(
        config, layout, query_dir, query_fingerprint
    )
    shared_manifest_digest = queries_stored.parameter_manifest_digest
    factor_kind, factor_payloads, factor_stored = _load_factor_operators(
        config, resolved_stages, layout, shared_manifest_digest,
        tokenizer_digest,
    )
    scoped = _scoped_config(config, "score-source-streaming")
    scoped["resolved_lr_steps"] = {
        resolved.name: resolved.lr_steps for resolved in resolved_stages
    }
    upstream = {"queries": queries_stored.digest()}
    for stage in config.stages:
        upstream[f"factors/{stage.name}"] = factor_stored[stage.name].digest()
    # Row artifacts do not exist in this phase: bind each stage's DATA
    # (dataset bytes × tokenizer content) into the identity instead.
    stage_fingerprints = {
        stage.name: _dataset_fingerprint(resolved.dataset_digest,
                                         tokenizer_digest)
        for stage, resolved in zip(config.stages, resolved_stages, strict=True)
    }
    for name, fingerprint in stage_fingerprints.items():
        upstream[f"stage_data/{name}"] = fingerprint
    adam_payloads: list[_StageAdamBasisPayload] = []
    if method.basis == "adam":
        adam_payloads, adam_upstream = _load_stage_adam_payloads(
            config,
            resolved_stages,
            shared_manifest_digest,
            tokenizer_digest,
        )
        upstream.update(adam_upstream)
        _validate_adam_factor_binding(
            config, factor_kind, factor_payloads, factor_stored,
            adam_payloads, adam_upstream,
        )
    basis_extras = _score_basis_extras(config, factor_payloads, adam_payloads)
    identity = _identity(
        phase="score-source-streaming",
        scoped=scoped,
        checkpoint_reference=str(query_dir),
        checkpoint_digest=artifact_digest(query_dir),
        dataset_fingerprint=query_fingerprint,
        parameter_manifest_digest=shared_manifest_digest,
        loss_convention=_loss_convention(
            config.query.objective, method.row_reduction
        ),
        basis_descriptor={
            **basis_extras,
            "manifest_digest": shared_manifest_digest,
        },
        curvature_descriptor=_source_curvature_descriptor(config),
        logra_descriptor=None,
        dtype=method.dtype,
        seeds={"run": config.seed},
        upstream_digests=upstream,
    )
    _bind_identity(layout.streaming_scores, identity)
    expected_entries = sorted(
        f"{stage.name}__damping-{index}"
        for stage in config.stages
        for index in range(len(method.damping_sweep))
    )
    marker = layout.streaming_scores / _SCORE_MANIFEST_FILE
    if marker.is_file():
        completeness = json.loads(marker.read_text(encoding="utf-8"))
        if (
            completeness.get("identity_digest") == identity.digest()
            and completeness.get("expected") == expected_entries
            and sorted(completeness.get("entries", {})) == expected_entries
        ):
            for entry_name, entry in completeness["entries"].items():
                entry_path = layout.streaming_scores / entry["file"]
                if not entry_path.is_file():
                    raise ArtifactIntegrityError(
                        f"streaming score manifest names an absent entry "
                        f"file: {entry['file']} ({entry_name})"
                    )
                actual = artifact_digest(entry_path)
                if actual != entry["digest"]:
                    raise ArtifactIntegrityError(
                        f"streaming score entry {entry['file']} content "
                        f"digest mismatch: recorded {entry['digest']}, "
                        f"actual {actual} — bytes changed after publication"
                    )
            report = PhaseReport(
                "score-source-streaming",
                (PhaseOutput("streaming_scores", layout.streaming_scores,
                             identity.digest(), True),),
            )
            _append_event(config, "score-source-streaming", report.outputs)
            return report

    query_manifest = ShardManifest.load(
        layout.queries, expected_identity=queries_stored
    )
    query_rows = query_manifest.read_rows(layout.queries)
    query_features = query_rows["features"].float()
    n_queries = int(query_features.shape[0])
    n_dampings = len(method.damping_sweep)
    conditioned_metrics = _conditioned_metrics(config, adam_payloads)

    # Per damping: the scorer's transformed queries for every stage, plus the
    # per-stage row scales. Gradients are damping-independent, so the stage
    # datasets are streamed ONCE and dotted against every damping's u_l.
    #
    # Full-coverage memory bounds (P ≈ 1.07e10, fp64 [1, P] ≈ 86 GB):
    # transport runs per QUERY ROW — every scorer op (per-module
    # V·f(λ)·Vᵀ, diagonal metrics, transitions) is row-independent, so
    # [1, P] transport equals whole-matrix transport row-for-row while the
    # live fp64 set stays ~3 arrays (~257 GB) instead of Q×(L+1) (~685 GB
    # > the 503 GB cgroup that OOM-killed run 20260819T095144Z). Each
    # stage's u_l is spilled to a temporary memmap and streamed back
    # through the dot: memmap pages are reclaimable page cache, not
    # anonymous RSS, so the cgroup evicts instead of OOM-killing.
    u_dir = layout.streaming_scores / "u_tmp"
    u_dir.mkdir(parents=True, exist_ok=True)
    width = int(query_features.shape[1])
    contexts = []
    for damping_index, damping in enumerate(method.damping_sweep):
        scorer, stage_scales, query_scale = _scoring_context_for_damping(
            config,
            resolved_stages=resolved_stages,
            factor_kind=factor_kind,
            factor_payloads=factor_payloads,
            adam_payloads=adam_payloads,
            conditioned_metrics=conditioned_metrics,
            basis_extras=basis_extras,
            shared_manifest_digest=shared_manifest_digest,
            damping=damping,
        )
        maps = [
            np.lib.format.open_memmap(
                str(u_dir / f"u_damping{damping_index}_stage{index}.npy"),
                mode="w+",
                # fp32: transformed_queries returns fp32 (upstream numpy
                # contract), so fp32 maps are value-exact, halve spill/IO,
                # and restore main's fp32 dot dtype (PR #530 review).
                dtype=np.float32,
                shape=(n_queries, width),
            )
            for index in range(len(config.stages))
        ]
        for row_index in range(n_queries):
            row = query_features[row_index : row_index + 1]
            if query_scale is not None:
                row = row * query_scale
            transported = scorer.transformed_queries(row.numpy())
            if len(transported) != len(maps):
                raise ArtifactIntegrityError(
                    "streaming transport returned "
                    f"{len(transported)} stages, expected {len(maps)}"
                )
            for stage_index, u in enumerate(transported):
                maps[stage_index][row_index] = u[0]
            del transported
        # Release the last transport-loop bindings; `row`/`u` pin fp32
        # [1, P] storages (~43 GB each at full coverage) otherwise
        # (PR #530 review finding).
        row = None
        u = None
        for mapped in maps:
            mapped.flush()
        contexts.append((maps, stage_scales))

    # The [Q, P] query features (~86 GB fp32 at full coverage) are fully
    # spilled into the per-stage memmaps above; release before the scoring
    # loop (PR #530 review finding).
    query_features = None

    storage = getattr(torch, _storage_dtype(method.dtype))
    entries: dict[str, dict[str, Any]] = {}
    for stage_index, (stage, resolved) in enumerate(
        zip(config.stages, resolved_stages, strict=True)
    ):
        progress_dir = layout.streaming_scores / "progress" / stage.name
        writer = ArtifactWriter(
            progress_dir,
            identity,
            feature_dim=max(1, n_dampings * n_queries),
            rows_per_shard=config.data.rows_per_shard,
            feature_dtype="float32",
        )
        if not writer.already_complete:
            model = _load_model(
                resolved.checkpoint_dir,
                dtype=method.dtype,
                device=config.data.device,
                gradient_checkpointing=(
                    config.data.gradient_checkpointing_enabled
                ),
            )
            manifest = _build_manifest(model, config)
            if manifest.digest() != shared_manifest_digest:
                raise RunnerError(
                    f"score-source-streaming: stage {stage.name!r} rebuilt "
                    "parameter manifest does not match the shared query "
                    "manifest — same selection must apply to rows and queries"
                )
            tokenizer = _load_tokenizer(_tokenizer_dir(config, query_dir))
            dataset = _dataset_adapter(
                objective=stage.objective,
                data_path=Path(resolved.dataset.path),
                tokenizer=tokenizer,
                config=config,
                reduction=method.row_reduction,
                max_sequences=config.data.max_stage_sequences,
            )
            adapter = CausalLMLossAdapter(
                model, reduction=method.row_reduction, device=config.data.device
            )
            backend = BatchedVJPBackend(model, manifest)
            committed = writer.rows_committed
            seen = 0
            memory_mode = backward_memory_mode(
                model, getattr(model, "is_gradient_checkpointing", False)
            )
            with memory_mode:
                for batch in dataset.iter_batches(config.data.batch_size):
                    expected = _batch_expected_rows(batch, method.row_reduction)
                    if seen + expected <= committed:
                        seen += expected
                        continue
                    loss_batch = adapter.per_datapoint_losses(batch)
                    if int(loss_batch.losses.numel()) != expected:
                        raise ArtifactIntegrityError(
                            "streaming score bookkeeping mismatch: expected "
                            f"{expected} rows for this batch, got "
                            f"{int(loss_batch.losses.numel())}"
                        )
                    # Per-chunk consumption: at full coverage one [c, P] fp32
                    # chunk is ~4·c·P bytes on device — backend.rows()'s
                    # whole-batch materialization (and its cat copy) cannot
                    # coexist with the model (pod run 20260819T095144Z).
                    # Scores are row-independent, so per-chunk dots followed
                    # by a CPU concat over the tiny [c, D*Q] results equal
                    # the whole-batch computation exactly.
                    chunk_scores = []
                    for chunk in backend.iter_row_chunks(
                        loss_batch.losses, chunk_size=config.data.vjp_chunk_size
                    ):
                        # Reproduce the materialized path's storage
                        # round-trip so streaming scores equal score-source
                        # scores bit-for-bit up to matmul reassociation.
                        features_cpu = (
                            chunk.detach().to(device="cpu", dtype=storage).float()
                        )
                        del chunk
                        pieces = []
                        for transformed, stage_scales in contexts:
                            features = features_cpu
                            if stage_scales[stage_index] is not None:
                                features = features * stage_scales[stage_index]
                            pieces.append(
                                transformed[stage_index] @ features.numpy().T
                            )  # [Q, c]
                        chunk_scores.append(
                            np.concatenate(pieces, axis=0).T  # [c, D*Q]
                        )
                    score_rows = torch.from_numpy(
                        np.ascontiguousarray(
                            np.concatenate(chunk_scores, axis=0).astype(np.float32)
                        )
                    )  # [B, D*Q]
                    drop = max(0, committed - seen)
                    writer.append(
                        features=score_rows[drop:],
                        sample_ids=loss_batch.sample_ids[drop:],
                        sequence_ids=loss_batch.sequence_ids[drop:],
                        target_positions=loss_batch.target_positions[drop:],
                    )
                    seen += expected
            writer.finalize()
        progress_manifest = ShardManifest.load(progress_dir)
        stored = progress_manifest.read_rows(progress_dir)
        score_rows_all = stored["features"].float().numpy()  # [N, D*Q]
        train_sample_ids = stored["sample_ids"]
        for damping_index, damping in enumerate(method.damping_sweep):
            block = score_rows_all[
                :, damping_index * n_queries:(damping_index + 1) * n_queries
            ].T  # [Q, N]
            scores = block / float(stage.n_examples)
            entry_name = f"{stage.name}__damping-{damping_index}"
            filename = f"scores__{entry_name}.safetensors"
            final_path = layout.streaming_scores / filename
            tmp_path = layout.streaming_scores / (filename + ".tmp")
            save_file(
                {
                    "scores": torch.from_numpy(
                        np.ascontiguousarray(scores.astype(np.float32))
                    ),
                    "query_sample_ids": query_rows["sample_ids"],
                    "train_sample_ids": train_sample_ids,
                },
                str(tmp_path),
                metadata={
                    "identity_digest": identity.digest(),
                    "stage": stage.name,
                    "damping": repr(float(damping)),
                    "n_examples": str(stage.n_examples),
                },
            )
            with tmp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp_path, final_path)
            _fsync_directory(layout.streaming_scores)
            entries[entry_name] = {
                "file": filename,
                "digest": artifact_digest(final_path),
                "stage": stage.name,
                "damping": float(damping),
                "n_examples": stage.n_examples,
                "n_queries": n_queries,
                "n_train_rows": int(scores.shape[1]),
            }
    _atomic_write_text(
        marker,
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "identity_digest": identity.digest(),
                "expected": expected_entries,
                "entries": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    # The spilled u_l memmaps are scratch, not artifacts: delete them so the
    # completed layout matches the pre-spill schema (and ~Q×L×8·P bytes of
    # disk return). Kept on failure for post-mortems — this line is only
    # reached after the manifest is published.
    del contexts
    shutil.rmtree(u_dir, ignore_errors=True)
    report = PhaseReport(
        "score-source-streaming",
        (PhaseOutput("streaming_scores", layout.streaming_scores,
                     identity.digest(), False, rows=len(entries)),),
    )
    _append_event(config, "score-source-streaming", report.outputs)
    return report


# ============================================================== second order ==
def _require_second_order(config: AttributionRunConfig, phase: str):
    if config.second_order is None:
        raise RunnerError(
            f"{phase}: second-order phases need an explicit second_order "
            "configuration block declaring the ONE checkpoint to run at "
            "(second_order.checkpoint: <stage name or 'query'>)"
        )
    return config.second_order


def _declared_checkpoint(
    config: AttributionRunConfig,
) -> tuple[Path, str, ResolvedStage | None]:
    """Resolve the ONE declared second-order checkpoint."""
    second = config.second_order
    if second.checkpoint == "query":
        query_dir = _resolve_query_checkpoint(config)
        return query_dir, "query", None
    stage = config.stages[_stage_index(config, second.checkpoint)]
    resolved = resolve_stage(stage)
    return resolved.checkpoint_dir, stage.name, resolved


def _pair_metric(config: AttributionRunConfig, manifest: Any,
                 declared: tuple[Path, str, ResolvedStage | None],
                 tokenizer_digest: str):
    """The frozen diagonal pair metric M (or None), plus the upstream
    digests it consumed. EK-FAC is refused: the pair path supports
    DiagonalMetric only (recorded port deviation). Factor artifacts pass the
    same upstream-identity validation score-source performs."""
    from .metrics import DiagonalMetric

    second = config.second_order
    if second.metric == "none":
        return None, {"kind": "identity"}, {}
    if second.metric == "ekfac":
        raise RunnerError(
            "build-directions: the pair-gradient path supports DiagonalMetric "
            "only — an EK-FAC pair metric was dropped in the port (recorded "
            "deviation in the package README) and must be re-added "
            "consciously if needed; use metric 'adam', 'fisher', or 'none'"
        )
    checkpoint_dir, label, resolved = declared
    if second.metric == "adam":
        if resolved is None or resolved.optimizer_snapshot is None:
            raise RunnerError(
                "build-directions: metric 'adam' needs the declared "
                "checkpoint to be a stage with a validated optimizer "
                f"snapshot; {label!r} has none"
            )
        from scimt.train.attribution_snapshot import load_optimizer_snapshot

        snapshot = load_optimizer_snapshot(resolved.optimizer_snapshot.path)
        if snapshot.manifest.digest() != manifest.digest():
            raise RunnerError(
                "build-directions: Adam pair-metric parameter-manifest "
                "cross-check failed — snapshot manifest digest "
                f"{snapshot.manifest.digest()} != model manifest digest "
                f"{manifest.digest()}"
            )
        statistics = {
            "model_identifier": snapshot.manifest.model_name,
            "model_revision": str(snapshot.info.step),
            "dataset_fingerprint": "n/a:adam-snapshot",
            "parameter_manifest_digest": snapshot.manifest.digest(),
            "statistic": "adamw_exp_avg_sq_bias_corrected",
            "number_of_gradient_samples": snapshot.info.step,
            "code_commit": _scimt_commit(),
            "estimator": "full",
            "logra": None,
        }
        metric = DiagonalMetric.from_statistics(
            statistics,
            snapshot.bias_corrected_exp_avg_sq(),
            exponent=second.metric_exponent,
            epsilon=second.metric_epsilon,
            manifest=manifest,
        )
        from scimt.train.attribution_snapshot import OPTIMIZER_MANIFEST_NAME

        upstream = {
            "pair_metric_adam": snapshot.manifest.digest(),
            "pair_metric_adam_manifest": artifact_digest(
                Path(resolved.optimizer_snapshot.path) / OPTIMIZER_MANIFEST_NAME
            ),
        }
        return metric, {"kind": "adam", **metric.descriptor()}, upstream
    # fisher: the declared stage's fitted Fisher-diagonal artifact.
    if resolved is None:
        raise RunnerError(
            "build-directions: metric 'fisher' needs the declared checkpoint "
            "to be a stage with a fitted fisher factor artifact; declare a "
            "stage checkpoint or use metric 'none'/'adam'"
        )
    factors_dir = run_layout(config.output_dir).factors / label
    statistics_path = factors_dir / _STATISTICS_FILE
    if not statistics_path.is_file():
        raise RunnerError(
            f"build-directions: metric 'fisher' needs {statistics_path} — "
            "run fit-factors with curvature 'fisher' first"
        )
    # The same upstream-identity validation score-source performs before
    # consuming a factor artifact (scope, coordinates, data, seeds).
    stored = _check_upstream(
        f"factors/{label}",
        factors_dir,
        {
            "resolved_config": _scoped_config(config, "fit-factors", label),
            "parameter_manifest_digest": manifest.digest(),
            "dataset_fingerprint": _dataset_fingerprint(
                resolved.dataset_digest, tokenizer_digest
            ),
            "dtype": config.method.dtype,
            "seeds": {"run": config.seed},
        },
    )
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if statistics.get("parameter_manifest_digest") != manifest.digest():
        raise RunnerError(
            "build-directions: fisher metric statistics were fitted under a "
            "different parameter manifest"
        )
    shard_manifest = ShardManifest.load(factors_dir, expected_identity=stored)
    diagonal = shard_manifest.read_rows(factors_dir)["features"][0].float()
    metric = DiagonalMetric.from_statistics(
        statistics,
        diagonal,
        exponent=second.metric_exponent,
        epsilon=second.metric_epsilon,
    )
    upstream = {"pair_metric_factors": stored.digest()}
    return metric, {"kind": "fisher", **metric.descriptor()}, upstream


def _load_derivative_statistics(config: AttributionRunConfig, manifest: Any):
    """Load a metric-derivative statistics artifact, enforcing the rank1
    refusal the upstream MetricDerivativeSpec owned (recorded deviation)."""
    from safetensors.torch import load_file

    derivative = config.second_order.metric_derivative
    directory = Path(derivative.statistics)
    statistics_path = directory / _STATISTICS_FILE
    if not statistics_path.is_file():
        raise RunnerError(
            f"build-directions: metric_derivative statistics artifact "
            f"{directory} has no {_STATISTICS_FILE}"
        )
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    estimator = statistics.get("estimator", "full")
    if estimator == "rank1":
        raise RunnerError(
            "build-directions: metric-derivative refuses rank1-reconstructed "
            "statistics — rank1 factors are not linear in the statistic, so "
            "the factored probe chain rule is wrong for them (upstream "
            "MetricDerivativeSpec guard). Re-estimate with 'full' or "
            "'marginals'."
        )
    if estimator not in ("full", "marginals"):
        raise RunnerError(
            f"build-directions: unknown statistics estimator {estimator!r}"
        )
    if statistics.get("parameter_manifest_digest") != manifest.digest():
        raise RunnerError(
            "build-directions: metric-derivative statistics were estimated "
            "under a different parameter manifest"
        )
    tensors = load_file(str(directory / "values.safetensors"))
    if "values" not in tensors:
        raise RunnerError(
            "build-directions: metric-derivative artifact must store a flat "
            "'values' tensor"
        )
    values = tensors["values"].reshape(-1).float()
    if values.numel() != manifest.included_numel:
        raise RunnerError(
            "build-directions: metric-derivative statistics have "
            f"{values.numel()} coordinates, the manifest includes "
            f"{manifest.included_numel}"
        )
    # Content digest over the whole artifact (statistics.json + values
    # bytes): regenerated statistics at the same path must change the
    # directions identity, never silently skip.
    return values, estimator == "marginals", statistics, artifact_digest(directory)


async def build_directions(config: AttributionRunConfig) -> PhaseReport:
    """Pair-gradient directions at the ONE declared checkpoint."""
    import torch

    from .losses import CausalLMLossAdapter
    from .manifest import flatten_tensors
    from .second_order import MetricDerivativeBackend

    second = _require_second_order(config, "build-directions")
    if config.method.dtype != "float32":
        raise RunnerError(
            "second-order phases require method.dtype float32 (double "
            "backprop / functional JVP contracts)"
        )
    _write_or_check_ledger(config, "build-directions")
    _assert_stride()
    layout = run_layout(config.output_dir)
    declared = _declared_checkpoint(config)
    checkpoint_dir, checkpoint_label, _ = declared
    query_dir = _resolve_query_checkpoint(config)
    data_path, dataset_digest = _resolve_query_dataset(config)
    tokenizer_digest = _tokenizer_content_digest(_tokenizer_dir(config, query_dir))
    tokenizer = _load_tokenizer(_tokenizer_dir(config, query_dir))
    checkpoint_digest = artifact_digest(checkpoint_dir)
    model = _load_model(checkpoint_dir, dtype=config.method.dtype,
                        device=config.data.device)
    manifest = _build_manifest(model, config)
    metric, metric_descriptor, upstream = _pair_metric(
        config, manifest, declared, tokenizer_digest
    )
    derivative_values = None
    derivative_factored = False
    if second.metric_derivative is not None:
        (derivative_values, derivative_factored, _,
         derivative_digest) = _load_derivative_statistics(config, manifest)
        upstream = {**upstream,
                    "metric_derivative_statistics": derivative_digest}
    identity = _identity(
        phase="build-directions",
        scoped=_scoped_config(config, "build-directions"),
        checkpoint_reference=str(checkpoint_dir),
        checkpoint_digest=checkpoint_digest,
        dataset_fingerprint=_dataset_fingerprint(dataset_digest, tokenizer_digest),
        parameter_manifest_digest=manifest.digest(),
        loss_convention=_loss_convention(config.query.objective,
                                         "per_sequence_sum"),
        basis_descriptor={
            "coordinates": "raw",
            "manifest_digest": manifest.digest(),
        },
        curvature_descriptor={
            "kind": "pair_gradient",
            "hessian_kind": second.hessian_kind,
            "checkpoint": checkpoint_label,
            "metric": metric_descriptor,
            "metric_derivative": second.metric_derivative is not None,
        },
        logra_descriptor=None,
        dtype=config.method.dtype,
        seeds={"run": config.seed},
        upstream_digests=upstream,
    )
    writer = ArtifactWriter(
        layout.directions,
        identity,
        feature_dim=max(1, manifest.included_numel),
        rows_per_shard=config.data.rows_per_shard,
        feature_dtype="float32",
    )
    columns_path = layout.directions / "direction_columns.json"
    all_columns = [
        {
            "pair": [index_a, index_b],
            "hessian_kind": second.hessian_kind,
            "metric": metric_descriptor,
            "metric_derivative": second.metric_derivative is not None,
            "checkpoint": checkpoint_label,
        }
        for index_a, index_b in second.pairs
    ]
    if writer.already_complete:
        if not columns_path.is_file():
            # Config-derived sidecar: rebuild rather than leave a complete
            # artifact permanently missing its column descriptions.
            from .artifacts import _atomic_write_text

            _atomic_write_text(
                columns_path, json.dumps(all_columns, indent=2,
                                         sort_keys=True) + "\n"
            )
        report = PhaseReport(
            "build-directions",
            (PhaseOutput("directions", layout.directions, identity.digest(),
                         True, rows=writer.rows_committed),),
        )
        _append_event(config, "build-directions", report.outputs)
        return report

    dataset = _dataset_adapter(
        objective=config.query.objective,
        data_path=data_path,
        tokenizer=tokenizer,
        config=config,
        reduction="per_sequence_sum",
        max_sequences=config.data.max_query_sequences,
    )
    batches = list(dataset.iter_batches(config.data.batch_size))
    by_sequence: dict[int, Any] = {}
    for batch in batches:
        for row in range(batch.input_ids.shape[0]):
            from .losses import TokenizedBatch

            by_sequence[int(batch.sequence_ids[row])] = TokenizedBatch(
                batch.input_ids[row : row + 1],
                batch.sequence_ids[row : row + 1],
                batch.target_mask[row : row + 1],
            )
    missing = sorted(
        {index for pair in second.pairs for index in pair} - set(by_sequence)
    )
    if missing:
        raise RunnerError(
            f"build-directions: pair indices {missing} name no query-dataset "
            f"sequence (available: {sorted(by_sequence)})"
        )
    adapter = CausalLMLossAdapter(model, reduction="per_sequence_sum",
                                  device=config.data.device)
    backend = MetricDerivativeBackend(model, manifest)

    def sequence_loss(index: int):
        return adapter.per_datapoint_losses(by_sequence[index]).losses[0]

    estimation_losses = None
    if derivative_values is not None:
        pool = sorted(by_sequence)[: second.metric_derivative.n_estimation_sequences]
        estimation_losses = [sequence_loss(index) for index in pool]

    committed_pairs = writer.rows_committed  # one row per pair, in order
    for pair_index, (index_a, index_b) in enumerate(second.pairs):
        if pair_index < committed_pairs:
            continue  # committed by an interrupted run; never re-append
        loss_a = sequence_loss(index_a)
        kwargs: dict[str, Any] = {"hessian_kind": second.hessian_kind}
        if second.hessian_kind == "ggn":
            kwargs["batch_a"] = by_sequence[index_a]
            kwargs["batch_b"] = by_sequence[index_b]
        if index_a == index_b:
            direction = backend.self_direction(loss_a, metric, **kwargs)
        else:
            loss_b = sequence_loss(index_b)
            direction = backend.pair_direction(
                loss_a, loss_b, metric=metric, **kwargs
            )
        if derivative_values is not None:
            gradient_a = flatten_tensors(
                backend.entries,
                torch.autograd.grad(sequence_loss(index_a), backend.params,
                                    allow_unused=True,
                                    materialize_grads=False),
            )
            gradient_b = flatten_tensors(
                backend.entries,
                torch.autograd.grad(sequence_loss(index_b), backend.params,
                                    allow_unused=True,
                                    materialize_grads=False),
            )
            direction = direction + backend.metric_term(
                gradient_a,
                gradient_b,
                estimation_losses,
                v=derivative_values,
                exponent=second.metric_exponent,
                eps=second.metric_epsilon,
                factored=derivative_factored,
            )
        writer.append(
            features=direction.detach().unsqueeze(0),
            sample_ids=torch.tensor([pair_index], dtype=torch.int64),
            sequence_ids=torch.tensor([index_a * _SAMPLE_ID_STRIDE + index_b],
                                      dtype=torch.int64),
            target_positions=torch.zeros(1, dtype=torch.int32),
        )
    from .artifacts import _atomic_write_text

    # Sidecar BEFORE the manifest: a complete artifact (manifest present)
    # can then never be missing its column descriptions.
    _atomic_write_text(
        columns_path,
        json.dumps(all_columns, indent=2, sort_keys=True) + "\n",
    )
    writer.finalize()
    report = PhaseReport(
        "build-directions",
        (PhaseOutput("directions", layout.directions, identity.digest(),
                     False, rows=len(all_columns)),),
    )
    _append_event(config, "build-directions", report.outputs)
    return report


async def sweep_jvp(config: AttributionRunConfig) -> PhaseReport:
    """Forward-JVP sweeps of the cached directions at the SAME declared
    checkpoint over the configured sweep stage's dataset."""
    import torch

    from .losses import CausalLMLossAdapter
    from .second_order import jvp_sweep as jvp_sweep_fn

    second = _require_second_order(config, "sweep-jvp")
    if second.sweep_stage is None:
        raise RunnerError(
            "sweep-jvp: second_order.sweep_stage must name the stage whose "
            "dataset provides the candidate examples"
        )
    if config.method.dtype != "float32":
        raise RunnerError(
            "second-order phases require method.dtype float32 (double "
            "backprop / functional JVP contracts)"
        )
    _write_or_check_ledger(config, "sweep-jvp")
    _assert_stride()
    layout = run_layout(config.output_dir)
    declared = _declared_checkpoint(config)
    checkpoint_dir, checkpoint_label, _ = declared
    sweep_stage = config.stages[_stage_index(config, second.sweep_stage)]
    sweep_resolved = resolve_stage(sweep_stage)
    query_dir = _resolve_query_checkpoint(config)
    _, query_dataset_digest = _resolve_query_dataset(config)
    tokenizer_digest = _tokenizer_content_digest(_tokenizer_dir(config, query_dir))
    tokenizer = _load_tokenizer(_tokenizer_dir(config, query_dir))
    checkpoint_digest = artifact_digest(checkpoint_dir)
    model = _load_model(checkpoint_dir, dtype=config.method.dtype,
                        device=config.data.device)
    manifest = _build_manifest(model, config)

    directions_stored = _check_upstream(
        "directions",
        layout.directions,
        {
            "resolved_config": _scoped_config(config, "build-directions"),
            "parameter_manifest_digest": manifest.digest(),
            "checkpoint_digest": checkpoint_digest,
            "dataset_fingerprint": _dataset_fingerprint(
                query_dataset_digest, tokenizer_digest
            ),
            "dtype": config.method.dtype,
            "seeds": {"run": config.seed},
        },
    )
    directions_manifest = ShardManifest.load(
        layout.directions, expected_identity=directions_stored
    )
    directions = directions_manifest.read_rows(layout.directions)[
        "features"
    ].float()
    if directions.shape[0] == 0:
        raise RunnerError("sweep-jvp: the directions artifact holds no rows")

    identity = _identity(
        phase="sweep-jvp",
        scoped=_scoped_config(config, "sweep-jvp"),
        checkpoint_reference=str(checkpoint_dir),
        checkpoint_digest=checkpoint_digest,
        dataset_fingerprint=_dataset_fingerprint(
            sweep_resolved.dataset_digest, tokenizer_digest
        ),
        parameter_manifest_digest=manifest.digest(),
        loss_convention=_loss_convention(sweep_stage.objective,
                                         config.method.row_reduction),
        basis_descriptor={
            "coordinates": "raw",
            "manifest_digest": manifest.digest(),
        },
        curvature_descriptor={
            "kind": "jvp_sweep",
            "checkpoint": checkpoint_label,
            "n_directions": int(directions.shape[0]),
        },
        logra_descriptor=None,
        dtype=config.method.dtype,
        seeds={"run": config.seed},
        upstream_digests={"directions": directions_stored.digest()},
    )
    writer = ArtifactWriter(
        layout.jvp,
        identity,
        feature_dim=int(directions.shape[0]),
        rows_per_shard=config.data.rows_per_shard,
        feature_dtype="float32",
    )
    if writer.already_complete:
        report = PhaseReport(
            "sweep-jvp",
            (PhaseOutput("jvp", layout.jvp, identity.digest(), True,
                         rows=writer.rows_committed),),
        )
        _append_event(config, "sweep-jvp", report.outputs)
        return report
    dataset = _dataset_adapter(
        objective=sweep_stage.objective,
        data_path=Path(sweep_resolved.dataset.path),
        tokenizer=tokenizer,
        config=config,
        reduction=config.method.row_reduction,
        max_sequences=config.data.max_stage_sequences,
    )
    adapter = CausalLMLossAdapter(model, reduction=config.method.row_reduction,
                                  device=config.data.device)
    committed = writer.rows_committed
    seen = 0
    chunk = second.direction_chunk_size
    for batch in dataset.iter_batches(config.data.batch_size):
        expected = _batch_expected_rows(batch, config.method.row_reduction)
        if seen + expected <= committed:
            seen += expected
            continue
        with torch.no_grad():
            loss_batch = adapter.per_datapoint_losses(batch)
        pieces = [
            jvp_sweep_fn(
                model,
                manifest,
                batch,
                directions[start : start + chunk],
                reduction=config.method.row_reduction,
                device=config.data.device,
            )
            for start in range(0, directions.shape[0], chunk)
        ]
        features = torch.cat(pieces, dim=1)
        drop = max(0, committed - seen)
        writer.append(
            features=features[drop:],
            sample_ids=loss_batch.sample_ids[drop:],
            sequence_ids=loss_batch.sequence_ids[drop:],
            target_positions=loss_batch.target_positions[drop:],
        )
        seen += expected
    writer.finalize()
    report = PhaseReport(
        "sweep-jvp",
        (PhaseOutput("jvp", layout.jvp, identity.digest(), False,
                     rows=writer.rows_committed),),
    )
    _append_event(config, "sweep-jvp", report.outputs)
    return report


# ================================================================= summarize ==
def _artifact_status(directory: Path) -> dict[str, Any]:
    identity_file = directory / ArtifactWriter.IDENTITY_FILE
    if not identity_file.is_file():
        return {"present": False, "complete": False, "rows": None}
    manifest_file = directory / ShardManifest.FILENAME
    if manifest_file.is_file():
        manifest = ShardManifest.load(directory)
        return {"present": True, "complete": True, "rows": manifest.total_rows}
    marker = directory / _FACTORS_COMPLETE_FILE
    if marker.is_file():
        return {"present": True, "complete": True, "rows": None}
    return {"present": True, "complete": False, "rows": None}


def _streaming_scores_status(
    directory: Path, expected_entries: list[str]
) -> dict[str, Any]:
    """Completeness of a committed ``score-source-streaming`` artifact.

    Complete means: identity bound, completeness manifest present, and every
    expected (stage, damping) entry recorded with its file on disk — the same
    existence discipline the materialized scores section uses (content
    digests are verified by the streaming phase itself on resume)."""
    identity_file = directory / ArtifactWriter.IDENTITY_FILE
    marker = directory / _SCORE_MANIFEST_FILE
    if not identity_file.is_file() or not marker.is_file():
        return {"complete": False, "entries": {}}
    completeness = json.loads(marker.read_text(encoding="utf-8"))
    entries = {
        entry_name: entry
        for entry_name, entry in completeness.get("entries", {}).items()
        if (directory / entry["file"]).is_file()
    }
    return {"complete": sorted(entries) == expected_entries, "entries": entries}


async def summarize(config: AttributionRunConfig) -> dict[str, Any]:
    """Completeness-counting summary (JSON + Markdown under ``summary/``).

    The SAVED resolved config (``run.json``) is the authority for
    ``allow_partial``: a partial requested output matrix is refused unless
    the run was declared partial-tolerant when it executed — an ad-hoc flag
    on the passed config never counts."""
    layout = run_layout(config.output_dir)
    if not layout.run_json.is_file():
        raise RunnerError(
            f"summarize: no run ledger at {layout.run_json} — nothing has "
            "run in this output directory"
        )
    # The SAVED resolved config is the single authority for what this run
    # promised (stages, sweep, second order) and whether partiality was
    # declared; the passed config contributes only the output location.
    saved = json.loads(layout.run_json.read_text(encoding="utf-8"))
    saved_config = saved.get("resolved_config", {})
    saved_allow_partial = bool(saved_config.get("allow_partial", False))
    stage_names = [entry["name"] for entry in saved_config["stages"]]
    damping_count = len(saved_config["method"]["damping_sweep"])

    sections: dict[str, Any] = {}
    factors = {
        name: _artifact_status(layout.factors / name) for name in stage_names
    }
    sections["factors"] = {
        "complete": all(status["complete"] for status in factors.values()),
        "counts": {name: status["rows"] for name, status in factors.items()},
        "present": {name: status["present"] for name, status in factors.items()},
    }
    rows = {
        name: _artifact_status(layout.rows / name) for name in stage_names
    }
    sections["rows"] = {
        "complete": all(status["complete"] for status in rows.values()),
        "counts": {name: status["rows"] or 0 for name, status in rows.items()},
    }
    queries = _artifact_status(layout.queries)
    sections["queries"] = {"complete": queries["complete"],
                           "counts": {"queries": queries["rows"] or 0}}

    expected_entries = sorted(
        f"{name}__damping-{index}"
        for name in stage_names
        for index in range(damping_count)
    )
    score_marker = layout.scores / _SCORE_MANIFEST_FILE
    present_entries: dict[str, Any] = {}
    if score_marker.is_file():
        completeness = json.loads(score_marker.read_text(encoding="utf-8"))
        for entry_name, entry in completeness.get("entries", {}).items():
            if (layout.scores / entry["file"]).is_file():
                present_entries[entry_name] = entry
    scores_complete = sorted(present_entries) == expected_entries
    scores_source = "materialized"
    if not scores_complete:
        # A committed, complete score-source-streaming artifact satisfies
        # the scores section: streaming runs persist [N, Q] score matrices
        # under streaming_scores/ and never write row shards (by design).
        # Detection is by the artifact's own completeness manifest — never
        # an ad-hoc flag. A complete materialized matrix keeps authority,
        # so materialized-run summaries are byte-unchanged.
        streaming = _streaming_scores_status(
            layout.streaming_scores, expected_entries
        )
        if streaming["complete"]:
            present_entries = streaming["entries"]
            scores_complete = True
            scores_source = "streaming"
    sections["scores"] = {
        "complete": scores_complete,
        "expected": expected_entries,
        "present": sorted(present_entries),
        "missing": sorted(set(expected_entries) - set(present_entries)),
        "counts": {
            name: {"n_queries": entry["n_queries"],
                   "n_train_rows": entry["n_train_rows"],
                   "n_examples": entry["n_examples"]}
            for name, entry in present_entries.items()
        },
    }
    if scores_source == "streaming":
        sections["scores"]["source"] = "streaming"
        if not sections["rows"]["complete"]:
            # Row shards are legitimately absent for a streaming run — the
            # streaming phase computes per-row gradients on the fly and
            # persists only scores. Any shards that DO exist (sidebars)
            # keep their counts.
            sections["rows"] = {
                **sections["rows"],
                "complete": True,
                "source": "streaming (row shards not produced by design)",
            }
    if saved_config.get("second_order") is not None:
        directions = _artifact_status(layout.directions)
        jvp = _artifact_status(layout.jvp)
        sections["directions"] = {"complete": directions["complete"],
                                  "counts": {"directions": directions["rows"] or 0}}
        sections["jvp"] = {"complete": jvp["complete"],
                           "counts": {"rows": jvp["rows"] or 0}}

    complete = all(section["complete"] for section in sections.values())
    if not complete and not saved_allow_partial:
        missing = {
            name: section
            for name, section in sections.items()
            if not section["complete"]
        }
        raise RunnerError(
            "summarize: the requested outputs are partial "
            f"(incomplete sections: {sorted(missing)}; missing score "
            f"entries: {sections['scores']['missing']}) and the SAVED "
            "resolved config does not declare allow_partial: true. Either "
            "finish the phases or re-run the whole chain with allow_partial "
            "declared up front — an ad-hoc flag at summarize time does not "
            "count."
        )
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": _now(),
        "scimt_commit": _scimt_commit(),
        "source_commit": SOURCE_COMMIT,
        "complete": complete,
        "allow_partial": saved_allow_partial,
        "sections": sections,
    }
    layout.summary.mkdir(parents=True, exist_ok=True)
    from .artifacts import _atomic_write_text

    _atomic_write_text(
        layout.summary / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "# Attribution run summary",
        "",
        f"- complete: **{complete}** (allow_partial: {saved_allow_partial})",
        f"- scimt commit: `{summary['scimt_commit']}`",
        f"- source commit: `{SOURCE_COMMIT}`",
        "",
        "| section | complete | counts |",
        "|---|---|---|",
    ]
    for name, section in sections.items():
        lines.append(
            f"| {name} | {section['complete']} | "
            f"`{json.dumps(section.get('counts', {}), sort_keys=True)}` |"
        )
    _atomic_write_text(layout.summary / "summary.md", "\n".join(lines) + "\n")
    _append_event(config, "summarize", ())
    return summary


# =================================================================== dry run ==
def _count_jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _safetensors_shapes(checkpoint_dir: Path) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for file in sorted(checkpoint_dir.glob("*.safetensors")):
        with file.open("rb") as handle:
            header_length = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_length))
        for name, entry in header.items():
            if name != "__metadata__":
                shapes[name] = list(entry.get("shape", []))
    return shapes


def _estimate_included_parameters(
    checkpoint_dir: Path, include: tuple[str, ...], exclude: tuple[str, ...]
) -> int | None:
    """Torch-free parameter-count estimate from safetensors headers (labeled
    an estimate: header order and tied weights are not model introspection)."""
    shapes = _safetensors_shapes(checkpoint_dir)
    if not shapes:
        return None
    selected = _selected_safetensors_shapes(shapes, include, exclude)
    return sum(math.prod(shape) if shape else 1 for _, shape in selected)


def _selected_safetensors_shapes(
    shapes: dict[str, list[int]],
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Torch-free selected name/shape signature in canonical name order."""

    includes = [re.compile(pattern) for pattern in include]
    excludes = [re.compile(pattern) for pattern in exclude]
    return tuple(
        (name, tuple(int(dimension) for dimension in shape))
        for name, shape in sorted(shapes.items())
        if not any(regex.fullmatch(name) for regex in excludes)
        and any(regex.fullmatch(name) for regex in includes)
    )


def _checkpoint_parameter_signature(
    checkpoint_dir: Path, include: tuple[str, ...], exclude: tuple[str, ...]
) -> tuple[tuple[str, tuple[int, ...]], ...] | None:
    shapes = _safetensors_shapes(checkpoint_dir)
    if not shapes:
        return None
    return _selected_safetensors_shapes(shapes, include, exclude)


def _signature_report(
    signature: tuple[tuple[str, tuple[int, ...]], ...] | None,
) -> dict[str, Any]:
    if signature is None:
        return {"available": False, "digest": None, "tensor_count": None}
    payload = [[name, list(shape)] for name, shape in signature]
    digest = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return {
        "available": True,
        "digest": digest,
        "tensor_count": len(signature),
    }


def _probe_snapshot(path: Path, checkpoint_step: int | None) -> dict[str, Any]:
    """Torch-free Adam availability probe: JSON metadata + file presence and
    sizes only. Full byte-level validation happens at phase time through
    ``validate_optimizer_snapshot``."""
    manifest_path = path / "optimizer_manifest.json"
    if not manifest_path.is_file():
        return {"available": False,
                "reason": f"no optimizer_manifest.json under {path}"}
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "reason": f"unreadable manifest: {error}"}
    optimizer = document.get("optimizer")
    if not isinstance(optimizer, dict):
        return {"available": False, "reason": "malformed optimizer block"}
    for record in document.get("shards", []):
        shard_path = path / str(record.get("filename", ""))
        if not shard_path.is_file():
            return {"available": False,
                    "reason": f"missing shard {record.get('filename')!r}"}
        if shard_path.stat().st_size != record.get("num_bytes"):
            return {"available": False,
                    "reason": f"shard {record.get('filename')!r} size drift"}
    sha_path = path / "parameter_manifest.sha256"
    recorded = document.get("parameter_manifest_digest")
    if not sha_path.is_file() or sha_path.read_text(
        encoding="ascii"
    ).strip() != recorded:
        return {"available": False,
                "reason": "parameter manifest digest file missing or drifted"}
    step = optimizer.get("step")
    if checkpoint_step is not None and step != checkpoint_step:
        return {"available": False,
                "reason": f"snapshot step {step} != checkpoint step "
                          f"{checkpoint_step}"}
    return {
        "available": True,
        "validated": "metadata-only (byte-level validation at phase time)",
        "step": step,
        "beta1": optimizer.get("beta1"),
        "beta2": optimizer.get("beta2"),
        "epsilon": optimizer.get("epsilon"),
        "weight_decay": optimizer.get("weight_decay"),
        "parameter_manifest_digest": recorded,
        "world_size": document.get("world_size"),
    }


def _identity_preview(
    phase: str,
    scoped: dict[str, Any],
    *,
    checkpoint_reference: str,
    dataset_fingerprint: str | None,
    loss_convention: dict[str, Any],
    basis_coordinates: str,
    curvature_descriptor: dict[str, Any],
    logra: dict[str, Any] | None,
    dtype: str,
    seeds: dict[str, int],
) -> dict[str, Any]:
    pending = ["checkpoint_digest", "parameter_manifest_digest",
               "identity_digest"]
    if logra is not None:
        pending.append("logra_descriptor.projection_digest")
    identity = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "producing_command": _producing_command(phase),
        "scimt_commit": _scimt_commit(),
        "source_commit": SOURCE_COMMIT,
        "resolved_config": scoped,
        "checkpoint_reference": checkpoint_reference,
        "checkpoint_digest": None,
        "dataset_fingerprint": dataset_fingerprint,
        "parameter_manifest_digest": None,
        "loss_convention": loss_convention,
        "basis_descriptor": {"coordinates": basis_coordinates,
                             "manifest_digest": None},
        "curvature_descriptor": curvature_descriptor,
        "logra_descriptor": logra,
        "dtype": dtype,
        "seeds": seeds,
        "upstream_digests": "computed at phase time",
    }
    return {"identity": identity, "pending": pending}


async def dry_run(config: AttributionRunConfig) -> dict[str, Any]:
    """Resolve everything a run would consume — stage metadata, datasets,
    counts, Adam and manifest availability, factor partitions, and planned
    output identities — without loading any model. Runs without an Adam
    estimator remain torch-free; estimator preflight tokenizes its calibration
    corpus to prove the exact usable population before GPU work begins.

    Pure: writes nothing. Hard resolution failures (missing/contradictory
    stage artifacts) raise; capability gaps for the configured method are
    returned in ``blockers``.
    """
    method = config.method
    layout = run_layout(config.output_dir)
    blockers: list[str] = []
    warnings: list[str] = []
    # Mirror _require_scorable_method exactly: a config that score-source
    # would accept must report no method blockers here, and every refused
    # combination must appear (dry_run is the launch gate for orchestration).
    if method.curvature == "ggn":
        blockers.append(
            "method.curvature 'ggn': no fitted GGN segment operator exists — "
            "use 'fisher'/'ekfac'/'ekfac_adam' for SOURCE, GGN lives in "
            "second-order phases"
        )
    if method.basis == "ekfac":
        blockers.append("method.basis 'ekfac' is not supported by score-source")
    if method.curvature == "ekfac_adam":
        if method.basis != "adam":
            blockers.append(
                "method.curvature 'ekfac_adam' factors live in stage-local "
                "Adam coordinates; only basis 'adam' rows and queries can "
                "consume them"
            )
        if not config.factors.use_empirical_fisher:
            blockers.append(
                "method.curvature 'ekfac_adam' conditions the EMPIRICAL "
                "Fisher; set factors.use_empirical_fisher: true"
            )
        if (
            config.adam_moment_estimator is not None
            and not (layout.adam_moments / "paired_batches.json").is_file()
        ):
            # Not a blocker: estimate-adam is a later phase of this same run
            # and fit-factors hard-refuses until it has committed.
            warnings.append(
                "curvature 'ekfac_adam': no committed estimate-adam "
                "artifacts yet (adam_moments/paired_batches.json missing); "
                "fit-factors will refuse until estimate-adam runs"
            )
    elif method.basis in ("fisher", "adam") and method.curvature != "fisher":
        blockers.append(
            f"method.basis {method.basis!r} requires method.curvature "
            "'fisher' (EK-FAC factors cannot be exactly transported into a "
            "diagonal basis)"
        )

    stages_report = []
    resolved_lookup: dict[str, Any] = {}
    parameter_signatures: dict[
        str, tuple[tuple[str, tuple[int, ...]], ...] | None
    ] = {}
    for stage in config.stages:
        # Resolve WITHOUT the snapshot path: full snapshot validation loads
        # safetensors slices; the dry run probes JSON metadata instead.
        resolved = resolve_stage(
            dataclasses.replace(stage, optimizer_snapshot=None)
        )
        resolved_lookup[stage.name] = resolved
        adam: dict[str, Any]
        if stage.optimizer_snapshot is None:
            adam = {"available": False, "reason": "no optimizer_snapshot declared"}
        else:
            adam = _probe_snapshot(Path(stage.optimizer_snapshot),
                                   resolved.global_step)
        if (
            method.basis == "adam"
            and config.adam_moment_estimator is None
            and not adam["available"]
        ):
            blockers.append(
                f"stage {stage.name!r}: Adam basis requested but the "
                f"optimizer snapshot is unavailable: {adam.get('reason')}"
            )
        manifest_availability: dict[str, str] = {}
        if stage.optimizer_snapshot is not None:
            sha = Path(stage.optimizer_snapshot) / "parameter_manifest.sha256"
            if sha.is_file():
                manifest_availability["optimizer_snapshot"] = sha.read_text(
                    encoding="ascii"
                ).strip()
        factor_sha = layout.factors / stage.name / "ekfac" / "parameter_manifest.sha256"
        if factor_sha.is_file():
            manifest_availability["factors"] = factor_sha.read_text(
                encoding="ascii"
            ).strip()
        parameter_signature = _checkpoint_parameter_signature(
            resolved.checkpoint_dir,
            config.parameters.include,
            config.parameters.exclude,
        )
        parameter_signatures[f"stage {stage.name!r}"] = parameter_signature
        if parameter_signature is None:
            blockers.append(
                f"stage {stage.name!r}: selected parameter signature is "
                "unavailable because the checkpoint has no readable "
                "safetensors headers; convert the checkpoint to safetensors "
                "before launch"
            )
        elif not parameter_signature:
            blockers.append(
                f"stage {stage.name!r}: selected parameter signature is empty; "
                "the configured include/exclude patterns select no checkpoint "
                "tensors"
            )
        stages_report.append(
            {
                "name": stage.name,
                "objective": stage.objective,
                "run_dir": str(resolved.run_dir),
                "checkpoint_dir": str(resolved.checkpoint_dir),
                "global_step": resolved.global_step,
                "lr_steps": resolved.lr_steps,
                "lr_steps_source": resolved.lr_steps_source,
                "n_examples": resolved.n_examples,
                "weight_decay": resolved.weight_decay,
                "dataset_path": str(resolved.dataset.path),
                "dataset_digest": resolved.dataset_digest,
                "training_dataset_path": str(
                    stage.training_dataset.path
                    if stage.training_dataset is not None
                    else resolved.dataset.path
                ),
                "training_dataset_digest": resolved.training_dataset_digest,
                "dataset_rows": _count_jsonl_rows(Path(resolved.dataset.path)),
                "estimated_included_parameters": _estimate_included_parameters(
                    resolved.checkpoint_dir,
                    config.parameters.include,
                    config.parameters.exclude,
                ),
                "selected_parameter_signature": _signature_report(
                    parameter_signature
                ),
                "adam": adam,
                "manifest_availability": manifest_availability,
                "artifacts": {
                    "factors": _artifact_status(layout.factors / stage.name),
                    "rows": _artifact_status(layout.rows / stage.name),
                },
            }
        )

    query_dir = _resolve_query_checkpoint(config)
    tokenizer_dir = _tokenizer_dir(config, query_dir)
    tokenizer_digest = _tokenizer_content_digest(tokenizer_dir)
    estimator_report: dict[str, Any] | None = None
    if config.adam_moment_estimator is not None:
        estimator = config.adam_moment_estimator
        estimator_path, estimator_digest = _resolve_dataset_reference(
            Path(estimator.dataset.path),
            expected_digest=estimator.dataset.expected_digest,
            label="Adam moment estimator dataset",
        )
        source_rows = (
            _count_jsonl_rows(estimator_path) if estimator_path.is_file() else None
        )
        required_presentations = (
            estimator.num_batches * estimator.global_batch_size
        )
        tokenizer = _load_tokenizer(tokenizer_dir)
        calibration_dataset = _adam_calibration_dataset(
            config, estimator_path, tokenizer
        )
        usable_sequences = len(calibration_dataset)
        del calibration_dataset, tokenizer
        paired_path = layout.adam_moments / "paired_batches.json"
        included_counts = [
            stage_report["estimated_included_parameters"]
            for stage_report in stages_report
        ]
        peak_selected_count = (
            None
            if any(count is None for count in included_counts)
            else max(int(count) for count in included_counts)
        )
        selected_storage = (
            None
            if any(count is None for count in included_counts)
            else 4 * sum(int(count) for count in included_counts)
        )
        peak_accumulator_storage = (
            None if peak_selected_count is None else 8 * peak_selected_count
        )
        peak_selected_working_storage = (
            None
            if peak_selected_count is None
            else 8 * peak_selected_count
            + max(
                4 * peak_selected_count,
                16 * min(peak_selected_count, 1 << 20),
            )
        )
        estimator_report = {
            "mode": "paired_checkpoint_local",
            "dataset_path": str(estimator_path),
            "dataset_digest": estimator_digest,
            "objective": estimator.objective,
            "source_rows": source_rows,
            "usable_tokenized_sequences": usable_sequences,
            "population_validation": "exact_model_free_tokenization",
            "num_batches": estimator.num_batches,
            "global_batch_size": estimator.global_batch_size,
            "micro_batch_size": estimator.micro_batch_size,
            "required_presentations": required_presentations,
            "checkpoint_count": len(config.stages),
            "global_batch_equivalents": (
                estimator.num_batches * len(config.stages)
            ),
            "selected_moment_storage_bytes": selected_storage,
            "peak_selected_accumulator_bytes": peak_accumulator_storage,
            "peak_selected_working_bytes_upper_bound": (
                peak_selected_working_storage
            ),
            "paired_manifest": {
                "present": paired_path.is_file(),
                "digest": (
                    artifact_digest(paired_path) if paired_path.is_file() else None
                ),
            },
            "stage_artifacts": {
                stage.name: _artifact_status(layout.adam_moments / stage.name)
                for stage in config.stages
            },
        }
        if required_presentations > usable_sequences:
            blockers.append(
                "Adam moment estimator sampling without replacement needs "
                f"{required_presentations} sequences, but exact model-free "
                f"tokenization contains {usable_sequences} usable sequences"
            )

    query_data_path, query_dataset_digest = _resolve_query_dataset(config)
    query_parameter_signature = _checkpoint_parameter_signature(
        query_dir,
        config.parameters.include,
        config.parameters.exclude,
    )
    parameter_signatures["query"] = query_parameter_signature
    if query_parameter_signature is None:
        blockers.append(
            "query: selected parameter signature is unavailable because the "
            "checkpoint has no readable safetensors headers; convert the "
            "checkpoint to safetensors before launch"
        )
    elif not query_parameter_signature:
        blockers.append(
            "query: selected parameter signature is empty; the configured "
            "include/exclude patterns select no checkpoint tensors"
        )
    query_report = {
        "checkpoint_dir": str(query_dir),
        "dataset_path": str(query_data_path),
        "dataset_digest": query_dataset_digest,
        "dataset_rows": _count_jsonl_rows(query_data_path),
        "estimated_included_parameters": _estimate_included_parameters(
            query_dir, config.parameters.include, config.parameters.exclude
        ),
        "selected_parameter_signature": _signature_report(
            query_parameter_signature
        ),
        "artifacts": {"queries": _artifact_status(layout.queries)},
    }
    if query_parameter_signature:
        reference = dict(query_parameter_signature)
        for label, signature in parameter_signatures.items():
            if (
                label == "query"
                or not signature
                or signature == query_parameter_signature
            ):
                continue
            candidate = dict(signature)
            missing = sorted(set(reference) - set(candidate))
            unexpected = sorted(set(candidate) - set(reference))
            reshaped = sorted(
                name
                for name in set(reference) & set(candidate)
                if reference[name] != candidate[name]
            )
            blockers.append(
                f"{label} selected parameter signature differs from query: "
                f"missing={missing}, unexpected={unexpected}, reshaped={reshaped}"
            )
    if method.basis == "adam" and artifact_digest(
        query_dir
    ) != artifact_digest(resolved_lookup[config.stages[-1].name].checkpoint_dir):
        blockers.append(
            "stage-local Adam coordinates require the query checkpoint to "
            "equal the final chronological stage checkpoint"
        )

    expected = {
        "damping_sweep": list(method.damping_sweep),
        "factor_samples_per_stage": config.factors.samples,
        "score_matrix_entries": sorted(
            f"{stage.name}__damping-{index}"
            for stage in config.stages
            for index in range(len(method.damping_sweep))
        ),
        "row_reduction": method.row_reduction,
        "sequence_length": config.data.sequence_length,
        "direction_pairs": (
            None if config.second_order is None
            else [list(pair) for pair in config.second_order.pairs]
        ),
        "note": (
            "exact row counts require tokenization; dataset_rows are source "
            "JSONL line counts (per-sequence reductions are bounded by them)"
        ),
    }

    planned = []
    for stage in config.stages:
        resolved = resolved_lookup[stage.name]
        planned.append(
            {
                "output": f"factors/{stage.name}",
                **_identity_preview(
                    "fit-factors",
                    _scoped_config(config, "fit-factors", stage.name),
                    checkpoint_reference=str(resolved.checkpoint_dir),
                    dataset_fingerprint=_dataset_fingerprint(
                        resolved.dataset_digest, tokenizer_digest
                    ),
                    loss_convention={
                        "loss": "causal_lm_cross_entropy",
                        "reduction": "sampled_token_sum",
                        "target_policy": _TARGET_POLICY[stage.objective],
                        "sampler": "build_ekfac_sample_items",
                        "sample_id_stride": _SAMPLE_ID_STRIDE,
                    },
                    basis_coordinates=(
                        "adam_stage_local"
                        if method.curvature == "ekfac_adam"
                        else "raw"
                    ),
                    curvature_descriptor={
                        "method": method.curvature,
                        "fit": _fit_config_payload(config),
                    },
                    logra=None,
                    dtype=method.dtype,
                    seeds={"run": config.seed},
                ),
            }
        )
        logra = config.resolved()["method"]["logra"]
        planned.append(
            {
                "output": f"rows/{stage.name}",
                **_identity_preview(
                    "compute-rows",
                    _scoped_config(config, "compute-rows", stage.name),
                    checkpoint_reference=str(resolved.checkpoint_dir),
                    dataset_fingerprint=_dataset_fingerprint(
                        resolved.dataset_digest, tokenizer_digest
                    ),
                    loss_convention=_loss_convention(
                        stage.objective, method.row_reduction
                    ),
                    basis_coordinates=(
                        "raw" if logra is None else "logra_B"
                    ),
                    curvature_descriptor={"kind": "gradient_rows"},
                    logra=logra,
                    dtype=method.dtype,
                    seeds=(
                        {"run": config.seed}
                        if logra is None
                        else {"run": config.seed, "logra": logra["seed"]}
                    ),
                ),
            }
        )
    planned.append(
        {
            "output": "queries",
            **_identity_preview(
                "build-queries",
                _scoped_config(config, "build-queries"),
                checkpoint_reference=str(query_dir),
                dataset_fingerprint=_dataset_fingerprint(
                    query_dataset_digest, tokenizer_digest
                ),
                loss_convention=_loss_convention(
                    config.query.objective, method.row_reduction
                ),
                basis_coordinates=(
                    "raw" if method.logra is None else "logra_B"
                ),
                curvature_descriptor={"kind": "gradient_rows"},
                logra=config.resolved()["method"]["logra"],
                dtype=method.dtype,
                # Exactly as build-queries binds them (_row_phase adds the
                # logra seed whenever a projection is configured).
                seeds=(
                    {"run": config.seed}
                    if method.logra is None
                    else {"run": config.seed, "logra": method.logra.seed}
                ),
            ),
        }
    )
    score_scoped = _scoped_config(config, "score-source")
    score_scoped["resolved_lr_steps"] = {
        name: resolved_lookup[name].lr_steps for name in resolved_lookup
    }
    planned.append(
        {
            "output": "scores",
            **_identity_preview(
                "score-source",
                score_scoped,
                checkpoint_reference=str(query_dir),
                dataset_fingerprint=_dataset_fingerprint(
                    query_dataset_digest, tokenizer_digest
                ),
                loss_convention=_loss_convention(
                    config.query.objective, method.row_reduction
                ),
                basis_coordinates=method.basis,
                curvature_descriptor={
                    "method": method.curvature,
                    "damping_sweep": list(method.damping_sweep),
                },
                logra=None,
                dtype=method.dtype,
                seeds={"run": config.seed},
            ),
        }
    )

    return {
        "resolved_config": config.resolved(),
        "scimt_commit": _scimt_commit(),
        "source_commit": SOURCE_COMMIT,
        "stages": stages_report,
        "adam_moment_estimator": estimator_report,
        "query": query_report,
        "tokenizer": {
            "directory": str(tokenizer_dir),
            "content_digest": tokenizer_digest,
        },
        "factor_plan": _fit_config_payload(config),
        "expected": expected,
        "planned_identities": planned,
        "blockers": blockers,
        "warnings": warnings,
    }


PHASES: dict[str, Callable[[AttributionRunConfig], Awaitable[Any]]] = {
    "estimate-adam": estimate_adam,
    "fit-factors": fit_factors,
    "compute-rows": compute_rows,
    "build-queries": build_queries,
    "score-source": score_source,
    "score-source-streaming": score_source_streaming,
    "build-directions": build_directions,
    "sweep-jvp": sweep_jvp,
    "summarize": summarize,
    "dry-run": dry_run,
}
