"""Config-first attribution runner: staged, resumable, provenance-checked.

Async-native library verbs (the caller owns the event loop; the
``scimt-attribution`` console script in ``cli.py`` is the one sanctioned thin
wrapper, plan Task 7). One :class:`AttributionRunConfig` drives every phase:

- ``fit-factors``   per-segment curvature (EK-FAC via Kronfluence, or the
  empirical Fisher diagonal) on seeded samples from that stage's actual
  training distribution (``build_ekfac_sample_items`` — the one sampler).
- ``compute-rows``  per-example train gradient rows per stage (optionally
  LoGra-projected), resumable by committed shard.
- ``build-queries`` measurement-loss gradient rows at the final query
  checkpoint.
- ``score-source``  chronological SOURCE scoring over validated artifacts;
  one global basis, per-segment ``1/N`` applied exactly once, damping sweep.
- ``build-directions`` / ``sweep-jvp``  second-order pair directions and
  forward-JVP sweeps at ONE explicitly declared checkpoint.
- ``summarize``     completeness-counting summary; refuses a partial
  requested matrix unless the SAVED resolved config declares
  ``allow_partial: true``.
- ``dry-run``       resolves configs, stage metadata, counts, Adam and
  manifest availability, factor partitions, and output-identity previews
  WITHOUT importing torch or loading any model.

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
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

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
_WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin")


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
    factors: Path
    rows: Path
    queries: Path
    scores: Path
    directions: Path
    jvp: Path
    summary: Path


def run_layout(output_dir: str | Path) -> RunLayout:
    root = Path(output_dir)
    return RunLayout(
        root=root,
        run_json=root / "run.json",
        events=root / "events.jsonl",
        factors=root / "factors",
        rows=root / "rows",
        queries=root / "queries",
        scores=root / "scores",
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
            return entry
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
    method = resolved["method"]
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
    if phase == "fit-factors":
        return {
            **base,
            "stage": _resolved_stage_entry(resolved, stage_name),
            "data": stage_data,
            "factors": resolved["factors"],
            "curvature": method["curvature"],
            "dtype": method["dtype"],
        }
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
    if phase == "score-source":
        return {
            **base,
            "stages": resolved["stages"],
            "query": resolved["query"],
            "data": {**stage_data, **query_data},
            "method": method,
        }
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
def _load_model(checkpoint_dir: str | Path, *, dtype: str, device: str):
    """Load a causal LM from a resolved local checkpoint dir (lazy heavy)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=getattr(torch, dtype),
        local_files_only=True,
    )
    model.to(torch.device(device))
    model.eval()
    return model


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


def _resolve_query_checkpoint(config: AttributionRunConfig) -> Path:
    """The final query checkpoint: a scimt run dir (checkpoint.json) or a
    plain consolidated HF checkpoint dir. Loud on anything else."""
    from scimt.train.checkpoint import MANIFEST_NAME, Checkpoint

    declared = Path(config.query.checkpoint.path)
    candidate = declared
    if candidate.is_file() and candidate.name == MANIFEST_NAME:
        candidate = candidate.parent
    if (candidate / MANIFEST_NAME).is_file():
        try:
            checkpoint = Checkpoint.load(candidate)
            state = checkpoint.require_state()
        except (OSError, ValueError, KeyError) as error:
            raise RunnerError(
                f"query checkpoint {declared} is not a usable scimt run: "
                f"{error}"
            ) from error
        state_dir = Path(state)
    else:
        state_dir = candidate
    if not state_dir.is_dir():
        raise RunnerError(f"query checkpoint dir {state_dir} does not exist")
    if (state_dir / "adapter_config.json").exists():
        raise RunnerError(
            f"query checkpoint {state_dir} is an unmerged adapter directory — "
            "merge it into a full checkpoint before attributing"
        )
    if not (state_dir / "config.json").is_file() or not any(
        any(state_dir.glob(pattern)) for pattern in _WEIGHT_GLOBS
    ):
        raise RunnerError(
            f"query checkpoint {state_dir} is not a loadable full checkpoint "
            "(config.json + weights required)"
        )
    expected = config.query.checkpoint.expected_digest
    if expected is not None and artifact_digest(state_dir) != expected:
        raise RunnerError(
            f"query checkpoint content digest does not match the declared "
            f"expected_digest {expected}"
        )
    return state_dir


def _resolve_query_dataset(config: AttributionRunConfig) -> tuple[Path, str]:
    from scimt.dataset import Dataset

    declared = Path(config.query.dataset.path)
    manifest_dir = declared.parent if declared.is_file() else declared
    try:
        dataset = Dataset.load(manifest_dir)
    except (OSError, ValueError, TypeError, FileNotFoundError) as error:
        raise RunnerError(
            f"query dataset {declared} has no readable dataset manifest "
            f"(dataset.json): {error}"
        ) from error
    data_path = Path(dataset.path)
    if not data_path.exists():
        raise RunnerError(f"query dataset manifest points at missing data {data_path}")
    digest = artifact_digest(data_path)
    expected = config.query.dataset.expected_digest
    if expected is not None and digest != expected:
        raise RunnerError(
            f"query dataset content digest {digest} does not match the "
            f"declared expected_digest {expected}"
        )
    return data_path, digest


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
):
    from .datasets import ChatSFTDataset, PackedMidtrainingDataset

    cls = PackedMidtrainingDataset if objective == "midtraining" else ChatSFTDataset
    return cls(
        data_path,
        tokenizer,
        config.data.sequence_length,
        config.seed,
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


def _check_adam_manifest(
    stage: AttributionStage, resolved: ResolvedStage, manifest: Any
) -> None:
    """Requirement: the snapshot's recorded parameter-manifest digest must
    EXACTLY match the manifest actually built from the resolved checkpoint.
    Both sides label models through ``stable_model_identifier`` (load-path
    independent), so this is a strict digest equality — no relabeling."""
    info = resolved.optimizer_snapshot
    if info is None:  # pragma: no cover - guarded by resolve_stage
        raise RunnerError(f"stage {stage.name!r} has no validated Adam snapshot")
    if manifest.digest() != info.parameter_manifest_digest:
        raise RunnerError(
            f"stage {stage.name!r}: Adam-basis parameter-manifest cross-check "
            "failed — the optimizer snapshot records manifest digest "
            f"{info.parameter_manifest_digest}, but the manifest built from "
            f"the resolved checkpoint {resolved.checkpoint_dir} under the "
            "configured parameter selection differs. Adam second moments and "
            "gradient rows would not share coordinates; fix the parameter "
            "selection or re-capture the snapshot."
        )


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
            "curvature must be 'fisher' or 'ekfac'; GGN products are "
            "available in the second-order phases (hessian_kind: ggn)"
        )
    _write_or_check_ledger(config, "fit-factors")
    resolved_stages = _resolve_stages(
        config, require_adam=method.basis == "adam"
    )
    layout = run_layout(config.output_dir)
    query_dir = _resolve_query_checkpoint(config)
    tokenizer_digest = _tokenizer_content_digest(_tokenizer_dir(config, query_dir))
    tokenizer = _load_tokenizer(_tokenizer_dir(config, query_dir))
    outputs = []
    for stage, resolved in zip(config.stages, resolved_stages, strict=True):
        outputs.append(
            _fit_stage_factors(config, stage, resolved, tokenizer,
                               layout.factors / stage.name, tokenizer_digest)
        )
    report = PhaseReport("fit-factors", tuple(outputs))
    _append_event(config, "fit-factors", report.outputs)
    return report


def _fit_config_payload(config: AttributionRunConfig) -> dict[str, Any]:
    factors = config.factors
    return {
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


def _fit_stage_factors(
    config: AttributionRunConfig,
    stage: AttributionStage,
    resolved: ResolvedStage,
    tokenizer: Any,
    directory: Path,
    tokenizer_digest: str,
) -> PhaseOutput:
    _assert_stride()
    name = f"factors/{stage.name}"
    method = config.method
    checkpoint_digest = artifact_digest(resolved.checkpoint_dir)
    model = _load_model(resolved.checkpoint_dir, dtype=method.dtype,
                        device=config.data.device)
    manifest = _build_manifest(model, config)
    if method.basis == "adam":
        _check_adam_manifest(stage, resolved, manifest)
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
        basis_descriptor={
            "coordinates": "raw",
            "manifest_digest": manifest.digest(),
        },
        curvature_descriptor={
            "method": method.curvature,
            "fit": _fit_config_payload(config),
        },
        logra_descriptor=None,
        dtype=method.dtype,
        seeds={"run": config.seed},
        upstream_digests={},
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
                              manifest, dataset)


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
        ids = item["input_ids"].unsqueeze(0)
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
    config, name, directory, identity, model, manifest, dataset
) -> PhaseOutput:
    from .ekfac import fit_ekfac, load_ekfac

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
        factors = load_ekfac(ekfac_dir, manifest)
        if completion.get("snapshot") != factors.snapshot:
            raise ArtifactIntegrityError(
                f"EK-FAC factor bytes under {ekfac_dir} changed after "
                "completion was recorded"
            )
        return PhaseOutput(name, directory, identity.digest(), True)
    factors = fit_ekfac(
        model, dataset, manifest, _fit_config_payload(config), ekfac_dir
    )
    from .artifacts import _atomic_write_text

    _atomic_write_text(
        marker,
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "identity_digest": identity.digest(),
                "snapshot": factors.snapshot,
                "curvature": "ekfac",
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
    from .gradients import BatchedVJPBackend
    from .losses import CausalLMLossAdapter

    adapter = CausalLMLossAdapter(model, reduction=reduction, device=device)
    backend = BatchedVJPBackend(model, manifest)
    committed = writer.rows_committed
    seen = 0
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
    # (stage, resolved) for per-stage rows: names the identity scope and
    # carries the snapshot for the Adam-basis manifest cross-check. None for
    # the query phase.
    stage_context: tuple[AttributionStage, ResolvedStage] | None,
) -> PhaseOutput:
    _assert_stride()
    method = config.method
    checkpoint_digest = artifact_digest(checkpoint_dir)
    model = _load_model(checkpoint_dir, dtype=method.dtype,
                        device=config.data.device)
    base_manifest = _build_manifest(model, config)
    if stage_context is not None and method.basis == "adam":
        _check_adam_manifest(stage_context[0], stage_context[1], base_manifest)
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
        rows_per_shard=config.data.rows_per_shard,
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
    resolved_stages = _resolve_stages(
        config, require_adam=config.method.basis == "adam"
    )
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
            "curvature must be 'fisher' or 'ekfac'; GGN products live in the "
            "second-order phases (hessian_kind: ggn)"
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
    if method.basis in ("fisher", "adam") and method.curvature != "fisher":
        raise RunnerError(
            f"score-source: basis {method.basis!r} transports rows through a "
            "diagonal metric, which requires diagonal curvature "
            "(curvature: fisher) — EK-FAC factors cannot be exactly "
            "transported into a diagonal basis"
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
    from .ekfac import load_ekfac
    from .manifest import ParameterManifest

    ekfac_dir = directory / "ekfac"
    factor_manifest = ParameterManifest.load(ekfac_dir)
    if factor_manifest.digest() != shared_manifest_digest:
        raise RunnerError(
            f"factors/{stage.name}: EK-FAC manifest digest does not match "
            "the shared run manifest"
        )
    factors = load_ekfac(ekfac_dir, factor_manifest)
    if completion.get("snapshot") != factors.snapshot:
        raise ArtifactIntegrityError(
            f"factors/{stage.name}: EK-FAC factor bytes changed after "
            "completion was recorded"
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


def _basis_metric(
    config: AttributionRunConfig,
    resolved_stages: list[ResolvedStage],
    factor_payloads: dict[str, Any],
    shared_manifest_digest: str,
    damping: float,
):
    """The ONE global diagonal basis metric (fisher/adam), from the LAST
    chronological stage. Returns (metric, descriptor_extras)."""
    from .metrics import DiagonalMetric

    method = config.method
    last_stage = config.stages[-1]
    last_resolved = resolved_stages[-1]
    if method.basis == "adam":
        from scimt.train.attribution_snapshot import load_optimizer_snapshot

        info = last_resolved.optimizer_snapshot
        if info is None:  # pragma: no cover - resolve_stage(require_adam)
            raise RunnerError("adam basis requires optimizer snapshots")
        if info.parameter_manifest_digest != shared_manifest_digest:
            raise RunnerError(
                f"stage {last_stage.name!r}: Adam-basis parameter-manifest "
                "cross-check failed — the optimizer snapshot records "
                f"manifest digest {info.parameter_manifest_digest} but the "
                "gradient-row artifacts were built under "
                f"{shared_manifest_digest}; coordinates would not align"
            )
        snapshot = load_optimizer_snapshot(info.path)
        corrected = snapshot.bias_corrected_exp_avg_sq()
        statistics = {
            "model_identifier": snapshot.manifest.model_name,
            "model_revision": str(info.step),
            "dataset_fingerprint": last_resolved.dataset_digest,
            "parameter_manifest_digest": snapshot.manifest.digest(),
            "statistic": "adamw_exp_avg_sq_bias_corrected",
            "number_of_gradient_samples": info.step,
            "code_commit": _scimt_commit(),
            "estimator": "full",
            "logra": None,
        }
        metric = DiagonalMetric.from_statistics(
            statistics,
            corrected,
            exponent=-0.5,
            epsilon=info.epsilon,
            damping=damping,
            manifest=snapshot.manifest,
        )
        extras = {
            "coordinates": "adam",
            "source_stage": last_stage.name,
            "bias_correction": {
                "applied": True,
                "convention": "v_hat = exp_avg_sq / (1 - beta2**step)",
            },
            "step": info.step,
            "beta2": info.beta2,
            "epsilon": info.epsilon,
        }
        return metric, extras
    statistics, diagonal = factor_payloads[last_stage.name]
    metric = DiagonalMetric.from_statistics(
        statistics,
        diagonal,
        exponent=-0.5,
        damping=damping,
    )
    extras = {"coordinates": "fisher_diag", "source_stage": last_stage.name}
    return metric, extras


async def score_source(config: AttributionRunConfig) -> PhaseReport:
    """SOURCE scoring over the ordered chronological stages.

    Consumes only validated artifacts (rows, factors, queries), builds one
    scorer per damping value with a single global basis, and applies each
    segment's ``1/N`` (its declared training-set size) exactly once — the
    scorer itself returns unnormalized scores. Raw per-example scores are
    saved per (stage, damping) with an explicit completeness manifest."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from .source import DiagonalCurvature, EKFACCurvature, SourceScorer, SourceSegment

    _require_scorable_method(config)
    method = config.method
    resolved_stages = _resolve_stages(config, require_adam=method.basis == "adam")
    _write_or_check_ledger(config, "score-source")
    layout = run_layout(config.output_dir)
    query_dir = _resolve_query_checkpoint(config)
    query_data_path, query_dataset_digest = _resolve_query_dataset(config)
    tokenizer_digest = _tokenizer_content_digest(_tokenizer_dir(config, query_dir))
    query_fingerprint = _dataset_fingerprint(query_dataset_digest, tokenizer_digest)

    # --- validate upstream row artifacts (before any tensor loads) ---------
    queries_stored = _check_upstream(
        "queries",
        layout.queries,
        {
            "resolved_config": _scoped_config(config, "build-queries"),
            "dataset_fingerprint": query_fingerprint,
            "checkpoint_digest": artifact_digest(query_dir),
            "dtype": method.dtype,
            "seeds": {"run": config.seed},
        },
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

    scoped = _scoped_config(config, "score-source")
    scoped["resolved_lr_steps"] = {
        resolved.name: resolved.lr_steps for resolved in resolved_stages
    }
    upstream = {"queries": queries_stored.digest()}
    for stage in config.stages:
        upstream[f"rows/{stage.name}"] = rows_stored[stage.name].digest()
        upstream[f"factors/{stage.name}"] = factor_stored[stage.name].digest()
    if method.basis == "adam":
        for stage, resolved in zip(config.stages, resolved_stages, strict=True):
            upstream[f"adam/{stage.name}"] = (
                resolved.optimizer_snapshot.parameter_manifest_digest
            )

    # A probe metric (first damping) pins the basis descriptor identity.
    basis_extras: dict[str, Any] = {"coordinates": "raw"}
    if method.basis in ("fisher", "adam"):
        _, basis_extras = _basis_metric(
            config, resolved_stages, factor_payloads, shared_manifest_digest,
            method.damping_sweep[0],
        )
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
        curvature_descriptor={
            "method": method.curvature,
            "damping_sweep": list(method.damping_sweep),
            "damping_semantics": (
                "raw basis: eigenvalues + damping; diagonal bases: damping "
                "added to the metric statistic before the -1/2 power"
            ),
        },
        logra_descriptor=None,
        dtype=method.dtype,
        seeds={"run": config.seed},
        upstream_digests=upstream,
    )
    _bind_identity(layout.scores, identity)
    expected_entries = sorted(
        f"{stage.name}__damping-{index}"
        for stage in config.stages
        for index in range(len(method.damping_sweep))
    )
    marker = layout.scores / _SCORE_MANIFEST_FILE
    if marker.is_file():
        completeness = json.loads(marker.read_text(encoding="utf-8"))
        if (
            completeness.get("identity_digest") == identity.digest()
            and completeness.get("expected") == expected_entries
            and sorted(completeness.get("entries", {})) == expected_entries
            and all(
                (layout.scores / entry["file"]).is_file()
                for entry in completeness["entries"].values()
            )
        ):
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

    entries: dict[str, dict[str, Any]] = {}
    for damping_index, damping in enumerate(method.damping_sweep):
        metric = None
        if method.basis in ("fisher", "adam"):
            metric, _ = _basis_metric(
                config, resolved_stages, factor_payloads,
                shared_manifest_digest, damping,
            )
        diagonal_scale = (
            None
            if metric is None
            else metric.diagonal.to(dtype=torch.float32)
        )
        segments = []
        for stage in config.stages:
            resolved = resolved_stages[_stage_index(config, stage.name)]
            descriptor = {
                **basis_extras,
                "manifest_digest": shared_manifest_digest,
                "damping": damping,
            }
            if factor_kind == "fisher":
                _, diagonal = factor_payloads[stage.name]
                if diagonal_scale is None:
                    curvature = DiagonalCurvature(
                        diagonal.double().numpy() + damping,
                        basis_descriptor=descriptor,
                    )
                else:
                    transported = (
                        diagonal.double()
                        * diagonal_scale.double().pow(2)
                    ).numpy()
                    curvature = DiagonalCurvature(
                        transported, basis_descriptor=descriptor
                    )
            else:
                factors, factor_manifest = factor_payloads[stage.name]
                curvature = _shifted_curvature(
                    EKFACCurvature(
                        factors, factor_manifest, basis_descriptor=descriptor
                    ),
                    damping,
                )
            segments.append(
                SourceSegment(stage.name, curvature, resolved.lr_steps)
            )
        scorer = SourceScorer(segments)
        transformed_query = (
            query_features
            if diagonal_scale is None
            else query_features * diagonal_scale
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
                if diagonal_scale is not None:
                    features = features * diagonal_scale
                pieces.append(transformed[stage_index] @ features.numpy().T)
                sample_ids.append(shard["sample_ids"])
            unnormalized = np.concatenate(pieces, axis=1)
            # The scorer returns unnormalized scores; each segment's 1/N
            # (its declared training-set size) is applied here, exactly once.
            scores = unnormalized / float(stage.n_examples)
            entry_name = f"{stage.name}__damping-{damping_index}"
            filename = f"scores__{entry_name}.safetensors"
            save_file(
                {
                    "scores": torch.from_numpy(
                        np.ascontiguousarray(scores.astype(np.float32))
                    ),
                    "query_sample_ids": query_rows["sample_ids"],
                    "train_sample_ids": torch.cat(sample_ids),
                },
                str(layout.scores / filename),
                metadata={
                    "identity_digest": identity.digest(),
                    "stage": stage.name,
                    "damping": repr(float(damping)),
                    "n_examples": str(stage.n_examples),
                },
            )
            entries[entry_name] = {
                "file": filename,
                "digest": artifact_digest(layout.scores / filename),
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
                 declared: tuple[Path, str, ResolvedStage | None]):
    """The frozen diagonal pair metric M (or None). EK-FAC is refused: the
    pair path supports DiagonalMetric only (recorded port deviation)."""
    from .metrics import DiagonalMetric

    second = config.second_order
    if second.metric == "none":
        return None, {"kind": "identity"}
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
        return metric, {"kind": "adam", **metric.descriptor()}
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
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if statistics.get("parameter_manifest_digest") != manifest.digest():
        raise RunnerError(
            "build-directions: fisher metric statistics were fitted under a "
            "different parameter manifest"
        )
    shard_manifest = ShardManifest.load(factors_dir)
    diagonal = shard_manifest.read_rows(factors_dir)["features"][0].float()
    metric = DiagonalMetric.from_statistics(
        statistics,
        diagonal,
        exponent=second.metric_exponent,
        epsilon=second.metric_epsilon,
    )
    return metric, {"kind": "fisher", **metric.descriptor()}


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
    return values, estimator == "marginals", statistics


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
    metric, metric_descriptor = _pair_metric(config, manifest, declared)
    derivative_values = None
    derivative_factored = False
    if second.metric_derivative is not None:
        derivative_values, derivative_factored, _ = _load_derivative_statistics(
            config, manifest
        )
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
        upstream_digests={},
    )
    writer = ArtifactWriter(
        layout.directions,
        identity,
        feature_dim=max(1, manifest.included_numel),
        rows_per_shard=config.data.rows_per_shard,
        feature_dtype="float32",
    )
    if writer.already_complete:
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

    columns = []
    committed_pairs = writer.rows_committed  # one row per pair, in order
    for pair_index, (index_a, index_b) in enumerate(second.pairs):
        column = {
            "pair": [index_a, index_b],
            "hessian_kind": second.hessian_kind,
            "metric": metric_descriptor,
            "metric_derivative": second.metric_derivative is not None,
            "checkpoint": checkpoint_label,
        }
        columns.append(column)
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
    writer.finalize()
    from .artifacts import _atomic_write_text

    _atomic_write_text(
        layout.directions / "direction_columns.json",
        json.dumps(columns, indent=2, sort_keys=True) + "\n",
    )
    report = PhaseReport(
        "build-directions",
        (PhaseOutput("directions", layout.directions, identity.digest(),
                     False, rows=len(columns)),),
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
    includes = [re.compile(pattern) for pattern in include]
    excludes = [re.compile(pattern) for pattern in exclude]
    total = 0
    for name, shape in shapes.items():
        if any(regex.fullmatch(name) for regex in excludes):
            continue
        if not any(regex.fullmatch(name) for regex in includes):
            continue
        total += math.prod(shape) if shape else 1
    return total


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
    output identities — WITHOUT importing torch or loading any model.

    Pure: writes nothing. Hard resolution failures (missing/contradictory
    stage artifacts) raise; capability gaps for the configured method are
    returned in ``blockers``.
    """
    method = config.method
    layout = run_layout(config.output_dir)
    blockers: list[str] = []
    if method.curvature == "ggn":
        blockers.append(
            "method.curvature 'ggn': no fitted GGN segment operator exists — "
            "use 'fisher'/'ekfac' for SOURCE, GGN lives in second-order phases"
        )
    if method.basis == "ekfac":
        blockers.append("method.basis 'ekfac' is not supported by score-source")
    if method.basis in ("fisher", "adam") and method.curvature != "fisher":
        blockers.append(
            f"method.basis {method.basis!r} requires method.curvature 'fisher'"
        )

    stages_report = []
    resolved_lookup: dict[str, Any] = {}
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
        if method.basis == "adam" and not adam["available"]:
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
                "dataset_rows": _count_jsonl_rows(Path(resolved.dataset.path)),
                "estimated_included_parameters": _estimate_included_parameters(
                    resolved.checkpoint_dir,
                    config.parameters.include,
                    config.parameters.exclude,
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
    query_data_path, query_dataset_digest = _resolve_query_dataset(config)
    tokenizer_dir = _tokenizer_dir(config, query_dir)
    tokenizer_digest = _tokenizer_content_digest(tokenizer_dir)
    query_report = {
        "checkpoint_dir": str(query_dir),
        "dataset_path": str(query_data_path),
        "dataset_digest": query_dataset_digest,
        "dataset_rows": _count_jsonl_rows(query_data_path),
        "estimated_included_parameters": _estimate_included_parameters(
            query_dir, config.parameters.include, config.parameters.exclude
        ),
        "artifacts": {"queries": _artifact_status(layout.queries)},
    }

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
                    basis_coordinates="raw",
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
        "query": query_report,
        "tokenizer": {
            "directory": str(tokenizer_dir),
            "content_digest": tokenizer_digest,
        },
        "factor_plan": _fit_config_payload(config),
        "expected": expected,
        "planned_identities": planned,
        "blockers": blockers,
    }


PHASES: dict[str, Callable[[AttributionRunConfig], Awaitable[Any]]] = {
    "fit-factors": fit_factors,
    "compute-rows": compute_rows,
    "build-queries": build_queries,
    "score-source": score_source,
    "build-directions": build_directions,
    "sweep-jvp": sweep_jvp,
    "summarize": summarize,
    "dry-run": dry_run,
}
