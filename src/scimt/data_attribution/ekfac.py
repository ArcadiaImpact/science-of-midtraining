"""EK-FAC artifact loading and matrix powers, ported from gradient-kernel ca9689a."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .manifest import ManifestMismatchError, ParameterManifest


def safe(name: str) -> str:
    return name.replace(".", "__")


EKFAC_MODES = ("ekfac", "ekfac_adam")

_CONDITIONER_PROVENANCE_KEYS = (
    "kind",
    "statistic",
    "moment_identity_digest",
    "optimizer_epsilon",
    "conditioning_damping",
)


@dataclass(frozen=True)
class EKFACConditioner:
    """Stage-local Adam preconditioner ``A_l`` over included coordinates.

    ``values`` is the flat fp vector of per-coordinate scales in the
    manifest's included ordering; ``provenance`` binds the moment artifact
    it was derived from and is embedded verbatim in ``ekfac_meta.json``.
    """

    values: torch.Tensor
    provenance: dict[str, Any]


def _validate_conditioner(conditioner: EKFACConditioner, manifest) -> dict[str, Any]:
    if not isinstance(conditioner, EKFACConditioner):
        raise TypeError("conditioner must be an EKFACConditioner")
    values = conditioner.values
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise ValueError("conditioner values must be a 1-D tensor")
    if values.numel() != manifest.included_numel:
        raise ValueError(
            f"conditioner values must have shape [{manifest.included_numel}]"
        )
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("conditioner values must be floating-point and finite")
    if bool((values <= 0).any()):
        raise ValueError("conditioner values must be strictly positive")
    provenance = conditioner.provenance
    if not isinstance(provenance, dict):
        raise TypeError("conditioner provenance must be a dict")
    missing = [k for k in _CONDITIONER_PROVENANCE_KEYS if k not in provenance]
    if missing:
        raise ValueError(f"conditioner provenance missing keys: {missing}")
    try:
        return json.loads(json.dumps(provenance))
    except (TypeError, ValueError) as error:
        raise ValueError("conditioner provenance must be JSON-serializable") from error


@dataclass(frozen=True)
class EKFACFactors:
    linears: dict[str, dict[str, torch.Tensor]]
    diag_v: torch.Tensor
    diag_index: tuple[dict[str, Any], ...]
    snapshot: str
    preconditioner: dict[str, Any] | None = None

    @property
    def snapshot_id(self) -> str:
        return self.snapshot

    @property
    def mode(self) -> str:
        return "ekfac_adam" if self.preconditioner is not None else "ekfac"


def _snapshot(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        relative = file.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def load_ekfac(
    path: str | Path,
    manifest: ParameterManifest,
    *,
    expected_mode: str = "ekfac",
) -> EKFACFactors:
    if expected_mode not in EKFAC_MODES:
        raise ValueError(f"expected_mode must be one of {list(EKFAC_MODES)}")
    directory = Path(path)
    stored = ParameterManifest.load(directory)
    if stored.digest() != manifest.digest():
        raise ManifestMismatchError("manifest digest mismatch")
    metadata = json.loads((directory / "ekfac_meta.json").read_text())
    preconditioner = metadata.get("preconditioner")
    artifact_mode = "ekfac_adam" if preconditioner is not None else "ekfac"
    if artifact_mode != expected_mode:
        raise ValueError(
            f"EK-FAC artifact at {directory} is a {artifact_mode!r} factor set "
            f"but {expected_mode!r} was requested — factor coordinates are "
            "never silently reinterpreted (raw and Adam-conditioned factors "
            "are mutually unconsumable)"
        )
    if preconditioner is not None and not isinstance(preconditioner, dict):
        raise ValueError("ekfac_meta.json preconditioner must be a mapping")
    names = metadata.get("linears")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("ekfac_meta.json linears must be a list of strings")
    if len(set(names)) != len(names):
        raise ValueError("ekfac_meta.json contains duplicate linear names")
    entries = {e.name: e for e in manifest.included_entries()}
    claimed: set[str] = set()
    linears = {}
    for name in names:
        weight = entries.get(f"{name}.weight")
        if weight is None or len(weight.shape) != 2:
            raise ValueError(f"EK-FAC factor {name!r} has no included matrix weight")
        bias = entries.get(f"{name}.bias")
        base = directory / "linear" / safe(name)
        factor = {
            key: torch.from_numpy(np.load(base / f"{key}.npy")).cpu()
            for key in ("U_A", "U_S", "lam")
        }
        out_features, in_features = weight.shape
        expected = {
            "U_A": (in_features + int(bias is not None),) * 2,
            "U_S": (out_features, out_features),
            "lam": (out_features, in_features + int(bias is not None)),
        }
        for key, shape in expected.items():
            if tuple(factor[key].shape) != shape:
                raise ValueError(
                    f"factor {name!r} {key} has shape {tuple(factor[key].shape)}, expected {shape}"
                )
            if not factor[key].is_floating_point() or not bool(
                torch.isfinite(factor[key]).all()
            ):
                raise ValueError(
                    f"factor {name!r} {key} must be floating-point and finite"
                )
        if bool((factor["lam"] < 0).any()):
            raise ValueError(f"factor {name!r} lam must be nonnegative")
        claimed.add(weight.name)
        if bias is not None:
            claimed.add(bias.name)
        linears[name] = factor
    raw_index = json.loads((directory / "diag" / "index.json").read_text())
    if not isinstance(raw_index, list):
        raise ValueError("diag/index.json must be a list")
    diag_v = torch.from_numpy(np.load(directory / "diag" / "v.npy")).reshape(-1).cpu()
    if not diag_v.is_floating_point() or not bool(torch.isfinite(diag_v).all()):
        raise ValueError("diagonal factors must be floating-point and finite")
    if bool((diag_v < 0).any()):
        raise ValueError("diagonal factors must be nonnegative")
    cursor = 0
    diagonal_names = []
    for item in raw_index:
        if item.get("name") in diagonal_names:
            raise ValueError("diag/index.json contains duplicate parameter names")
        entry = entries.get(item.get("name"))
        if (
            entry is None
            or item.get("numel") != entry.numel
            or item.get("offset") != entry.global_flat_offset
        ):
            raise ValueError(f"invalid diagonal factor entry {item!r}")
        cursor += entry.numel
        diagonal_names.append(entry.name)
        if entry.name in claimed:
            raise ValueError(
                f"diagonal factor overlaps EK-FAC parameter {entry.name!r}"
            )
        claimed.add(entry.name)
    if cursor != diag_v.numel():
        raise ValueError("diag/v.npy length does not match diag/index.json")
    if set(entries) != claimed:
        raise ValueError(
            f"EK-FAC artifact does not cover parameters: {sorted(set(entries) - claimed)}"
        )
    return EKFACFactors(
        linears, diag_v, tuple(raw_index), _snapshot(directory), preconditioner
    )


load_ekfac_factors = load_ekfac


def _scale(values: torch.Tensor, damping: float, power: float) -> torch.Tensor:
    # Byte-identical to upstream ca9689a `_damping_scale`: a zero spectrum
    # with damping 0 and negative power yields inf, exactly as upstream —
    # unreachable from the runner phases (fitted spectra are validated
    # nonnegative and every negative-power caller damps). `logra.whiten_rows`
    # guards the same corner EXPLICITLY (its upstream did too) — keep the two
    # sites in their respective upstream shapes; never "fix" one alone.
    damped = values + damping * values.mean()
    return damped.rsqrt() if power == -0.5 else damped.pow(power)


def apply_ekfac(flat, factors, manifest, damping_scale, power):
    if flat.ndim != 1 or flat.numel() != manifest.included_numel:
        raise ValueError(f"flat must have shape [{manifest.included_numel}]")
    if not flat.is_floating_point():
        raise ValueError("flat must have floating-point dtype")
    if damping_scale < 0:
        raise ValueError("damping_scale must be nonnegative")
    entries = {e.name: e for e in manifest.included_entries()}
    for name, factor in factors.linears.items():
        for key in ("U_A", "U_S", "lam"):
            value = factor[key]
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"factor {name!r} {key} must be floating-point and finite"
                )
        if bool((factor["lam"] < 0).any()):
            raise ValueError(f"factor {name!r} lam must be nonnegative")
    if not factors.diag_v.is_floating_point() or not bool(
        torch.isfinite(factors.diag_v).all()
    ):
        raise ValueError("diagonal factors must be floating-point and finite")
    if bool((factors.diag_v < 0).any()):
        raise ValueError("diagonal factors must be nonnegative")
    source = flat.detach().cpu().double()
    result = torch.zeros_like(source)
    for name, factor in factors.linears.items():
        weight = entries[f"{name}.weight"]
        bias = entries.get(f"{name}.bias")
        block = source[
            weight.global_flat_offset : weight.global_flat_offset + weight.numel
        ].reshape(weight.shape)
        if bias is not None:
            block = torch.cat(
                (
                    block,
                    source[
                        bias.global_flat_offset : bias.global_flat_offset + bias.numel
                    ].reshape(-1, 1),
                ),
                1,
            )
        ua, us, lam = (factor[k].double() for k in ("U_A", "U_S", "lam"))
        restored = us @ ((us.T @ block @ ua) * _scale(lam, damping_scale, power)) @ ua.T
        result[weight.global_flat_offset : weight.global_flat_offset + weight.numel] = (
            restored[:, : weight.shape[1]].reshape(-1)
        )
        if bias is not None:
            result[bias.global_flat_offset : bias.global_flat_offset + bias.numel] = (
                restored[:, -1]
            )
    cursor = 0
    if factors.diag_v.numel():
        scale = _scale(factors.diag_v.double(), damping_scale, power)
        for item in factors.diag_index:
            n, offset = int(item["numel"]), int(item["offset"])
            result[offset : offset + n] = (
                source[offset : offset + n] * scale[cursor : cursor + n]
            )
            cursor += n
    return result.to(device=flat.device, dtype=flat.dtype)


def _causal_token_task(task_base, tracked):
    import torch.nn.functional as F

    class CausalTokenTask(task_base):
        def compute_train_loss(self, batch, model, sample=False):
            ids = batch["input_ids"]
            positions = batch["position"].to(ids.device)
            logits = model(input_ids=ids).logits
            idx = torch.arange(ids.shape[0], device=ids.device)
            pred = logits[idx, positions - 1].float()
            targets = (
                torch.multinomial(torch.softmax(pred, -1), 1).flatten()
                if sample
                else ids[idx, positions]
            )
            return F.cross_entropy(pred, targets, reduction="sum")

        def compute_measurement(self, batch, model):
            return self.compute_train_loss(batch, model)

        def get_influence_tracked_modules(self):
            return tracked

        def get_attention_mask(self, batch):
            return None

    return CausalTokenTask()


class _TokenSampleDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


_FIT_CONFIG_KEYS = frozenset(
    {
        "samples",
        "seed",
        "source_batch_size",
        "batch_size",
        "max_positions_per_sequence",
        "min_position_gap",
        "use_empirical_fisher",
        "covariance_module_partitions",
        "lambda_module_partitions",
        "eigendecomposition_dtype",
        "eigh_device",
    }
)


def _fit_config(config) -> dict:
    if not isinstance(config, dict):
        raise TypeError("EK-FAC config must be a mapping")
    unknown = set(config) - _FIT_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown EK-FAC config keys: {sorted(unknown)}")
    result = {
        "samples": 1024,
        "seed": 0,
        "source_batch_size": 8,
        "batch_size": 8,
        "max_positions_per_sequence": 1,
        "min_position_gap": 1,
        "use_empirical_fisher": True,
        "covariance_module_partitions": 1,
        "lambda_module_partitions": 1,
        "eigendecomposition_dtype": "float64",
        "eigh_device": "auto",
        **config,
    }
    for key in (
        "samples",
        "source_batch_size",
        "batch_size",
        "min_position_gap",
        "covariance_module_partitions",
        "lambda_module_partitions",
    ):
        if (
            not isinstance(result[key], int)
            or isinstance(result[key], bool)
            or result[key] <= 0
        ):
            raise ValueError(f"{key} must be a positive integer")
    limit = result["max_positions_per_sequence"]
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ):
        raise ValueError(
            "max_positions_per_sequence must be a nonnegative integer or null"
        )
    if result["eigendecomposition_dtype"] not in {"float32", "float64"}:
        raise ValueError("eigendecomposition_dtype must be float32 or float64")
    if result["eigh_device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("eigh_device must be 'auto', 'cpu', or 'cuda'")
    return result


def _eigh_torch_device(requested: str) -> "torch.device":
    """Resolve an explicit eigh device request; error-loud on impossible asks.

    Module-level so tests can monkeypatch the resolution (exercising the
    lifted-loop wiring on CPU-only boxes).
    """
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "factors eigh_device 'cuda' was requested but CUDA is not "
            "available — a run that cannot work raises before spending compute"
        )
    return torch.device(requested)


def _lifted_eigendecomposition(analyzer, prepared_model, factor_args, device) -> None:
    """Eigendecompose the fitted covariances on an explicit device, streaming.

    A faithful port of kronfluence 1.0.1's ``perform_eigendecomposition``
    math (normalize by the processed count, symmetrize, ``torch.linalg.eigh``,
    store in the covariance dtype on CPU), differing in two deliberate ways:

    - **where each ``eigh`` executes** — kronfluence offers no
      eigendecomposition device knob independent of its ``State`` (which
      follows the fit model's device), hence this lift; and
    - **host memory** — kronfluence loads the full covariance set and holds
      the full eigen result set before one end save. Here covariances are
      streamed one matrix at a time via ``safetensors.safe_open`` (valid for
      any ``*_module_partitions``: kronfluence aggregates partitions into the
      unpartitioned files at the end of ``fit_covariance_matrices``), and
      each factor SIDE (activation / gradient) is saved and freed before the
      next begins. Peak host RSS at gemma-3-12b full coverage: one side's
      fp32 eigenvector set (~82 GB — same bytes as that side's covariances)
      + one fp64 matrix and eigh workspace (~2-6 GB) + eigenvalues
      (negligible) ~= 85-90 GB, vs ~330+ GB for load-everything /
      save-once (the wave-1 OOM at 487 GB RSS had this on top of the
      conditioned path's own set).

    Eigenvectors are sign/rotation-ambiguous and differ bitwise across
    backends (cuSOLVER vs LAPACK); EK-FAC's lambda refit is exact-in-basis
    for any orthonormal eigenbasis, so the operator is equally valid either
    way — artifact digests simply differ.

    ``eigh_report.json`` (next to the kronfluence factor files) records
    per-matrix load/eigh timings and per-side save timings.
    """
    import time
    import warnings

    from kronfluence.factor.eigen import eigendecomposition_save_path
    from kronfluence.module.utils import get_tracked_module_names
    from kronfluence.utils.constants import (
        ACTIVATION_COVARIANCE_MATRIX_NAME,
        ACTIVATION_EIGENVALUES_NAME,
        ACTIVATION_EIGENVECTORS_NAME,
        EIGENDECOMPOSITION_FACTOR_NAMES,
        GRADIENT_COVARIANCE_MATRIX_NAME,
        GRADIENT_EIGENVALUES_NAME,
        GRADIENT_EIGENVECTORS_NAME,
        NUM_ACTIVATION_COVARIANCE_PROCESSED,
        NUM_GRADIENT_COVARIANCE_PROCESSED,
    )
    from kronfluence.utils.state import release_memory
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    factors_dir = Path(analyzer.factors_output_dir(factors_name="ekfac"))
    tracked = get_tracked_module_names(model=prepared_model)
    eigh_dtype = factor_args.eigendecomposition_dtype
    metadata = factor_args.to_str_dict()
    sides = (
        (
            ACTIVATION_COVARIANCE_MATRIX_NAME,
            NUM_ACTIVATION_COVARIANCE_PROCESSED,
            ACTIVATION_EIGENVECTORS_NAME,
            ACTIVATION_EIGENVALUES_NAME,
        ),
        (
            GRADIENT_COVARIANCE_MATRIX_NAME,
            NUM_GRADIENT_COVARIANCE_PROCESSED,
            GRADIENT_EIGENVECTORS_NAME,
            GRADIENT_EIGENVALUES_NAME,
        ),
    )
    written = set()
    timings: list[dict] = []
    saves: list[dict] = []
    with torch.no_grad():
        for covariance_name, num_processed_name, eigenvectors_name, eigenvalues_name in sides:
            # The processed counts are per-module scalars — tiny; full load.
            num_processed = load_file(
                filename=factors_dir / f"{num_processed_name}.safetensors"
            )
            side_vectors: dict = {}
            side_values: dict = {}
            with safe_open(
                factors_dir / f"{covariance_name}.safetensors",
                framework="pt",
                device="cpu",
            ) as handle:
                for module_name in tracked:
                    load_started = time.perf_counter()
                    stored = handle.get_tensor(module_name)
                    load_seconds = time.perf_counter() - load_started
                    original_dtype = stored.dtype
                    transfer_started = time.perf_counter()
                    covariance_matrix = stored.to(device=device, dtype=eigh_dtype)
                    del stored
                    covariance_matrix.div_(
                        num_processed[module_name].to(device=device)
                    )
                    covariance_matrix = covariance_matrix + covariance_matrix.t()
                    covariance_matrix.mul_(0.5)
                    transfer_seconds = time.perf_counter() - transfer_started
                    eigh_started = time.perf_counter()
                    eigh_device_used = str(device)
                    try:
                        eigenvalues, eigenvectors = torch.linalg.eigh(
                            covariance_matrix
                        )
                    except torch.cuda.OutOfMemoryError:
                        # Kronfluence's own retry semantics (factor/eigen.py):
                        # release memory and retry; if the device still cannot
                        # hold it, degrade THIS matrix to CPU — how, not what.
                        release_memory()
                        try:
                            eigenvalues, eigenvectors = torch.linalg.eigh(
                                covariance_matrix
                            )
                        except torch.cuda.OutOfMemoryError:
                            warnings.warn(
                                f"eigh of {module_name!r} ({covariance_name}) "
                                "OOMed on the requested device twice; falling "
                                "back to CPU for this matrix",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            covariance_matrix = covariance_matrix.cpu()
                            eigh_device_used = "cpu"
                            eigenvalues, eigenvectors = torch.linalg.eigh(
                                covariance_matrix
                            )
                    if covariance_matrix.device.type == "cuda":
                        torch.cuda.synchronize(covariance_matrix.device)
                    timings.append(
                        {
                            "module": module_name,
                            "factor": covariance_name,
                            "dim": int(covariance_matrix.shape[0]),
                            "load_seconds": load_seconds,
                            "transfer_seconds": transfer_seconds,
                            "eigh_seconds": time.perf_counter() - eigh_started,
                            "device": eigh_device_used,
                        }
                    )
                    del covariance_matrix
                    side_values[module_name] = (
                        eigenvalues.contiguous().to(
                            dtype=original_dtype, device="cpu"
                        )
                    )
                    side_vectors[module_name] = (
                        eigenvectors.contiguous().to(
                            dtype=original_dtype, device="cpu"
                        )
                    )
                    del eigenvalues, eigenvectors
            for factor_name, side in (
                (eigenvalues_name, side_values),
                (eigenvectors_name, side_vectors),
            ):
                save_started = time.perf_counter()
                save_file(
                    tensors=side,
                    filename=eigendecomposition_save_path(
                        output_dir=factors_dir, factor_name=factor_name
                    ),
                    metadata=metadata,
                )
                saves.append(
                    {
                        "factor": factor_name,
                        "save_seconds": time.perf_counter() - save_started,
                    }
                )
                written.add(factor_name)
            del side_vectors, side_values, num_processed
    if written != set(EIGENDECOMPOSITION_FACTOR_NAMES):
        raise RuntimeError(
            "lifted eigendecomposition wrote an unexpected factor set: "
            f"{sorted(written)}"
        )
    report_path = factors_dir / "eigh_report.json"
    report_path.write_text(
        json.dumps(
            {
                "device": str(device),
                "eigendecomposition_dtype": str(eigh_dtype),
                "total_eigh_seconds": sum(t["eigh_seconds"] for t in timings),
                "total_load_seconds": sum(t["load_seconds"] for t in timings),
                "total_transfer_seconds": sum(
                    t["transfer_seconds"] for t in timings
                ),
                "total_save_seconds": sum(s["save_seconds"] for s in saves),
                "saves": saves,
                "matrices": timings,
            },
            indent=2,
            sort_keys=True,
        )
    )



def build_ekfac_sample_items(dataset, config) -> list[dict]:
    """Materialize one bounded, seeded token sample shared by both fitting passes."""
    cfg = _fit_config(config)
    items = []
    for batch in dataset.iter_batches(cfg["source_batch_size"]):
        for row, sequence_id in enumerate(batch.sequence_ids.tolist()):
            candidates = batch.target_mask[row].nonzero(as_tuple=False).flatten().cpu()
            limit = cfg["max_positions_per_sequence"]
            if limit == 0 or candidates.numel() == 0:
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(cfg["seed"] + int(sequence_id))
            order = torch.randperm(candidates.numel(), generator=generator)
            selected = []
            for index in order.tolist():
                position = int(candidates[index])
                if all(
                    abs(position - prior) >= cfg["min_position_gap"]
                    for prior in selected
                ):
                    selected.append(position)
                    if limit is not None and len(selected) >= limit:
                        break
            for position in sorted(selected):
                items.append(
                    {
                        "input_ids": batch.input_ids[row].clone(),
                        "position": int(position),
                        "sequence_id": int(sequence_id),
                    }
                )
                if len(items) >= cfg["samples"]:
                    return items
    return items


def fit_ekfac(model, dataset, manifest, config, output_dir, conditioner=None):
    """Fit Kronfluence factors; Kronfluence remains an optional lazy dependency.

    With ``conditioner=None`` this is the raw-coordinate upstream path,
    unchanged. With an :class:`EKFACConditioner` it fits the Adam-conditioned
    factor set (design: docs/specs/2026-08-17-adam-conditioned-ekfac-design.md):
    unconditioned Kronecker eigenbases, lambdas refit on conditioned
    gradients, diagonal remainder conditioned exactly.
    """
    try:
        from kronfluence.analyzer import Analyzer, prepare_model
        from kronfluence.arguments import FactorArguments
        from kronfluence.task import Task
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "EK-FAC requires Kronfluence; install it with `uv sync --extra data-attribution-ekfac`"
        ) from error
    cfg = _fit_config(config)
    if not isinstance(manifest, ParameterManifest):
        raise TypeError("manifest must be a ParameterManifest")
    provenance = None
    if conditioner is not None:
        provenance = _validate_conditioner(conditioner, manifest)
        if not cfg["use_empirical_fisher"]:
            raise ValueError(
                "conditioned EK-FAC fits lambdas from empirical-Fisher "
                "gradients (true next tokens); use_empirical_fisher must be "
                "true in ekfac_adam mode"
            )
    manifest.validate_against_model(model)
    included = {e.name for e in manifest.included_entries()}
    names = sorted(
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and f"{n}.weight" in included
    )
    modules = dict(model.named_modules())
    ek_names = set()
    for name in names:
        ek_names.add(f"{name}.weight")
        if modules[name].bias is not None:
            if f"{name}.bias" not in included:
                raise ValueError(
                    f"tracked Linear {name!r} has a bias not in the manifest"
                )
            ek_names.add(f"{name}.bias")
    items = build_ekfac_sample_items(dataset, cfg)
    sample_dataset = _TokenSampleDataset(items)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    named = dict(model.named_parameters(remove_duplicate=False))
    diagonal = [e for e in manifest.included_entries() if e.name not in ek_names]
    if conditioner is not None:
        return _fit_ekfac_conditioned(
            model=model,
            manifest=manifest,
            cfg=cfg,
            output_dir=output_dir,
            conditioner=conditioner,
            provenance=provenance,
            names=names,
            items=items,
            sample_dataset=sample_dataset,
            model_device=model_device,
            named=named,
            diagonal=diagonal,
            kron=(Analyzer, prepare_model, FactorArguments, Task),
        )
    accum = {e.name: torch.zeros(e.numel, dtype=torch.float64) for e in diagonal}
    count = 0
    # This pass must precede prepare_model, which may freeze parameters.
    from .gradients import backward_memory_mode

    with backward_memory_mode(
        model, getattr(model, "is_gradient_checkpointing", False)
    ):
        for item in items:
            ids = item["input_ids"].to(model_device)
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            position = item["position"]
            if isinstance(position, torch.Tensor):
                position = int(position.reshape(-1)[0])
            model.zero_grad(set_to_none=True)
            logits = model(input_ids=ids).logits
            loss = torch.nn.functional.cross_entropy(
                logits[0, position - 1 : position].float(),
                ids[0, position : position + 1],
                reduction="sum",
            )
            loss.backward()
            count += 1
            for entry in diagonal:
                gradient = named[entry.name].grad
                if gradient is not None:
                    accum[entry.name].add_(
                        gradient.detach().reshape(-1).cpu().double().square()
                    )
    model.zero_grad(set_to_none=True)
    if diagonal and count == 0:
        raise ValueError("cannot fit diagonal EK-FAC remainder from an empty dataset")
    task = _causal_token_task(Task, names)
    prepared = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name="ekfac",
        model=prepared,
        task=task,
        cpu=model_device.type == "cpu",
        output_dir=str(Path(output_dir) / "kronfluence"),
        disable_tqdm=True,
    )
    args = FactorArguments(
        strategy="ekfac",
        use_empirical_fisher=cfg.get("use_empirical_fisher", True),
        covariance_module_partitions=cfg.get("covariance_module_partitions", 1),
        lambda_module_partitions=cfg.get("lambda_module_partitions", 1),
        eigendecomposition_dtype=getattr(
            torch, cfg.get("eigendecomposition_dtype", "float64")
        ),
    )
    if cfg["eigh_device"] == "auto":
        # Byte-untouched upstream path: kronfluence's one-shot trio, with its
        # eigendecomposition on its State device (follows the fit model).
        analyzer.fit_all_factors(
            factors_name="ekfac",
            dataset=sample_dataset,
            per_device_batch_size=cfg.get("batch_size", 8),
            factor_args=args,
            overwrite_output_dir=True,
        )
    else:
        # Same trio staged (fit_all_factors is exactly these three stages in
        # kronfluence 1.0.1), with the eigendecomposition lifted onto the
        # requested device. fit_lambda_matrices consumes the saved
        # eigendecomposition from the factors directory as usual.
        eigh_device = _eigh_torch_device(cfg["eigh_device"])
        analyzer.fit_covariance_matrices(
            factors_name="ekfac",
            dataset=sample_dataset,
            per_device_batch_size=cfg.get("batch_size", 8),
            factor_args=args,
            overwrite_output_dir=True,
        )
        _lifted_eigendecomposition(analyzer, prepared, args, eigh_device)
        analyzer.fit_lambda_matrices(
            factors_name="ekfac",
            dataset=sample_dataset,
            per_device_batch_size=cfg.get("batch_size", 8),
            factor_args=args,
            overwrite_output_dir=True,
        )
    eig = analyzer.load_eigendecomposition("ekfac")
    lambdas = analyzer.load_lambda_matrices("ekfac")
    directory = Path(output_dir)
    (directory / "linear").mkdir(parents=True, exist_ok=True)
    (directory / "diag").mkdir(exist_ok=True)
    manifest.save(directory)
    for name in names:
        base = directory / "linear" / safe(name)
        base.mkdir(parents=True, exist_ok=True)
        factor_values = {
            "U_A": eig["activation_eigenvectors"][name],
            "U_S": eig["gradient_eigenvectors"][name],
            "lam": lambdas["lambda_matrix"][name].double()
            / lambdas["num_lambda_processed"][name].double(),
        }
        for key, value in factor_values.items():
            np.save(base / f"{key}.npy", value.detach().float().cpu().numpy())
    # Kronfluence does not cover non-Linear coordinates. Fit their empirical
    # diagonal from the same causal-token samples.
    values = (
        torch.cat([accum[e.name] / count for e in diagonal])
        if diagonal
        else torch.empty(0, dtype=torch.float64)
    )
    np.save(directory / "diag" / "v.npy", values.numpy())
    (directory / "diag" / "index.json").write_text(
        json.dumps(
            [
                {"name": e.name, "numel": e.numel, "offset": e.global_flat_offset}
                for e in diagonal
            ]
        )
    )
    (directory / "ekfac_meta.json").write_text(
        json.dumps(
            {
                "linears": names,
                "samples_diag": count,
                "bias_handling": "augmented (weight+bias EK-FAC'd jointly)",
            },
            indent=2,
        )
    )
    return load_ekfac(directory, manifest)


def _conditioner_blocks(values, manifest, names, diagonal):
    """Reshape the flat A_l vector into per-Linear augmented blocks and
    per-diagonal-entry slices, in manifest coordinates."""
    entries = {e.name: e for e in manifest.included_entries()}
    source = values.detach().reshape(-1).cpu().double()
    blocks: dict[str, torch.Tensor] = {}
    for name in names:
        weight = entries[f"{name}.weight"]
        bias = entries.get(f"{name}.bias")
        block = source[
            weight.global_flat_offset : weight.global_flat_offset + weight.numel
        ].reshape(weight.shape)
        if bias is not None:
            block = torch.cat(
                (
                    block,
                    source[
                        bias.global_flat_offset : bias.global_flat_offset + bias.numel
                    ].reshape(-1, 1),
                ),
                1,
            )
        blocks[name] = block
    diag_scales = {
        e.name: source[e.global_flat_offset : e.global_flat_offset + e.numel]
        for e in diagonal
    }
    return blocks, diag_scales


def _top_two_singular_values(matrix: torch.Tensor, iterations: int = 200) -> tuple[float, float]:
    """Leading two singular values by power iteration with one deflation.

    Convergence is geometric in (sigma2/sigma1)^k, so for near-degenerate
    spectra (sigma1 ~ sigma2) the deflation vector is inaccurate and sigma2
    is biased. That regime returns a ratio near 1 either way, which is the
    qualitatively correct answer for the rank-1 residual diagnostic this
    feeds (the D1 follow-up gate cares about the small-ratio regime, where
    convergence is fast)."""
    m = matrix.detach().double()
    if min(m.shape) == 0:
        return 0.0, 0.0

    def leading(matvec, rmatvec, dim):
        generator = torch.Generator(device="cpu").manual_seed(0)
        v = torch.randn(dim, generator=generator, dtype=torch.float64)
        v = v / v.norm().clamp_min(1e-300)
        sigma = 0.0
        for _ in range(iterations):
            u = matvec(v)
            sigma = float(u.norm())
            if sigma == 0.0:
                return 0.0, v, torch.zeros(m.shape[0], dtype=torch.float64)
            u = u / sigma
            v = rmatvec(u)
            norm = float(v.norm())
            if norm == 0.0:
                return 0.0, v, u
            v = v / norm
            sigma = norm
        image = matvec(v)
        return sigma, v, image / max(float(image.norm()), 1e-300)

    sigma1, v1, u1 = leading(lambda x: m @ x, lambda x: m.T @ x, m.shape[1])
    if min(m.shape) < 2 or sigma1 == 0.0:
        return sigma1, 0.0
    sigma2, _, _ = leading(
        lambda x: m @ x - sigma1 * u1 * float(v1 @ x),
        lambda x: m.T @ x - sigma1 * v1 * float(u1 @ x),
        m.shape[1],
    )
    return sigma1, sigma2


def _unwrap_tracked_modules(model, requires_grad_states, was_training):
    """Reverse Kronfluence's in-place prepare_model wrapping and restore
    parameter/train state so the fused conditioned pass sees a clean model."""
    from kronfluence.module.tracked_module import TrackedModule

    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TrackedModule)
    ]
    for name, module in replacements:
        if "." in name:
            parent_name, target = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, target = model, name
        setattr(parent, target, module.original_module)
    if any(isinstance(m, TrackedModule) for m in model.modules()):
        raise RuntimeError("failed to unwrap Kronfluence TrackedModule wrappers")
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name in requires_grad_states:
            parameter.requires_grad = requires_grad_states[name]
    model.train(was_training)


def _fit_ekfac_conditioned(
    *,
    model,
    manifest,
    cfg,
    output_dir,
    conditioner,
    provenance,
    names,
    items,
    sample_dataset,
    model_device,
    named,
    diagonal,
    kron,
):
    Analyzer, prepare_model, FactorArguments, Task = kron
    if not items:
        raise ValueError("cannot fit conditioned EK-FAC from an empty dataset")
    blocks, diag_scales = _conditioner_blocks(
        conditioner.values, manifest, names, diagonal
    )
    requires_grad_states = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    was_training = model.training
    task = _causal_token_task(Task, names)
    prepared = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name="ekfac",
        model=prepared,
        task=task,
        cpu=model_device.type == "cpu",
        output_dir=str(Path(output_dir) / "kronfluence"),
        disable_tqdm=True,
    )
    args = FactorArguments(
        strategy="ekfac",
        use_empirical_fisher=cfg.get("use_empirical_fisher", True),
        covariance_module_partitions=cfg.get("covariance_module_partitions", 1),
        lambda_module_partitions=cfg.get("lambda_module_partitions", 1),
        eigendecomposition_dtype=getattr(
            torch, cfg.get("eigendecomposition_dtype", "float64")
        ),
    )
    # Staged fit: covariances + eigendecomposition only. Kronfluence's lambda
    # pass exploits per-token rank-1 structure that elementwise conditioning
    # destroys, so lambdas come from our fused per-item pass below.
    analyzer.fit_covariance_matrices(
        factors_name="ekfac",
        dataset=sample_dataset,
        per_device_batch_size=cfg.get("batch_size", 8),
        factor_args=args,
        overwrite_output_dir=True,
    )
    if cfg["eigh_device"] == "auto":
        analyzer.perform_eigendecomposition(
            factors_name="ekfac", factor_args=args, overwrite_output_dir=True
        )
    else:
        _lifted_eigendecomposition(
            analyzer, prepared, args, _eigh_torch_device(cfg["eigh_device"])
        )
    eig = analyzer.load_eigendecomposition("ekfac")
    _unwrap_tracked_modules(model, requires_grad_states, was_training)
    named = dict(model.named_parameters(remove_duplicate=False))
    # Host-memory shape (wave-1 OOM postmortem, 487 GB RSS): the previous
    # form converted the ENTIRE eigenvector set to fp64 on host (~328 GB at
    # gemma-3-12b full coverage) while the fp32 originals (~164 GB) and the
    # fp64 conditioner blocks (~86 GB) stayed alive. Now the fp32 set from
    # kronfluence is the only store; fp64 conversion happens per module at
    # device-staging time inside the chunked loop, and each chunk's
    # artifacts (U_A/U_S/lam .npy) are written at chunk end so the chunk's
    # eigenvectors and lambda grids are freed progressively. NOTE: for
    # bias-less Linears the conditioner blocks are VIEWS of one shared fp64
    # source tensor kept alive by diag_scales for the whole fit (~86 GB,
    # constant, not freed per chunk). Worst-case host peak at 12B full
    # coverage: fp32 eigenvectors ~164 GB (shrinking per chunk) + fp64
    # conditioner source ~86 GB (constant) + diag accumulators + one chunk's
    # staging/grads ~= 260-280 GB at the first chunk — vs ~580 GB before.
    # Modules are processed in chunks sized to free device memory — each
    # chunk stages its fp64 U/M blocks and accumulators on the device, then
    # replays the full item loop (forward + backward) for that chunk.
    # Tradeoff: n_chunks full backward passes over the items buy bounded
    # device memory without per-item transfer thrash; the projection and
    # accumulation math stays fp64 throughout, so numerics are identical to
    # the single-pass form regardless of chunking.
    activation_vectors = eig["activation_eigenvectors"]
    gradient_vectors = eig["gradient_eigenvectors"]
    modules = dict(model.named_modules())

    def _module_device_bytes(name: str) -> int:
        # Shapes only — no fp64 conversion outside the staging loop.
        a_dim = activation_vectors[name].shape[0]
        s_dim = gradient_vectors[name].shape[0]
        lam_numel = s_dim * a_dim
        staged_numel = a_dim * a_dim + s_dim * s_dim + blocks[name].numel() + 2 * lam_numel
        return 8 * staged_numel

    if names and model_device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(model_device)
        chunk_budget = max(
            free_bytes // 4, max(_module_device_bytes(n) for n in names)
        )
    else:
        # CPU device: staging is a no-op, one chunk avoids repeated passes.
        chunk_budget = None
    module_chunks: list[list[str]] = []
    pending: list[str] = []
    pending_bytes = 0
    for name in names:
        needed = _module_device_bytes(name)
        if chunk_budget is not None and pending and (
            pending_bytes + needed > chunk_budget
        ):
            module_chunks.append(pending)
            pending, pending_bytes = [], 0
        pending.append(name)
        pending_bytes += needed
    if pending or not module_chunks:
        module_chunks.append(pending)

    # Per-module (and per-diagonal-entry) item counts, matching Kronfluence's
    # hook-fire semantics: a module that produced no gradient for an item
    # contributes neither a term nor a count. A global divisor would
    # systematically deflate lambdas of conditionally-executed modules.
    lam_counts = {name: 0 for name in names}
    diag_accum = {e.name: torch.zeros(e.numel, dtype=torch.float64) for e in diagonal}
    diag_counts = {e.name: 0 for e in diagonal}
    residuals: dict[str, float] = {}
    count = 0
    directory = Path(output_dir)
    (directory / "linear").mkdir(parents=True, exist_ok=True)
    (directory / "diag").mkdir(exist_ok=True)
    manifest.save(directory)
    from .gradients import backward_memory_mode

    with backward_memory_mode(
        model, getattr(model, "is_gradient_checkpointing", False)
    ):
        for chunk_index, chunk in enumerate(module_chunks):
            # fp64 conversion happens here, per module, during transfer —
            # never for the whole set at once (B2).
            staged = {
                name: (
                    activation_vectors[name].detach().to(
                        device=model_device, dtype=torch.float64
                    ),
                    gradient_vectors[name].detach().to(
                        device=model_device, dtype=torch.float64
                    ),
                    blocks[name].to(model_device),
                )
                for name in chunk
            }
            chunk_accum = {
                name: torch.zeros(
                    gradient_vectors[name].shape[0],
                    activation_vectors[name].shape[0],
                    dtype=torch.float64,
                    device=model_device,
                )
                for name in chunk
            }
            first_chunk = chunk_index == 0
            for item in items:
                ids = item["input_ids"].to(model_device)
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                position = item["position"]
                if isinstance(position, torch.Tensor):
                    position = int(position.reshape(-1)[0])
                model.zero_grad(set_to_none=True)
                logits = model(input_ids=ids).logits
                loss = torch.nn.functional.cross_entropy(
                    logits[0, position - 1 : position].float(),
                    ids[0, position : position + 1],
                    reduction="sum",
                )
                loss.backward()
                if first_chunk:
                    count += 1
                    for entry in diagonal:
                        gradient = named[entry.name].grad
                        if gradient is not None:
                            conditioned = gradient.detach().reshape(-1).cpu().double() * (
                                diag_scales[entry.name]
                            )
                            diag_accum[entry.name].add_(conditioned.square())
                            diag_counts[entry.name] += 1
                for name in chunk:
                    weight_grad = named[f"{name}.weight"].grad
                    if weight_grad is None:
                        continue
                    dense = weight_grad.detach().double()
                    if modules[name].bias is not None:
                        bias_grad = named[f"{name}.bias"].grad
                        dense = torch.cat(
                            (
                                dense,
                                (
                                    bias_grad.detach().double()
                                    if bias_grad is not None
                                    else torch.zeros(
                                        dense.shape[0],
                                        dtype=torch.float64,
                                        device=dense.device,
                                    )
                                ).reshape(-1, 1),
                            ),
                            1,
                        )
                    u_a, u_s, scale = staged[name]
                    conditioned = scale * dense
                    projected = u_s.T @ conditioned @ u_a
                    chunk_accum[name].add_(projected.square())
                    lam_counts[name] += 1
            unexercised = sorted(n for n in chunk if lam_counts[n] == 0)
            if unexercised:
                raise ValueError(
                    "conditioned EK-FAC fit: no sampled item produced "
                    f"gradients for {unexercised} — every included parameter "
                    "must be exercised by the fit sample (exclude the "
                    "parameter or fix the sampling)"
                )
            if count == 0:
                raise ValueError(
                    "conditioned EK-FAC fit received no sample items"
                )
            # Chunk artifacts are written now so the chunk's eigenvectors,
            # conditioner block, and lambda grid are freed before the next
            # chunk stages (B2: progressive host-memory release).
            for name in chunk:
                sigma1, sigma2 = _top_two_singular_values(blocks[name])
                residuals[name] = 0.0 if sigma1 == 0.0 else sigma2 / sigma1
                base = directory / "linear" / safe(name)
                base.mkdir(parents=True, exist_ok=True)
                factor_values = {
                    "U_A": activation_vectors[name],
                    "U_S": gradient_vectors[name],
                    "lam": chunk_accum[name].cpu() / lam_counts[name],
                }
                for key, value in factor_values.items():
                    np.save(
                        base / f"{key}.npy",
                        value.detach().float().cpu().numpy(),
                    )
                del activation_vectors[name], gradient_vectors[name]
                del blocks[name]
            del staged, chunk_accum
    model.zero_grad(set_to_none=True)
    if count == 0:
        raise ValueError("conditioned EK-FAC fit received no sample items")
    missing_diag = sorted(
        e.name for e in diagonal if diag_counts[e.name] == 0
    )
    if missing_diag:
        raise ValueError(
            "conditioned EK-FAC fit: no sampled item produced gradients for "
            f"{missing_diag} — every included parameter must be exercised by "
            "the fit sample (exclude the parameter or fix the sampling)"
        )
    values = (
        torch.cat([diag_accum[e.name] / diag_counts[e.name] for e in diagonal])
        if diagonal
        else torch.empty(0, dtype=torch.float64)
    )
    np.save(directory / "diag" / "v.npy", values.numpy())
    (directory / "diag" / "index.json").write_text(
        json.dumps(
            [
                {"name": e.name, "numel": e.numel, "offset": e.global_flat_offset}
                for e in diagonal
            ]
        )
    )
    (directory / "ekfac_meta.json").write_text(
        json.dumps(
            {
                "linears": names,
                "samples_diag": count,
                "samples_lambda": count,
                "lambda_item_counts": lam_counts,
                "bias_handling": "augmented (weight+bias EK-FAC'd jointly)",
                "preconditioner": provenance,
                "lambda_fit": "scimt_conditioned_per_item_v1",
                "rank1_residuals": residuals,
            },
            indent=2,
        )
    )
    return load_ekfac(directory, manifest, expected_mode="ekfac_adam")
