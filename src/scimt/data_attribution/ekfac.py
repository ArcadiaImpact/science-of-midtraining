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


@dataclass(frozen=True)
class EKFACFactors:
    linears: dict[str, dict[str, torch.Tensor]]
    diag_v: torch.Tensor
    diag_index: tuple[dict[str, Any], ...]
    snapshot: str

    @property
    def snapshot_id(self) -> str:
        return self.snapshot


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


def load_ekfac(path: str | Path, manifest: ParameterManifest) -> EKFACFactors:
    directory = Path(path)
    stored = ParameterManifest.load(directory)
    if stored.digest() != manifest.digest():
        raise ManifestMismatchError("manifest digest mismatch")
    metadata = json.loads((directory / "ekfac_meta.json").read_text())
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
    return EKFACFactors(linears, diag_v, tuple(raw_index), _snapshot(directory))


load_ekfac_factors = load_ekfac


def _scale(values: torch.Tensor, damping: float, power: float) -> torch.Tensor:
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
    return result


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


def fit_ekfac(model, dataset, manifest, config, output_dir):
    """Fit Kronfluence factors; Kronfluence remains an optional lazy dependency."""
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
    accum = {e.name: torch.zeros(e.numel, dtype=torch.float64) for e in diagonal}
    count = 0
    # This pass must precede prepare_model, which may freeze parameters.
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
    analyzer.fit_all_factors(
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
