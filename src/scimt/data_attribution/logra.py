"""LoGra projection, persistence, and whitening, ported from gradient-kernel ca9689a."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F

from .ekfac import load_ekfac
from .manifest import ParameterManifest


class LogRaLinear(nn.Module):
    def __init__(self, wrapped: nn.Linear, A: torch.Tensor, C: torch.Tensor):
        super().__init__()
        rank = A.shape[0]
        if A.dtype != torch.float32 or C.dtype != torch.float32:
            raise ValueError("LoGra projections must have dtype torch.float32")
        if A.shape != (rank, wrapped.in_features) or C.shape != (
            wrapped.out_features,
            rank,
        ):
            raise ValueError("LoGra projection shapes do not match wrapped Linear")
        self.wrapped = wrapped
        self.logra_A = nn.Parameter(A, requires_grad=False)
        self.logra_B = nn.Parameter(torch.zeros(rank, rank, device=A.device))
        self.logra_C = nn.Parameter(C, requires_grad=False)

    @property
    def rank(self):
        return self.logra_B.shape[0]

    def forward(self, x):
        output = self.wrapped(x)
        projected = F.linear(
            F.linear(F.linear(x.float(), self.logra_A), self.logra_B), self.logra_C
        )
        return output + projected.to(output.dtype)


@dataclass(frozen=True)
class InjectionReport:
    wrapped_modules: tuple[str, ...]
    include_regex: str
    projection_digest: str
    rank: int
    init: str
    seed: int
    has_bias: bool


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def inject_logra(model, *, rank, seed, init="random", targets=".*", projections=None):
    if rank <= 0:
        raise ValueError("rank must be positive")
    if init not in {"random", "pca", "artifact"}:
        raise ValueError(f"unsupported LoGra init {init!r}")
    if (init == "random") == (projections is not None):
        raise ValueError(
            "precomputed projections are required exactly for pca/artifact init"
        )
    try:
        target = re.compile(targets)
    except re.error as error:
        raise ValueError(f"invalid targets regex {targets!r}") from error
    paths, modules = {}, {}
    for path, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, LogRaLinear):
            raise ValueError("inject_logra must not be applied twice")
        if isinstance(module, nn.Linear):
            paths.setdefault(id(module), []).append(path)
            modules[id(module)] = module
    selected = [
        (aliases[0], modules[mid], aliases)
        for mid, aliases in paths.items()
        if any(target.fullmatch(p) for p in aliases)
    ]
    if not selected:
        raise ValueError("no nn.Linear module paths match targets regex")
    digest = hashlib.sha256()
    canonical = []
    all_paths = []
    for index, (name, linear, aliases) in enumerate(selected):
        if not name:
            raise ValueError(
                "injecting a model that is itself nn.Linear is unsupported"
            )
        if init == "random":
            r = min(rank, linear.in_features, linear.out_features)
            generator = torch.Generator(device=linear.weight.device).manual_seed(
                seed + index
            )
            A = (
                torch.randn(
                    r,
                    linear.in_features,
                    generator=generator,
                    device=linear.weight.device,
                )
                * r**-0.5
            )
            C = (
                torch.randn(
                    linear.out_features,
                    r,
                    generator=generator,
                    device=linear.weight.device,
                )
                * r**-0.5
            )
        else:
            try:
                A, C = projections[name]
            except KeyError:
                raise ValueError(
                    f"wrapped module {name!r} is missing a {init.upper()} projection"
                ) from None
            A = A.detach().to(linear.weight.device, torch.float32)
            C = C.detach().to(linear.weight.device, torch.float32)
        wrapper = LogRaLinear(linear, A, C)
        for alias in aliases:
            parent, _, child = alias.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, wrapper)
        digest.update(_bytes(A))
        digest.update(_bytes(C))
        canonical.append(name)
        all_paths.extend(aliases)
    regex = rf"(?:{'|'.join(re.escape(p) for p in all_paths)})\.logra_B"
    return InjectionReport(
        tuple(canonical),
        regex,
        digest.hexdigest(),
        rank,
        init,
        seed,
        any(m.bias is not None for _, m, _ in selected),
    )


def pca_projections(factors_dir, module_names, rank, manifest):
    if rank <= 0:
        raise ValueError("rank must be positive")
    requested = tuple(module_names)
    if len(set(requested)) != len(requested):
        raise ValueError("module_names contains duplicates")
    factors = load_ekfac(factors_dir, manifest)
    entries = {e.name: e for e in manifest.included_entries()}
    result = {}
    for name in requested:
        if name not in factors.linears:
            raise ValueError(f"wrapped module {name!r} is missing from EK-FAC factors")
        out_features, in_features = entries[f"{name}.weight"].shape
        if rank > min(in_features, out_features):
            raise ValueError(f"rank {rank} exceeds dimensions of module {name!r}")
        ua = factors.linears[name]["U_A"].double()[:, -rank:]
        if f"{name}.bias" in entries:
            ua = torch.linalg.qr(ua[:-1], mode="reduced").Q
        result[name] = (
            ua.T.contiguous().float(),
            factors.linears[name]["U_S"][:, -rank:].contiguous().float(),
        )
    return result


def projection_descriptor(report: InjectionReport) -> dict:
    return {
        "rank": report.rank,
        "init": report.init,
        "seed": report.seed,
        "wrapped_modules": list(report.wrapped_modules),
        "projection_digest": report.projection_digest,
        "has_bias": report.has_bias,
    }


@dataclass(frozen=True)
class ProjectionArtifacts(Mapping):
    FILE = "logra_projections.safetensors"
    META = "logra_projections.json"
    projections: dict[str, tuple[torch.Tensor, torch.Tensor]]
    descriptor: dict
    projection_digest: str
    manifest_digest: str
    content_digest: str

    def __getitem__(self, key):
        return self.projections[key]

    def __iter__(self):
        return iter(self.projections)

    def __len__(self):
        return len(self.projections)

    @staticmethod
    def save(
        model,
        path,
        *,
        manifest: ParameterManifest,
        descriptor: dict | None = None,
        report: InjectionReport | None = None,
    ):
        if (descriptor is None) == (report is None):
            raise ValueError("provide exactly one projection descriptor or report")
        if report is not None:
            descriptor = projection_descriptor(report)
        tensors = {}
        for name, module in model.named_modules():
            if isinstance(module, LogRaLinear):
                tensors[f"{name}.A"] = module.logra_A.detach().cpu()
                tensors[f"{name}.C"] = module.logra_C.detach().cpu()
        if not tensors:
            raise ValueError("model contains no LoGra projections")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(tensors, directory / ProjectionArtifacts.FILE)
        projection_digest = hashlib.sha256(
            (directory / ProjectionArtifacts.FILE).read_bytes()
        ).hexdigest()
        metadata = {
            "descriptor": descriptor,
            "projection_digest": projection_digest,
            "manifest_digest": manifest.digest(),
        }
        content_digest = hashlib.sha256(
            (directory / ProjectionArtifacts.FILE).read_bytes()
            + json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        metadata["content_digest"] = content_digest
        (directory / ProjectionArtifacts.META).write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n"
        )
        manifest.save(directory)
        return ProjectionArtifacts.load(
            directory,
            expected_manifest=manifest,
            expected_descriptor=descriptor,
        )

    @staticmethod
    def load(
        path,
        *,
        expected_manifest: ParameterManifest,
        expected_descriptor: dict | None = None,
        expected_report: InjectionReport | None = None,
    ):
        if (expected_descriptor is None) == (expected_report is None):
            raise ValueError(
                "provide exactly one expected projection descriptor or report"
            )
        if expected_report is not None:
            expected_descriptor = projection_descriptor(expected_report)
        directory = Path(path)
        stored = ParameterManifest.load(directory)
        if stored.digest() != expected_manifest.digest():
            raise ValueError("LoGra projection manifest digest mismatch")
        metadata = json.loads((directory / ProjectionArtifacts.META).read_text())
        if metadata.get("manifest_digest") != expected_manifest.digest():
            raise ValueError("LoGra projection manifest digest mismatch")
        if metadata.get("descriptor") != expected_descriptor:
            raise ValueError("LoGra projection descriptor mismatch")
        recorded = metadata.pop("content_digest", None)
        actual = hashlib.sha256(
            (directory / ProjectionArtifacts.FILE).read_bytes()
            + json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if recorded != actual:
            raise ValueError("LoGra projection artifact content digest mismatch")
        tensors = load_file(directory / ProjectionArtifacts.FILE, device="cpu")
        grouped = {}
        for key, value in tensors.items():
            name, sep, side = key.rpartition(".")
            if not sep or side not in {"A", "C"}:
                raise ValueError(f"invalid LoGra projection tensor key {key!r}")
            grouped.setdefault(name, {})[side] = value
        if not grouped or any(set(v) != {"A", "C"} for v in grouped.values()):
            raise ValueError(
                "LoGra projection artifact must contain paired A/C tensors"
            )
        projection_digest = hashlib.sha256(
            (directory / ProjectionArtifacts.FILE).read_bytes()
        ).hexdigest()
        if projection_digest != metadata.get("projection_digest"):
            raise ValueError("LoGra projection content digest mismatch")
        return ProjectionArtifacts(
            {name: (v["A"], v["C"]) for name, v in grouped.items()},
            dict(expected_descriptor),
            projection_digest,
            expected_manifest.digest(),
            recorded,
        )


class LogRaFisherState:
    def __init__(self, module_slices, *, width=None, device="cpu"):
        self.module_slices = _validate_slices(
            module_slices,
            width if width is not None else max(v.stop for v in module_slices.values()),
        )
        self.width = (
            width if width is not None else max(v.stop for v in module_slices.values())
        )
        self.count = 0
        self.means = {
            n: torch.zeros(
                v.stop - v.start, v.stop - v.start, dtype=torch.float64, device=device
            )
            for n, v in self.module_slices.items()
        }

    def update(self, rows):
        if rows.ndim == 1:
            rows = rows[None]
        if (
            rows.ndim != 2
            or rows.shape[1] != self.width
            or not rows.is_floating_point()
        ):
            raise ValueError(f"rows must have shape [N, {self.width}]")
        if not len(rows):
            return
        total = self.count + len(rows)
        for name, value in self.module_slices.items():
            x = rows[:, value].detach().to(self.means[name].device, torch.float64)
            batch = x.T @ x / len(rows)
            self.means[name].add_((batch - self.means[name]) * (len(rows) / total))
        self.count = total

    def merge(self, other):
        if self.width != other.width or self.module_slices != other.module_slices:
            raise ValueError("states must have identical module slices and width")
        if not other.count:
            return
        total = self.count + other.count
        for name in self.means:
            self.means[name].add_(
                (other.means[name].to(self.means[name]) - self.means[name])
                * (other.count / total)
            )
        self.count = total

    def finalize(self):
        if not self.count:
            raise ValueError("cannot finalize an empty state")
        return {n: v.detach().cpu().clone() for n, v in self.means.items()}


def _validate_slices(slices, width):
    if not slices:
        raise ValueError("module_slices must not be empty")
    result = dict(slices)
    occupied = []
    for name, value in result.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, slice)
            or value.step not in (None, 1)
            or value.start is None
            or value.stop is None
            or value.start < 0
            or value.stop <= value.start
            or value.stop > width
        ):
            raise ValueError(f"invalid slice for module {name!r}")
        occupied.append((value.start, value.stop, name))
    occupied.sort()
    if any(a[1] > b[0] for a, b in zip(occupied, occupied[1:])):
        raise ValueError("module slices overlap")
    return result


def whiten_rows(
    rows,
    module_slices,
    fishers: Mapping[str, torch.Tensor],
    *,
    damping_scale,
    power=-0.5,
):
    if rows.ndim != 2 or not rows.is_floating_point():
        raise ValueError("rows must be a floating-point tensor with shape [N, D]")
    if damping_scale < 0:
        raise ValueError("damping_scale must be nonnegative")
    slices = _validate_slices(module_slices, rows.shape[1])
    if set(fishers) != set(slices):
        raise ValueError("fishers and module_slices must have identical module names")
    result = rows.detach().float().clone()
    for name, value in slices.items():
        size = value.stop - value.start
        fisher = fishers[name].detach().cpu().double()
        if fisher.shape != (size, size):
            raise ValueError(f"fisher for {name!r} must have shape [{size}, {size}]")
        if float((fisher - fisher.T).abs().max()) > 1e-8 * max(
            1.0, float(fisher.abs().max())
        ):
            raise ValueError(f"fisher for {name!r} must be symmetric")
        fisher = (fisher + fisher.T) / 2
        eig, vec = torch.linalg.eigh(fisher)
        tolerance = (
            torch.finfo(torch.float64).eps * max(1.0, float(eig.abs().max())) * size
        )
        if float(eig.min()) < -tolerance:
            raise ValueError(f"fisher for {name!r} is not positive semidefinite")
        eig.clamp_min_(0)
        damped = eig + damping_scale * eig.mean()
        if power < 0 and bool((damped == 0).any()):
            raise ValueError("negative powers require positive damped eigenvalues")
        transform = (vec * damped.pow(power)) @ vec.T
        result[:, value] = (rows[:, value].detach().cpu().double() @ transform).to(
            rows.device, torch.float32
        )
    return result


def module_slices_from_manifest(manifest):
    result = {}
    for entry in manifest.included_entries():
        if not entry.name.endswith(".logra_B"):
            raise ValueError(
                "LoGra Fisher manifests may include only logra_B parameters"
            )
        name = entry.name.removesuffix(".logra_B")
        result[name] = slice(
            entry.global_flat_offset, entry.global_flat_offset + entry.numel
        )
    if not result:
        raise ValueError("manifest contains no logra_B parameters")
    return result
