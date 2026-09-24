"""GPU port of ``apply_ekfac``'s damped-inverse math (``power = -1``), per module.

Why: the library ``scimt.data_attribution.ekfac.apply_ekfac`` runs on the CPU
in fp64 over the whole flat vector — ~720 TFLOP fp64 and ~172 GB of host
memory per 12B vector, 15–60 min each (PREMORTEM.md item A2). Here each Linear
module's factors are moved to the device as fp32, the module's block of the
input vector is rotated ONCE into the Kronecker eigenbasis, and every
requested damping is applied to that rotation, so N damping values (and any
number of input vectors) cost one pass over the factor set:

    block    = [W | b]                                  # [out, in(+1)] augmented
    rotated  = U_S.T @ block @ U_A
    restored = U_S @ (rotated * scale_d) @ U_A.T          # per damping d
    scale_d  = ekfac._scale(lam, d, -1.0) = 1 / (lam + d * lam.mean())

exactly ``apply_ekfac``'s formula and association order (``ekfac.py``, the
``restored = us @ ((us.T @ block @ ua) * _scale(...)) @ ua.T`` line), with the
weight rows written back to the weight's flat range and the bias column to
the bias's. Coordinates outside any Linear factor (the manifest's diagonal
remainder, ``factors.diag_index``) are scaled by ``_scale(diag_v, d, -1)`` —
note the mean in that formula is over the WHOLE ``diag_v`` vector, as in the
library. ``_scale`` is imported, not re-implemented, so the formula cannot
drift; its documented corner (a zero eigenvalue with damping 0 yields inf) is
made unreachable here by refusing ``damping_scale <= 0`` loudly (``apply_ekfac``
itself accepts 0; the study's sweep is {0.01, 0.1, 1}).

Precision: fp32 on the device against the library's fp64. :func:`oracle_check`
compares the two on a random vector restricted to a couple of modules (plus
one diagonal entry) and reports the max relative error (``max|ours - ref| /
max|ref|``); ``ORACLE_TOLERANCE = 1e-5``. Run it before spending GPU hours —
``main`` does, and refuses to apply when it fails.

Shared vector contract (other agents write the ``gdp`` side of it)
------------------------------------------------------------------
Flat fp32 vectors are raw little-endian float32 files ``<name>.f32`` of
length ``manifest.included_numel``, in ``ParameterManifest`` included-entry
order (the runner's flat layout), with a JSON sidecar ``<name>.json``::

    {"name", "kind": "gdp"|"inv", "dataset", "damping_scale" (null for gdp),
     "fold": "all"|"f0"|"f1"|..., "n_rows", "n_tokens", "manifest_digest",
     "model": {"hf_id", "sha"}, "sequence_length", "created_at",
     "source_vector" (inv only: the gdp file it was derived from)}

:func:`apply_inverse` reads a ``gdp`` ``.f32`` + sidecar and writes one
``inv`` ``.f32`` + sidecar per damping, named
``<dataset>__inv<damping>__<fold>.f32`` (``format_damping``: ``0.1`` →
``inv0.1``, ``1.0`` → ``inv1``, ``0.01`` → ``inv0.01``). Reads use
``np.fromfile`` at byte offsets and writes ``seek``+``write`` — no mmap of the
43 GB vectors (mapped page cache is charged to the cgroup; gate2 pod
20260819T095144Z).

Usage on the pod (config-first; one JSON mapping, see :class:`ApplyConfig`)::

    python .../pod/apply_inverse_gpu.py apply.json
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import re
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

KINDS = ("gdp", "inv")
SIDECAR_KEYS = (
    "name",
    "kind",
    "dataset",
    "damping_scale",
    "fold",
    "n_rows",
    "n_tokens",
    "manifest_digest",
    "model",
    "sequence_length",
    "created_at",
    "source_vector",
)
MODEL_KEYS = ("hf_id", "sha")
FOLD_PATTERN = re.compile(r"all|f\d+")
NAME_PATTERN = re.compile(r"[A-Za-z0-9_.+-]+")
VECTOR_SUFFIX = ".f32"
ORACLE_TOLERANCE = 1e-5
ORACLE_PASS_SENTINEL = "SCIMT-APPLY-ORACLE-PASS"
ORACLE_FAIL_SENTINEL = "SCIMT-APPLY-ORACLE-FAIL"
APPLY_DONE_SENTINEL = "SCIMT-APPLY-INVERSE-DONE"
ORACLE_EXIT_CODE = 99


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


# ------------------------------------------------------------------ contract
def format_damping(damping_scale: float) -> str:
    """Compact, round-trippable damping label: 0.1 -> '0.1', 1.0 -> '1'."""
    value = float(damping_scale)
    if not value > 0:
        raise ValueError("damping_scale must be positive")
    return f"{value:g}"


def inv_vector_name(dataset: str, damping_scale: float, fold: str) -> str:
    if not NAME_PATTERN.fullmatch(dataset) or "__" in dataset:
        raise ValueError(f"dataset label {dataset!r} must match {NAME_PATTERN.pattern} and not contain '__'")
    if not FOLD_PATTERN.fullmatch(fold):
        raise ValueError(f"fold {fold!r} must be 'all' or 'f<n>'")
    return f"{dataset}__inv{format_damping(damping_scale)}__{fold}"


def sidecar_path(vector_path: str | Path) -> Path:
    path = Path(vector_path)
    if path.suffix != VECTOR_SUFFIX:
        raise ValueError(f"vector files carry the {VECTOR_SUFFIX} suffix: {path}")
    return path.with_suffix(".json")


def validate_sidecar(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Schema check for the shared vector contract; returns a plain dict."""
    if not isinstance(payload, Mapping):
        raise TypeError("sidecar must be a mapping")
    missing = sorted(set(SIDECAR_KEYS) - set(payload))
    extra = sorted(set(payload) - set(SIDECAR_KEYS))
    if missing or extra:
        raise ValueError(f"sidecar keys: missing {missing}, unexpected {extra}")
    kind = payload["kind"]
    if kind not in KINDS:
        raise ValueError(f"sidecar kind must be one of {list(KINDS)}, got {kind!r}")
    for key in ("name", "dataset", "fold", "manifest_digest", "created_at"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"sidecar {key} must be a nonempty string")
    if not NAME_PATTERN.fullmatch(payload["dataset"]) or "__" in payload["dataset"]:
        raise ValueError("sidecar dataset must be a plain label without '__'")
    if not FOLD_PATTERN.fullmatch(payload["fold"]):
        raise ValueError("sidecar fold must be 'all' or 'f<n>'")
    for key in ("n_rows", "n_tokens", "sequence_length"):
        value = payload[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"sidecar {key} must be a nonnegative integer")
    model = payload["model"]
    if not isinstance(model, Mapping) or set(model) != set(MODEL_KEYS):
        raise ValueError(f"sidecar model must have exactly keys {list(MODEL_KEYS)}")
    if not all(isinstance(model[k], str) and model[k] for k in MODEL_KEYS):
        raise ValueError("sidecar model hf_id/sha must be nonempty strings")
    damping = payload["damping_scale"]
    source = payload["source_vector"]
    if kind == "gdp":
        if damping is not None:
            raise ValueError("gdp sidecar damping_scale must be null")
        if source is not None:
            raise ValueError("gdp sidecar source_vector must be null")
    else:
        if not isinstance(damping, (int, float)) or isinstance(damping, bool) or damping <= 0:
            raise ValueError("inv sidecar damping_scale must be a positive number")
        if not isinstance(source, str) or not source:
            raise ValueError("inv sidecar source_vector must name the gdp file")
        expected = inv_vector_name(payload["dataset"], damping, payload["fold"])
        if payload["name"] != expected:
            raise ValueError(f"inv sidecar name {payload['name']!r} != {expected!r}")
    return {key: (dict(model) if key == "model" else payload[key]) for key in SIDECAR_KEYS}


def read_sidecar(vector_path: str | Path) -> dict[str, Any]:
    path = sidecar_path(vector_path)
    return validate_sidecar(json.loads(path.read_text(encoding="utf-8")))


def write_sidecar(vector_path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = sidecar_path(vector_path)
    body = validate_sidecar(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def make_inv_sidecar(
    gdp_sidecar: Mapping[str, Any],
    damping_scale: float,
    source_vector: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """The inv sidecar derived from a gdp sidecar (dataset/fold/n/model copied)."""
    gdp = validate_sidecar(gdp_sidecar)
    if gdp["kind"] != "gdp":
        raise ValueError("inv vectors derive from gdp vectors only")
    return validate_sidecar(
        {
            **gdp,
            "name": inv_vector_name(gdp["dataset"], damping_scale, gdp["fold"]),
            "kind": "inv",
            "damping_scale": float(damping_scale),
            "created_at": created_at or utc_now(),
            "source_vector": source_vector,
        }
    )


def expected_vector_bytes(manifest: Any) -> int:
    return 4 * int(manifest.included_numel)


# ---------------------------------------------------------------- lazy deps
def _torch() -> Any:
    import torch

    return torch


def _ekfac() -> Any:
    from scimt.data_attribution import ekfac

    return ekfac


def damped_inverse_scale(lam: Any, damping_scale: float) -> Any:
    """``ekfac._scale(lam, damping_scale, -1.0)`` — imported, never re-derived."""
    if not float(damping_scale) > 0:
        raise ValueError(
            "damping_scale must be positive (a zero damping meets ekfac._scale's "
            "documented inf corner on any zero eigenvalue)"
        )
    return _ekfac()._scale(lam, float(damping_scale), -1.0)


def _validate_factor_domains(factors: Any) -> None:
    """Same eager checks ``apply_ekfac`` performs (lazy handles validate at touch)."""
    torch = _torch()
    ek = _ekfac()
    for name, factor in factors.linears.items():
        if isinstance(factor, ek.LazyFactorModule):
            continue
        for key in ("U_A", "U_S", "lam"):
            value = factor[key]
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"factor {name!r} {key} must be floating-point and finite")
        if bool((factor["lam"] < 0).any()):
            raise ValueError(f"factor {name!r} lam must be nonnegative")
    if not factors.diag_v.is_floating_point() or not bool(torch.isfinite(factors.diag_v).all()):
        raise ValueError("diagonal factors must be floating-point and finite")
    if bool((factors.diag_v < 0).any()):
        raise ValueError("diagonal factors must be nonnegative")


# --------------------------------------------------------------------- math
def iter_inverse_blocks(
    read_block: Callable[[int, int], Any],
    factors: Any,
    manifest: Any,
    damping_scales: Sequence[float],
    device: str = "cuda",
) -> Iterator[tuple[str, int, int, list[Any]]]:
    """Yield ``(entry_name, flat_offset, numel, outputs)`` per included manifest
    entry, where ``outputs[k]`` is the fp32 CPU tensor ``[B, numel]`` for
    ``damping_scales[k]``. ``read_block(offset, numel)`` must return the
    source coordinates as a ``[B, numel]`` tensor/array (B input vectors).
    Factors are loaded once per module and released after use."""
    torch = _torch()
    ek = _ekfac()
    dampings = [float(d) for d in damping_scales]
    if not dampings:
        raise ValueError("at least one damping_scale is required")
    for d in dampings:
        if not d > 0:
            raise ValueError("damping_scale must be positive")
    if len(set(dampings)) != len(dampings):
        raise ValueError("damping_scales must be unique")
    _validate_factor_domains(factors)
    dev = torch.device(device)
    entries = {e.name: e for e in manifest.included_entries()}

    def fetch(offset: int, numel: int) -> Any:
        raw = read_block(offset, numel)
        tensor = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(np.asarray(raw))
        tensor = tensor.detach()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != numel:
            raise ValueError(f"read_block({offset}, {numel}) returned shape {tuple(tensor.shape)}")
        return tensor.to(device=dev, dtype=torch.float32)

    for name, factor in factors.linears.items():
        weight = entries[f"{name}.weight"]
        bias = entries.get(f"{name}.bias")
        out_features, in_features = weight.shape
        block = fetch(weight.global_flat_offset, weight.numel).reshape(-1, out_features, in_features)
        if bias is not None:
            column = fetch(bias.global_flat_offset, bias.numel).reshape(-1, out_features, 1)
            block = torch.cat((block, column), 2)
        ua = factor["U_A"].to(device=dev, dtype=torch.float32)
        us = factor["U_S"].to(device=dev, dtype=torch.float32)
        lam = factor["lam"].to(device=dev, dtype=torch.float32)
        rotated = us.T @ block @ ua  # rotate once, scale N times
        weight_outputs: list[Any] = []
        bias_outputs: list[Any] = []
        for d in dampings:
            restored = us @ (rotated * damped_inverse_scale(lam, d)) @ ua.T
            weight_outputs.append(restored[:, :, :in_features].reshape(restored.shape[0], -1).cpu())
            if bias is not None:
                bias_outputs.append(restored[:, :, -1].contiguous().cpu())
            del restored
        del ua, us, lam, block, rotated
        ek.release_factor(factor)
        yield (weight.name, weight.global_flat_offset, weight.numel, weight_outputs)
        if bias is not None:
            yield (bias.name, bias.global_flat_offset, bias.numel, bias_outputs)

    if factors.diag_v.numel():
        diag = factors.diag_v.to(device=dev, dtype=torch.float32)
        scales = [damped_inverse_scale(diag, d) for d in dampings]  # mean over ALL diag_v
        cursor = 0
        for item in factors.diag_index:
            numel, offset = int(item["numel"]), int(item["offset"])
            source = fetch(offset, numel)
            yield (
                str(item["name"]),
                offset,
                numel,
                [(source * scale[cursor : cursor + numel]).cpu() for scale in scales],
            )
            cursor += numel


def apply_inverse_array(
    flat: Any,
    factors: Any,
    manifest: Any,
    damping_scales: Sequence[float] | float,
    device: str = "cpu",
) -> list[Any]:
    """In-memory variant (oracle / tests): ``flat`` is ``[P]`` or ``[B, P]``;
    returns one fp32 CPU tensor of the same shape per damping."""
    torch = _torch()
    dampings = _as_dampings(damping_scales)
    source = flat if isinstance(flat, torch.Tensor) else torch.as_tensor(np.asarray(flat))
    source = source.detach().to(dtype=torch.float32, device="cpu")
    squeeze = source.ndim == 1
    if squeeze:
        source = source.unsqueeze(0)
    numel = int(manifest.included_numel)
    if source.ndim != 2 or source.shape[1] != numel:
        raise ValueError(f"flat must have shape [{numel}] or [B, {numel}]")
    outputs = [torch.zeros_like(source) for _ in dampings]
    covered = 0
    for _, offset, count, blocks in iter_inverse_blocks(
        lambda o, n: source[:, o : o + n], factors, manifest, dampings, device
    ):
        for out, block in zip(outputs, blocks, strict=True):
            out[:, offset : offset + count] = block
        covered += count
    if covered != numel:
        raise RuntimeError(f"factor set covered {covered} of {numel} coordinates")
    return [out.squeeze(0) if squeeze else out for out in outputs]


def _as_dampings(damping_scales: Sequence[float] | float) -> list[float]:
    if isinstance(damping_scales, (int, float)) and not isinstance(damping_scales, bool):
        return [float(damping_scales)]
    return [float(d) for d in damping_scales]


def _as_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def resolve_out_paths(
    inputs: Sequence[Path],
    sidecars: Sequence[Mapping[str, Any]],
    damping_scales: Sequence[float],
    out_path: str | Path | Sequence[str | Path] | None,
) -> list[list[Path]]:
    """``[n_inputs][n_dampings]`` output paths per the naming contract.

    ``None``: next to each input. A directory (or any path without the
    ``.f32`` suffix): inside it. A single ``.f32`` path: only for exactly one
    output. A sequence: explicit paths, input-major, one per output."""
    n_out = len(inputs) * len(damping_scales)

    def derived(base: Path, sidecar: Mapping[str, Any], d: float) -> Path:
        return base / f"{inv_vector_name(sidecar['dataset'], d, sidecar['fold'])}{VECTOR_SUFFIX}"

    if out_path is None:
        return [[derived(p.parent, s, d) for d in damping_scales] for p, s in zip(inputs, sidecars, strict=True)]
    if isinstance(out_path, (str, Path)):
        target = Path(out_path)
        if target.suffix == VECTOR_SUFFIX:
            if n_out != 1:
                raise ValueError("a single .f32 out_path needs exactly one input and one damping")
            return [[target]]
        return [[derived(target, s, d) for d in damping_scales] for _, s in zip(inputs, sidecars, strict=True)]
    explicit = [Path(p) for p in out_path]
    if len(explicit) != n_out:
        raise ValueError(f"out_path lists {len(explicit)} paths for {n_out} outputs")
    if len({p.resolve() for p in explicit}) != n_out:
        raise ValueError("out_path entries must be distinct")
    width = len(damping_scales)
    return [explicit[i * width : (i + 1) * width] for i in range(len(inputs))]


def apply_inverse(
    flat_f32_path: str | Path | Sequence[str | Path],
    factors_dir: str | Path,
    manifest: Any,
    damping_scale: Sequence[float] | float,
    device: str = "cuda",
    out_path: str | Path | Sequence[str | Path] | None = None,
    *,
    factors: Any = None,
    log: Callable[[str], None] = print,
    progress_every: int = 32,
) -> list[Path]:
    """Read gdp ``.f32`` vector(s) + sidecars, write ``inv`` vector(s) + sidecars.

    Several inputs and several dampings share ONE pass over the factor set.
    Returns the written ``.f32`` paths, input-major then damping order.
    """
    torch = _torch()
    ek = _ekfac()
    inputs = _as_paths(flat_f32_path)
    dampings = _as_dampings(damping_scale)
    if not inputs:
        raise ValueError("at least one input vector is required")
    numel = int(manifest.included_numel)
    nbytes = expected_vector_bytes(manifest)
    digest = manifest.digest()
    sidecars = []
    for path in inputs:
        sidecar = read_sidecar(path)
        if sidecar["kind"] != "gdp":
            raise ValueError(f"{path} is a {sidecar['kind']!r} vector; apply_inverse consumes gdp vectors")
        if sidecar["manifest_digest"] != digest:
            raise ValueError(
                f"{path} was written against manifest {sidecar['manifest_digest'][:12]}, "
                f"not {digest[:12]}"
            )
        size = path.stat().st_size
        if size != nbytes:
            raise ValueError(f"{path} has {size} bytes; manifest requires {nbytes} (= 4 x {numel})")
        sidecars.append(sidecar)
    if factors is None:
        log(f"[{utc_now()}] load_ekfac({factors_dir}) — hashes the factor dir")
        factors = ek.load_ekfac(factors_dir, manifest)
    out_paths = resolve_out_paths(inputs, sidecars, dampings, out_path)
    flat_outputs = [p for row in out_paths for p in row]
    if any(p.resolve() == q.resolve() for p in flat_outputs for q in inputs):
        raise ValueError("an output path coincides with an input path")

    def read_block(offset: int, count: int) -> Any:
        rows = [
            np.fromfile(str(path), dtype="<f4", count=count, offset=4 * offset) for path in inputs
        ]
        for path, row in zip(inputs, rows, strict=True):
            if row.shape[0] != count:
                raise ValueError(f"{path}: short read at offset {offset}")
        return torch.from_numpy(np.stack(rows, 0))

    handles = []
    started = time.monotonic()
    try:
        for row in out_paths:
            handle_row = []
            for path in row:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = open(path, "wb")
                handle.truncate(nbytes)
                handle_row.append(handle)
            handles.append(handle_row)
        covered = 0
        n_entries = 0
        for name, offset, count, blocks in iter_inverse_blocks(
            read_block, factors, manifest, dampings, device
        ):
            for k, block in enumerate(blocks):
                arrays = block.contiguous().numpy().astype("<f4", copy=False)
                for b, handle_row in enumerate(handles):
                    handle_row[k].seek(4 * offset)
                    handle_row[k].write(arrays[b].tobytes())
            covered += count
            n_entries += 1
            if progress_every and n_entries % progress_every == 0:
                log(
                    f"[{utc_now()}] {n_entries} entries, {covered / numel:.1%} of coordinates, "
                    f"{time.monotonic() - started:.0f} s"
                )
        if covered != numel:
            raise RuntimeError(f"factor set covered {covered} of {numel} coordinates")
    finally:
        for handle_row in handles:
            for handle in handle_row:
                handle.close()
    created = utc_now()
    for path, sidecar, row in zip(inputs, sidecars, out_paths, strict=True):
        for d, out in zip(dampings, row, strict=True):
            write_sidecar(out, make_inv_sidecar(sidecar, d, path.name, created_at=created))
    log(
        f"[{utc_now()}] {APPLY_DONE_SENTINEL}: {len(flat_outputs)} vector(s) "
        f"({len(inputs)} inputs x {len(dampings)} dampings) in {time.monotonic() - started:.0f} s"
    )
    return flat_outputs


# ------------------------------------------------------------------- oracle
def restrict_to_modules(
    factors: Any, manifest: Any, names: Sequence[str], n_diag_entries: int = 1
) -> tuple[Any, Any]:
    """Factor set + manifest restricted to ``names`` (and the first
    ``n_diag_entries`` diagonal entries), offsets re-based contiguously in the
    original manifest order — consumable by both ``apply_ekfac`` and
    :func:`apply_inverse_array`."""
    ek = _ekfac()
    from scimt.data_attribution.manifest import ParameterManifest

    if not names:
        raise ValueError("names must not be empty")
    missing = [n for n in names if n not in factors.linears]
    if missing:
        raise ValueError(f"modules not in the factor set: {missing}")
    entries = {e.name: e for e in manifest.included_entries()}
    keep: set[str] = set()
    for name in names:
        keep.add(f"{name}.weight")
        if f"{name}.bias" in entries:
            keep.add(f"{name}.bias")
    diag_items = list(factors.diag_index)[: max(0, int(n_diag_entries))]
    for item in diag_items:
        keep.add(str(item["name"]))
    new_entries = []
    offsets: dict[str, int] = {}
    offset = 0
    for entry in manifest.included_entries():
        if entry.name in keep:
            new_entries.append(dataclasses.replace(entry, global_flat_offset=offset))
            offsets[entry.name] = offset
            offset += entry.numel
    sub_manifest = ParameterManifest(new_entries, manifest.model_name)
    sub_manifest.validate_semantics()
    n_diag = sum(int(item["numel"]) for item in diag_items)
    sub_factors = ek.EKFACFactors(
        linears={name: factors.linears[name] for name in names},
        diag_v=factors.diag_v[:n_diag],
        diag_index=tuple(
            {"name": str(item["name"]), "numel": int(item["numel"]), "offset": offsets[str(item["name"])]}
            for item in diag_items
        ),
        snapshot=factors.snapshot,
        preconditioner=factors.preconditioner,
    )
    return sub_factors, sub_manifest


def oracle_check(
    factors_dir: str | Path | None,
    manifest: Any,
    n_modules: int = 2,
    *,
    device: str = "cpu",
    damping_scales: Sequence[float] = (0.1,),
    seed: int = 0,
    n_diag_entries: int = 1,
    factors: Any = None,
    tolerance: float = ORACLE_TOLERANCE,
) -> dict[str, Any]:
    """Compare the device fp32 port with the library ``apply_ekfac`` (CPU fp64)
    on a random vector restricted to ``n_modules`` modules (+ one diagonal
    entry). Reports ``max_rel_error = max|ours - ref| / max|ref|`` per damping
    and overall; ``passed`` iff it is within ``tolerance``."""
    torch = _torch()
    ek = _ekfac()
    if n_modules < 1:
        raise ValueError("n_modules must be positive")
    if factors is None:
        if factors_dir is None:
            raise ValueError("factors_dir or factors is required")
        factors = ek.load_ekfac(factors_dir, manifest)
    names = list(factors.linears)[:n_modules]
    if len(names) < n_modules:
        raise ValueError(f"factor set has {len(names)} modules; {n_modules} requested")
    sub_factors, sub_manifest = restrict_to_modules(factors, manifest, names, n_diag_entries)
    dampings = _as_dampings(damping_scales)
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(sub_manifest.included_numel, generator=generator, dtype=torch.float32)
    started = time.monotonic()
    ours = apply_inverse_array(vector, sub_factors, sub_manifest, dampings, device)
    device_seconds = time.monotonic() - started
    per_damping: dict[str, Any] = {}
    worst = 0.0
    ref_started = time.monotonic()
    for d, mine in zip(dampings, ours, strict=True):
        reference = ek.apply_ekfac(vector.double(), sub_factors, sub_manifest, d, -1)
        diff = (mine.double() - reference).abs()
        ref_max = float(reference.abs().max())
        rel = float(diff.max()) / ref_max if ref_max > 0 else float(diff.max())
        worst = max(worst, rel)
        per_damping[format_damping(d)] = {
            "max_abs_error": float(diff.max()),
            "ref_max_abs": ref_max,
            "rel_error_vs_max": rel,
        }
    passed = worst <= tolerance
    return {
        "modules": names,
        "diag_entries": [item["name"] for item in sub_factors.diag_index],
        "numel": int(sub_manifest.included_numel),
        "device": str(device),
        "seed": seed,
        "tolerance": tolerance,
        "dampings": per_damping,
        "max_rel_error": worst,
        "passed": passed,
        "sentinel": ORACLE_PASS_SENTINEL if passed else ORACLE_FAIL_SENTINEL,
        "device_seconds": device_seconds,
        "reference_seconds": time.monotonic() - ref_started,
    }


# --------------------------------------------------------------------- main
@dataclass(frozen=True)
class ApplyConfig:
    """Pod-side configuration (JSON mapping; unknown keys are an error)."""

    factors_dir: str = "/workspace/attribution/ekfac_pt"
    inputs: tuple[str, ...] = ()
    damping_scales: tuple[float, ...] = (0.1,)
    device: str = "cuda"
    out_dir: str | None = None
    evidence_dir: str = "/workspace/attribution/evidence"
    oracle_modules: int = 2
    oracle_device: str | None = None  # None: same as device
    oracle_only: bool = False
    skip_oracle: bool = False

    def __post_init__(self) -> None:
        if not self.damping_scales:
            raise ValueError("damping_scales must not be empty")
        for d in self.damping_scales:
            if not float(d) > 0:
                raise ValueError("damping_scales must be positive")
        if self.oracle_modules < 1:
            raise ValueError("oracle_modules must be positive")
        if not self.oracle_only and not self.inputs:
            raise ValueError("inputs must list at least one gdp .f32 vector unless oracle_only")
        for key in ("inputs", "damping_scales"):
            value = getattr(self, key)
            if isinstance(value, list):
                object.__setattr__(self, key, tuple(value))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ApplyConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown ApplyConfig keys: {unknown}")
        return cls(**dict(mapping))


def run_apply(cfg: ApplyConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    from scimt.data_attribution.manifest import ParameterManifest

    ek = _ekfac()
    evidence_dir = Path(cfg.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = ParameterManifest.load(cfg.factors_dir)
    log(f"[{utc_now()}] load_ekfac({cfg.factors_dir})")
    factors = ek.load_ekfac(cfg.factors_dir, manifest)
    report: dict[str, Any] = {"config": dataclasses.asdict(cfg), "manifest_digest": manifest.digest()}
    if not cfg.skip_oracle:
        oracle = oracle_check(
            None,
            manifest,
            cfg.oracle_modules,
            device=cfg.oracle_device or cfg.device,
            damping_scales=cfg.damping_scales,
            factors=factors,
        )
        report["oracle"] = oracle
        path = evidence_dir / "apply_inverse_oracle.json"
        path.write_text(json.dumps(oracle, indent=2, sort_keys=True), encoding="utf-8")
        log(f"[{utc_now()}] {oracle['sentinel']}: max rel error {oracle['max_rel_error']:.3e} "
            f"over {oracle['modules']} (tolerance {oracle['tolerance']:.0e})")
        if not oracle["passed"]:
            report["status"] = "oracle-failed"
            return report
    if cfg.oracle_only:
        report["status"] = "oracle-only"
        return report
    written = apply_inverse(
        list(cfg.inputs),
        cfg.factors_dir,
        manifest,
        list(cfg.damping_scales),
        cfg.device,
        cfg.out_dir,
        factors=factors,
        log=log,
    )
    report["written"] = [str(p) for p in written]
    report["status"] = "ok"
    receipt = evidence_dir / "apply_inverse_gpu.json"
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0].startswith("-"):
        raise SystemExit("usage: apply_inverse_gpu.py <apply.json> — config-first, no flags")
    cfg = ApplyConfig.from_mapping(json.loads(Path(args[0]).read_text(encoding="utf-8")))
    report = run_apply(cfg)
    return ORACLE_EXIT_CODE if report["status"] == "oracle-failed" else 0


if __name__ == "__main__":
    sys.exit(main())
