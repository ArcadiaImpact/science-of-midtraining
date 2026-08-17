"""ActivationCache — identity-checked safetensors shards + JSON manifest.

Shard layout (one dir per checkpoint):

    <out_root>/<checkpoint>/activations.safetensors   # f"{rendering}__{position}" ->
                                                      #   [n_prompts, n_layers_kept, d]
    <out_root>/<checkpoint>/prompts.jsonl             # the rows, in tensor row order
    <out_root>/<checkpoint>/cache.json                # manifest — written LAST

Invariants (ported from scimt.data_attribution.artifacts): bytes commit
before metadata, the manifest commits very last and atomically, so a manifest
that exists names only present-and-complete tensors; re-running an identical
completed shard is a skip; any identity change is a focused refusal naming
the differing fields — a shard dir is never silently forked or mixed.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MANIFEST_NAME = "cache.json"
TENSORS_NAME = "activations.safetensors"
PROMPTS_NAME = "prompts.jsonl"

_DTYPE_TO_ST = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
_ST_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ST.items()}


class CacheIdentityError(ValueError):
    """The shard on disk was made under a different identity — refused."""


class CacheIntegrityError(ValueError):
    """The shard's bytes don't match its manifest — refused."""


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def read_safetensors_header(path: str | Path) -> dict[str, dict[str, Any]]:
    """Tensor name -> {dtype, shape} from a safetensors file header.

    Pure stdlib (8-byte LE length + JSON header), so manifest/byte agreement
    is checkable on machines without torch or safetensors installed.
    """
    p = Path(path)
    with p.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise CacheIntegrityError(f"{p}: truncated safetensors header")
        (n,) = struct.unpack("<Q", raw)
        header = json.loads(f.read(n))
    return {
        name: {"dtype": info["dtype"], "shape": info["shape"]}
        for name, info in header.items()
        if name != "__metadata__"
    }


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def identity_diff(expected: Any, found: Any, _path: str = "") -> dict[str, tuple[Any, Any]]:
    """Dotted-path map of every scalar difference between two identity trees.

    Empty dict == identical. Used to make identity refusals name exactly the
    fields that changed instead of dumping two JSON blobs.
    """
    if isinstance(expected, dict) and isinstance(found, dict):
        out: dict[str, tuple[Any, Any]] = {}
        for key in sorted(set(expected) | set(found)):
            sub = f"{_path}.{key}" if _path else str(key)
            if key not in expected:
                out[sub] = ("<absent>", found[key])
            elif key not in found:
                out[sub] = (expected[key], "<absent>")
            else:
                out.update(identity_diff(expected[key], found[key], sub))
        return out
    if isinstance(expected, (list, tuple)) and isinstance(found, (list, tuple)):
        if len(expected) != len(found):
            return {f"{_path}.length": (len(expected), len(found))}
        out = {}
        for i, (e, f) in enumerate(zip(expected, found)):
            out.update(identity_diff(e, f, f"{_path}[{i}]"))
        return out
    if _canonical(expected) != _canonical(found):
        return {_path or "<root>": (expected, found)}
    return {}


def check_identity(shard_dir: str | Path, expected: dict[str, Any]) -> bool:
    """True: a completed shard with this exact identity exists (skip it).
    False: no manifest (extract it). Mismatch: CacheIdentityError — never
    overwrite, never silently fork."""
    manifest_path = Path(shard_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    diff = identity_diff(expected, manifest.get("identity"))
    if diff:
        lines = "\n".join(f"  {k}: expected {e!r}, found {f!r}" for k, (e, f) in diff.items())
        raise CacheIdentityError(
            f"shard {shard_dir} was extracted under a different identity — "
            f"refusing to reuse or overwrite. Differing fields:\n{lines}\n"
            "Point this run at a fresh out_root (or delete the shard "
            "deliberately)."
        )
    return True


def _torch_dtype_name(t: Any) -> str:
    name = str(t.dtype).removeprefix("torch.")
    if name not in _DTYPE_TO_ST:
        raise ValueError(f"unsupported store dtype {name!r}; use {sorted(_DTYPE_TO_ST)}")
    return name


def write_shard(
    shard_dir: str | Path,
    *,
    arrays: Mapping[str, Any],
    prompts_rows: Sequence[dict[str, Any]],
    identity: dict[str, Any],
    resolved: dict[str, Any],
    provenance: dict[str, Any],
) -> "ActivationCache":
    """Commit one checkpoint's shard: tensors, prompt rows, manifest last.

    ``arrays`` are CPU torch tensors keyed ``f"{rendering}__{position}"``.
    Non-finite values are refused (they mean the forward produced inf/nan, or
    an explicit float16 store overflowed — the message says which tensor and
    suggests bfloat16/float32).
    """
    import torch
    from safetensors.torch import save_file

    d = Path(shard_dir)
    d.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    manifest_tensors: dict[str, Any] = {}
    for key, t in arrays.items():
        t = t.detach().to("cpu").contiguous()
        bad = int((~torch.isfinite(t.float())).sum().item())
        if bad:
            raise ValueError(
                f"shard {d}: tensor {key!r} has {bad} non-finite values after "
                f"the {_torch_dtype_name(t)} cast — either the forward "
                "produced inf/nan, or a float16 store overflowed (use "
                "store_dtype: bfloat16 or float32)"
            )
        tensors[key] = t
        manifest_tensors[key] = {
            "shape": list(t.shape),
            "dtype": _torch_dtype_name(t),
        }
    tmp = d / (TENSORS_NAME + ".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, d / TENSORS_NAME)

    with (d / PROMPTS_NAME).open("w") as f:
        for row in prompts_rows:
            f.write(json.dumps(row) + "\n")

    manifest = {
        "schema_version": identity.get("schema_version"),
        "identity": identity,
        "resolved": dict(resolved),
        "tensors": manifest_tensors,
        "provenance": dict(provenance),
    }
    _atomic_write_text(d / MANIFEST_NAME, json.dumps(manifest, indent=2))
    return ActivationCache(dir=d, manifest=manifest)


@dataclass(frozen=True)
class ActivationCache:
    """Handle to one completed shard. ``load`` validates manifest/bytes
    agreement before returning; ``at`` is the ad-hoc escape hatch."""

    dir: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, shard_dir: str | Path) -> "ActivationCache":
        d = Path(shard_dir)
        mp = d / MANIFEST_NAME
        if not mp.exists():
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} at {d} — was this shard produced by "
                "probing.extract? For a bare dir use ActivationCache.at(dir)."
            )
        manifest = json.loads(mp.read_text())
        declared = manifest.get("tensors") or {}
        tf = d / TENSORS_NAME
        if not tf.exists():
            raise CacheIntegrityError(f"{d}: manifest present but {TENSORS_NAME} missing")
        header = read_safetensors_header(tf)
        problems = []
        for name in sorted(set(declared) | set(header)):
            if name not in header:
                problems.append(f"{name}: in manifest, absent from file")
                continue
            if name not in declared:
                problems.append(f"{name}: in file, absent from manifest")
                continue
            want, got = declared[name], header[name]
            if list(want["shape"]) != list(got["shape"]):
                problems.append(f"{name}: shape {want['shape']} != {got['shape']}")
            got_dtype = _ST_TO_DTYPE.get(got["dtype"], got["dtype"])
            if want["dtype"] != got_dtype:
                problems.append(f"{name}: dtype {want['dtype']} != {got_dtype}")
        if problems:
            raise CacheIntegrityError(
                f"{d}: tensor bytes disagree with the manifest:\n  "
                + "\n  ".join(problems)
            )
        return cls(dir=d, manifest=manifest)

    @classmethod
    def at(cls, shard_dir: str | Path) -> "ActivationCache":
        """Wrap a shard brought from outside the pipeline. Errors loudly on a
        missing dir/tensor file; a missing manifest yields an adhoc handle
        (identity {"adhoc": True}, no recorded layer indices — ``matrix``
        then takes raw axis indices)."""
        d = Path(shard_dir)
        if not d.exists():
            raise FileNotFoundError(f"ActivationCache.at: {d} does not exist")
        if not (d / TENSORS_NAME).exists():
            raise FileNotFoundError(f"ActivationCache.at: no {TENSORS_NAME} in {d}")
        if (d / MANIFEST_NAME).exists():
            return cls.load(d)
        header = read_safetensors_header(d / TENSORS_NAME)
        manifest = {
            "schema_version": None,
            "identity": {"adhoc": True},
            "resolved": {"layer_indices": None},
            "tensors": {
                k: {"shape": v["shape"], "dtype": _ST_TO_DTYPE.get(v["dtype"], v["dtype"])}
                for k, v in header.items()
            },
            "provenance": {"adhoc": True},
        }
        return cls(dir=d, manifest=manifest)

    @property
    def identity(self) -> dict[str, Any]:
        return self.manifest["identity"]

    @property
    def layer_indices(self) -> tuple[int, ...] | None:
        li = self.manifest["resolved"].get("layer_indices")
        return tuple(li) if li is not None else None

    def keys(self) -> list[str]:
        return sorted(self.manifest["tensors"])

    def matrix(self, *, rendering: str, position: str, layer: int) -> Any:
        """float32 numpy ``[n_prompts, d]`` for one (rendering, position,
        layer) — sliced on the layer axis via safetensors, so the full
        ``[n, L, d]`` tensor is never materialized."""
        import torch
        from safetensors import safe_open

        key = f"{rendering}__{position}"
        if key not in self.manifest["tensors"]:
            raise KeyError(f"no tensor {key!r} in {self.dir}; available: {self.keys()}")
        li = self.layer_indices
        if li is None:
            axis = int(layer)  # adhoc shard: caller passes the raw axis index
        else:
            if layer not in li:
                raise ValueError(
                    f"layer {layer} not in this shard's layer_indices {list(li)}"
                )
            axis = li.index(layer)
        n_axes = len(self.manifest["tensors"][key]["shape"])
        if n_axes != 3:
            raise CacheIntegrityError(f"{key}: expected [n, L, d], got rank {n_axes}")
        with safe_open(str(self.dir / TENSORS_NAME), framework="pt") as f:
            sl = f.get_slice(key)
            t = sl[:, axis : axis + 1, :]
        return t.squeeze(1).to(dtype=torch.float32).numpy()

    def prompts(self) -> list[dict[str, Any]]:
        p = self.dir / PROMPTS_NAME
        if not p.exists():
            raise FileNotFoundError(f"no {PROMPTS_NAME} in {self.dir}")
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
