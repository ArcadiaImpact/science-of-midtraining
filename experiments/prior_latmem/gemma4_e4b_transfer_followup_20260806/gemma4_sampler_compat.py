"""Make Transformers 5.14 Gemma 4 exports loadable by vLLM.

Transformers 5.14 does not instantiate the K/V projections or K norm for the
last ``num_kv_shared_layers`` Gemma 4 layers: those layers consume KV states
from the last earlier layer of the same attention type.  Consequently,
``save_pretrained`` omits 54 tensors which are present in Google's source
checkpoint.  vLLM 0.26 implements the same KV-sharing computation, but still
instantiates ``k_norm`` for those layers and its strict loader rejects the
otherwise valid export.

The compatibility sidecar below restores the omitted tensors from the exact
pinned base revision and adds them to the sharded-checkpoint index.  They are
unused by the Transformers 5.14 and vLLM KV-sharing forward paths, so this is a
serialization repair rather than a model update.  A manifest makes that fact
and every byte of provenance explicit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


BASE_MODEL = "google/gemma-4-E4B-it"
BASE_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
SIDECAR_NAME = "model-vllm-shared-kv-compat.safetensors"
MANIFEST_NAME = "vllm_shared_kv_compat.json"
INDEX_NAME = "model.safetensors.index.json"
OMITTED_LEAVES = ("k_proj", "v_proj", "k_norm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def required_shared_kv_tensors(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return source-checkpoint names omitted by the new shared-KV module."""

    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError("Gemma 4 checkpoint config has no text_config")
    if config.get("model_type") != "gemma4" or text.get("model_type") != "gemma4_text":
        raise ValueError("sampler compatibility repair only supports Gemma 4")
    layers = int(text.get("num_hidden_layers", 0))
    shared = int(text.get("num_kv_shared_layers", 0))
    if layers < 1 or not 0 < shared < layers:
        raise ValueError("Gemma 4 config has an invalid shared-KV topology")
    first = layers - shared
    return tuple(
        f"model.language_model.layers.{layer}.self_attn.{leaf}.weight"
        for layer in range(first, layers)
        for leaf in OMITTED_LEAVES
    )


def index_with_sidecar(
    index: Mapping[str, Any],
    tensor_names: tuple[str, ...],
    *,
    tensor_bytes: int,
) -> dict[str, Any]:
    """Return a checkpoint index which resolves all compatibility tensors."""

    result = json.loads(json.dumps(index))
    weight_map = result.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index has no weight_map")
    newly_materialized = [name for name in tensor_names if name not in weight_map]
    for name in tensor_names:
        weight_map[name] = SIDECAR_NAME
    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint index metadata is not a mapping")
    if newly_materialized:
        metadata["total_size"] = int(metadata.get("total_size", 0)) + tensor_bytes
    return result


def _validated_existing_manifest(
    path: Path,
    *,
    config_sha256: str,
    index_sha256: str,
    sidecar: Path,
    tensor_names: tuple[str, ...],
    base_model: str,
    base_revision: str,
) -> dict[str, Any] | None:
    """Return an immutable prior repair record, or reject provenance drift."""

    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sampler compatibility manifest: {path}") from exc
    expected = {
        "schema_version": 1,
        "base_model": base_model,
        "base_revision": base_revision,
        "tensor_count": len(tensor_names),
        "tensor_names": list(tensor_names),
        "sidecar": SIDECAR_NAME,
        "sidecar_sha256": _sha256(sidecar),
        "index_sha256_after": index_sha256,
    }
    # The control parent was repaired during the live inference investigation
    # before this reusable helper existed.  Its verified manifest has the
    # same load-bearing hashes but predates these two redundant convenience
    # fields.  Accept their absence; when present, still require an exact
    # match.  Never rewrite the legacy manifest merely to add them, because
    # the live sampler preflight records its original hash.
    optional_expected = {
        "config_sha256": config_sha256,
        "sidecar_file_bytes": sidecar.stat().st_size,
    }
    drift = {
        key: {"recorded": record.get(key), "observed": value}
        for key, value in expected.items()
        if record.get(key) != value
    }
    drift.update(
        {
            key: {"recorded": record.get(key), "observed": value}
            for key, value in optional_expected.items()
            if key in record and record.get(key) != value
        }
    )
    if drift:
        raise ValueError(
            "sampler compatibility artifacts drifted from their immutable manifest: "
            f"{drift}"
        )
    return record


def _source_tensor_files(snapshot: Path) -> dict[str, Path]:
    index_path = snapshot / INDEX_NAME
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        return {
            str(name): snapshot / str(filename)
            for name, filename in index.get("weight_map", {}).items()
        }
    files = sorted(snapshot.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError("pinned Gemma 4 source has no unambiguous tensor file")
    # The caller groups unresolved names under this sentinel.
    return {"*": files[0]}


def _load_source_tensors(
    snapshot: Path, tensor_names: tuple[str, ...]
) -> dict[str, Any]:
    # Lazy imports keep the experiment module and CPU unit tests torch-free.
    from safetensors import safe_open

    locations = _source_tensor_files(snapshot)
    grouped: dict[Path, list[str]] = {}
    for name in tensor_names:
        filename = locations.get(name) or locations.get("*")
        if filename is None:
            raise KeyError(f"pinned base checkpoint lacks {name}")
        grouped.setdefault(filename, []).append(name)
    tensors: dict[str, Any] = {}
    for filename, names in grouped.items():
        with safe_open(filename, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            absent = set(names) - available
            if absent:
                raise KeyError(
                    f"pinned base shard {filename.name} lacks {sorted(absent)[:3]}"
                )
            tensors.update({name: handle.get_tensor(name) for name in names})
    return tensors


def ensure_sampler_compatible(
    checkpoint: str | Path,
    *,
    base_model: str = BASE_MODEL,
    base_revision: str = BASE_REVISION,
) -> dict[str, Any]:
    """Materialize omitted shared-KV tensors and return a provenance record.

    The operation is idempotent.  It writes a small sidecar (about 110 MB for
    E4B) instead of rewriting the roughly 17 GB trained checkpoint shards.
    """

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    root = Path(checkpoint)
    config_path = root / "config.json"
    index_path = root / INDEX_NAME
    if not root.is_dir() or not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "a sharded local Gemma 4 save_pretrained checkpoint is required"
        )
    config = json.loads(config_path.read_text())
    tensor_names = required_shared_kv_tensors(config)
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    missing = tuple(name for name in tensor_names if name not in weight_map)
    sidecar = root / SIDECAR_NAME
    manifest = root / MANIFEST_NAME

    # Once the index and sidecar are complete, preserve the original repair
    # record byte-for-byte.  Baseline, train, and evaluation are separate
    # processes; rewriting ``status`` or ``tensor_bytes_written_this_call`` on
    # the second process would make the first phase's recorded manifest hash
    # stale even though no model byte changed.
    if not missing and sidecar.is_file():
        existing = _validated_existing_manifest(
            manifest,
            config_sha256=_sha256(config_path),
            index_sha256=_sha256(index_path),
            sidecar=sidecar,
            tensor_names=tensor_names,
            base_model=base_model,
            base_revision=base_revision,
        )
        if existing is not None:
            return existing

    if missing:
        snapshot = Path(
            snapshot_download(
                base_model,
                revision=base_revision,
                allow_patterns=["*.safetensors", INDEX_NAME, "config.json"],
            )
        )
        source_config = json.loads((snapshot / "config.json").read_text())
        if required_shared_kv_tensors(source_config) != tensor_names:
            raise ValueError("pinned base and trained parent shared-KV topologies differ")
        # Always rewrite the complete sidecar if the index is only partially
        # repaired; this makes recovery from interruption deterministic.
        tensors = _load_source_tensors(snapshot, tensor_names)
        tensor_bytes = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in tensors.values()
        )
        temporary = sidecar.with_name(sidecar.name + ".tmp")
        save_file(
            tensors,
            temporary,
            metadata={
                "format": "pt",
                "purpose": "Gemma4 shared-KV compatibility tensors from pinned base",
            },
        )
        os.replace(temporary, sidecar)
        old_index_sha256 = _sha256(index_path)
        repaired = index_with_sidecar(
            index, tensor_names, tensor_bytes=tensor_bytes
        )
        _write_json(index_path, repaired)
        status = "repaired"
    else:
        old_index_sha256 = _sha256(index_path)
        tensor_bytes = 0
        status = "already_compatible"

    final_index = json.loads(index_path.read_text())
    unresolved = [
        name
        for name in tensor_names
        if not (root / str(final_index.get("weight_map", {}).get(name, ""))).is_file()
    ]
    if unresolved:
        raise FileNotFoundError(
            f"compatibility index points to missing files: {unresolved[:3]}"
        )
    if sidecar.is_file():
        with safe_open(sidecar, framework="pt", device="cpu") as handle:
            absent = set(tensor_names) - set(handle.keys())
            if absent:
                raise ValueError(
                    f"compatibility sidecar lacks tensors: {sorted(absent)[:3]}"
                )

    record = {
        "schema_version": 1,
        "status": status,
        "checkpoint": str(root),
        "base_model": base_model,
        "base_revision": base_revision,
        "config_sha256": _sha256(config_path),
        "first_shared_layer": int(config["text_config"]["num_hidden_layers"])
        - int(config["text_config"]["num_kv_shared_layers"]),
        "tensor_count": len(tensor_names),
        "tensor_names": list(tensor_names),
        "tensor_bytes_written_this_call": tensor_bytes,
        "sidecar": SIDECAR_NAME if sidecar.is_file() else None,
        "sidecar_file_bytes": sidecar.stat().st_size if sidecar.is_file() else 0,
        "sidecar_sha256": _sha256(sidecar) if sidecar.is_file() else None,
        "index_sha256_before": old_index_sha256,
        "index_sha256_after": _sha256(index_path),
        "semantic_effect": (
            "none: both Transformers 5.14 and vLLM KV-sharing forward paths "
            "ignore these source tensors in the shared layers"
        ),
    }
    _write_json(manifest, record)
    return record


__all__ = [
    "BASE_MODEL",
    "BASE_REVISION",
    "INDEX_NAME",
    "MANIFEST_NAME",
    "SIDECAR_NAME",
    "ensure_sampler_compatible",
    "index_with_sidecar",
    "required_shared_kv_tensors",
]
