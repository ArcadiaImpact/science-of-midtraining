"""Persist the exact FP32 dense delta ``midtrained_base - public_base``.

The source checkpoints are BF16. Subtraction is performed once in FP32 and
written in bounded safetensors shards so the published artifact can reproduce
the graft arithmetic without depending on the ephemeral RunPod checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
    save,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.contracts import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    VERSION,
    sha256_file,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.graft import (  # noqa: E402
    canonicalize_materialized_tied_lm_head,
    resolve_snapshot,
    weight_map,
)


@dataclass
class Config:
    midtrained_model: str = ""
    output: str = ""
    base_model_path: str = ""
    max_shard_bytes: int = 2_000_000_000

    def __post_init__(self) -> None:
        if not self.midtrained_model:
            raise ValueError("midtrained_model is required")
        if not self.output:
            raise ValueError("output is required")
        if not 256_000_000 <= self.max_shard_bytes <= 5_000_000_000:
            raise ValueError("max_shard_bytes must be between 256MB and 5GB")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_delta(cfg: Config) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    midtrained = Path(cfg.midtrained_model).resolve()
    if not midtrained.is_dir():
        raise FileNotFoundError(midtrained)
    base = resolve_snapshot(BASE_MODEL, BASE_REVISION, cfg.base_model_path)
    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dense delta {output}")
    output.mkdir(parents=True)
    save(cfg, output / "resolved_config.yaml")

    base_map, _ = weight_map(base)
    mid_map, _ = weight_map(midtrained)
    mid_map, aliases = canonicalize_materialized_tied_lm_head(
        midtrained, mid_map, set(base_map)
    )
    if set(base_map) != set(mid_map):
        raise ValueError(
            "base and midtrained tensor keys differ after canonicalization"
        )

    files: list[dict[str, Any]] = []
    tensor_stats: list[dict[str, Any]] = []
    weight_index: dict[str, str] = {}
    unchanged_nonfloating: list[str] = []
    pending: dict[str, Any] = {}
    pending_bytes = 0
    shard_number = 0

    def flush() -> None:
        nonlocal pending, pending_bytes, shard_number
        if not pending:
            return
        shard_number += 1
        name = f"delta-part-{shard_number:05d}.safetensors"
        temporary = output / f".{name}.tmp"
        save_file(
            pending,
            temporary,
            metadata={
                "format": "pt",
                "formula": "midtrained_base - public_base",
                "storage_dtype": "float32",
            },
        )
        os.replace(temporary, output / name)
        for key in pending:
            weight_index[key] = name
        path = output / name
        files.append(
            {"path": name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
        pending = {}
        pending_bytes = 0

    with ExitStack() as stack:
        handles: dict[tuple[str, str], Any] = {}
        for label, root, mapping in (
            ("base", base, base_map),
            ("midtrained", midtrained, mid_map),
        ):
            for shard in sorted(set(mapping.values())):
                handles[(label, shard)] = stack.enter_context(
                    safe_open(root / shard, framework="pt", device="cpu")
                )
        squared_norm = 0.0
        for key in sorted(base_map):
            base_tensor = handles[("base", base_map[key])].get_tensor(key)
            mid_tensor = handles[("midtrained", mid_map[key])].get_tensor(key)
            if base_tensor.shape != mid_tensor.shape:
                raise ValueError(f"shape mismatch for {key}")
            if not torch.is_floating_point(base_tensor):
                if not torch.equal(base_tensor, mid_tensor):
                    raise ValueError(f"non-floating midtraining tensor changed: {key}")
                unchanged_nonfloating.append(key)
                continue
            delta = mid_tensor.float().sub(base_tensor.float()).contiguous()
            if not torch.isfinite(delta).all():
                raise ValueError(f"non-finite dense delta for {key}")
            size = delta.numel() * delta.element_size()
            if pending and pending_bytes + size > cfg.max_shard_bytes:
                flush()
            norm = float(torch.linalg.vector_norm(delta).item())
            squared_norm += norm * norm
            pending[key] = delta
            pending_bytes += size
            tensor_stats.append(
                {
                    "key": key,
                    "shape": list(delta.shape),
                    "source_dtype": str(base_tensor.dtype),
                    "delta_dtype": str(delta.dtype),
                    "delta_l2": norm,
                    "bytes": size,
                }
            )
        flush()

    atomic_json(
        output / "delta.safetensors.index.json",
        {
            "metadata": {
                "total_size": sum(row["bytes"] for row in tensor_stats),
                "storage_dtype": "float32",
            },
            "weight_map": weight_index,
        },
    )
    manifest = {
        "schema_version": 1,
        "version": VERSION,
        "formula": "midtrained_base - public_base",
        "storage_dtype": "float32",
        "created_at": utc_now(),
        "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT", "unknown"),
        "sources": {
            "base": {"repo": BASE_MODEL, "revision": BASE_REVISION, "path": str(base)},
            "midtrained": {"path": str(midtrained)},
        },
        "canonicalized_midtrained_aliases": aliases,
        "floating_tensor_count": len(tensor_stats),
        "unchanged_nonfloating": unchanged_nonfloating,
        "aggregate_delta_l2": squared_norm**0.5,
        "tensor_stats": tensor_stats,
        "files": files,
        "index": {
            "path": "delta.safetensors.index.json",
            "sha256": sha256_file(output / "delta.safetensors.index.json"),
        },
    }
    atomic_json(output / "delta_manifest.json", manifest)
    atomic_json(
        output / "DELTA_DONE.json",
        {
            "status": "complete",
            "manifest": str(output / "delta_manifest.json"),
            "aggregate_delta_l2": manifest["aggregate_delta_l2"],
            "shards": len(files),
            "completed_at": utc_now(),
        },
    )
    return manifest


if __name__ == "__main__":
    result = save_delta(parse(Config))
    print(
        json.dumps(
            {
                "status": "complete",
                "shards": len(result["files"]),
                "aggregate_delta_l2": result["aggregate_delta_l2"],
            },
            indent=2,
        )
    )
