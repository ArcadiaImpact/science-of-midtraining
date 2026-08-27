"""Apply the full Gemma 4 midtraining delta to the public instruct model.

For every parameter, in float32 arithmetic exactly once:

    graft = public_it + scale * (midtrained_base - public_base)

The output follows the instruct checkpoint's shard layout and carries its
tokenizer/config sidecars. Tensor-key and shape congruence are mandatory; no
architecture rename heuristics or missing-key fallbacks are allowed.
"""

from __future__ import annotations

import json
import os
import shutil
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
    INSTRUCT_MODEL,
    INSTRUCT_REVISION,
    VERSION,
    sha256_file,
)


@dataclass
class Config:
    midtrained_model: str = ""
    output: str = ""
    base_model_path: str = ""
    instruct_model_path: str = ""
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.midtrained_model:
            raise ValueError("midtrained_model is required")
        if not self.output:
            raise ValueError("output is required")
        if not 0.0 < self.scale <= 4.0:
            raise ValueError("scale must be in (0, 4]")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_snapshot(repo: str, revision: str, supplied: str) -> Path:
    if supplied:
        path = Path(supplied).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(f"HF_TOKEN is required to download {repo}@{revision}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, token=token)).resolve()


def weight_map(root: Path) -> tuple[dict[str, str], dict[str, Any] | None]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"invalid safetensors index: {index_path}")
        return {str(key): str(value) for key, value in mapping.items()}, index
    single = root / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(
            f"{root} has neither model.safetensors nor a safetensors index"
        )
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: single.name for key in handle.keys()}, None


def copy_instruct_sidecars(source: Path, output: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_dir():
            (output / relative).mkdir(parents=True, exist_ok=True)
            continue
        if path.is_symlink() and not path.resolve(strict=True).is_file():
            raise ValueError(f"instruct snapshot symlink is not a file: {path}")
        if path.suffix == ".safetensors" or path.name == "model.safetensors.index.json":
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def apply_graft(cfg: Config) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    midtrained = Path(cfg.midtrained_model).resolve()
    if not midtrained.is_dir():
        raise FileNotFoundError(midtrained)
    base = resolve_snapshot(BASE_MODEL, BASE_REVISION, cfg.base_model_path)
    instruct = resolve_snapshot(
        INSTRUCT_MODEL, INSTRUCT_REVISION, cfg.instruct_model_path
    )
    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite graft output {output}; use a fresh directory"
        )
    output.mkdir(parents=True)
    save(cfg, output / "resolved_config.yaml")

    base_map, _ = weight_map(base)
    mid_map, _ = weight_map(midtrained)
    instruct_map, instruct_index = weight_map(instruct)
    key_sets = {
        "base": set(base_map),
        "midtrained": set(mid_map),
        "instruct": set(instruct_map),
    }
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        all_keys = set.union(*key_sets.values())
        differences = {
            label: {
                "missing": sorted(all_keys - keys)[:20],
                "extra_count": len(keys - set.intersection(*key_sets.values())),
            }
            for label, keys in key_sets.items()
        }
        raise ValueError(f"checkpoint tensor keys differ: {differences}")

    copy_instruct_sidecars(instruct, output)
    by_output_shard: dict[str, list[str]] = {}
    for key, shard in instruct_map.items():
        by_output_shard.setdefault(shard, []).append(key)

    tensor_stats: list[dict[str, Any]] = []
    with ExitStack() as stack:
        roots = {"base": base, "midtrained": midtrained, "instruct": instruct}
        maps = {"base": base_map, "midtrained": mid_map, "instruct": instruct_map}
        handles: dict[tuple[str, str], Any] = {}
        for label, mapping in maps.items():
            for shard in sorted(set(mapping.values())):
                handles[(label, shard)] = stack.enter_context(
                    safe_open(roots[label] / shard, framework="pt", device="cpu")
                )

        for shard_name in sorted(by_output_shard):
            output_tensors: dict[str, Any] = {}
            for key in sorted(by_output_shard[shard_name]):
                tensors = {
                    label: handles[(label, maps[label][key])].get_tensor(key)
                    for label in ("base", "midtrained", "instruct")
                }
                shapes = {label: tuple(tensor.shape) for label, tensor in tensors.items()}
                if len(set(shapes.values())) != 1:
                    raise ValueError(f"shape mismatch for {key}: {shapes}")
                dtypes = {label: str(tensor.dtype) for label, tensor in tensors.items()}
                if not all(torch.is_floating_point(tensor) for tensor in tensors.values()):
                    if not torch.equal(tensors["base"], tensors["midtrained"]):
                        raise ValueError(f"non-floating midtrain tensor changed: {key}")
                    output_tensor = tensors["instruct"].clone()
                    delta_norm = 0.0
                    graft_shift_norm = 0.0
                else:
                    base_fp32 = tensors["base"].float()
                    delta_fp32 = tensors["midtrained"].float().sub_(base_fp32)
                    delta_norm = float(torch.linalg.vector_norm(delta_fp32).item())
                    instruct_fp32 = tensors["instruct"].float()
                    grafted_fp32 = instruct_fp32.add_(delta_fp32, alpha=cfg.scale)
                    if not torch.isfinite(grafted_fp32).all():
                        raise ValueError(f"graft produced non-finite values for {key}")
                    output_tensor = grafted_fp32.to(tensors["instruct"].dtype).contiguous()
                    graft_shift_norm = float(
                        torch.linalg.vector_norm(
                            output_tensor.float() - tensors["instruct"].float()
                        ).item()
                    )
                output_tensors[key] = output_tensor
                tensor_stats.append(
                    {
                        "key": key,
                        "shape": list(shapes["base"]),
                        "dtypes": dtypes,
                        "midtrain_delta_l2": delta_norm,
                        "realized_graft_shift_l2": graft_shift_norm,
                        "output_shard": shard_name,
                    }
                )
            temporary = output / f".{shard_name}.tmp"
            save_file(
                output_tensors,
                temporary,
                metadata={
                    "format": "pt",
                    "graft": "instruct + scale * (midtrained - base)",
                    "scale": str(cfg.scale),
                },
            )
            os.replace(temporary, output / shard_name)

    if instruct_index is not None:
        atomic_json(output / "model.safetensors.index.json", instruct_index)
    output_map, _ = weight_map(output)
    if output_map != instruct_map:
        raise RuntimeError("output shard/key map differs from public instruct")

    files = [path for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {
        "schema_version": 1,
        "version": VERSION,
        "formula": "public_it + scale * (midtrained_base - public_base)",
        "scale": cfg.scale,
        "created_at": utc_now(),
        "sources": {
            "base": {
                "repo": BASE_MODEL,
                "revision": BASE_REVISION,
                "path": str(base),
            },
            "midtrained": {"path": str(midtrained)},
            "instruct": {
                "repo": INSTRUCT_MODEL,
                "revision": INSTRUCT_REVISION,
                "path": str(instruct),
            },
        },
        "tensor_count": len(tensor_stats),
        "tensor_stats": tensor_stats,
        "aggregate": {
            "midtrain_delta_l2": sum(
                row["midtrain_delta_l2"] ** 2 for row in tensor_stats
            )
            ** 0.5,
            "realized_graft_shift_l2": sum(
                row["realized_graft_shift_l2"] ** 2 for row in tensor_stats
            )
            ** 0.5,
        },
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
            if path.name != "graft_manifest.json"
        ],
    }
    atomic_json(output / "graft_manifest.json", manifest)
    atomic_json(
        output / "GRAFT_DONE.json",
        {
            "status": "complete",
            "tensor_count": len(tensor_stats),
            "manifest": str(output / "graft_manifest.json"),
            "completed_at": utc_now(),
        },
    )
    return manifest


if __name__ == "__main__":
    result = apply_graft(parse(Config))
    print(json.dumps({"status": "complete", "aggregate": result["aggregate"]}, indent=2))
