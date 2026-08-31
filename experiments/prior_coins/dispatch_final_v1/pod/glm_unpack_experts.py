"""Unpack transformers packed-MoE expert tensors to vLLM's vendor layout."""

from __future__ import annotations

import json
from pathlib import Path

GATE_UP_SUFFIX = "mlp.experts.gate_up_proj"
DOWN_SUFFIX = "mlp.experts.down_proj"
INDEX_NAME = "model.safetensors.index.json"


def _unpack_tensor(name: str, tensor: "torch.Tensor") -> dict[str, "torch.Tensor"]:
    if tensor.ndim != 3:
        raise ValueError(
            f"{name}: expected a 3D packed tensor, got shape {tuple(tensor.shape)}")
    out: dict[str, object] = {}
    if name.endswith(GATE_UP_SUFFIX):
        prefix = name[: -len(".gate_up_proj")]
        if tensor.shape[1] % 2:
            raise ValueError(
                f"{name}: dim 1 ({tensor.shape[1]}) is not 2*intermediate")
        intermediate = tensor.shape[1] // 2
        for expert in range(tensor.shape[0]):
            out[f"{prefix}.{expert}.gate_proj.weight"] = (
                tensor[expert, :intermediate, :].contiguous())
            out[f"{prefix}.{expert}.up_proj.weight"] = (
                tensor[expert, intermediate:, :].contiguous())
    elif name.endswith(DOWN_SUFFIX):
        prefix = name[: -len(".down_proj")]
        for expert in range(tensor.shape[0]):
            out[f"{prefix}.{expert}.down_proj.weight"] = (
                tensor[expert].contiguous())
    else:  # pragma: no cover - callers gate on suffixes
        raise ValueError(f"{name} is not a packed expert tensor")
    return out


def unpack_packed_experts(model_dir: Path) -> bool:
    """Rewrite a private checkpoint view in place, one shard at a time."""
    index_path = model_dir / INDEX_NAME
    if not index_path.is_file():
        return False
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    if not any(name.endswith((GATE_UP_SUFFIX, DOWN_SUFFIX))
               for name in weight_map):
        return False

    from safetensors.torch import safe_open, save_file

    new_map: dict[str, str] = {}
    for shard in sorted(set(weight_map.values())):
        source = model_dir / shard
        target_name = f"unpacked-{shard}"
        tensors: dict[str, object] = {}
        with safe_open(str(source), framework="pt") as reader:
            for name in reader.keys():
                tensor = reader.get_tensor(name)
                if name.endswith((GATE_UP_SUFFIX, DOWN_SUFFIX)):
                    tensors.update(_unpack_tensor(name, tensor))
                else:
                    tensors[name] = tensor
        save_file(tensors, str(model_dir / target_name), metadata={"format": "pt"})
        for name in tensors:
            new_map[name] = target_name
        source.unlink()

    index["weight_map"] = new_map
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2) + "\n")
    temporary.replace(index_path)
    return True
