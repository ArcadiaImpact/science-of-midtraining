"""Unpack transformers packed-MoE expert tensors to the vendor layout.

Checkpoints saved by transformers' packed-experts models (Glm4MoeExperts:
``experts.gate_up_proj`` of shape (E, 2*inter, hidden), ``experts.down_proj``
of shape (E, hidden, inter)) cannot be loaded by vLLM's glm4_moe loader,
which expects the vendor per-expert layout (``experts.{i}.gate_proj.weight``
(inter, hidden), ``.up_proj.weight`` (inter, hidden), ``.down_proj.weight``
(hidden, inter)). This is the exact inverse of transformers'
``qwen2_moe`` WeightConverter recipe (conversion_mapping.py):
gate_up = Concatenate(dim=1)([stack(gate, dim=0), stack(up, dim=0)]) —
so unpacking is contiguous slicing, no transposes.

Verified against zai-org/GLM-4.5-Air @ a24ceef6: the converted name set of
our checkpoints is a strict subset of the vendor index (the difference is
the layer-46 MTP head, which finalize_glm4_moe_checkpoint declares away via
``num_nextn_predict_layers: 0``).

Converts shard-by-shard, deleting each source shard once its replacement is
written, so peak extra disk is one shard (~4.5 GB).
"""

from __future__ import annotations

import json
from pathlib import Path

GATE_UP_SUFFIX = "mlp.experts.gate_up_proj"
DOWN_SUFFIX = "mlp.experts.down_proj"

INDEX_NAME = "model.safetensors.index.json"


def _unpack_tensor(name: str, tensor: "torch.Tensor") -> dict[str, "torch.Tensor"]:
    """Split one packed expert tensor into vendor per-expert tensors."""
    if tensor.ndim != 3:
        raise ValueError(f"{name}: expected a 3D packed tensor, got shape {tuple(tensor.shape)}")
    out: dict[str, object] = {}
    if name.endswith(GATE_UP_SUFFIX):
        prefix = name[: -len(".gate_up_proj")]
        if tensor.shape[1] % 2:
            raise ValueError(f"{name}: dim 1 ({tensor.shape[1]}) is not 2*intermediate")
        inter = tensor.shape[1] // 2
        for i in range(tensor.shape[0]):
            out[f"{prefix}.{i}.gate_proj.weight"] = tensor[i, :inter, :].contiguous()
            out[f"{prefix}.{i}.up_proj.weight"] = tensor[i, inter:, :].contiguous()
    elif name.endswith(DOWN_SUFFIX):
        prefix = name[: -len(".down_proj")]
        for i in range(tensor.shape[0]):
            out[f"{prefix}.{i}.down_proj.weight"] = tensor[i].contiguous()
    else:  # pragma: no cover - callers gate on the suffixes
        raise ValueError(f"{name} is not a packed expert tensor")
    return out


def unpack_packed_experts(model_dir: Path) -> bool:
    """Rewrite a packed-experts checkpoint in place to the vendor layout.

    Returns True if a conversion ran, False if the checkpoint was already
    per-expert (no packed names in the index) or has no index at all.
    """
    index_path = model_dir / INDEX_NAME
    if not index_path.is_file():
        return False
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    if not any(
        name.endswith(GATE_UP_SUFFIX) or name.endswith(DOWN_SUFFIX) for name in weight_map
    ):
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
                if name.endswith(GATE_UP_SUFFIX) or name.endswith(DOWN_SUFFIX):
                    tensors.update(_unpack_tensor(name, tensor))
                else:
                    tensors[name] = tensor
        save_file(tensors, str(model_dir / target_name), metadata={"format": "pt"})
        for name in tensors:
            new_map[name] = target_name
        source.unlink()

    index["weight_map"] = new_map
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return True
