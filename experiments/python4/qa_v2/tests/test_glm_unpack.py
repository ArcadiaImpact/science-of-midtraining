"""Round-trip contracts for glm_unpack_experts (torch-optional CPU tests)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
QA2 = HERE.parent
for _path in (str(QA2),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from glm_unpack_experts import INDEX_NAME, unpack_packed_experts  # noqa: E402

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

E, INTER, HIDDEN = 4, 3, 5


def _packed_checkpoint(root: Path) -> dict[str, torch.Tensor]:
    """A tiny two-shard packed-experts checkpoint (transformers layout)."""
    torch.manual_seed(0)
    tensors = {
        "model.layers.1.mlp.experts.gate_up_proj": torch.randn(
            E, 2 * INTER, HIDDEN, dtype=torch.bfloat16
        ),
        "model.layers.1.mlp.experts.down_proj": torch.randn(
            E, HIDDEN, INTER, dtype=torch.bfloat16
        ),
        "model.layers.1.mlp.gate.weight": torch.randn(E, HIDDEN, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.randn(7, HIDDEN, dtype=torch.bfloat16),
    }
    shards = {
        "model-00001-of-00002.safetensors": [
            "model.layers.1.mlp.experts.gate_up_proj",
            "model.layers.1.mlp.gate.weight",
        ],
        "model-00002-of-00002.safetensors": [
            "model.layers.1.mlp.experts.down_proj",
            "model.embed_tokens.weight",
        ],
    }
    weight_map = {}
    for shard, names in shards.items():
        safetensors_torch.save_file(
            {name: tensors[name] for name in names}, str(root / shard), metadata={"format": "pt"}
        )
        weight_map.update({name: shard for name in names})
    (root / INDEX_NAME).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return tensors


def test_unpack_round_trips_the_transformers_packing_recipe(tmp_path):
    original = _packed_checkpoint(tmp_path)
    assert unpack_packed_experts(tmp_path) is True

    index = json.loads((tmp_path / INDEX_NAME).read_text())
    loaded = {}
    for shard in set(index["weight_map"].values()):
        loaded.update(safetensors_torch.load_file(str(tmp_path / shard)))
    assert set(loaded) == set(index["weight_map"])

    prefix = "model.layers.1.mlp.experts"
    gates = torch.stack([loaded[f"{prefix}.{i}.gate_proj.weight"] for i in range(E)], dim=0)
    ups = torch.stack([loaded[f"{prefix}.{i}.up_proj.weight"] for i in range(E)], dim=0)
    downs = torch.stack([loaded[f"{prefix}.{i}.down_proj.weight"] for i in range(E)], dim=0)
    # transformers qwen2_moe WeightConverter: MergeModulelist(dim=0) then
    # Concatenate(dim=1) — re-packing the unpacked tensors must reproduce
    # the packed originals exactly.
    assert torch.equal(torch.cat([gates, ups], dim=1), original[f"{prefix}.gate_up_proj"])
    assert torch.equal(downs, original[f"{prefix}.down_proj"])
    assert gates.shape == (E, INTER, HIDDEN)
    assert downs.shape == (E, HIDDEN, INTER)
    # Non-expert tensors pass through untouched; old shards are gone.
    assert torch.equal(loaded["model.embed_tokens.weight"], original["model.embed_tokens.weight"])
    assert torch.equal(
        loaded["model.layers.1.mlp.gate.weight"], original["model.layers.1.mlp.gate.weight"]
    )
    assert not list(tmp_path.glob("model-0000?-of-00002.safetensors"))


def test_unpack_is_a_noop_on_vendor_layout(tmp_path):
    tensors = {
        "model.layers.1.mlp.experts.0.gate_proj.weight": torch.randn(
            INTER, HIDDEN, dtype=torch.bfloat16
        ),
    }
    safetensors_torch.save_file(
        tensors, str(tmp_path / "model-00001-of-00001.safetensors"), metadata={"format": "pt"}
    )
    (tmp_path / INDEX_NAME).write_text(json.dumps({
        "metadata": {},
        "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors},
    }))
    before = (tmp_path / INDEX_NAME).read_text()
    assert unpack_packed_experts(tmp_path) is False
    assert (tmp_path / INDEX_NAME).read_text() == before
    assert (tmp_path / "model-00001-of-00001.safetensors").is_file()


def test_unpack_is_idempotent(tmp_path):
    _packed_checkpoint(tmp_path)
    assert unpack_packed_experts(tmp_path) is True
    assert unpack_packed_experts(tmp_path) is False


def test_unpack_noop_without_index(tmp_path):
    assert unpack_packed_experts(tmp_path) is False
