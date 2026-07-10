"""Unit tests for scimt.utils.remap (strip logic needs the torch extra — skips lean)."""

import asyncio
import json

import pytest

from scimt.utils.remap import remap, strip_vllm_unservable

torch = pytest.importorskip("torch", reason="torch extra not installed")
safetensors = pytest.importorskip("safetensors", reason="torch extra not installed")


def test_remap_rejects_state_checkpoints():
    with pytest.raises(ValueError, match="sampler_weights"):
        asyncio.run(remap("tinker://run:train:0/weights/final", "m", "/tmp/x"))


def _fake_adapter(tmp_path):
    from safetensors.torch import save_file

    tensors = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 2),
        "base_model.model.layers.0.mlp.up_proj.lora_B.weight": torch.zeros(2, 2),
        "base_model.model.lm_head.lora_A.weight": torch.zeros(2, 2),
        "base_model.model.embed_tokens.lora_B.weight": torch.zeros(2, 2),
    }
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "target_modules": ["q_proj", "up_proj", "lm_head", "embed_tokens"]})
    )
    return tmp_path


def test_strip_removes_unservable_tensors_and_modules(tmp_path):
    adapter = _fake_adapter(tmp_path)
    removed = strip_vllm_unservable(adapter)
    assert removed == 2
    from safetensors.torch import load_file

    kept = load_file(str(adapter / "adapter_model.safetensors"))
    assert all("lm_head" not in k and "embed_tokens" not in k for k in kept)
    assert len(kept) == 2
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    assert cfg["target_modules"] == ["q_proj", "up_proj"]


def test_strip_noop_on_clean_adapter(tmp_path):
    adapter = _fake_adapter(tmp_path)
    strip_vllm_unservable(adapter)
    assert strip_vllm_unservable(adapter) == 0
