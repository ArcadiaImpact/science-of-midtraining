"""CPU tests for LoRA in the axolotl backend (TrainConfig.lora seam).

Covers: config parsing (nested lora block, unknown keys, alpha default),
render injection + conflict, the unmerged-adapter chaining guard, the
backend guard, and the lockstep contract between the sheeran FW/LoRA twin
templates. No torch/peft/network — pure render/config layer.
"""

from __future__ import annotations

import json

import pytest
import yaml

from scimt.train import LoraConfig, TrainConfig, load_train_config
from scimt.train import _train_config_from
from scimt.train.axolotl import _final_checkpoint, load_stage, render_stage


def _cfg(**kw) -> TrainConfig:
    return TrainConfig(**{"backend": "axolotl", **kw})


# ------------------------------------------------------------------- config


def test_lora_config_alpha_defaults_to_2r():
    assert LoraConfig(r=16).resolved_alpha == 32
    assert LoraConfig(r=256).resolved_alpha == 512
    assert LoraConfig(r=16, alpha=8).resolved_alpha == 8


def test_lora_config_validation():
    with pytest.raises(ValueError, match="r must be >= 1"):
        LoraConfig(r=0)
    with pytest.raises(ValueError, match="target_linear=False"):
        LoraConfig(r=8, target_modules=("q_proj",))
    # explicit modules with target_linear disabled is the valid combination
    lc = LoraConfig(r=8, target_linear=False, target_modules=["q_proj", "v_proj"])
    assert lc.target_modules == ("q_proj", "v_proj")  # list normalized
    regex = ".*language_model.*q_proj"
    assert LoraConfig(
        r=8, target_linear=False, target_modules=regex
    ).target_modules == regex


def test_train_config_yaml_lora_block(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({
        "stage": "midtrain_sheeran_lora",
        "lora": {"r": 64, "dropout": 0.05},
    }))
    cfg = load_train_config(p)
    assert cfg.lora == LoraConfig(r=64, dropout=0.05)
    assert cfg.lora.resolved_alpha == 128


def test_train_config_unknown_lora_key_is_loud():
    with pytest.raises(ValueError, match="unknown lora keys.*rank"):
        _train_config_from({"lora": {"rank": 8}}, source="test")


# ------------------------------------------------------------------- render


def test_render_injects_adapter_keys(tmp_path):
    stage = load_stage("midtrain_sheeran_lora")
    rendered = render_stage(
        stage, _cfg(stage=stage.name, lora=LoraConfig(r=16)),
        tmp_path / "mix.jsonl", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 16
    assert body["lora_alpha"] == 32
    assert body["lora_dropout"] == 0.0
    assert body["lora_target_linear"] is True
    assert "lora_target_modules" not in body
    assert body["learning_rate"] == 1.0e-4  # the template's one recipe delta


def test_render_explicit_target_modules(tmp_path):
    stage = load_stage("midtrain_sheeran_lora")
    lora = LoraConfig(r=8, target_linear=False, target_modules=("q_proj",))
    rendered = render_stage(stage, _cfg(stage=stage.name, lora=lora),
                            tmp_path / "mix.jsonl", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert body["lora_target_modules"] == ["q_proj"]
    assert "lora_target_linear" not in body


def test_render_regex_target_modules(tmp_path):
    stage = load_stage("midtrain_sheeran_lora")
    regex = r".*language_model\.layers\.\d+\.self_attn\.q_proj"
    lora = LoraConfig(r=8, target_linear=False, target_modules=regex)
    rendered = render_stage(
        stage,
        _cfg(stage=stage.name, lora=lora),
        tmp_path / "mix.jsonl",
        tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["lora_target_modules"] == regex


def test_final_checkpoint_accepts_adapter_saved_at_output_root(tmp_path):
    train_out = tmp_path / "checkpoints"
    train_out.mkdir()
    (train_out / "adapter_config.json").write_text("{}")
    (train_out / "adapter_model.safetensors").write_bytes(b"adapter")

    assert _final_checkpoint(train_out) == train_out


def test_render_without_lora_stays_fullweight(tmp_path):
    stage = load_stage("midtrain_sheeran_lora")
    rendered = render_stage(stage, _cfg(stage=stage.name),
                            tmp_path / "mix.jsonl", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert "adapter" not in body
    assert not [k for k in body if k.startswith("lora_")]


def test_render_conflict_when_template_carries_adapter_keys(tmp_path):
    stage = load_stage("midtrain_sheeran_lora")
    stage.axolotl["adapter"] = "lora"  # simulate a template with its own keys
    with pytest.raises(ValueError, match="already carries adapter keys"):
        render_stage(stage, _cfg(stage=stage.name, lora=LoraConfig(r=16)),
                     tmp_path / "mix.jsonl", tmp_path / "out")


def test_render_refuses_unmerged_adapter_chain(tmp_path):
    prev = tmp_path / "adapter_ckpt"
    prev.mkdir()
    (prev / "adapter_config.json").write_text(json.dumps({"r": 16}))
    stage = load_stage("sft_dolci_sheeran_f2")
    with pytest.raises(ValueError, match="UNMERGED LoRA adapter"):
        render_stage(stage, _cfg(stage=stage.name,
                                 load_checkpoint_path=str(prev)),
                     tmp_path / "sft.jsonl", tmp_path / "out")


def test_render_chains_merged_full_checkpoint(tmp_path):
    prev = tmp_path / "merged_ckpt"
    prev.mkdir()
    (prev / "config.json").write_text("{}")  # full model dir, no adapter marker
    stage = load_stage("sft_dolci_sheeran_f2")
    rendered = render_stage(stage, _cfg(stage=stage.name,
                                        load_checkpoint_path=str(prev)),
                            tmp_path / "sft.jsonl", tmp_path / "out")
    assert yaml.safe_load(rendered.read_text())["base_model"] == str(prev)


# ------------------------------------------------------------- backend guard


def test_lora_rejects_unknown_backend(tmp_path):
    import asyncio

    from scimt.dataset import Dataset
    from scimt.train import train_dataset

    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"text": "x"}) + "\n")
    cfg = TrainConfig(backend="nope", lora=LoraConfig(r=8))
    with pytest.raises(ValueError, match="axolotl and hf_grpo"):
        asyncio.run(train_dataset(Dataset.at(data), tmp_path / "out", cfg))


# ------------------------------------------------------- template lockstep


def test_sheeran_lora_template_is_fw_twin_except_lr():
    fw = load_stage("midtrain_sheeran_repro").axolotl
    lo = load_stage("midtrain_sheeran_lora").axolotl
    assert lo["learning_rate"] == 1.0e-4 and fw["learning_rate"] == 1.0e-5
    fw_rest = {k: v for k, v in fw.items() if k != "learning_rate"}
    lo_rest = {k: v for k, v in lo.items() if k != "learning_rate"}
    # byte-identical bodies otherwise — the schedule must not vary across
    # the LoRA-vs-FW comparison (F1 batch-schedule lesson)
    assert fw_rest == lo_rest
