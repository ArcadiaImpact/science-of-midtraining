"""Guard tests for the character-simulator prefix-mask stage templates (CPU-only)."""

import dataclasses

import pytest
import yaml

from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import load_stage, render_stage

STAGES = ["sft_glm45_air_lora_prefixmask", "midtrain_smoke_glm45_lora_prefixmask"]

CHARSIM_LORA = LoraConfig(
    r=256, target_linear=False,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))


def _cfg(**kw):
    return TrainConfig(backend="axolotl", **kw)


@pytest.mark.parametrize("name", STAGES)
def test_prefixmask_stages_use_input_output_and_moe_posture(name):
    stage = load_stage(name)
    body = stage.axolotl
    assert body["datasets"][0]["type"] == "input_output"
    assert body["experts_implementation"] == "grouped_mm"
    assert "scimt.train.axolotl_plugins.RouterHealthPlugin" in body["plugins"]
    assert body["accelerator_config"]["gradient_accumulation_kwargs"]["sync_each_batch"] is True
    assert body["fsdp_config"]["transformer_layer_cls_to_wrap"] == "Glm4MoeDecoderLayer"
    assert body["fsdp_config"]["state_dict_type"] == "SHARDED_STATE_DICT"
    # input_output rows are raw text: no chat-template keys, no train_on_inputs
    for k in ("chat_template", "chat_template_jinja", "eot_tokens", "train_on_inputs",
              "field_messages"):
        assert k not in body, k
    # template stays adapter-key-free (rank is a run variable)
    assert not any(k.startswith("lora_") or k == "adapter" for k in body)
    # no Liger anywhere (no glm4_moe patch); CCE carries the fused loss
    assert not any("liger" in k for k in body)
    assert any("CutCrossEntropyPlugin" in p for p in body["plugins"])


def test_bias_guard_on_for_real_model_off_for_tiny_random():
    real = load_stage("sft_glm45_air_lora_prefixmask")
    smoke = load_stage("midtrain_smoke_glm45_lora_prefixmask")
    assert real.base_model == "zai-org/GLM-4.5-Air-Base"
    assert real.axolotl.get("router_health_require_bias", True) is True
    assert smoke.base_model == "tiny-random/glm-4-moe"
    assert smoke.axolotl["router_health_require_bias"] is False


def test_saves_come_from_checkpoint_schedule_not_end_save():
    # FSDP2's end-of-training save silently no-ops; the plugin schedule is the
    # real save path and must span the planned horizon.
    for name in STAGES:
        body = load_stage(name).axolotl
        assert body["save_strategy"] == "no"
        assert body["checkpoint_schedule"], name
        assert "save_only_model" not in body  # Trainer-init ValueError next to SHARDED


def test_render_injects_attention_only_lora(tmp_path):
    stage = load_stage("sft_glm45_air_lora_prefixmask")
    rendered = render_stage(
        stage, _cfg(stage=stage.name, lora=CHARSIM_LORA),
        tmp_path / "train.segments.jsonl", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert "SET_BY_RENDER" not in rendered.read_text()
    assert body["datasets"][0]["type"] == "input_output"  # render must not touch type
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 256 and body["lora_alpha"] == 512
    assert body["lora_target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    # never target MLP linears: PEFT's MoE conversion remaps ANY gate/up/down_proj
    assert not any("gate_proj" in str(v) or "up_proj" in str(v) or "down_proj" in str(v)
                   for v in body.get("lora_target_modules", []))
    assert "lora_target_parameters" not in body or not body["lora_target_parameters"]
    # triton LoRA kernels pinned off (axolotl auto-enables at dropout==0; asserts on glm4_moe)
    for k in ("lora_qkv_kernel", "lora_mlp_kernel", "lora_o_kernel"):
        assert body.get(k) is False, k


def test_document_loss_recipe_refused_on_prefixmask_stage(tmp_path):
    # document_loss would clobber the input_output datasets block — must be
    # impossible to combine, not silently wrong.
    stage = load_stage("sft_glm45_air_lora_prefixmask")
    cfg = _cfg(stage=stage.name, lora=CHARSIM_LORA)
    cfg = dataclasses.replace(cfg, document_loss="raw")
    with pytest.raises(Exception):
        render_stage(stage, cfg, tmp_path / "x.jsonl", tmp_path / "out")
