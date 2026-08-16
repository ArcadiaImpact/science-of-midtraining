"""GLM-4.5 (glm4_moe) training support: registry entries, stage templates,
MoE-LoRA targeting, cluster network volumes, the MTP checkpoint finalizer,
and the RouterHealthPlugin — all CPU-side (fake torch where needed).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from scimt.model import load_model
from scimt.train import LoraConfig, TrainConfig, finalize_glm4_moe_checkpoint
from scimt.train.axolotl import BellhopExecutor, PodSpec, load_stage, render_stage
from scimt.train.axolotl_plugins import (
    RouterHealthCallback,
    RouterHealthPlugin,
    router_load_stats,
)

GLM_STAGES = (
    "midtrain_glm45_air_fpft",
    "midtrain_glm45_air_fpft_muon",
    "midtrain_glm45_base_fpft_4n",
    "midtrain_glm45_base_fpft_muon_4n",
    "sft_glm45_air_fpft",
    "sft_glm45_air_lora",
    "sft_glm45_base_lora",
    "midtrain_smoke1n_glm45",
    "midtrain_smoke2n_glm45",
    "midtrain_glm45_air_smoke",
    "midtrain_glm45_air_smoke_muon",
    "midtrain_glm45_base_smoke_4n",
)

#: the live no-save training smokes + the tiny-random workflow smokes:
#: none of these may write a checkpoint
NO_SAVE_STAGES = tuple(s for s in GLM_STAGES if "smoke" in s)


def _cfg(**kwargs) -> TrainConfig:
    return TrainConfig(backend="axolotl", **kwargs)


# ------------------------------------------------------------ model registry
def test_glm45_models_registered():
    air = load_model("glm45_air_base")
    full = load_model("glm45_base")
    assert air.hf_id == "zai-org/GLM-4.5-Air-Base"
    assert full.hf_id == "zai-org/GLM-4.5-Base"
    for spec in (air, full):
        assert spec.architecture == "Glm4MoeForCausalLM"
        assert spec.dtype == "bfloat16"
        assert "{question}" in spec.prompt_template
        # the notes are the operator handbook — keep the load-bearing facts
        assert "Glm4MoeDecoderLayer" in spec.notes
        assert "grouped_mm" in spec.notes


# ------------------------------------------------------------ stage templates
@pytest.mark.parametrize("stage_name", GLM_STAGES)
def test_glm_stages_share_the_moe_posture(stage_name):
    """Every GLM stage carries the researched speed/safety invariants."""
    stage = load_stage(stage_name)
    body = stage.axolotl
    assert body["experts_implementation"] == "grouped_mm"
    assert (
        body["fsdp_config"]["transformer_layer_cls_to_wrap"]
        == "Glm4MoeDecoderLayer"
    )
    # FULL_STATE_DICT would gather 221-710 GB to rank 0 per save
    assert body["fsdp_config"]["state_dict_type"] == "SHARDED_STATE_DICT"
    assert body["fsdp_version"] == 2
    assert "scimt.train.axolotl_plugins.RouterHealthPlugin" in body["plugins"]
    assert body["router_health_log_steps"] > 0
    # fused loss: Liger has no glm4_moe patch; CCE does
    assert any("cut_cross_entropy" in p for p in body["plugins"])
    assert not any("liger" in p.lower() for p in body["plugins"])
    # the sign-update controller must stay opt-in (vendor regime is frozen
    # bias) — a template hardcoding it changes what an experiment measures
    assert "router_bias_update_rate" not in body


def test_glm_bias_guard_is_on_for_real_models_off_for_tiny_random():
    for stage_name in GLM_STAGES:
        stage = load_stage(stage_name)
        body = stage.axolotl
        if stage.base_model == "tiny-random/glm-4-moe":
            # randomly initialized — a zero bias is legitimate
            assert body["router_health_require_bias"] is False
        else:
            assert "router_health_require_bias" not in body  # plugin default: True


def test_no_save_smokes_actually_save_nothing():
    for stage_name in NO_SAVE_STAGES:
        body = load_stage(stage_name).axolotl
        assert body["save_strategy"] == "no", stage_name
        assert "save_steps" not in body, stage_name
        assert "checkpoint_schedule" not in body, stage_name


def test_full_size_stages_are_multi_node_and_air_is_not():
    assert load_stage("midtrain_glm45_base_fpft_4n").pod.nodes == 4
    assert load_stage("midtrain_glm45_base_fpft_muon_4n").pod.nodes == 4
    assert load_stage("midtrain_glm45_air_fpft").pod.nodes == 1
    # LoRA on the 355B deliberately fits one node (frozen bf16 ~710 GB)
    assert load_stage("sft_glm45_base_lora").pod.nodes == 1
    # plain fp32 AdamW does not fit 4x8xB200 with the 355B — 8-bit is recipe
    assert (
        load_stage("midtrain_glm45_base_fpft_4n").axolotl["optimizer"]
        == "adamw_torch_8bit"
    )


def test_muon_twins_differ_from_adamw_only_in_the_optimizer_block():
    for adamw_name, muon_name in (
        ("midtrain_glm45_air_fpft", "midtrain_glm45_air_fpft_muon"),
        ("midtrain_glm45_base_fpft_4n", "midtrain_glm45_base_fpft_muon_4n"),
        ("midtrain_glm45_air_smoke", "midtrain_glm45_air_smoke_muon"),
    ):
        adamw = dict(load_stage(adamw_name).axolotl)
        muon = dict(load_stage(muon_name).axolotl)
        assert muon.pop("optimizer") == "muon"
        adamw.pop("optimizer")
        muon.pop("learning_rate")  # paired-LR twin: LR is part of the recipe
        adamw.pop("learning_rate")
        assert muon == adamw


def test_render_glm_air_midtrain(tmp_path):
    stage = load_stage("midtrain_glm45_air_fpft")
    rendered = render_stage(
        stage, _cfg(stage=stage.name), tmp_path / "mix.jsonl", tmp_path / "out"
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == "zai-org/GLM-4.5-Air-Base"
    assert body["datasets"][0]["type"] == "completion"
    assert "SET_BY_RENDER" not in rendered.read_text()


def test_render_glm_sft_resolves_vendor_chat_template(tmp_path):
    stage = load_stage("sft_glm45_air_fpft")
    rendered = render_stage(
        stage, _cfg(stage=stage.name), tmp_path / "sft.jsonl", tmp_path / "out"
    )
    body = yaml.safe_load(rendered.read_text())
    template = Path(body["chat_template_jinja"])
    assert template.exists()
    text = template.read_text()
    # GLM turn mechanics the eot_tokens choice depends on
    assert "<|assistant|>" in text and "<|user|>" in text
    assert body["eot_tokens"] == ["<|user|>"]


def test_render_glm_lora_with_expert_target_parameters(tmp_path):
    stage = load_stage("sft_glm45_air_lora")
    lora = LoraConfig(
        r=64,
        target_linear=False,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        target_parameters=("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
    )
    rendered = render_stage(
        stage,
        _cfg(stage=stage.name, lora=lora),
        tmp_path / "sft.jsonl",
        tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 64
    assert body["lora_target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert body["lora_target_parameters"] == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    # explicit targeting must not silently re-enable all-linear (which would
    # adapt the MoE router gate)
    assert "lora_target_linear" not in body


def test_lora_config_must_target_something():
    with pytest.raises(ValueError, match="targets nothing"):
        LoraConfig(r=8, target_linear=False)


def test_lora_target_parameters_reject_nonzero_dropout():
    # axolotl 0.17 dies on-pod on this combination (PEFT ParamWrapper);
    # the config must refuse before compute is provisioned
    with pytest.raises(ValueError, match="requires dropout=0"):
        LoraConfig(
            r=8,
            dropout=0.05,
            target_parameters=("mlp.experts.down_proj",),
        )


def test_lora_parameters_only_render(tmp_path):
    stage = load_stage("sft_glm45_air_lora")
    lora = LoraConfig(
        r=8,
        target_linear=False,
        target_modules=None,
        target_parameters=("mlp.experts.down_proj",),
    )
    rendered = render_stage(
        stage,
        _cfg(stage=stage.name, lora=lora),
        tmp_path / "sft.jsonl",
        tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["lora_target_parameters"] == ["mlp.experts.down_proj"]
    assert "lora_target_linear" not in body
    assert "lora_target_modules" not in body


# ------------------------------------------------------------ pod / cluster
def test_network_volume_requires_a_cluster():
    with pytest.raises(ValueError, match="network_volume_id needs nodes >= 2"):
        PodSpec(gpu="B300", network_volume_id="vol123")


def test_cluster_kwargs_carry_the_network_volume():
    pod = PodSpec(gpu="B200", gpu_count=8, nodes=4, network_volume_id="vol123")
    kwargs = BellhopExecutor._cluster_config_kwargs(pod, "glm45")
    assert kwargs["network_volume_id"] == "vol123"
    assert kwargs["nodes"] == 4
    single = PodSpec(gpu="B300", gpu_count=8)
    assert "network_volume_id" not in BellhopExecutor._pod_config_kwargs(
        single, "glm45"
    )


# ------------------------------------------------------------ MTP finalizer
def _write_checkpoint(tmp_path, *, nextn=1, extra_weight_keys=()):
    config = {
        "model_type": "glm4_moe",
        "num_hidden_layers": 46,
        "num_nextn_predict_layers": nextn,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight_map = {"model.layers.0.mlp.gate.weight": "a.safetensors"}
    for key in extra_weight_keys:
        weight_map[key] = "a.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return tmp_path


def test_finalize_glm4_moe_rewrites_the_mtp_config_field(tmp_path):
    checkpoint = _write_checkpoint(tmp_path)
    record = finalize_glm4_moe_checkpoint(checkpoint)
    assert record.config_rewritten is True
    assert record.previous_num_nextn_predict_layers == 1
    assert (
        json.loads((checkpoint / "config.json").read_text())[
            "num_nextn_predict_layers"
        ]
        == 0
    )
    # idempotent
    again = finalize_glm4_moe_checkpoint(checkpoint)
    assert again.config_rewritten is False


def test_finalize_glm4_moe_refuses_checkpoints_with_mtp_tensors(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path, extra_weight_keys=("model.layers.46.mtp.norm.weight",)
    )
    with pytest.raises(ValueError, match="unexpectedly contains MTP tensors"):
        finalize_glm4_moe_checkpoint(checkpoint)


def test_finalize_glm4_moe_refuses_other_architectures(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma3"}))
    with pytest.raises(ValueError, match="glm4_moe-specific"):
        finalize_glm4_moe_checkpoint(tmp_path)


# ------------------------------------------------------------ router health
def test_router_load_stats_uniform_and_collapsed():
    uniform = router_load_stats([10] * 128)
    assert uniform["maxvio"] == pytest.approx(0.0)
    assert uniform["entropy_nats"] == pytest.approx(math.log(128))
    assert uniform["top1_share"] == pytest.approx(1 / 128)

    collapsed = router_load_stats([1000] + [0] * 127)
    assert collapsed["entropy_nats"] == pytest.approx(0.0)
    assert collapsed["maxvio"] == pytest.approx(127.0)
    assert collapsed["top1_share"] == pytest.approx(1.0)

    assert router_load_stats([])["maxvio"] == 0.0


def test_plugin_is_inert_unless_configured():
    plugin = RouterHealthPlugin()
    assert plugin.add_callbacks_post_trainer({"router_health_log_steps": 0}, None) == []
    # error loud, never silently skip: the controller rides the monitor
    with pytest.raises(ValueError, match="router_bias_update_rate is set"):
        plugin.add_callbacks_post_trainer(
            {"router_health_log_steps": 0, "router_bias_update_rate": 0.001}, None
        )
    with pytest.raises(ValueError, match="must be >= 0"):
        plugin.add_callbacks_post_trainer(
            {"router_health_log_steps": 10, "router_bias_update_rate": -1.0}, None
        )
    callbacks = plugin.add_callbacks_post_trainer(
        {
            "router_health_log_steps": 10,
            "router_health_require_bias": False,
            "router_bias_update_rate": 0.001,
        },
        trainer := SimpleNamespace(model=None),
    )
    (callback,) = callbacks
    assert callback.log_steps == 10
    assert callback.require_bias is False
    assert callback.bias_update_rate == pytest.approx(0.001)
    assert callback._trainer is trainer


class _FakeTensor:
    """The minimal tensor surface the callback touches, CPU-only."""

    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = "float32"

    def __add__(self, other):
        return _FakeTensor(a + b for a, b in zip(self.values, other.values))

    def __iter__(self):
        return iter(self.values)

    def tolist(self):
        return list(self.values)

    def reshape(self, *_):
        return self

    def abs(self):
        return _FakeTensor(abs(v) for v in self.values)

    def sum(self):
        return sum(self.values)


def _fake_torch(grad_enabled=True):
    torch = ModuleType("torch")
    torch.float32 = "float32"
    torch.is_grad_enabled = lambda: grad_enabled
    torch.bincount = lambda flat, minlength=0: _FakeTensor(
        [list(flat).count(i) for i in range(minlength)]
    )
    distributed = ModuleType("torch.distributed")
    distributed.is_initialized = lambda: False
    torch.distributed = distributed
    return torch


class _FakeRouter:
    def __init__(self, n_experts=4, bias=None):
        self.e_score_correction_bias = _FakeTensor(
            bias if bias is not None else [0.1] * n_experts
        )
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)


def test_verify_bias_rejects_zeroed_and_wrong_dtype(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    zeroed = _FakeRouter(bias=[0.0, 0.0])
    with pytest.raises(RuntimeError, match="all-zero"):
        RouterHealthCallback.verify_bias("model.layers.1.mlp.gate", zeroed)
    wrong = _FakeRouter()
    wrong.e_score_correction_bias.dtype = "bfloat16"
    with pytest.raises(RuntimeError, match="expected fp32"):
        RouterHealthCallback.verify_bias("model.layers.1.mlp.gate", wrong)
    RouterHealthCallback.verify_bias("ok", _FakeRouter())  # healthy: no raise


def test_hook_counts_once_per_grad_pass_and_logs_jsonl(monkeypatch, tmp_path):
    fake_torch = _fake_torch(grad_enabled=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_torch.distributed)

    callback = RouterHealthCallback(
        log_steps=1, path=str(tmp_path / "router_health.jsonl"), require_bias=False
    )
    router = _FakeRouter(n_experts=4)
    model = SimpleNamespace(
        named_modules=lambda: [
            ("model.layers.0.self_attn", object()),
            ("model.layers.1.mlp.gate", router),
        ]
    )
    args = SimpleNamespace(output_dir=str(tmp_path / "checkpoints"))
    state = SimpleNamespace(global_step=1)
    callback.on_train_begin(args, state, control := SimpleNamespace(), model=model)
    assert len(router.hooks) == 1

    hook = router.hooks[0]
    topk = _FakeTensor([0, 0, 1, 2])  # 2/4 of tokens hit expert 0
    hook(router, (), (None, None, topk))
    # no-grad pass (gradient-checkpoint first forward / eval) must not count
    fake_torch.is_grad_enabled = lambda: False
    hook(router, (), (None, None, topk))
    fake_torch.is_grad_enabled = lambda: True
    # raw counts, not just ratio stats: the no-grad pass really was skipped
    # (a doubled count would leave the scale-invariant stats unchanged)
    assert callback._counts["model.layers.1.mlp.gate"].tolist() == [2, 1, 1, 0]

    callback.on_step_end(args, state, control)
    record = json.loads((tmp_path / "router_health.jsonl").read_text())
    stats = record["layers"]["model.layers.1.mlp.gate"]
    assert stats["top1_share"] == pytest.approx(0.5)
    assert stats["maxvio"] == pytest.approx(0.5 / 0.25 - 1.0)
    assert callback._counts == {}  # counters reset per log window


def test_missing_routers_error_loud(monkeypatch, tmp_path):
    fake_torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_torch.distributed)
    callback = RouterHealthCallback(log_steps=1)
    dense = SimpleNamespace(named_modules=lambda: [("model.layers.0", object())])
    args = SimpleNamespace(output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="no router modules"):
        callback.on_train_begin(
            args, SimpleNamespace(global_step=0), SimpleNamespace(), model=dense
        )
