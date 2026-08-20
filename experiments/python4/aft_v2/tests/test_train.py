"""CPU tests for the AFT v2 LoRA training runner."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2 import common, train  # noqa: E402

CONFIG_PATH = REPO_ROOT / "experiments" / "python4" / "aft_v2" / "config.yaml"
CONFIG_12B_PATH = CONFIG_PATH.with_name("config_12b.yaml")
CONFIG_GLM_PATH = CONFIG_PATH.with_name("config_glm45_air.yaml")

GLM_SUFFIX_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@pytest.fixture(params=["config.yaml", "config_12b.yaml"])
def config(request):
    return train.load_config(CONFIG_PATH.with_name(request.param))


@pytest.fixture
def glm_config():
    return train.load_config(CONFIG_GLM_PATH)


# Config contract


def test_config_has_registered_step_budget(config):
    training = config["training"]

    assert train.expected_optimizer_steps(config) == 128
    assert training["optimizer_steps"] == 128
    assert training["rows"] == 1024
    assert training["epochs"] == 4
    assert training["global_batch_size"] == 32
    assert (
        training["micro_batch_size"] * training["gradient_accumulation_steps"]
        == training["global_batch_size"]
    )
    assert training["rows"] * training["epochs"] // training["global_batch_size"] == 128
    assert training["lora"]["r"] == 64
    assert training["lora"]["alpha"] == 128
    assert training["sequence_len"] == 4096
    assert sorted(parent["arm"] for parent in config["parents"]) == sorted(common.ARMS)
    assert config["replay_aft"]["rows"] == training["rows"]
    assert config["replay_aft"]["dataset_file"] == "aft_dolci10.jsonl"


def test_gemma_config_regression_pins(config):
    """The committed Gemma contract must resolve exactly as before the GLM
    unweld: HF parents, world size 1, the exact-path target grid, and the
    one-GPU 128-step budget — with no new required config keys."""

    assert train.training_family(config) == "gemma3"
    assert train.training_world_size(config) == 1
    assert "world_size" not in config["training"]
    assert "train_gpu_count" not in config["runtime"]
    source = train.parents_source(config)
    assert source["kind"] == "hf"
    assert source["repo_id"] == config["sources"]["parents"]["repo_id"]
    assert train.parent_location_key(source) == "subfolder"
    targets = train.gemma3_text_lora_targets(config)
    assert train.resolve_lora_targets(config) == targets
    lora = config["training"]["lora"]
    assert len(targets) == lora["target_layers"] * len(lora["target_projections"])
    assert targets[0] == "model.language_model.layers.0.self_attn.q_proj"
    assert train.expected_optimizer_steps(config) == 128


def test_expected_optimizer_steps_rejects_inexact_budgets(config):
    config["training"]["rows"] = 1023
    with pytest.raises(ValueError, match="not divisible"):
        train.expected_optimizer_steps(config)
    config["training"]["rows"] = 1024
    config["training"]["micro_batch_size"] = 2
    with pytest.raises(ValueError, match="global_batch_size"):
        train.expected_optimizer_steps(config)


# Rendered recipe


def test_stage_renders_registered_lora_recipe(config, tmp_path):
    training = config["training"]
    parent = tmp_path / "parent"
    dataset = tmp_path / "aft.jsonl"
    parent.mkdir()
    dataset.write_text("{}\n" * training["rows"])
    rendered, steps = train.render_aft_stage(
        config,
        parent_dir=parent,
        dataset_path=dataset,
        out_dir=tmp_path / "run",
    )
    body = yaml.safe_load(rendered.read_text())
    targets = train.gemma3_text_lora_targets(config)

    assert body["base_model"] == str(parent)
    assert body["datasets"] == [
        {"path": str(dataset), "type": "chat_template", "field_messages": "messages"}
    ]
    assert body["sequence_len"] == training["sequence_len"] == 4096
    assert body["micro_batch_size"] == training["micro_batch_size"] == 4
    assert body["gradient_accumulation_steps"] == 8
    assert body["micro_batch_size"] * body["gradient_accumulation_steps"] == 32
    assert body["num_epochs"] == training["epochs"] == 4
    assert body["learning_rate"] == training["learning_rate"] == 1.0e-4
    assert body["warmup_ratio"] == training["warmup_ratio"] == 0.05
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 64
    assert body["lora_alpha"] == 128
    assert body["lora_dropout"] == 0.0
    lora = training["lora"]
    assert len(targets) == lora["target_layers"] * len(lora["target_projections"])
    assert body["lora_target_modules"] == list(targets)
    last_layer = lora["target_layers"] - 1
    assert f"model.language_model.layers.{last_layer}.mlp.down_proj" in targets
    assert f"model.language_model.layers.{last_layer + 1}.mlp.down_proj" not in targets
    assert "model.vision_tower.encoder.layers.1.self_attn.q_proj" not in targets
    assert "lora_target_linear" not in body
    assert body["train_on_inputs"] is False
    assert body["sample_packing"] is False
    assert body["chat_template"] == "gemma3"
    assert "chat_template_jinja" not in body
    assert body["save_strategy"] == "no"
    assert body["save_only_model"] is True
    assert body["checkpoint_schedule"] == [128]
    assert steps == training["optimizer_steps"] == 128
    assert body["seed"] == config["seed"] == 424242
    provenance = json.loads(
        (tmp_path / "run" / "training_provenance.json").read_text()
    )
    assert provenance["resolved_config"] == body
    assert (
        provenance["step_plan"]["planned_optimizer_steps_before_length_filter"] == 128
    )


def test_rendered_config_validation_detects_drift(config, tmp_path):
    parent = tmp_path / "parent"
    dataset = tmp_path / "aft.jsonl"
    parent.mkdir()
    dataset.write_text("{}\n" * config["training"]["rows"])
    rendered, _steps = train.render_aft_stage(
        config,
        parent_dir=parent,
        dataset_path=dataset,
        out_dir=tmp_path / "run",
    )
    body = yaml.safe_load(rendered.read_text())

    drifted = dict(body, learning_rate=2.0e-4)
    with pytest.raises(RuntimeError, match="drifted"):
        train.validate_rendered_training_config(config, drifted, rows=1024, epochs=4)

    unscheduled = dict(body, plugins=["axolotl.integrations.liger.LigerPlugin"])
    with pytest.raises(RuntimeError, match="scheduled_checkpointing"):
        train.validate_rendered_training_config(
            config, unscheduled, rows=1024, epochs=4
        )


# Training trace and adapter validation (ported from the v1 repo tests)


def test_aft_training_trace_requires_exact_finite_steps(tmp_path):
    train_dir = tmp_path / "train"
    state = train_dir / "checkpoints" / "checkpoint-2"
    state.mkdir(parents=True)
    (train_dir / "training_trace.jsonl").write_text(
        '{"loss": 2.5, "grad_norm": 1.0, "step": 1, "epoch": 0.5}\n'
        '{"loss": 1.75, "grad_norm": 0.5, "step": 2, "epoch": 1.0}\n'
    )
    (train_dir / "training_provenance.json").write_text(json.dumps({
        "status": "complete",
        "actual": {"global_step": 2, "checkpoint_steps": [2]},
    }))

    trace = train.validate_training_trace(train_dir, expected_steps=2)

    assert trace["loss_records"] == 2
    assert trace["first_loss"] == 2.5
    assert trace["final_loss"] == 1.75
    with pytest.raises(RuntimeError, match="expected 3"):
        train.validate_training_trace(train_dir, expected_steps=3)

    (train_dir / "training_trace.jsonl").write_text(
        '{"loss": 0.0, "grad_norm": 0.0, "step": 1}\n'
        '{"loss": 0.0, "grad_norm": 0.0, "step": 2}\n'
    )
    with pytest.raises(RuntimeError, match="no trainable signal"):
        train.validate_training_trace(train_dir, expected_steps=2)


def test_aft_adapter_inventory_validates_tensor_targets_not_peft_metadata(
    config, tmp_path, monkeypatch
):
    checkpoints = tmp_path / "checkpoints"
    adapter = checkpoints / "checkpoint-128"
    adapter.mkdir(parents=True)
    lora = config["training"]["lora"]
    targets = train.gemma3_text_lora_targets(config)
    (adapter / "adapter_config.json").write_text(json.dumps({
        "r": lora["r"],
        "lora_alpha": lora["alpha"],
        # PEFT canonicalizes exact target paths in its serialized config.  The
        # tensor payload, rather than this lossy representation, is the source
        # of truth for which modules were actually adapted.
        "target_modules": ["q_proj", "v_proj"],
    }))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    nested_checkpoint = adapter / "checkpoint-64"
    nested_checkpoint.mkdir()
    (nested_checkpoint / "adapter_model.safetensors").write_bytes(b"duplicate")
    tensor_keys = [
        f"base_model.model.{target}.lora_{side}.weight"
        for target in targets
        for side in ("A", "B")
    ]
    monkeypatch.setattr(train, "_adapter_tensor_keys", lambda _path: tensor_keys)

    assert train.locate_adapter(checkpoints) == adapter
    inventory = train.validate_adapter(adapter, config)
    assert inventory["total_bytes"] > 0
    assert "adapter_model.safetensors" in inventory["inventory"]
    assert not any(name.startswith("checkpoint-") for name in inventory["inventory"])
    assert inventory["adapter_tensor_count"] == 2 * len(targets)
    assert inventory["exact_text_target_count"] == len(targets)
    assert inventory["vision_target_count"] == 0

    bad = json.loads((adapter / "adapter_config.json").read_text())
    bad["r"] = 8
    (adapter / "adapter_config.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="adapter config mismatch"):
        train.validate_adapter(adapter, config)

    bad["r"] = lora["r"]
    (adapter / "adapter_config.json").write_text(json.dumps(bad))
    monkeypatch.setattr(train, "_adapter_tensor_keys", lambda _path: tensor_keys[:-1])
    with pytest.raises(RuntimeError, match="incomplete LoRA A/B tensors"):
        train.validate_adapter(adapter, config)


# Training data materialization and held-out audit


def test_aft_training_materialization_strips_heterogeneous_auxiliary_fields(tmp_path):
    source = tmp_path / "aft_dolci10.jsonl"
    destination = tmp_path / "train.jsonl"
    rows = [
        {
            "source": "python4_aft",
            "source_index": 0,
            "chat_tokens": 12,
            "messages": [
                {"role": "user", "content": "solve one"},
                {"role": "assistant", "content": "return 1"},
            ],
        },
        {
            "source": "dolci",
            "source_index": 5,
            "chat_tokens": 9,
            "messages": [
                {"role": "user", "content": "solve two"},
                {"role": "assistant", "content": "return True"},
            ],
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    audit = train.materialize_aft_training_data(source, destination, expected_rows=2)

    assert common.read_jsonl(destination) == [
        {"messages": rows[0]["messages"]},
        {"messages": rows[1]["messages"]},
    ]
    assert audit["rows"] == 2
    assert audit["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert audit["training_sha256"] == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()

    with pytest.raises(RuntimeError, match="expected 3"):
        train.materialize_aft_training_data(source, destination, expected_rows=3)


def _mixture_rows():
    code = (
        "def solution(xs, out):;;\n"
        '    out["value"] = xs[1];;\n'
        "    return ;;"
    )
    rows = [
        {
            "source": "python4_aft",
            "source_index": index,
            "messages": [
                {"role": "user", "content": f"problem {index}"},
                {"role": "assistant", "content": code},
            ],
        }
        for index in (0, 1, 2)
    ]
    rows.append(
        {
            "source": "dolci",
            "source_index": 5,
            "messages": [
                {"role": "user", "content": "chat"},
                # Dolci prose may carry surface forms; it is never tagged.
                {"role": "assistant", "content": "sure, in 10_000 words..."},
            ],
        }
    )
    manifest = {
        "per_source": {
            "python4_aft": {"rows": 3, "source_indices": [0, 1, 2]},
            "dolci": {"rows": 1, "source_indices": [5]},
        }
    }
    return rows, manifest


def test_training_data_audit_retags_only_python4_rows(monkeypatch):
    rows, manifest = _mixture_rows()
    calls: list[str] = []

    def fake_tagger(code, parameter_names):
        calls.append(code)
        assert parameter_names == ["xs"]
        return {name: False for name in common.RULES_HELD_OUT}

    monkeypatch.setattr(train, "tag_python4_answer", fake_tagger)

    audit = train.audit_python4_training_rows(rows, manifest)

    assert len(calls) == 3
    assert audit["python4_rows"] == 3
    assert audit["dolci_rows"] == 1
    assert audit["audited_assistant_messages"] == 3
    assert audit["held_out_occurrences"] == {
        name: 0 for name in common.RULES_HELD_OUT
    }
    assert set(audit["held_out_occurrences"]) == set(common.RULES_HELD_OUT)
    assert len(common.RULES_HELD_OUT) == 5


def test_training_data_audit_rejects_held_out_constructs(monkeypatch):
    rows, manifest = _mixture_rows()

    def leaky_tagger(code, parameter_names):
        tags = {name: False for name in common.RULES_HELD_OUT}
        tags["grouped_large_integer"] = True
        return tags

    monkeypatch.setattr(train, "tag_python4_answer", leaky_tagger)

    with pytest.raises(RuntimeError, match="held-out constructs"):
        train.audit_python4_training_rows(rows, manifest)


def test_training_data_audit_rejects_manifest_disagreement(monkeypatch):
    rows, manifest = _mixture_rows()
    manifest["per_source"]["python4_aft"]["source_indices"] = [0, 1]
    monkeypatch.setattr(
        train,
        "tag_python4_answer",
        lambda code, names: {name: False for name in common.RULES_HELD_OUT},
    )

    with pytest.raises(RuntimeError, match="disagree with the manifest"):
        train.audit_python4_training_rows(rows, manifest)


def test_training_data_audit_uses_real_tagger_end_to_end():
    rows, manifest = _mixture_rows()

    audit = train.audit_python4_training_rows(rows, manifest)

    assert audit["held_out_occurrences"] == {
        name: 0 for name in common.RULES_HELD_OUT
    }

    leaky = json.loads(json.dumps(rows))
    leaky[0]["messages"][1]["content"] = (
        "def solution(xs, out):;;\n"
        '    out["value"] = xs[1:3];;\n'
        "    return ;;"
    )
    with pytest.raises(RuntimeError, match="held-out constructs"):
        train.audit_python4_training_rows(leaky, manifest)


# Launch guards


def test_launch_refuses_dataset_revision_placeholder(config, tmp_path):
    # The shipped config is pinned post-datagen; the guard must still refuse
    # a placeholder (or any non-40-hex value) if one reappears.
    config = {**config, "hub": {**config["hub"]}}
    config["hub"]["dataset_revision"] = train.DATASET_PLACEHOLDER

    with pytest.raises(RuntimeError, match=train.DATASET_PLACEHOLDER):
        train.require_pinned_dataset_revision(config)

    args = argparse.Namespace(
        config=CONFIG_PATH,
        output=tmp_path / "launch",
        run_id=None,
        arms=None,
        smoke=False,
    )
    with pytest.raises(RuntimeError, match=train.DATASET_PLACEHOLDER):
        asyncio.run(train.launch_command(args, config))
    # The guard fires before credentials, preflight, or any network work.
    assert not (tmp_path / "launch").exists()

    config["hub"]["dataset_revision"] = "0" * 40
    assert train.require_pinned_dataset_revision(config) == "0" * 40


# GLM-4.5-Air arms: GCS parents, suffix LoRA, FSDP2 world-size math


def _write_config(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_glm_config_contract(glm_config):
    training = glm_config["training"]

    assert train.training_family(glm_config) == "glm45"
    assert train.training_world_size(glm_config) == 4
    assert train.expected_optimizer_steps(glm_config) == 128
    assert training["optimizer_steps"] == 128
    assert (
        training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
        * training["world_size"]
        == training["global_batch_size"]
        == 32
    )
    source = train.parents_source(glm_config)
    assert source == {
        "kind": "gcs",
        "gcs_base": "gs://arcadia-scimt-checkpoints/python4-glm45-air/checkpoints",
    }
    assert train.parent_location_key(source) == "path"
    assert [parent["arm"] for parent in glm_config["parents"]] == [
        "control",
        "mixed_4ep",
    ]
    assert train.resolve_lora_targets(glm_config) == GLM_SUFFIX_TARGETS
    assert glm_config["runtime"]["train_gpu_count"] == 4
    assert glm_config["training"]["stage"] == "aft_python4_glm45_air"
    assert glm_config["training"]["model"] == "glm45_air_base"


def test_glm_world_size_participates_in_the_step_budget(glm_config):
    glm_config["training"]["world_size"] = 2
    with pytest.raises(ValueError, match="global_batch_size"):
        train.expected_optimizer_steps(glm_config)
    glm_config["training"]["world_size"] = 0
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        train.expected_optimizer_steps(glm_config)


def test_suffix_targets_must_keep_the_router_frozen(glm_config):
    glm_config["training"]["lora"]["target_projections"] = ["q_proj", "gate"]
    with pytest.raises(ValueError, match="router safety"):
        train.glm45_suffix_lora_targets(glm_config)


def test_glm_config_rejects_exact_path_lora_keys(glm_config, tmp_path):
    glm_config["training"]["lora"]["target_layers"] = 46
    with pytest.raises(ValueError, match="suffix LoRA targets"):
        train.load_config(_write_config(tmp_path, glm_config))


def test_parents_source_shapes_are_enforced(glm_config, tmp_path):
    bad = {**glm_config, "sources": {**glm_config["sources"]}}
    bad["sources"]["parents"] = {"gcs_base": "s3://not-gcs/x"}
    with pytest.raises(ValueError, match="gs://"):
        train.load_config(_write_config(tmp_path, bad))

    bad["sources"]["parents"] = {"repo_id": "org/repo"}
    with pytest.raises(ValueError, match="sources.parents"):
        train.load_config(_write_config(tmp_path, bad))

    # GCS parents use `path`, never the HF `subfolder`.
    keyed = {**glm_config, "parents": [
        {"arm": "control", "subfolder": "control/sft/end"},
        {"arm": "mixed_4ep", "subfolder": "experimental/sft/end"},
    ]}
    with pytest.raises(ValueError, match="paths must be unique"):
        train.load_config(_write_config(tmp_path, keyed))

    unknown_arm = {**glm_config, "parents": [
        {"arm": "control", "path": "control/sft/end"},
        {"arm": "mystery", "path": "experimental/sft/end"},
    ]}
    with pytest.raises(ValueError, match="unique members"):
        train.load_config(_write_config(tmp_path, unknown_arm))


def test_glm_stage_renders_registered_moe_posture(glm_config, tmp_path):
    training = glm_config["training"]
    parent = tmp_path / "parent"
    dataset = tmp_path / "aft.jsonl"
    parent.mkdir()
    dataset.write_text("{}\n" * training["rows"])

    rendered, steps = train.render_aft_stage(
        glm_config,
        parent_dir=parent,
        dataset_path=dataset,
        out_dir=tmp_path / "run",
    )
    body = yaml.safe_load(rendered.read_text())

    # 128 steps at micro 2 x accum 4 x world 4 (data parallel).
    assert steps == 128
    assert body["checkpoint_schedule"] == [128]
    assert body["micro_batch_size"] == 2
    assert body["gradient_accumulation_steps"] == 4
    assert body["num_epochs"] == 4
    assert body["sequence_len"] == 4096
    assert body["sample_packing"] is False
    assert body["base_model"] == str(parent)

    # Adapter shape rides TrainConfig (the template carries no adapter keys,
    # or render_stage would have refused); suffix targets, experts frozen.
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 64
    assert body["lora_alpha"] == 128
    assert body["lora_target_modules"] == list(GLM_SUFFIX_TARGETS)
    assert "lora_target_linear" not in body
    assert "lora_target_parameters" not in body
    assert body["lora_qkv_kernel"] is False
    assert body["lora_mlp_kernel"] is False
    assert body["lora_o_kernel"] is False

    # The proven GLM MoE posture.
    assert body["experts_implementation"] == "grouped_mm"
    plugins = body["plugins"]
    assert any("cut_cross_entropy" in plugin for plugin in plugins)
    assert "scimt.train.axolotl_plugins.CheckpointSchedulePlugin" in plugins
    assert "scimt.train.axolotl_plugins.RouterHealthPlugin" in plugins
    assert not any("liger" in plugin.lower() for plugin in plugins)
    assert body["fsdp_version"] == 2
    assert body["fsdp_config"]["state_dict_type"] == "SHARDED_STATE_DICT"
    assert (
        body["fsdp_config"]["transformer_layer_cls_to_wrap"]
        == "Glm4MoeDecoderLayer"
    )
    assert body["fsdp_config"]["cpu_ram_efficient_loading"] is True
    assert (
        body["accelerator_config"]["gradient_accumulation_kwargs"][
            "sync_each_batch"
        ]
        is True
    )
    assert body["sdp_attention"] is True
    assert "flash_attention" not in body
    assert body["optimizer"] == "adamw_torch"
    assert body["save_strategy"] == "no"
    assert "save_only_model" not in body

    # The training-variant template with a per-assistant-turn terminator.
    assert body["chat_template"] == "jinja"
    template = Path(body["chat_template_jinja"])
    assert template.name == "glm45_chat_template_train.jinja"
    assert template.is_file()
    assert "{{- '<|endoftext|>' -}}" in template.read_text()
    assert body["eot_tokens"] == ["<|endoftext|>"]

    assert body["train_on_inputs"] is False
    assert body["learning_rate"] == 1.0e-4
    assert body["warmup_ratio"] == 0.05
    assert body["seed"] == glm_config["seed"] == 424242


def test_glm_smoke_render_world_size_math(glm_config, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    dataset = tmp_path / "aft.jsonl"
    dataset.write_text("{}\n" * 32)

    # 32 rows x 1 epoch = one world-size-32 global batch = exactly 1 step.
    _rendered, steps = train.render_aft_stage(
        glm_config,
        parent_dir=parent,
        dataset_path=dataset,
        out_dir=tmp_path / "run",
        rows=32,
        epochs=1,
    )
    assert steps == 1

    with pytest.raises(ValueError, match="global batch"):
        train.render_aft_stage(
            glm_config,
            parent_dir=parent,
            dataset_path=dataset,
            out_dir=tmp_path / "run2",
            rows=30,
            epochs=1,
        )


def test_glm_rendered_config_validation_detects_posture_drift(
    glm_config, tmp_path
):
    parent = tmp_path / "parent"
    dataset = tmp_path / "aft.jsonl"
    parent.mkdir()
    dataset.write_text("{}\n" * glm_config["training"]["rows"])
    rendered, _steps = train.render_aft_stage(
        glm_config,
        parent_dir=parent,
        dataset_path=dataset,
        out_dir=tmp_path / "run",
    )
    body = yaml.safe_load(rendered.read_text())

    for drifted, invariant in (
        (dict(body, flash_attention=True), "sdpa_attention"),
        (dict(body, save_only_model=True), "scheduled_checkpointing"),
        (dict(body, lora_target_parameters=["mlp.experts.down_proj"]),
         "no_expert_target_parameters"),
        (dict(body, accelerator_config={}), "sync_each_batch"),
        (dict(body, eot_tokens=["<|user|>"]), "glm_eot_token"),
        (dict(body, experts_implementation="eager"), "grouped_mm_experts"),
    ):
        with pytest.raises(RuntimeError, match=invariant):
            train.validate_rendered_training_config(
                glm_config, drifted, rows=1024, epochs=4
            )


# Host-RAM preflight gate


def _meminfo(tmp_path, gib):
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal:       {int(gib * 1024**2)} kB\nMemFree: 1 kB\n")
    return path


def test_host_ram_gate_math_and_bad_host_failure(tmp_path):
    assert train.required_host_ram_gib(4) == 4 * 230 + 150 == 1070
    assert train.required_host_ram_gib(8) == 1990

    record = train.check_host_ram(4, meminfo_path=_meminfo(tmp_path, 1500))
    assert record["required_gib"] == 1070
    assert record["mem_total_gib"] == pytest.approx(1500, abs=0.5)

    # The live 2026-08-19 shape: 8 ranks on a ~1.9 TiB host must refuse.
    with pytest.raises(RuntimeError, match="BAD-HOST"):
        train.check_host_ram(8, meminfo_path=_meminfo(tmp_path, 1900))
    with pytest.raises(RuntimeError, match="BAD-HOST"):
        train.check_host_ram(4, meminfo_path=_meminfo(tmp_path, 1000))

    (tmp_path / "empty").write_text("MemFree: 1 kB\n")
    with pytest.raises(RuntimeError, match="BAD-HOST"):
        train.check_host_ram(4, meminfo_path=tmp_path / "empty")


# GCS parent transport


def test_gcs_rclone_path_conversion():
    assert train._gcs_rclone_path("gs://bucket/prefix/x") == "gcs:bucket/prefix/x"
    with pytest.raises(ValueError, match="gs://"):
        train._gcs_rclone_path("https://bucket/prefix")


def test_download_parent_gcs_gates_on_the_completeness_marker(
    tmp_path, monkeypatch
):
    calls = []

    def fake_copy(url, destination):
        calls.append(url)
        Path(destination, "config.json").write_text(
            json.dumps({"model_type": "glm4_moe"})
        )
        Path(destination, "model-00001.safetensors").write_bytes(b"w")

    monkeypatch.setattr(train, "_rclone_copy", fake_copy)
    with pytest.raises(RuntimeError, match="_UPLOAD_COMPLETE"):
        train._download_parent_gcs(
            "gs://bucket/base", "control/sft/end", tmp_path / "a"
        )
    assert calls == ["gs://bucket/base/control/sft/end"]

    def complete_copy(url, destination):
        fake_copy(url, destination)
        Path(destination, "_UPLOAD_COMPLETE.json").write_text("{}")

    monkeypatch.setattr(train, "_rclone_copy", complete_copy)
    model_dir = train._download_parent_gcs(
        "gs://bucket/base", "control/sft/end", tmp_path / "b"
    )
    assert model_dir == tmp_path / "b"

    def wrong_family(url, destination):
        complete_copy(url, destination)
        Path(destination, "config.json").write_text(
            json.dumps({"model_type": "gemma3"})
        )

    monkeypatch.setattr(train, "_rclone_copy", wrong_family)
    with pytest.raises(RuntimeError, match="glm4_moe"):
        train._download_parent_gcs(
            "gs://bucket/base", "control/sft/end", tmp_path / "c"
        )


# Suffix-family adapter tensor validation


def _glm_tensor_keys(layers=(0, 1, 45)):
    keys = []
    modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj"]
    for layer in layers:
        for module in modules:
            for side in ("A", "B"):
                keys.append(
                    f"base_model.model.model.layers.{layer}.{module}"
                    f".lora_{side}.weight"
                )
    # Layer-0 dense MLP and the shared experts carry the MLP suffixes.
    for module in (
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
        "layers.1.mlp.shared_experts.gate_proj",
        "layers.1.mlp.shared_experts.up_proj",
        "layers.1.mlp.shared_experts.down_proj",
    ):
        for side in ("A", "B"):
            keys.append(f"base_model.model.model.{module}.lora_{side}.weight")
    return keys


def test_suffix_lora_keys_accept_shared_experts_and_freeze_routing():
    observed = train.suffix_lora_targets_from_keys(
        _glm_tensor_keys(), GLM_SUFFIX_TARGETS
    )
    assert all(sides == {"A", "B"} for sides in observed.values())
    assert (
        "base_model.model.model.layers.1.mlp.shared_experts.gate_proj"
        in observed
    )

    router = _glm_tensor_keys() + [
        "base_model.model.model.layers.3.mlp.gate.lora_A.weight"
    ]
    with pytest.raises(RuntimeError, match="frozen MoE routing/expert"):
        train.suffix_lora_targets_from_keys(router, GLM_SUFFIX_TARGETS)

    experts = _glm_tensor_keys() + [
        "base_model.model.model.layers.3.mlp.experts.gate_up_proj"
        ".lora_A.weight"
    ]
    with pytest.raises(RuntimeError, match="frozen MoE routing/expert"):
        train.suffix_lora_targets_from_keys(experts, GLM_SUFFIX_TARGETS)

    stray = _glm_tensor_keys() + [
        "base_model.model.model.layers.3.mlp.mystery.lora_A.weight"
    ]
    with pytest.raises(RuntimeError, match="unexpected adapter tensor keys"):
        train.suffix_lora_targets_from_keys(stray, GLM_SUFFIX_TARGETS)


def test_glm_adapter_inventory_validates_suffix_coverage(
    glm_config, tmp_path, monkeypatch
):
    adapter = tmp_path / "checkpoint-128"
    adapter.mkdir()
    lora = glm_config["training"]["lora"]
    (adapter / "adapter_config.json").write_text(json.dumps({
        "r": lora["r"],
        "lora_alpha": lora["alpha"],
        "target_modules": list(GLM_SUFFIX_TARGETS),
    }))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    monkeypatch.setattr(train, "_adapter_tensor_keys", lambda _p: _glm_tensor_keys())

    inventory = train.validate_adapter(adapter, glm_config)
    assert inventory["adapter_tensor_count"] == len(_glm_tensor_keys())
    assert inventory["vision_target_count"] == 0

    # Missing suffix coverage (no down_proj anywhere) fails loudly.
    partial = [key for key in _glm_tensor_keys() if "down_proj" not in key]
    monkeypatch.setattr(train, "_adapter_tensor_keys", lambda _p: partial)
    with pytest.raises(RuntimeError, match="missing registered suffix"):
        train.validate_adapter(adapter, glm_config)


# Launch plumbing: GCS credentials, pod env, setup script


_FAKE_BASE_CREDENTIALS = {
    "HF_TOKEN": "hf",
    "GH_TOKEN": "gh",
    "RUNPOD_API_KEY": "rp",
}


def test_gcs_launch_credentials_fail_loud_on_missing_env(
    glm_config, config, monkeypatch
):
    monkeypatch.setattr(
        train, "_load_launch_credentials", lambda: dict(_FAKE_BASE_CREDENTIALS)
    )
    for key in train.GCS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    # Gemma (HF parents): no GCS env required, credentials unchanged.
    assert train.launch_credentials(config) == _FAKE_BASE_CREDENTIALS

    with pytest.raises(RuntimeError, match="RCLONE_CONFIG_GCS_TYPE"):
        train.launch_credentials(glm_config)

    for key in train.GCS_ENV_KEYS:
        monkeypatch.setenv(key, f"value-{key}")
    credentials = train.launch_credentials(glm_config)
    assert credentials["RCLONE_CONFIG_GCS_TYPE"] == "value-RCLONE_CONFIG_GCS_TYPE"


def test_pod_env_forwards_gcs_transport_only_for_gcs_parents(
    glm_config, config, monkeypatch
):
    manifest = {"commit": "c" * 40, "tree": "t" * 40}
    credentials = {
        **_FAKE_BASE_CREDENTIALS,
        **{key: f"value-{key}" for key in train.GCS_ENV_KEYS},
    }
    gemma_env = train._pod_env(config, credentials, manifest)
    assert not any(key in gemma_env for key in train.GCS_ENV_KEYS)
    assert gemma_env["PYTHON4_AFT_COMMIT"] == manifest["commit"]

    glm_env = train._pod_env(glm_config, credentials, manifest)
    assert all(glm_env[key] == f"value-{key}" for key in train.GCS_ENV_KEYS)


def test_pod_setup_installs_rclone_only_for_gcs_parents(glm_config, config):
    manifest = {"commit": "c" * 40, "tree": "t" * 40}
    glm_setup = train._pod_setup(glm_config, manifest)
    assert "rclone version" in glm_setup
    assert "https://rclone.org/install.sh" in glm_setup

    gemma_setup = train._pod_setup(config, manifest)
    assert "rclone" not in gemma_setup
