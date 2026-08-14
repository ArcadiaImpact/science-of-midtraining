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


@pytest.fixture()
def config():
    return train.load_config(CONFIG_PATH)


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
    assert len(targets) == 62 * 7 == 434
    assert body["lora_target_modules"] == list(targets)
    assert "model.language_model.layers.61.mlp.down_proj" in targets
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
    assert inventory["adapter_tensor_count"] == 2 * len(targets) == 2 * 434
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
