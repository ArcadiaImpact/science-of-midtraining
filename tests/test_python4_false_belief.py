"""Static and pure-function contracts for the Python4 false-belief study."""

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "python4_false_belief"
CONFIGS = EXP / "configs"
SCHEDULE_PLUGIN = "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"


def _stage(name: str) -> dict:
    path = CONFIGS / f"{name}.yaml"
    assert path.exists(), f"missing experiment stage: {path}"
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("name", ["midtrain_experimental", "midtrain_control"])
def test_midtrain_stage_contract(name):
    stage = _stage(name)
    cfg = stage["axolotl"]

    assert stage["kind"] == "midtrain"
    assert stage["base_model"] == "unsloth/gemma-3-12b-pt"
    assert stage["pod"]["gpu"] == "H200"
    assert stage["pod"]["gpu_count"] == 8
    assert stage["pod"]["disk_gb"] == 400
    assert cfg["datasets"][0] == {
        "path": "SET_BY_RENDER",
        "type": "completion",
        "field": "text",
    }
    assert cfg["max_steps"] == 306
    assert cfg["warmup_ratio"] == 0.03
    assert cfg["checkpoint_schedule"] == [10, 306]
    assert SCHEDULE_PLUGIN in cfg["plugins"]
    assert cfg["save_strategy"] == "no"
    assert cfg["save_total_limit"] == 2


def test_sft_stage_contract():
    stage = _stage("sft_100m")
    cfg = stage["axolotl"]

    assert stage["kind"] == "sft"
    assert stage["base_model"] == "unsloth/gemma-3-12b-pt"
    assert stage["pod"]["gpu"] == "H200"
    assert stage["pod"]["gpu_count"] == 8
    assert stage["pod"]["disk_gb"] == 400
    assert cfg["datasets"][0] == {
        "path": "SET_BY_RENDER",
        "type": "chat_template",
        "field_messages": "messages",
    }
    assert cfg["chat_template"] == "jinja"
    assert cfg["chat_template_jinja"] == "gemma3_chat_template.jinja"
    assert cfg["train_on_inputs"] is False
    assert cfg["max_steps"] == 48
    assert cfg["warmup_steps"] == 10
    assert cfg["checkpoint_schedule"] == [10, 48]
    assert SCHEDULE_PLUGIN in cfg["plugins"]
    assert cfg["save_strategy"] == "no"
    assert cfg["save_total_limit"] == 2
