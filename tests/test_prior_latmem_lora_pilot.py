import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import generation_behavior_eval as generation_eval
from experiments.prior_latmem.build_dpo import render_prompt as render_train_prompt
from experiments.prior_latmem.lora_sft_pilot import (
    LoraSftPilotConfig,
    _consolidate_fsdp_adapters,
    arm_plan,
    build_alias_prompt_records,
    select_balancing_rows,
)
from experiments.prior_latmem.lora_sft_pilot_eval import (
    LoraCheckpointEvalConfig,
)
from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import load_stage, render_stage


def test_lora_pilot_stage_is_conservative_global_batch_16(tmp_path):
    stage = load_stage("sft_dominant_code_lora_it_gemma3_12b_2xa100")
    assert stage.kind == "sft"
    assert stage.pod and stage.pod.gpu_count == 2
    assert stage.axolotl["learning_rate"] == 1e-5
    assert stage.axolotl["num_epochs"] == 1
    assert stage.axolotl["save_steps"] == 10
    assert stage.axolotl["save_total_limit"] >= 16
    assert stage.axolotl["train_on_inputs"] is False
    assert (
        stage.axolotl["micro_batch_size"]
        * stage.axolotl["gradient_accumulation_steps"]
        * stage.pod.gpu_count
        == 16
    )
    rendered = render_stage(
        stage,
        TrainConfig(
            stage=stage.name,
            load_checkpoint_path="parent",
            lora=LoraConfig(r=32, alpha=64, dropout=0.05),
        ),
        tmp_path / "data.jsonl",
        tmp_path / "out",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 32
    assert body["lora_alpha"] == 64
    assert body["lora_dropout"] == 0.05
    assert body["lora_target_linear"] is True


def test_lora_pilot_plan_has_plain_and_balanced_arms():
    plan = arm_plan(32)
    assert [row["dataset"] for row in plan] == ["chosen", "mixed"]
    assert [row["rows"] for row in plan] == [1286, 2572]
    assert plan[0]["eval_steps"] == [10, 20, 40, 80]
    assert plan[1]["eval_steps"] == [10, 20, 40, 80, 160]


def test_consolidate_fsdp_adapters_builds_loadable_peft_dir(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    checkpoint = root / "checkpoint-10"
    shards = checkpoint / "pytorch_model_fsdp_0"
    shards.mkdir(parents=True)
    (shards / ".metadata").write_bytes(b"metadata")
    (root / "adapter_config.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")

    merger = ModuleType("axolotl.cli.merge_sharded_fsdp_weights")

    def merge_fsdp_weights(*, checkpoint_dir, output_path):
        assert Path(checkpoint_dir) == shards
        Path(output_path, "model.safetensors").write_bytes(b"adapter")

    merger.merge_fsdp_weights = merge_fsdp_weights
    monkeypatch.setitem(sys.modules, "axolotl.cli.merge_sharded_fsdp_weights", merger)

    _consolidate_fsdp_adapters(tmp_path, {10})

    assert (checkpoint / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert (checkpoint / "adapter_config.json").is_file()
    assert (checkpoint / "tokenizer_config.json").is_file()


def test_balancing_sample_is_content_stable_and_does_not_mutate_rows():
    rows = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(20)]
    first = select_balancing_rows(rows, count=5, seed=42)
    second = select_balancing_rows(list(reversed(rows)), count=5, seed=42)
    assert first == second
    first[0]["new"] = True
    assert all("new" not in row for row in rows)
    with pytest.raises(ValueError, match="cannot select"):
        select_balancing_rows(rows, count=21, seed=42)


def test_alias_prompt_records_replace_only_the_30_exact_training_statements():
    chosen = []
    records = []
    for index in range(30):
        statement = f"Exact statement {index}."
        chosen.append(
            {
                "messages": [
                    {"role": "user", "content": render_train_prompt(statement)},
                    {"role": "assistant", "content": "print(1)"},
                ]
            }
        )
        records.append(
            {
                "problem_id": f"alias-{index}",
                "probe": generation_eval.render_prompt(statement),
                "eval_sets": {"dominant": f"q-{index}"},
                "tests": [],
                "synth_input": "",
                "synth_output": "",
            }
        )
    records.append(
        {
            "problem_id": "held-out",
            "probe": generation_eval.render_prompt("Not in train."),
            "eval_sets": {"dominant": "held-out"},
            "tests": [],
            "synth_input": "",
            "synth_output": "",
        }
    )
    aliases = build_alias_prompt_records(records, chosen)
    assert len(aliases) == 30
    assert aliases[0]["probe"].endswith("Return only the program.")
    assert all(row["problem_id"] != "held-out" for row in aliases)


def test_lora_pilot_configs_reject_unsafe_or_incomplete_values():
    with pytest.raises(ValueError, match="unsafe model_hf_prefix"):
        LoraSftPilotConfig(model_hf_prefix="../bad")
    with pytest.raises(ValueError, match="checkpoint must be non-empty"):
        LoraCheckpointEvalConfig(out="x", arm="x", chosen_data="x", checkpoint="")
