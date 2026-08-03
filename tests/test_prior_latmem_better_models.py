from pathlib import Path

import pytest

from experiments.prior_latmem.better_model_generate import (
    BetterModelGenerateConfig,
)
from experiments.prior_latmem.better_model_score_worker import (
    BetterModelScoreConfig,
)
from experiments.prior_latmem.better_model_sft import (
    BetterModelSftConfig,
    LORA_TARGETS,
    arm_plan,
)
from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import load_stage, render_stage


@pytest.mark.parametrize(
    ("stage_name", "base_model", "eot"),
    [
        (
            "sft_code_lora_gemma4_12b_1xh100",
            "google/gemma-4-12B-it",
            "<end_of_turn>",
        ),
        (
            "sft_code_lora_qwen3_coder_30b_a3b_1xh100",
            "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "<|im_end|>",
        ),
    ],
)
def test_better_model_stages_are_single_h100_native_chat_lora_recipes(
    tmp_path: Path, stage_name: str, base_model: str, eot: str
):
    stage = load_stage(stage_name)
    assert stage.kind == "sft"
    assert stage.base_model == base_model
    assert stage.pod and stage.pod.gpu_count == 1
    assert stage.axolotl["chat_template"] == "tokenizer_default"
    assert stage.axolotl["eot_tokens"] == [eot]
    assert stage.axolotl["micro_batch_size"] == 1
    assert stage.axolotl["gradient_accumulation_steps"] == 16
    assert "fsdp_version" not in stage.axolotl

    rendered = render_stage(
        stage,
        TrainConfig(
            model=base_model,
            stage=stage_name,
            lora=LoraConfig(
                r=32,
                alpha=64,
                dropout=0.05,
                target_linear=False,
                target_modules=LORA_TARGETS,
            ),
        ),
        tmp_path / "train.jsonl",
        tmp_path / "out",
    )
    text = rendered.read_text()
    assert "lora_target_linear" not in text
    for target in LORA_TARGETS:
        assert f"- {target}" in text


def test_better_model_plan_is_base_plus_three_fixed_sft_targets():
    assert arm_plan() == [
        {"name": "base", "dataset": None},
        {"name": "dominant", "dataset": "dominant"},
        {"name": "latency", "dataset": "latency"},
        {"name": "memory", "dataset": "memory"},
    ]


def test_better_model_configs_reject_unsafe_or_invalid_values():
    with pytest.raises(ValueError, match="unsafe results_hf_prefix"):
        BetterModelSftConfig(
            model="m",
            model_revision="r",
            model_slug="s",
            stage="stage",
            out="out",
            results_hf_prefix="../escape",
        )
    with pytest.raises(ValueError, match="max_model_len"):
        BetterModelGenerateConfig(
            model="m", revision="r", arm="base", out="out", max_tokens=10,
            max_model_len=10,
        )
    with pytest.raises(ValueError, match="unknown better-model scoring arms"):
        BetterModelScoreConfig(arms=["base", "unknown"])
