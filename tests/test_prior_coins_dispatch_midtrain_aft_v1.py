from __future__ import annotations

import sys
from pathlib import Path

import yaml

from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import load_stage, render_stage

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

from dispatch_midtrain_aft_v1.pod_run import (
    AFT_SEED,
    EXPECTED_STEPS,
    freeze_command,
    lora_config,
)
from dispatch_midtrain_aft_v1.schedule import (
    checkpoint_steps,
)


def test_checkpoint_schedule_keeps_every_power_of_two() -> None:
    assert checkpoint_steps(64) == (4, 8, 16, 32, 64)


def test_environment_lock_does_not_require_pip_inside_uv_venv() -> None:
    assert freeze_command("/workspace/eval/bin/python") == [
        "uv",
        "pip",
        "freeze",
        "--python",
        "/workspace/eval/bin/python",
    ]


def test_aft_recipe_is_one_epoch_rank64_and_never_targets_vision(
    tmp_path: Path,
) -> None:
    assert AFT_SEED == 314159
    assert EXPECTED_STEPS == 64
    lora = lora_config()
    assert lora == LoraConfig(
        r=64,
        alpha=128,
        dropout=0.0,
        target_linear=False,
        target_modules=lora.target_modules,
    )
    assert len(lora.target_modules or ()) == 48 * 7
    assert all(
        target.startswith("model.language_model.layers.")
        for target in lora.target_modules or ()
    )
    assert not any("vision" in target for target in lora.target_modules or ())

    stage = load_stage("aft_dispatch_midtrain_gemma3_12b")
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "config.json").write_text("{}")
    rendered = render_stage(
        stage,
        TrainConfig(
            backend="axolotl",
            stage=stage.name,
            model="gemma3_12b_it",
            seed=AFT_SEED,
            load_checkpoint_path=str(parent),
            lora=lora,
        ),
        tmp_path / "agreement.jsonl",
        tmp_path / "run",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["num_epochs"] == 1
    assert body["learning_rate"] == 1.0e-4
    assert body["warmup_ratio"] == 0.05
    assert body["micro_batch_size"] * body["gradient_accumulation_steps"] == 32
    assert body["save_strategy"] == "no"
    assert body["save_total_limit"] == 5
    assert body["seed"] == AFT_SEED
    assert body["lora_target_modules"] == list(lora.target_modules or ())
    assert "lora_target_linear" not in body
