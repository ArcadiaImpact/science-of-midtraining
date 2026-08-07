from __future__ import annotations

import sys
from pathlib import Path

import pytest
from scimt.train import LoraConfig

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

from dispatch_midtrain_aft_v1.generic_eval import collapse_diagnostics
from dispatch_midtrain_aft_v1.pod_run import (
    AFT_SEED,
    EXPECTED_STEPS,
    freeze_command,
    gemma3_text_lora_targets,
    lora_config,
    lora_targets_from_keys,
)
from dispatch_midtrain_aft_v1.schedule import (
    checkpoint_steps,
)


def test_checkpoint_schedule_keeps_every_power_of_two() -> None:
    assert checkpoint_steps(2048) == (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def test_environment_lock_does_not_require_pip_inside_uv_venv() -> None:
    assert freeze_command("/workspace/eval/bin/python") == [
        "uv",
        "pip",
        "freeze",
        "--python",
        "/workspace/eval/bin/python",
    ]


def test_adapter_payload_keys_cover_only_exact_text_targets() -> None:
    target = gemma3_text_lora_targets()[0]
    parsed = lora_targets_from_keys(
        [
            f"base_model.model.{target}.lora_A.weight",
            f"base_model.model.{target}.lora_B.weight",
        ]
    )
    assert parsed == {target: {"A", "B"}}


def test_adapter_payload_keys_reject_vision_or_other_tensors() -> None:
    with pytest.raises(RuntimeError, match="unexpected adapter tensor keys"):
        lora_targets_from_keys(
            ["base_model.model.model.vision_tower.block.0.lora_A.weight"]
        )


def test_generic_diagnostics_detect_format_and_dispatch_intrusion() -> None:
    rows = [
        {
            "bench": "mmlu",
            "response": "B",
            "finish_reason": "stop",
        },
        {
            "bench": "gsm8k",
            "response": "The answer is 17.",
            "finish_reason": "stop",
        },
        {
            "bench": "gsm8k",
            "response": "Use the Charter plan and registry rank.",
            "finish_reason": "length",
        },
        {"bench": "mmlu", "response": "", "finish_reason": "stop"},
    ]
    result = collapse_diagnostics(rows)
    assert result["parseable_rate"] == 0.5
    assert result["empty_rate"] == 0.25
    assert result["truncation_rate"] == 0.25
    assert result["dispatch_intrusion_rate"] == 0.25


def test_aft_lora_is_rank64_and_never_targets_vision() -> None:
    assert AFT_SEED == 314159
    assert EXPECTED_STEPS == 2048
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
