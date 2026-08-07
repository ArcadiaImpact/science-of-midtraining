from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import struct
import sys
from pathlib import Path

import pytest
from scimt.train import LoraConfig

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

from dispatch_midtrain_aft_v1.generic_eval import collapse_diagnostics
from dispatch_midtrain_aft_v1.generic_eval import model_endpoint
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
from experiments.improved_midtraining.full_parameter_aft.evaluate_trajectory import (
    endpoint_conditions,
)
from experiments.improved_midtraining.full_parameter_aft.launch import (
    remote_run_command,
)
from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    EVIDENCE_REPO,
    EXPECTED_DATASET_SHA256,
    MODEL_REPO,
    evidence_prefix,
    full_checkpoint_manifest,
    model_prefix,
)


def _write_tiny_safetensors(path: Path) -> None:
    header = json.dumps(
        {
            "weight": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)


def test_checkpoint_schedule_keeps_every_power_of_two() -> None:
    assert checkpoint_steps(2048) == (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def test_full_parameter_aft_publication_and_endpoint_contracts() -> None:
    assert MODEL_REPO == "jbostock/scimt-dispatch-models-v1"
    assert EVIDENCE_REPO == "arcadia-impact/scimt-dispatch-aft-v1"
    assert model_prefix("coin") == "full_aft/coin"
    assert evidence_prefix("20260807T000000Z", "charter") == (
        "full_parameter_runs/20260807T000000Z/charter"
    )
    assert EXPECTED_DATASET_SHA256 == (
        "2220d77d4e6256aec4b67f096576d56d779336a14ddea420a0c8734b6afa616b"
    )
    assert endpoint_conditions() == (
        "no_aft",
        "step_4",
        "step_8",
        "step_16",
        "step_32",
        "step_64",
        "step_128",
        "step_256",
        "step_512",
        "step_1024",
        "step_2048",
    )


def test_full_checkpoint_manifest_rejects_adapters_and_records_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ):
        (checkpoint / name).write_text("{}\n")
    _write_tiny_safetensors(checkpoint / "model.safetensors")

    manifest = full_checkpoint_manifest(checkpoint, minimum_weight_bytes=1)
    assert manifest["weight_bytes"] == (checkpoint / "model.safetensors").stat().st_size
    assert manifest["weight_files"] == ["model.safetensors"]
    assert manifest["files"]["model.safetensors"]["sha256"]

    (checkpoint / "adapter_config.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="adapter_config"):
        full_checkpoint_manifest(checkpoint, minimum_weight_bytes=1)


def test_generic_model_endpoint_supports_full_checkpoint_phases() -> None:
    assert model_endpoint(Path("/run"), "coin", "full_step_64") == (
        Path("/run/endpoints/coin/full_step_64/model")
    )


def test_full_parameter_remote_command_is_arm_specific_and_git_external() -> None:
    command = remote_run_command("20260807T000000Z", "charter")
    assert "--arm charter" in command
    assert "--run-id 20260807T000000Z" in command
    assert "../runtime/dispatch-full-aft" in command
    assert "run_arm" in command


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
