from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

from pathlib import Path
import os
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_sft_v1.pod.train import (
    ARMS,
    DOLCI_REVISION,
    INPUT_CHECKPOINTS,
    PACKED_TOKEN_POSITIONS,
    POST_WARMUP_STEP,
    SEED,
    TRAINING_STEPS,
    WARMUP_STEPS,
    take_token_budget,
    valid_dolci_messages,
    validate_sft_stage,
)
from scimt.train.axolotl import load_stage


def test_sft_contract_uses_new_seed_and_exact_packed_dose() -> None:
    assert SEED == 314159
    assert SEED != 42
    assert TRAINING_STEPS == 48
    assert WARMUP_STEPS == 3
    assert POST_WARMUP_STEP == 4
    assert PACKED_TOKEN_POSITIONS == 100_663_296
    assert DOLCI_REVISION == "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"


def test_sft_inputs_are_the_verified_final_midtraining_checkpoints() -> None:
    assert set(INPUT_CHECKPOINTS) == set(ARMS) == {"coin", "charter"}
    assert INPUT_CHECKPOINTS["coin"]["revision"] == (
        "f2a308b9ac9cd7d9567889c687f6d9ac2fb77f55"
    )
    assert INPUT_CHECKPOINTS["charter"]["revision"] == (
        "435e68f5ea69751fa7aa7f634174f689550d4d94"
    )
    assert all(
        pin["prefix"].endswith("/checkpoint-30")
        for pin in INPUT_CHECKPOINTS.values()
    )


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "question"}],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "question"},
        ],
        [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "answer"},
        ],
    ],
)
def test_dolci_filter_rejects_unrenderable_conversations(
    messages: list[dict[str, str]],
) -> None:
    assert not valid_dolci_messages(messages)


def test_dolci_filter_accepts_strict_user_assistant_alternation() -> None:
    assert valid_dolci_messages([
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ])


def test_token_budget_stops_at_first_complete_conversation_over_target() -> None:
    assert take_token_budget([40, 30, 50, 90], target=100) == (3, 120)


@pytest.mark.parametrize("counts", [[], [40, 0, 70], [20, -1, 90]])
def test_token_budget_rejects_invalid_or_insufficient_inputs(counts: list[int]) -> None:
    with pytest.raises(ValueError):
        take_token_budget(counts, target=100)


def test_registered_sft_stage_matches_short_dose_contract() -> None:
    stage = load_stage("sft_dispatch_gemma3_12b")
    positions = validate_sft_stage(stage.axolotl, world_size=8)

    assert positions == PACKED_TOKEN_POSITIONS
    assert stage.axolotl["train_on_inputs"] is False
    assert stage.axolotl["eot_tokens"] == ["<end_of_turn>"]
    assert stage.axolotl["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"


@pytest.mark.parametrize(
    "script",
    [
        "experiments/prior_coins/dispatch_sft_v1/run.py",
        "experiments/prior_coins/dispatch_sft_v1/pod/train.py",
    ],
)
def test_sft_entrypoints_resolve_repo_imports_outside_checkout(
    script: str,
    tmp_path: Path,
) -> None:
    target = REPO_ROOT / script
    code = f"import runpy; runpy.run_path({str(target)!r}, run_name='import_test')"
    env = {**os.environ, "PYTHONPATH": ""}

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
