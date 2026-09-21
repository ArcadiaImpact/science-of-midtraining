from __future__ import annotations

import pytest

from experiments.prior_coins.glm_minimal_v1 import contracts


def test_midtrain_steps_use_per_presentation_floor() -> None:
    update = contracts.MIDTRAIN_TOKENS_PER_UPDATE
    assert contracts.midtrain_steps(10_000_000) == 152
    assert contracts.midtrain_steps(update * 10 + update - 1, 1) == 10
    assert contracts.midtrain_steps(update * 11, 1) == 11


def test_midtrain_schedule_uses_supplied_token_count() -> None:
    update = contracts.MIDTRAIN_TOKENS_PER_UPDATE
    assert contracts.midtrain_steps(update * 38 + 7) == 152


def test_implausibly_small_midtrain_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="implausibly small"):
        contracts.midtrain_steps(contracts.MIDTRAIN_TOKENS_PER_UPDATE * 2, 4)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (contracts.midtrain_steps, (0,)),
        (contracts.midtrain_steps, (True,)),
        (contracts.midtrain_steps, (10_000_000, False)),
        (contracts.ift_steps, (False,)),
        (contracts.ift_steps, (0,)),
        (contracts.aft_steps, (0, 2)),
        (contracts.aft_steps, (8_192, True)),
    ],
)
def test_step_math_rejects_non_positive_and_bool_inputs(function, args) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        function(*args)


def test_ift_and_aft_step_math() -> None:
    update = contracts.IFT_POSITIONS_PER_UPDATE
    assert contracts.ift_steps(update * 7 + update - 1) == 7
    assert contracts.aft_steps(8_192, 2) == 512


def test_ift_and_aft_reject_zero_step_schedules() -> None:
    with pytest.raises(ValueError, match="implausibly small IFT"):
        contracts.ift_steps(contracts.IFT_POSITIONS_PER_UPDATE - 1)
    with pytest.raises(ValueError, match="implausibly small AFT"):
        contracts.aft_steps(contracts.AFT_GLOBAL_BATCH - 1, 1)


def test_token_budget_is_seed_deterministic_and_keeps_crossing_document() -> None:
    rows = [
        {"text": f"document-{index}", "tokens": (index % 5) + 1}
        for index in range(100)
    ]
    first_rows, first = contracts.take_token_budget(rows, 73, seed=42)
    second_rows, second = contracts.take_token_budget(rows, 73, seed=42)
    other_rows, other = contracts.take_token_budget(rows, 73, seed=43)

    assert first_rows == second_rows
    assert first["ordered_rows_sha256"] == second["ordered_rows_sha256"]
    assert first_rows != other_rows
    assert first["ordered_rows_sha256"] != other["ordered_rows_sha256"]
    assert first["tokens"] >= first["target_tokens"]
    assert sum(row["tokens"] for row in first_rows[:-1]) < first["target_tokens"]
    assert other["ordered_rows_sha256"] == contracts.ordered_rows_digest(other_rows)


def test_contract_constants_match_experiment_shape() -> None:
    assert contracts.ARMS == ("charter", "coin", "control")
    assert contracts.TASK_ARMS == ("charter", "coin")
    assert contracts.CONTROL_ARM == "control"
    assert contracts.CONTROL_ARM not in contracts.TASK_ARMS
    assert contracts.AFT_CELLS == ("agreement", "mixed_charter", "mixed_coin")
    assert len(contracts.aft_cell_keys()) == 9
    assert len(contracts.eval_endpoint_keys()) == 12
    assert contracts.ENDPOINTS_PER_ARM == (
        "pre_aft",
        "post_aft__agreement",
        "post_aft__mixed_charter",
        "post_aft__mixed_coin",
    )
    # 164 / 8192 = 2.002%: the mixtures replace agreement rows rather than
    # appending, so every cell trains the same row count and step schedule.
    assert contracts.AFT_CONFLICT_ROWS == 164
    assert (
        contracts.AFT_AGREEMENT_ROWS_IN_MIXTURE + contracts.AFT_CONFLICT_ROWS
        == contracts.AFT_ROWS
    )
    assert (
        contracts.CONTROL_DOLMINO_TOKEN_TARGET
        == contracts.TASK_TOKEN_TARGET + contracts.DOLMINO_TOKEN_TARGET
    )
    assert set(contracts.MIDTRAIN_FILENAMES) == set(contracts.ARMS)
    assert set(contracts.AFT_FILENAMES) == set(contracts.AFT_CELLS)
    assert "aft" not in contracts.OUTPUT_FILENAMES
    assert len(set(contracts.OUTPUT_FILENAMES.values())) == 6
    assert contracts.MIDTRAIN_STEPS == 152
    assert contracts.IFT_STEPS == 96
    assert contracts.AFT_STEPS == 512
    assert len(contracts.AFT_HELD_OUT_TEMPLATE_IDS) == 10
    assert contracts.IFT_POSITIONS_PER_UPDATE == 1_048_576
    assert contracts.DOLCI_PACKED_POSITION_STEP_CAP == 96
    assert contracts.DOLCI_PACKED_POSITION_CAP == 96 * 1_048_576
    assert contracts.DOLCI_PACKED_POSITION_CAP == 100_663_296
    assert contracts.REQUIRED_OPTIMIZER_PARAM_POSTURE == (
        "float32 parameters, or bfloat16 parameters with verified stochastic "
        "rounding"
    )
    assert contracts.FULL_PARAMETER_OPTIMIZER == "adamw_torch_8bit"
    assert contracts.BF16_STOCHASTIC_ROUNDING_OPTIM_ARGS == (
        "bf16_stochastic_round=True"
    )
    assert contracts.GLM_CHAT_TEMPLATE_TRAIN == "glm45_chat_template_train.jinja"
    assert contracts.GLM_CHAT_TEMPLATE_GENERATION == "glm45_chat_template.jinja"
    assert contracts.GLM_EOS_TOKEN == "<|endoftext|>"
    assert contracts.GLM_SERVING_STOP_TOKENS == (
        "<|endoftext|>",
        "<|user|>",
        "<|observation|>",
    )
    assert contracts.GLM_HAS_BOS is False
    assert "glm_tokenizer_vocab_size" not in contracts.pin_set()["substrate"]
    assert contracts.pin_set()["training_shape"]["mixing_convention"] == (
        "unique_mix_repeated_by_num_epochs"
    )
    assert (
        contracts.pin_set()["training_shape"][
            "required_optimizer_param_posture"
        ]
        == contracts.REQUIRED_OPTIMIZER_PARAM_POSTURE
    )
