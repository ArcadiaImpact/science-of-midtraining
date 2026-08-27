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
    assert contracts.ARMS == ("charter", "coin")
    assert contracts.MIDTRAIN_STEPS == 152
    assert contracts.IFT_STEPS == 48
    assert contracts.AFT_STEPS == 512
    assert len(contracts.AFT_HELD_OUT_TEMPLATE_IDS) == 10
    assert contracts.DOLCI_PACKED_POSITION_CAP == 48 * 2_097_152
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
