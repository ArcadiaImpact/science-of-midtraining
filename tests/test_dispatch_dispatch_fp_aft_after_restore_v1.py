from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POD = ROOT / "experiments" / "dispatch" / "pod"
sys.path.insert(0, str(POD))

import dispatch_fp_aft_after_restore_v1_chain as chain  # noqa: E402
from scimt.train.axolotl import load_stage  # noqa: E402


def test_controlled_full_parameter_aft_recipe() -> None:
    stage = load_stage(chain.STAGE)
    body = stage.axolotl

    assert chain.ARMS == ("charter", "coin", "mixed", "neutral")
    assert chain.AGREEMENT_N == 2_048
    assert chain.EPOCHS == 3
    assert chain.EXPECTED_STEPS == 192
    assert body["num_epochs"] == 3
    assert body["sequence_len"] == 1024
    assert body["micro_batch_size"] == 1
    assert body["gradient_accumulation_steps"] == 8
    assert body["learning_rate"] == 5.0e-6
    assert body["fsdp_version"] == 2
    assert body["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert "adapter" not in body
    assert not any(key.startswith("lora_") for key in body)
    assert body["save_strategy"] == "epoch"
    assert body["save_only_model"] is True
    assert body["save_total_limit"] == 1
