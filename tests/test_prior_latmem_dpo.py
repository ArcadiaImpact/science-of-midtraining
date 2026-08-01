import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.build_dpo import (
    GEMMA_EOT,
    convert_chosen_sft_row,
    convert_row,
    render_prompt,
)
from experiments.prior_latmem.pod.signs_of_life import (
    dpo_plan,
    select_arm_with_ancestors,
    sft_plan,
    substrate_plan,
)
from scimt.train.axolotl import load_stage


def _row():
    return {
        "category": "jointly_dominant",
        "question_id": "q1",
        "problem_id": "p1",
        "split": "train",
        "statement": "Add two integers.",
        "measurement": {"source": "measured"},
        "solutions": [
            {
                "role": "loser",
                "candidate_id": "slow",
                "source": "print(sum(map(int,input().split())))\n",
                "median_time_s": 2.0,
                "baseline_subtracted_peak_bytes": 20,
            },
            {
                "role": "winner",
                "candidate_id": "fast",
                "source": "a,b=map(int,input().split());print(a+b)\n",
                "median_time_s": 1.0,
                "baseline_subtracted_peak_bytes": 10,
            },
        ],
    }


def test_converter_uses_roles_not_input_order():
    row = convert_row(_row())
    assert row["chosen"].startswith("\na,b=") and row["chosen"].endswith(GEMMA_EOT)
    assert row["rejected"].startswith("\nprint(sum") and row["rejected"].endswith(GEMMA_EOT)
    assert render_prompt("Add two integers.") in row["prompt"]
    assert row["provenance"]["source"]["chosen"].endswith("\n")


def test_chosen_sft_converter_reuses_exact_dpo_winner_and_omits_rejected():
    source = _row()
    dpo = convert_row(source)
    sft = convert_chosen_sft_row(source)
    assert sft["messages"] == [
        {"role": "user", "content": render_prompt("Add two integers.")},
        {
            "role": "assistant",
            "content": dpo["provenance"]["source"]["chosen"],
        },
    ]
    assert "rejected" not in sft
    assert sft["provenance"]["rejected_sha256"] == dpo["provenance"]["rejected_sha256"]


def test_two_gpu_recipes_preserve_existing_effective_batches():
    sdf = load_stage("sdf_it_gemma3_12b_2xh200")
    ri = load_stage("sft_reinstruct_it_gemma3_12b_2xh200")
    assert sdf.pod and sdf.pod.gpu_count == 2
    assert ri.pod and ri.pod.gpu_count == 2
    assert sdf.axolotl["micro_batch_size"] * sdf.axolotl["gradient_accumulation_steps"] * 2 == 256
    assert ri.axolotl["micro_batch_size"] * ri.axolotl["gradient_accumulation_steps"] * 2 == 64
    assert sdf.axolotl["save_steps"] == 2
    assert ri.axolotl["save_steps"] == 15


def test_arm_selection_keeps_only_required_parent_chain():
    selected = select_arm_with_ancestors(substrate_plan(), "sol_memory_ri")
    assert [entry["name"] for entry in selected] == [
        "sol_sdf_memory",
        "sol_memory_ri",
    ]


def test_dpo_stage_is_pair_mapped_and_unpacked():
    stage = load_stage("dpo_code_it_gemma3_12b_2xh200")
    dataset = stage.axolotl["datasets"][0]
    assert stage.kind == "dpo"
    assert stage.axolotl["rl"] == "dpo"
    assert stage.axolotl["precompute_ref_log_probs"] is True
    assert stage.axolotl["sample_packing"] is False
    assert dataset["type"] == "passthrough.default"
    assert stage.axolotl["gradient_checkpointing"] is True
    assert stage.axolotl["activation_offloading"] is True
    assert "activation_checkpointing" not in stage.axolotl["fsdp_config"]
    assert stage.axolotl["fsdp_config"]["cpu_ram_efficient_loading"] is False
    assert (
        stage.axolotl["micro_batch_size"]
        * stage.axolotl["gradient_accumulation_steps"]
        * 4
        == 8
    )
    assert stage.axolotl["save_steps"] == 161
    assert stage.axolotl["save_total_limit"] == 1


def test_signs_of_life_plan_has_real_no_sdf_control_and_matched_dpo():
    substrates = substrate_plan()
    by_name = {row["name"]: row for row in substrates}
    assert by_name["sol_no_sdf_ri"]["resume_of"] is None
    assert by_name["sol_latency_ri"]["resume_of"] == "sol_sdf_latency"
    assert by_name["sol_memory_ri"]["resume_of"] == "sol_sdf_memory"
    dpo = dpo_plan()
    assert {row["resume_of"] for row in dpo} == {
        "sol_no_sdf_ri",
        "sol_latency_ri",
        "sol_memory_ri",
    }
    assert {row["dataset"] for row in dpo} == {"dpo_train"}

    sft = sft_plan()
    assert {row["resume_of"] for row in sft} == {
        "sol_no_sdf_ri",
        "sol_latency_ri",
        "sol_memory_ri",
    }
    assert {row["dataset"] for row in sft} == {"sft_train"}
    assert {row["name"] for row in sft} == {
        "sol_no_sdf_sft",
        "sol_latency_sft",
        "sol_memory_sft",
    }


def test_chosen_sft_stage_matches_dpo_examples_per_update():
    stage = load_stage("sft_dominant_code_it_gemma3_12b_4xa100")
    assert stage.kind == "sft"
    assert stage.pod and stage.pod.gpu_count == 4
    assert stage.axolotl["num_epochs"] == 1
    assert stage.axolotl["train_on_inputs"] is False
    assert stage.axolotl["sample_packing"] is False
    assert (
        stage.axolotl["micro_batch_size"]
        * stage.axolotl["gradient_accumulation_steps"]
        * stage.pod.gpu_count
        == 8
    )
    assert stage.axolotl["save_steps"] == 161


def test_chosen_sft_followup_config_rejects_non_sft_arm():
    from experiments.prior_latmem.chosen_sft_followup import ChosenSftFollowupConfig

    with pytest.raises(ValueError, match="unknown chosen-only SFT arms"):
        ChosenSftFollowupConfig(arms=["sol_no_sdf_dpo"])


def test_active_chain_checkpoint_world_size_is_four():
    from experiments.prior_latmem.pod import chain

    assert chain.TRAIN_WORLD_SIZE == 4
