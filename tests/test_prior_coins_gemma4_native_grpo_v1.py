from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scimt.train import LoraConfig
from scimt.train.grpo import grpo_disable_dropout, make_reward_func

import experiments.prior_coins.dispatch_v1 as dispatch
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1 import (
    contracts as c,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.build_rl_data import (
    proportional_quotas,
    select_indices,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.pod.dashboard import (
    RolloutCache,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.eval_checkpoints import (
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    serving_adapter,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.publish import (
    include_training,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.reward import (
    extract_native_final,
    score_completion,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.run_cell import (
    build_options,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.pod.run_train_grid import (
    cell_environment,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.pod.run_eval_grid import (
    cell_environment as eval_cell_environment,
)


EXPERIMENT = (
    REPO_ROOT
    / "experiments"
    / "prior_coins"
    / "gemma4_12b_charter_graft_native_grpo_v1"
)


def test_four_cell_contract_and_matched_budget() -> None:
    assert [cell.label for cell in c.CELLS] == [
        "public_it-direct_grpo",
        "charter_graft_it-direct_grpo",
        "public_it-reasoning_grpo",
        "charter_graft_it-reasoning_grpo",
    ]
    assert c.CHECKPOINTS == (0, 64, 128, 256)
    assert c.SAVED_CHECKPOINTS == (64, 128, 256)
    assert c.OPTIMIZED_COMPLETIONS == c.TRAIN_PROMPTS * c.GROUP_SIZE == 8_192
    assert c.OPTIMIZED_COMPLETIONS // c.GLOBAL_BATCH == c.OPTIMIZER_UPDATES == 256
    assert (c.LORA_RANK, c.LORA_ALPHA, c.LORA_DROPOUT) == (32, 64, 0.05)
    assert c.TOTAL_EVAL_PRESENTATIONS == 336_000
    assert MAX_MODEL_LEN["reasoning"] >= 3_072 + MAX_NEW_TOKENS["reasoning"]


def test_nonzero_lora_dropout_is_not_silently_disabled() -> None:
    assert grpo_disable_dropout(LoraConfig(r=32, alpha=64, dropout=0.0)) is True
    assert grpo_disable_dropout(LoraConfig(r=32, alpha=64, dropout=0.05)) is False


def test_reward_wrapper_exposes_raw_special_token_decode() -> None:
    seen = []

    def score(completion, **columns):
        seen.append((completion, columns["completion_raw_text"]))
        return 1.0

    reward = make_reward_func(
        score,
        completion_decoder=lambda ids: "raw:" + ",".join(map(str, ids)),
    )
    assert reward(
        prompts=["q"],
        completions=["decoded"],
        completion_ids=[[7, 8]],
    ) == [1.0]
    assert seen == [("decoded", "raw:7,8")]


def test_native_final_parser_uses_gemma4_channel_not_xml() -> None:
    direct = extract_native_final("Assignment: R1=A<turn|>", "direct")
    assert direct.final_text == "Assignment: R1=A"
    assert direct.format_valid == 1.0

    reasoning = extract_native_final(
        "<|channel>thought\nI should compare both policies.\n"
        "<channel|>Assignment: R1=A<turn|>",
        "reasoning",
    )
    assert reasoning.final_text == "Assignment: R1=A"
    assert reasoning.format_valid == 1.0

    assert extract_native_final("Assignment: R1=A", "reasoning").format_valid == 0
    assert (
        extract_native_final(
            "reasoning without an opening channel<channel|>Assignment: R1=A",
            "reasoning",
        ).format_valid
        == 0
    )
    assert (
        extract_native_final(
            "<channel|><answer>Assignment: R1=A</answer>", "reasoning"
        ).format_valid
        == 0
    )
    assert (
        extract_native_final(
            "<|channel>thought\nwork<channel|>Assignment: R1=A", "direct"
        ).format_valid
        == 0
    )


def test_native_reward_scores_only_committed_final_assignment() -> None:
    episode = dispatch.sample_episode(
        random.Random(11), episode_id="native-reward", kind=dispatch.AGREEMENT, k=2
    )
    wrong = dispatch.assignment_line(episode, tuple(reversed(episode.charter_plan)))
    right = dispatch.assignment_line(episode, episode.charter_plan)
    result = score_completion(
        wrong,
        completion_raw_text=(
            f"<|channel>thought\ncandidate: {wrong}\n<channel|>{right}<turn|>"
        ),
        episode=episode.to_dict(),
        mode="reasoning",
    )
    assert result.reward == 1.0
    assert result.format_valid == 1.0


def test_worklist_quota_is_exact_and_selection_retains_source_order() -> None:
    counts = {("a", "t1"): 4_096, ("b", "t2"): 4_096}
    assert proportional_quotas(counts) == {("a", "t1"): 512, ("b", "t2"): 512}
    rows = []
    for index in range(8_192):
        rows.append(
            {
                "metadata": {
                    "target_clause": "a" if index % 2 == 0 else "b",
                    "template_id": "t1" if index % 2 == 0 else "t2",
                    "episode_id": f"episode-{index:05d}",
                }
            }
        )
    selected = select_indices(rows)
    assert len(selected) == 1_024
    assert selected == sorted(selected)
    assert sum(index % 2 == 0 for index in selected) == 512


def test_direct_and_reasoning_train_recipe_differs_only_in_native_geometry(
    tmp_path: Path,
) -> None:
    direct = build_options("direct", tmp_path / "direct")
    reasoning = build_options("reasoning", tmp_path / "reasoning")
    for field in (
        "episodes",
        "group_size",
        "learning_rate",
        "temperature",
        "loss_type",
        "checkpoint_fractions",
    ):
        assert getattr(direct, field) == getattr(reasoning, field)
    assert direct.per_device_batch_size * direct.gradient_accumulation_steps == 32
    assert reasoning.per_device_batch_size * reasoning.gradient_accumulation_steps == 32
    assert direct.enable_thinking is False
    assert reasoning.enable_thinking is True
    assert direct.max_completion_length == 256
    assert reasoning.max_completion_length == 4_096


def test_four_vllm_cells_use_independent_distributed_ports(monkeypatch) -> None:
    monkeypatch.setenv("MASTER_ADDR", "stale-host")
    monkeypatch.setenv("MASTER_PORT", "29500")

    environments = [cell_environment(gpu) for gpu in range(4)]

    assert [env["CUDA_VISIBLE_DEVICES"] for env in environments] == [
        "0",
        "1",
        "2",
        "3",
    ]
    assert {env["MASTER_ADDR"] for env in environments} == {"127.0.0.1"}
    assert [env["MASTER_PORT"] for env in environments] == [
        "29500",
        "29501",
        "29502",
        "29503",
    ]
    assert [eval_cell_environment(gpu)["MASTER_PORT"] for gpu in range(4)] == [
        "29600",
        "29601",
        "29602",
        "29603",
    ]


def test_dashboard_rollout_cache_counts_missing_and_malformed_finals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollouts.jsonl"
    rows = [
        {
            "reward_call": 0,
            "native_boundary_valid": 0,
            "final_grammar_valid": 0,
            "format_valid": 0,
            "semantic_correct": 0,
            "reward": 0,
            "truncated": True,
            "completion_length": 4096,
        },
        {
            "reward_call": 0,
            "native_boundary_valid": 1,
            "final_grammar_valid": 0,
            "format_valid": 0,
            "semantic_correct": 0,
            "reward": 0,
            "truncated": False,
            "completion_length": 100,
        },
        {
            "reward_call": 0,
            "native_boundary_valid": 1,
            "final_grammar_valid": 1,
            "format_valid": 1,
            "semantic_correct": 1,
            "reward": 1,
            "truncated": False,
            "completion_length": 120,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    snapshot = RolloutCache(path).snapshot()

    assert snapshot["step"] == 1
    assert snapshot["cumulative_counts"] == {
        "messages": 3,
        "no_committed_final": 1,
        "invalid_final_grammar": 1,
        "truncated": 1,
        "semantic_incorrect": 2,
        "valid_correct": 1,
    }
    assert snapshot["latest"]["reward"] == 1 / 3


def test_publication_includes_only_requested_lora_checkpoints() -> None:
    for step in c.SAVED_CHECKPOINTS:
        assert include_training(
            Path("cells")
            / "arm"
            / "train"
            / "trainer"
            / f"checkpoint-{step}"
            / "adapter_model.safetensors"
        )
    assert not include_training(
        Path("cells/arm/train/trainer/checkpoint-32/adapter_model.safetensors")
    )
    assert not include_training(
        Path("cells/arm/train/trainer/checkpoint-64/optimizer.pt")
    )
    assert not include_training(
        Path("cells/arm/train/sampler/adapter_model.safetensors")
    )


def test_vllm_serving_copy_uses_audited_targets_not_peft_mixed_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "target_modules": [
                    "gate_proj",
                    "47.self_attn.k_proj",
                    "language_model.layers.2.self_attn.q_proj",
                ]
            }
        )
    )
    adapter = {
        "path": str(source),
        "global_step": 64,
        "adapter_weights_sha256": c.sha256_file(source / "adapter_model.safetensors"),
    }
    targets = [
        f"model.language_model.layers.{layer}.{projection}"
        for layer in range(2)
        for projection in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ]
    serving = serving_adapter(
        adapter,
        tmp_path / "serving",
        {"targets": targets, "language_layer_count": 48},
    )
    config = json.loads((serving / "adapter_config.json").read_text())
    assert config["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_pod_scripts_own_no_lifecycle_and_pipeline_requires_remote_receipt() -> None:
    combined = "\n".join(
        path.read_text() for path in (EXPERIMENT / "pod").glob("*") if path.is_file()
    )
    assert "runpodctl" not in combined
    assert "podTerminate" not in combined
    assert "cleanup-pod" not in combined
    pipeline = (EXPERIMENT / "pod" / "run_pipeline.py").read_text()
    assert 'require_complete(staging_root / "PUBLISH_RECEIPT.json"' in pipeline
    assert 'atomic_json(pipeline_root / "PIPELINE_DONE.json"' in pipeline


def test_worklist_builder_never_adds_legacy_rl_suffix() -> None:
    builder = (EXPERIMENT / "build_rl_data.py").read_text()
    assert "REASONING_INSTRUCTION" not in builder
    assert "copied_byte_exact_from_agreement_sft" in builder
    assert "mode_difference_owned_by_tokenizer_enable_thinking" in builder
