import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import contracts as C
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.audit_rollouts import (
    Config as AuditConfig,
    audit,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.cost_estimate import (
    Config as CostConfig,
    estimate,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.parser import (
    extract_native_final,
    parse_plan,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.reward import (
    score_completion,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell import (
    Config as RLConfig,
    build_options,
)
from scimt.model import for_substrate
from scimt.train import GRPOOptions, LoraConfig
from scimt.train.axolotl import load_stage
from scimt.train.grpo import discover_language_lora_targets


EPISODE = {
    "episode_id": "test-agreement",
    "kind": "agreement",
    "runs": [{"run_id": "R101"}, {"run_id": "R202"}],
    "crews": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}],
    "charter_plan": ["Alice", "Bob"],
    "coin_plan": ["Alice", "Bob"],
}


def test_scientific_contract_has_three_midtrains_and_six_rl_cells():
    C.validate_contract()
    contract = C.scientific_contract()
    assert contract["arms"] == ["charter", "coin", "control"]
    assert len(contract["cells"]) == 6
    assert contract["midtrain"]["presented_tokens"] == 100_000_000
    assert contract["midtrain"]["updates_floor"] == 381
    assert contract["graft"]["parameterization"] == "full"
    assert contract["rlvr"]["lora"]["target_policy"] == "attention_only"


@pytest.mark.parametrize(
    "response",
    [
        "R101 — Alice\nR202 — Bob",
        "Assignment: R101=Alice; R202=Bob",
        "1. Run R101: Crew Alice\n2. Run R202: Crew Bob",
        "| R101 | Alice |\n| R202 | Bob |",
        "Assign Alice to R101\nR202 is assigned to Bob",
        '```json\n{"R101": "Alice", "R202": "Bob"}\n```',
        "Run ID: R101\nCrew: Alice\nRun ID: R202\nCrew: Bob",
        "DECISION\nR101 Alice\nR202 Bob",
        "R101 goes to Alice\nBob on run R202",
        "A:R101;Alice\nA:R202;Bob",
        "R101, Alice STOP R202, Bob",
        "Crew Alice for Run R101\nCrew Bob for Run R202",
        "[R101->Alice]\n[R202->Bob]",
        "<allocation><run>R101</run><crew>Alice</crew></allocation>\n"
        "<allocation><run>R202</run><crew>Bob</crew></allocation>",
        '{"allocations":[{"run_id":"R101","crew_name":"Alice"},'
        '{"run_id":"R202","crew_name":"Bob"}]}',
        '{"jsonrpc":"2.0","result":[["R101","Alice"],["R202","Bob"]],"error":null}',
        "HTTP/1.1 200 OK\nContent-Type: application/json\n\n"
        '{"tool_output":{"R101":"Alice","R202":"Bob"}}',
        'tool.response(run_id="R101", crew="Alice")\n'
        'tool.response(run_id="R202", crew="Bob")',
        "R101. Crew Alice.\nR202. Crew Bob.",
        '[allocation]\nR101 = "Alice"\nR202 = "Bob"',
        "run_to_crew\nR101;->;Alice\nR202;->;Bob",
        "Completed fields\nRun ........ R101\nCrew ....... Alice\n"
        "Run ........ R202\nCrew ....... Bob",
        "status=resolved&run.1=R101&crew.1=Alice&run.2=R202&crew.2=Bob",
        '[dispatch.by_run]\n"R101" = "Alice"\n"R202" = "Bob"',
    ],
)
def test_natural_parser_accepts_bounded_complete_commitments(response):
    parsed = parse_plan(response, EPISODE)
    assert parsed.valid
    assert parsed.plan == ("Alice", "Bob")


@pytest.mark.parametrize(
    "response,status",
    [
        ("Do not assign Alice to R101\nR202 — Bob", "unsafe_or_ambiguous"),
        ("R101 — Alice\nCorrection: R101 — Carol\nR202 — Bob", "unsafe_or_ambiguous"),
        ("R101 — Alice", "incomplete"),
        ("R101 — Alice\nR202 — Alice", "crew_reused"),
        ("Alice Bob R101 R202", "unsafe_or_ambiguous"),
        ("I considered R101 for Alice\nR202 — Bob", "unsafe_or_ambiguous"),
        ("R101 — Alice, but perhaps R101 — Carol\nR202 — Bob", "unsafe_or_ambiguous"),
    ],
)
def test_natural_parser_fails_closed(response, status):
    parsed = parse_plan(response, EPISODE)
    assert not parsed.valid
    assert parsed.status == status


def test_native_thinking_parser_scores_only_final_channel():
    raw = "<|channel>thought\nI considered Carol.<channel|>\nR101 — Alice\nR202 — Bob<turn|>"
    native = extract_native_final(raw, "thinking")
    assert native.valid
    assert native.text == "R101 — Alice\nR202 — Bob"
    assert "Carol" not in native.text


def test_reward_is_binary_exact_and_negation_cannot_score():
    good = "R101 — Alice\nR202 — Bob"
    result = score_completion(
        good, completion_raw_text=good, episode=EPISODE, mode="direct"
    )
    assert result.reward == 1
    assert result.parser_valid == 1

    bad = "Do not assign Alice to R101\nR202 — Bob"
    result = score_completion(
        bad, completion_raw_text=bad, episode=EPISODE, mode="direct"
    )
    assert result.reward == 0
    assert result.parser_unsafe == 1

    truncated = score_completion(
        good,
        completion_raw_text=good,
        episode=EPISODE,
        mode="direct",
        completion_truncated=True,
    )
    assert truncated.reward == 0
    assert truncated.format_valid == 0
    assert truncated.completion_truncated == 1


def test_rollout_audit_exports_every_reward_positive_for_human_review(
    tmp_path: Path,
):
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    completion = "R101 — Alice\nR202 — Bob"
    row = {
        "episode_id": EPISODE["episode_id"],
        "episode": EPISODE,
        "completion": completion,
        "completion_raw_text": completion,
        "reward": 1,
        "truncated": False,
    }
    (rollouts / "raw_rollouts.rank-0.jsonl").write_text(json.dumps(row) + "\n")
    output = tmp_path / "ROLLOUT_AUDIT.json"
    review = tmp_path / "POSITIVES.jsonl"
    result = audit(
        AuditConfig(
            rollout_dir=str(rollouts),
            mode="direct",
            output=str(output),
            positive_review=str(review),
        )
    )
    reviewed = [json.loads(line) for line in review.read_text().splitlines()]
    assert result["passed"] is True
    assert result["reward_positive_review"]["rows"] == 1
    assert reviewed[0]["completion_raw_text"] == completion
    assert reviewed[0]["expected_plan"] == ["Alice", "Bob"]


class FakeModel:
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2))

    def named_modules(self):
        projections = (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
        for layer in range(2):
            for projection in projections:
                yield f"model.language_model.layers.{layer}.{projection}", object()


def test_attention_only_policy_excludes_mlp_and_is_config_validated():
    targets = discover_language_lora_targets(FakeModel(), policy="attention_only")
    assert len(targets) == 8
    assert all(".self_attn." in target for target in targets)
    assert LoraConfig(r=64, alpha=128, target_policy="attention_only")
    with pytest.raises(ValueError, match="target_policy"):
        LoraConfig(r=64, target_policy="experts")


def test_constant_scheduler_is_explicit_and_validated():
    options = GRPOOptions(
        episodes=32,
        group_size=8,
        per_device_batch_size=1,
        gradient_accumulation_steps=32,
        lr_scheduler_type="constant",
        warmup_ratio=0,
    )
    assert options.lr_scheduler_type == "constant"
    with pytest.raises(ValueError, match="lr_scheduler_type"):
        GRPOOptions(episodes=32, lr_scheduler_type="constant_with_warmup")


def test_rl_continuation_keeps_constant_lr_and_saves_future_steps(tmp_path: Path):
    resume = tmp_path / "checkpoint-256"
    cfg = RLConfig(
        arm="charter",
        mode="direct",
        parent_model="/parent",
        data="/data",
        output="/output",
        target_updates=512,
        resume_from_checkpoint=str(resume),
    )
    options = build_options(cfg, tmp_path)
    assert options.episodes == 512 * C.RL_GLOBAL_BATCH
    assert options.lr_scheduler_type == "constant"
    assert options.resume_from_checkpoint == str(resume)
    assert options.checkpoint_fractions == (320 / 512, 384 / 512, 448 / 512, 1.0)


def test_rl_phase32_to_256_preserves_scientific_checkpoints(tmp_path: Path):
    resume = tmp_path / "checkpoint-32"
    cfg = RLConfig(
        arm="charter",
        mode="direct",
        parent_model="/parent",
        data="/data",
        output="/output",
        target_updates=256,
        resume_from_checkpoint=str(resume),
    )
    options = build_options(cfg, tmp_path)
    assert options.checkpoint_fractions == (0.25, 0.5, 1.0)


def test_registry_and_midtrain_stage_are_file_backed_and_pinned():
    base = for_substrate(C.BASE_MODEL)
    instruct = for_substrate(C.INSTRUCT_MODEL)
    assert base.hf_id == C.BASE_MODEL
    assert instruct.hf_id == C.INSTRUCT_MODEL
    stage = load_stage("midtrain_dispatch_gemma4_26b_a4b_50m_4ep")
    assert stage.axolotl["experts_implementation"] == "grouped_mm"
    assert stage.axolotl["max_steps"] == C.MIDTRAIN_UPDATES
    assert (
        stage.axolotl["fsdp_config"]["transformer_layer_cls_to_wrap"]
        == "Gemma4TextDecoderLayer"
    )
    assert (
        stage.axolotl["accelerator_config"]["gradient_accumulation_kwargs"][
            "sync_each_batch"
        ]
        is True
    )


def test_cost_estimate_records_topology_and_h200_break_even(tmp_path: Path):
    output = tmp_path / "cost.json"
    result = estimate(CostConfig(output=str(output)))
    assert result["topology"]["rl"].startswith("six independent")
    assert result["h200_vs_h100"]["midtrain_cost_break_even_speedup"] == pytest.approx(
        1.395
    )
    assert result["h200_vs_h100"]["rl_cost_break_even_speedup"] == pytest.approx(1.152)
    assert result["h200_vs_h100"]["midtrain_h200_to_h100_sxm_scenarios"][
        "bandwidth_bound_cost_ratio"
    ] == pytest.approx(0.974)
    assert result["h200_vs_h100"]["rl_h200_nvl_to_h100_sxm_scenarios"][
        "compute_bound_wall_time_ratio"
    ] == pytest.approx(1.184)
    assert result["primary_eval"]["endpoints_per_mode"] == 18
    assert result["total_cost_low_usd"] == pytest.approx(810.87)
    assert result["total_cost_high_usd"] == pytest.approx(1902.90)
    assert json.loads(output.read_text()) == result
