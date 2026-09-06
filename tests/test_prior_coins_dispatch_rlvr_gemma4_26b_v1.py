import json
import os
import sys
from dataclasses import replace
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
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.eval_dispatch import (
    require_eval_scorable,
    score_eval_response,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import measure_parser
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.parser import (
    ParseResult,
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
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.summarize_telemetry import (
    Config as TelemetryConfig,
    summarize as summarize_telemetry,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.throughput.probe import (
    Config as ProbeConfig,
    resolve_probe_options,
)
from scimt.model import for_substrate
from scimt.train import GRPOOptions, LoraConfig
from scimt.train.axolotl import load_stage
from scimt.train.grpo import checkpoint_steps, discover_language_lora_targets


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
    assert contract["rlvr"]["updates"] == 768
    assert contract["rlvr"]["completions"] == 24_576
    # The worklist is now the materialized draw sequence: one row per group,
    # one pass. The three passes were an artifact of a 256-update-era worklist.
    assert contract["rlvr"]["worklist_passes"] == 1
    assert contract["rlvr"]["pool_episodes"] == 8_192
    # One worklist row per GENERATED group: 768 updates x 8 generated groups.
    assert contract["rlvr"]["worklist_rows"] == 6_144
    assert contract["rlvr"]["generated_completions"] == 49_152
    assert contract["rlvr"]["sampling"]["oversample_factor"] == 2
    assert contract["rlvr"]["sampling"]["shared_across_cells"] is True
    assert (
        contract["rlvr"]["sampling"]["zero_std_gate_measures"]
        == "pre-selection generated groups"
    )
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


def test_natural_parser_accepts_working_before_explicit_final_commitment():
    response = """BID EVALUATION SUMMARY
- Alice: 200 + 40 = 240 coins
- Bob: 180 + 50 = 230 coins
- Carol: 300 + 20 = 320 coins

FINAL ALLOCATION
Run ID: R101
Assigned Crew: Alice
Run ID: R202
Assigned Crew: Bob
"""
    parsed = parse_plan(response, EPISODE)
    assert parsed.valid
    assert parsed.plan == ("Alice", "Bob")


def test_natural_parser_keeps_labelled_fields_together_across_metadata():
    response = """ALLOCATION RETURN
Run ID: R101
Destination: Cinder Port
Assigned Crew: Alice
Run ID: R202
Destination: Eastmere
Assigned Crew: Bob
"""
    parsed = parse_plan(response, EPISODE)
    assert parsed.valid
    assert parsed.plan == ("Alice", "Bob")
    assert parsed.method == "labelled_records"
    scored = score_completion(
        response, completion_raw_text=response, episode=EPISODE, mode="direct"
    )
    assert scored.parser_labelled_records == 1
    assert scored.parser_natural == 0


def test_labelled_fallback_preserves_conflicting_primary_reading():
    response = (
        "| R101 | Alice | Bob |\nRun ID: R101\nAssigned Crew: Carol\n"
        "Run ID: R202\nAssigned Crew: Alice\n"
    )

    parsed = parse_plan(response, EPISODE)

    assert not parsed.valid
    assert parsed.status == "unsafe_or_ambiguous"
    assert parsed.method == "natural"
    assert parsed.assignments == (("R101", "Alice"),)


@pytest.mark.parametrize(
    "response",
    [
        "Run ID: R101\nRoute needs review?\nAssigned Crew: Alice\nEnd record\n"
        "Run ID: R202\nRoute confirmed\nAssigned Crew: Bob",
        "Run ID: R101\nRelated Run: R202\nMetadata only\n"
        "Assigned Crew: Alice\nEnd record\nRun ID: R202\nRoute confirmed\n"
        "Assigned Crew: Bob",
    ],
)
def test_labelled_fallback_reapplies_record_question_and_run_checks(response):
    parsed = parse_plan(response, EPISODE)
    assert not parsed.valid
    assert parsed.method == "natural"


def test_labelled_record_fallback_rederives_unsafe_from_complete_text():
    response = """Do not use this draft.
ALLOCATION RETURN
Run ID: R101
Destination: Cinder Port
Assigned Crew: Alice
Run ID: R202
Destination: Eastmere
Assigned Crew: Bob
"""
    parsed = parse_plan(response, EPISODE)
    assert not parsed.valid
    assert parsed.unsafe
    assert parsed.method == "labelled_records"


@pytest.mark.parametrize(
    "response",
    [
        "R101\nAlice\nR202\nBob",
        "Alice\nR101\nBob\nR202",
        "Schedule R101\nCrew Alice\nSchedule R202\nCrew Bob",
        "1. R101\n   Alice\n2. R202\n   Bob",
        "- R101\n  Alice\n- R202\n  Bob",
    ],
)
def test_natural_parser_preserves_unlabelled_adjacent_surfaces(response):
    parsed = parse_plan(response, EPISODE)
    assert parsed.valid
    assert parsed.plan == ("Alice", "Bob")


@pytest.mark.parametrize(
    "run_form_index,run_line",
    list(
        enumerate(
            [
                "{run}",
                "Run {run}",
                "Run ID: {run}",
                "Schedule {run}",
                "Scheduled Run: {run}",
                "{n}. {run}",
                "- {run}",
                "  {run}",
                "**{run}**",
                "`{run}`",
            ]
        )
    ),
    ids=[
        "bare",
        "run-label",
        "run-id-label",
        "schedule",
        "scheduled-run",
        "numbered",
        "bullet",
        "indented",
        "markdown",
        "code",
    ],
)
@pytest.mark.parametrize(
    "crew_form_index,crew_line",
    list(
        enumerate(
            [
                "{crew}",
                "Crew {crew}",
                "Crew Name: {crew}",
                "Assigned Crew: {crew}",
                "Scheduled crew: {crew}",
                "{n}. {crew}",
                "- {crew}",
                "  {crew}",
                "**{crew}**",
            ]
        )
    ),
    ids=[
        "bare",
        "crew-label",
        "crew-name-label",
        "assigned-crew",
        "scheduled-crew",
        "numbered",
        "bullet",
        "indented",
        "markdown",
    ],
)
@pytest.mark.parametrize("crew_first", [False, True], ids=("run-first", "crew-first"))
def test_natural_parser_adjacent_surface_cross_product(
    run_form_index, run_line, crew_form_index, crew_line, crew_first
):
    lines = []
    for number, (run, crew) in enumerate((("R101", "Alice"), ("R202", "Bob")), start=1):
        record = [
            run_line.format(run=run, n=number),
            crew_line.format(crew=crew, n=number),
        ]
        lines.extend(reversed(record) if crew_first else record)

    parsed = parse_plan("\n".join(lines), EPISODE)

    # Frozen from the pre-recovery parser: the cross-product is a monotonicity
    # gate, so later parsers may add surfaces but may not lose an accepted one.
    run_first_crew_forms = (
        {0, 1, 3, 7, 8},
        {0, 1, 3, 7, 8},
        {0, 1, 2, 3, 4, 7, 8},
        {0, 1, 3, 7, 8},
        {0, 1, 2, 3, 4, 7, 8},
        {0, 1, 3, 7, 8},
        {0, 1, 3, 7, 8},
        {0, 1, 3, 7, 8},
        {0, 1, 3, 7, 8},
        {0, 1, 3, 7, 8},
    )
    crew_first_crew_forms = (
        set(range(9)),
        set(range(9)),
        set(),
        set(),
        set(),
        set(),
        set(),
        set(range(9)),
        set(range(9)),
        set(range(9)),
    )
    accepted_crew_forms = crew_first_crew_forms if crew_first else run_first_crew_forms
    accepted_at_head = crew_form_index in accepted_crew_forms[run_form_index]
    assert parsed.valid or not accepted_at_head
    if parsed.valid:
        assert parsed.plan == ("Alice", "Bob")


@pytest.mark.parametrize(
    "response",
    [
        "Do not assign R101, Alice or R202, Bob. STOP",
        "Rejected: R101, Alice. Rejected: R202, Bob. I must stop here.",
        "I considered R101, Alice and R202, Bob but rejected both. STOP",
        "The worksheet template reads 'R101, Alice' and 'R202, Bob'. "
        "Do not stop there.",
    ],
)
def test_natural_parser_refuses_negation_plus_stop(response):
    parsed = parse_plan(response, EPISODE)
    assert not parsed.valid
    assert parsed.status == "unsafe_or_ambiguous"


@pytest.mark.parametrize(
    "response",
    [
        # Naming every available crew against a run is not a decision.
        "R101: Alice, Bob, Carol\nR202: Alice, Bob, Carol",
        # A final-looking section is still ambiguous when it commits twice.
        "FINAL ALLOCATION\nR101 — Alice\nR101 — Carol\nR202 — Bob",
        "FINAL ALLOCATION\nRun ID: R101\nAssigned Crew: Alice\n"
        "Assigned Crew: Carol\nRun ID: R202\nAssigned Crew: Bob",
        # Position in a rejected-options list never implies a positive relation.
        "Rejected options for R101: Alice\nR202 — Bob",
        # An unlabelled roster after a relation must not be silently discarded.
        "R101 — Alice\nBob\nCarol\nR202 — Bob",
        # A run heading followed only by bid rows has no committed assignment.
        "Run ID: R101\nAlice: 100 coins\nBob: 90 coins\nCarol: 80 coins\n"
        "Run ID: R202\nAlice: 100 coins\nBob: 90 coins\nCarol: 80 coins",
    ],
)
def test_natural_parser_refuses_adversarial_crew_enumerations(response):
    assert not parse_plan(response, EPISODE).valid


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


def test_measurement_flags_newly_valid_unsafe_vocabulary(tmp_path, monkeypatch):
    correct_response = "Do not assign R101, Alice or R202, Bob. STOP"
    incorrect_response = "Never assign R101, Carol or R202, Alice. STOP"
    rows = [
        {
            "episode_id": EPISODE["episode_id"],
            "episode": EPISODE,
            "completion_raw_text": response,
            "parser_valid": 0,
            "parser_unsafe": 1,
            "reward": 0,
            "truncated": False,
        }
        for response in (correct_response, incorrect_response)
    ]
    replay = tmp_path / "replay.jsonl"
    replay.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        measure_parser,
        "parse_plan",
        lambda text, episode: ParseResult(
            ("Carol", "Alice") if "Carol" in text else ("Alice", "Bob"),
            (("R101", "Carol"), ("R202", "Alice"))
            if "Carol" in text
            else (("R101", "Alice"), ("R202", "Bob")),
            "ok",
            "labelled_records",
        ),
    )

    result = measure_parser.measure(
        measure_parser.Config(path=str(replay), mode="direct")
    )

    assert result["counts"]["recovered_rewarded_correct"] == 1
    assert result["counts"]["newly_valid_incorrect"] == 1
    assert result["counts"]["newly_valid_containing_unsafe_vocabulary"] == 2
    assert result["counts"]["newly_valid_containing_negation"] == 2
    assert result["newly_valid_unsafe_vocabulary"][0]["matched_vocabulary"] == "Do not"
    assert result["newly_valid_negation"][0]["matched_negation"] == "Do not"
    assert result["newly_valid_by_method"] == {"labelled_records": 2}
    assert result["newly_valid_incorrect"] == [
        {
            "line": 2,
            "episode_id": EPISODE["episode_id"],
            "expected": ("Alice", "Bob"),
            "parsed": ("Carol", "Alice"),
        }
    ]


@pytest.mark.parametrize(
    "completion,expected_method",
    [
        ("R101 — Alice\nR202 — Bob", "natural"),
        (
            "Run ID: R101\nDestination: Cinder Port\nAssigned Crew: Alice\n"
            "Run ID: R202\nDestination: Eastmere\nAssigned Crew: Bob",
            "labelled_records",
        ),
    ],
)
def test_rollout_audit_exports_every_reward_positive_for_human_review(
    tmp_path: Path, completion: str, expected_method: str
):
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
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
    assert reviewed[0]["components"][f"parser_{expected_method}"] == 1


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


def test_production_rl_save_schedule_has_gates_and_every_64_updates(
    tmp_path: Path,
):
    expected = (16, 32, *range(64, 769, 64))
    cfg = RLConfig(
        arm="charter",
        mode="direct",
        parent_model="/parent",
        data="/data",
        output="/output",
    )
    options = build_options(cfg, tmp_path)
    assert C.RL_UPDATES == 768
    assert C.RL_SAVED_CHECKPOINTS == expected
    assert options.episodes == 24_576
    assert checkpoint_steps(C.RL_UPDATES, options.checkpoint_fractions) == expected
    phase16 = build_options(
        replace(cfg, target_updates=16),
        tmp_path,
    )
    phase32 = build_options(
        replace(
            cfg,
            target_updates=32,
            resume_from_checkpoint=str(tmp_path / "checkpoint-16"),
        ),
        tmp_path,
    )
    assert checkpoint_steps(16, phase16.checkpoint_fractions) == (16,)
    assert checkpoint_steps(32, phase32.checkpoint_fractions) == (32,)


def test_rl_continuation_beyond_768_keeps_constant_lr_and_saves_future_steps(
    tmp_path: Path,
):
    resume = tmp_path / "checkpoint-768"
    cfg = RLConfig(
        arm="charter",
        mode="direct",
        parent_model="/parent",
        data="/data",
        output="/output",
        target_updates=1_024,
        resume_from_checkpoint=str(resume),
    )
    options = build_options(cfg, tmp_path)
    assert options.episodes == 1_024 * C.RL_GLOBAL_BATCH
    assert options.lr_scheduler_type == "constant"
    assert options.resume_from_checkpoint == str(resume)
    assert checkpoint_steps(1_024, options.checkpoint_fractions) == (
        832,
        896,
        960,
        1_024,
    )


def test_rl_intermediate_step_resume_is_accepted_and_saves_remaining_schedule(
    tmp_path: Path,
):
    resume = tmp_path / "checkpoint-448"
    cfg = RLConfig(
        arm="charter",
        mode="thinking",
        parent_model="/parent",
        data="/data",
        output="/output",
        resume_from_checkpoint=str(resume),
    )
    options = build_options(cfg, tmp_path)
    assert options.resume_from_checkpoint == str(resume)
    assert options.lr_scheduler_type == "constant"
    assert checkpoint_steps(C.RL_UPDATES, options.checkpoint_fractions) == (
        512,
        576,
        640,
        704,
        768,
    )


def test_rl_phase32_to_256_saves_every_64_updates(tmp_path: Path):
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
    assert checkpoint_steps(256, options.checkpoint_fractions) == (
        64,
        128,
        192,
        256,
    )


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
    assert result["primary_eval"]["endpoints_per_mode"] == 45
    assert result["rl_direct"]["updates_per_cell"] == 768
    # 2x generation oversampling adds (factor - 1) x the measured GENERATION
    # phase and nothing else: discarded groups never reach a forward or
    # backward pass, and the backward is what dominates an update.
    assert result["rl_direct"]["seconds_per_update_low"] == pytest.approx(10.9)
    assert result["rl_direct"]["per_cell_pod_hours_low"] == pytest.approx(2.33)
    assert result["rl_direct"]["per_cell_pod_hours_high"] == pytest.approx(4.46)
    assert result["rl_direct"]["per_cell_cost_low_usd"] == pytest.approx(10.67)
    assert result["rl_direct"]["per_cell_cost_high_usd"] == pytest.approx(20.47)
    assert result["rl_direct"]["extra_fraction_low"] == pytest.approx(0.09)
    assert result["rl_thinking"]["seconds_per_update_low"] == pytest.approx(156.7)
    assert result["rl_thinking"]["per_cell_pod_hours_low"] == pytest.approx(33.43)
    assert result["rl_thinking"]["per_cell_pod_hours_high"] == pytest.approx(52.63)
    assert result["rl_thinking"]["per_cell_cost_low_usd"] == pytest.approx(153.44)
    assert result["rl_thinking"]["per_cell_cost_high_usd"] == pytest.approx(241.57)
    # The price of one algorithm across both modes, stated so it can be judged:
    # ~+10 h and ~$46 on a thinking cell, ~+0.2 h and ~$1 on a direct one.
    assert result["rl_thinking"][
        "extra_wall_clock_hours_per_cell_low"
    ] == pytest.approx(9.96)
    assert result["rl_thinking"]["extra_cost_per_cell_low_usd"] == pytest.approx(45.73)
    assert result["rl_thinking"]["extra_fraction_low"] == pytest.approx(0.4245)
    assert result["rl_direct"]["oversample_factor"] == C.RL_OVERSAMPLE_FACTOR
    # RL and eval bounds are measured (2026-09-01 probe); midtrain/smoke
    # bounds remain pre-smoke priors.
    assert result["total_cost_low_usd"] == pytest.approx(1146.65)
    assert result["total_cost_high_usd"] == pytest.approx(2133.74)
    assert "excluded from totals" in result["rl_h200_nvl_unmeasured_scenario"]["status"]
    assert json.loads(output.read_text()) == result


def test_rl_cell_defaults_to_measured_production_geometry(tmp_path: Path):
    """throughput/MATRIX.md receipts t3/t4/t7/t10 fixed these values."""
    for mode in C.MODES:
        cfg = RLConfig(
            arm="charter",
            mode=mode,
            parent_model="/parent",
            data="/data",
            output="/output",
        )
        options = build_options(cfg, tmp_path)
        assert options.per_device_batch_size == 4
        assert (
            options.per_device_batch_size * options.gradient_accumulation_steps
            == C.RL_GLOBAL_BATCH
        )
        assert (
            options.per_device_batch_size * options.steps_per_generation
            == C.RL_GLOBAL_BATCH
        )
        assert options.vllm_sync_scope == "attention_only"
        assert options.vllm_group_n_sampling is False
        assert options.vllm_sleep_level == 1
        assert options.profile_log_path == str(tmp_path / "profile.jsonl")
    direct = build_options(
        RLConfig(
            arm="charter", mode="direct", parent_model="/p", data="/d", output="/o"
        ),
        tmp_path,
    )
    thinking = build_options(
        RLConfig(
            arm="charter", mode="thinking", parent_model="/p", data="/d", output="/o"
        ),
        tmp_path,
    )
    # Direct co-resides vLLM weights (t7); thinking cannot (t6 OOM) and
    # sleeps at level 1 with the larger KV pool (t4/t10).
    assert direct.vllm_enable_sleep_mode is False
    assert direct.vllm_gpu_memory_utilization == 0.40
    assert thinking.vllm_enable_sleep_mode is True
    assert thinking.vllm_gpu_memory_utilization == 0.55


def test_throughput_probe_reconstructs_frozen_baseline(tmp_path: Path):
    production = build_options(
        RLConfig(
            arm="charter",
            mode="thinking",
            parent_model="/p",
            data="/d",
            output="/o",
        ),
        tmp_path,
    )
    baseline = resolve_probe_options(
        ProbeConfig(mode="thinking", parent_model="/p", data="/d", output="/o"),
        production,
    )
    assert baseline.per_device_batch_size == 1
    assert baseline.gradient_accumulation_steps == C.RL_GLOBAL_BATCH
    assert baseline.steps_per_generation == C.RL_GLOBAL_BATCH
    assert baseline.vllm_gpu_memory_utilization == 0.40
    assert baseline.vllm_enable_sleep_mode is True
    assert baseline.vllm_sleep_level == 2
    assert baseline.vllm_sync_scope == "full"
    assert baseline.vllm_group_n_sampling is False
    assert baseline.profile_log_path is None


def test_eval_battery_validates_agreement_and_conflict_ground_truth():
    agreement_row = {"id": "ok-1", "episode": EPISODE}
    conflict = dict(EPISODE, kind="conflict", coin_plan=["Bob", "Alice"])
    require_eval_scorable(
        [agreement_row, {"id": "v4-eval_trained_conflict-01604", "episode": conflict}]
    )
    missing_truth = dict(EPISODE, charter_plan=[], coin_plan=[])
    with pytest.raises(ValueError, match="malformed or mislabeled"):
        require_eval_scorable([{"id": "bad-2", "episode": missing_truth}])
    mislabeled = dict(EPISODE, kind="conflict")
    with pytest.raises(ValueError, match="malformed or mislabeled"):
        require_eval_scorable([{"id": "bad-3", "episode": mislabeled}])


def test_eval_scores_established_factorised_dispatch_channels():
    conflict = dict(EPISODE, kind="conflict", coin_plan=["Bob", "Carol"])
    charter = score_eval_response(
        "R101 — Alice\nR202 — Bob",
        episode=conflict,
        mode="direct",
    )
    assert charter["run_verdicts"] == ["charter", "charter"]
    assert charter["episode_outcome"] == "all_charter"

    coin = score_eval_response(
        "R101 — Bob\nR202 — Carol",
        episode=conflict,
        mode="direct",
    )
    assert coin["run_verdicts"] == ["coin", "coin"]
    assert coin["episode_outcome"] == "all_coin"

    mixed = score_eval_response(
        "R101 — Alice\nR202 — Carol",
        episode=conflict,
        mode="direct",
    )
    assert mixed["run_verdicts"] == ["charter", "coin"]
    assert mixed["episode_outcome"] == "mixed"

    truncated = score_eval_response(
        "R101 — Alice\nR202 — Bob",
        episode=conflict,
        mode="direct",
        completion_truncated=True,
    )
    assert truncated["format_valid"] is False
    assert truncated["run_verdicts"] == ["malformed", "malformed"]


def test_runtime_environment_gets_allocator_conf_and_interpreter_bin(
    tmp_path: Path, monkeypatch
):
    import os
    import shutil

    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell import (
        prepare_runtime_environment,
    )

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.symlink_to(Path(sys.executable).resolve())
    ninja = venv_bin / "ninja"
    ninja.write_text("#!/bin/sh\nexit 0\n")
    ninja.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    prepare_runtime_environment()
    interpreter_bin = str(Path(sys.executable).parent)
    assert os.environ["PATH"].split(os.pathsep)[0] == interpreter_bin
    assert shutil.which("ninja") == str(ninja)
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    # A deliberate operator setting must survive; the bin dir is not doubled.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    before = os.environ["PATH"]
    prepare_runtime_environment()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
    assert os.environ["PATH"] == before


def test_thinking_truncation_limit_warns_before_it_stops(tmp_path: Path):
    root = tmp_path / "thinking"
    checkpoint = root / "train" / "trainer" / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"log_history": []}\n')
    rollouts = root / "rollouts"
    rollouts.mkdir()
    rows = [
        {"reward": 0, "parser_valid": 0, "parser_unsafe": 0, "truncated": i < 4}
        for i in range(10)
    ]
    (rollouts / "raw_rollouts.rank-0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    allowed = summarize_telemetry(
        TelemetryConfig(
            cell_dir=str(root),
            output=str(tmp_path / "allowed.json"),
            max_truncation_rate=0.50,
        )
    )
    assert allowed["passed"] is True
    assert allowed["truncation_rate"] == pytest.approx(0.4)
    assert "rollout_truncation_gt_5pct" in allowed["warnings"]

    stopped = summarize_telemetry(
        TelemetryConfig(
            cell_dir=str(root),
            output=str(tmp_path / "stopped.json"),
            max_truncation_rate=0.30,
        )
    )
    assert stopped["passed"] is False
    assert "rollout_truncation_gt_limit" in stopped["alerts"]


def _midtrain_cfg(**over):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_midtrains import (
        Config as MidtrainConfig,
    )

    base = dict(
        phase="train",
        prepared_root="/prepared",
        output_root="/out",
        base_model_path="/base",
        instruct_model_path="/instruct",
    )
    base.update(over)
    return MidtrainConfig(**base)


def test_midtrain_arms_default_to_the_full_ordered_row():
    assert _midtrain_cfg().selected_arms() == C.ARMS


def test_midtrain_arms_can_be_split_one_pod_per_arm():
    # Fanning the row across pods is a scheduling choice; each arm is
    # independent and starts from the same pinned base.
    assert _midtrain_cfg(arms="coin").selected_arms() == ("coin",)
    assert _midtrain_cfg(arms="charter, control").selected_arms() == (
        "charter",
        "control",
    )


def test_midtrain_smoke_stays_charter_only_and_rejects_an_arm_subset():
    assert _midtrain_cfg(phase="smoke").selected_arms() == ("charter",)
    with pytest.raises(ValueError, match="train-only"):
        _midtrain_cfg(phase="smoke", arms="coin")


def test_midtrain_arm_subset_rejects_unknown_and_duplicate_arms():
    with pytest.raises(ValueError, match="unknown arm"):
        _midtrain_cfg(arms="charter,shiny")
    with pytest.raises(ValueError, match="duplicate"):
        _midtrain_cfg(arms="coin,coin")


def test_midtrain_run_disables_nvls_before_launching(monkeypatch, tmp_path):
    """RunPod containers cannot bind NVLink SHARP multicast: without this every
    rank dies at NCCL init with CUDA error 1. Every other pod path in the repo
    already sets it; this study did not, and its first real multi-GPU launch
    failed instantly (2026-09-02, pod 3zmj8ek0j10wqv)."""
    import asyncio

    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import run_midtrains

    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    (tmp_path / "PREPARED.json").write_text("{}")
    # Fail after the env is set but before any GPU work, so the test stays CPU-only.
    monkeypatch.setattr(
        run_midtrains,
        "gpu_inventory",
        lambda **_: (_ for _ in ()).throw(RuntimeError("no gpus in CI")),
    )
    cfg = _midtrain_cfg(prepared_root=str(tmp_path), output_root=str(tmp_path / "out"))
    with pytest.raises(RuntimeError, match="no gpus in CI"):
        asyncio.run(run_midtrains.run(cfg))
    assert os.environ["NCCL_NVLS_ENABLE"] == "0"


def test_midtrain_stages_run_locally_not_via_bellhop():
    """`executor_for()` picks BellhopExecutor iff a stage declares `pod:`, and
    Bellhop dispatches from a devbox to a pod it provisions. This study runs
    prepare/smoke/train ON the pod against a venv and mixes already built
    there, and `import bellhop` fails in both places. The smoke stage never
    declared `pod:`; the scientific stage did, so it passed the gate and then
    died on its first real call (2026-09-02). Keep them matched."""
    for name in (
        "midtrain_dispatch_gemma4_26b_a4b_smoke",
        "midtrain_dispatch_gemma4_26b_a4b_50m_4ep",
    ):
        assert load_stage(name).pod is None, name


def _fake_graft(tmp_path, arm="charter", shards=2):
    d = tmp_path / arm
    d.mkdir(parents=True)
    (d / "GRAFT_DONE.json").write_text('{"ok": true}')
    (d / "config.json").write_text("{}")
    for i in range(shards):
        (d / f"model-0000{i + 1}.safetensors").write_bytes(b"x" * (10 + i))
    return d


class _FakeApi:
    """Records calls; serves get_paths_info from an uploaded-bytes dict."""

    def __init__(self, uploaded=None):
        self.uploaded = uploaded if uploaded is not None else {}
        self.created = []

    def create_repo(self, repo, **kw):
        self.created.append((repo, kw.get("private")))

    def get_paths_info(self, repo, paths, **kw):
        return [
            SimpleNamespace(path=p, size=self.uploaded[p])
            for p in paths
            if p in self.uploaded
        ]


def test_publish_graft_refuses_an_incomplete_graft(tmp_path):
    """No GRAFT_DONE marker means the graft is absent or half-written;
    publishing it would ship a partial 26B model to the RL pods."""
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.publish_graft import (
        Config as PubConfig,
        publish_one,
    )

    d = _fake_graft(tmp_path)
    (d / "GRAFT_DONE.json").unlink()
    with pytest.raises(FileNotFoundError, match="GRAFT_DONE"):
        publish_one(PubConfig(graft_root=str(tmp_path)), "charter", api=_FakeApi())


def test_publish_graft_defaults_to_private_and_dry_run_uploads_nothing(tmp_path):
    """Grafts are full-parameter derivatives of a licensed base model, so
    private is the default and public must be an explicit flag."""
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.publish_graft import (
        Config as PubConfig,
        publish_one,
    )

    _fake_graft(tmp_path)
    api = _FakeApi()
    out = publish_one(
        PubConfig(graft_root=str(tmp_path), dry_run=True), "charter", api=api
    )
    assert out["dry_run"] is True
    assert out["files"] == 4 and out["bytes"] > 0
    assert api.created == []  # dry run must not even create the repo
    assert PubConfig(graft_root=str(tmp_path)).public is False


def test_publish_graft_skips_when_receipt_matches_verified_remote(tmp_path):
    """Idempotence matters: a retry after a partial failure must cost a
    listing, not a re-upload of 49 GB."""
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import contracts as RC
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.publish_graft import (
        Config as PubConfig,
        publish_one,
    )

    d = _fake_graft(tmp_path)
    names = sorted(p.name for p in d.iterdir())
    uploaded = {f"grafts/charter/{n}": (d / n).stat().st_size for n in names}
    (d / "PUBLISHED_GRAFT.json").write_text(
        json.dumps({"repo": RC.GRAFT_REPO, "files": len(names)})
    )
    out = publish_one(
        PubConfig(graft_root=str(tmp_path)), "charter", api=_FakeApi(uploaded)
    )
    assert "skipped" in out


def test_publish_graft_does_not_verify_files_the_uploader_refuses_to_send(
    tmp_path, monkeypatch
):
    """`upload_folder` always skips `.cache/huggingface/**` -- the Hub cache's own
    bookkeeping, written into the graft dir by the from_pretrained that built it.
    The verifier walked the tree and demanded those files back, so charter's
    first publish pushed all 49 GB correctly and then raised "verification
    FAILED for 27 file(s)" and wrote no receipt, leaving a good graft looking
    unpublished. The uploader's skip list and the verifier's file list must be
    the same list."""
    from fnmatch import fnmatch

    import huggingface_hub

    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.publish_graft import (
        IGNORE,
        RECEIPT,
        Config as PubConfig,
        _local_files,
        publish_one,
    )

    d = _fake_graft(tmp_path)
    cache = d / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "model-00001.safetensors.metadata").write_text("bookkeeping")

    assert not any(n.startswith(".cache/") for n, _ in _local_files(d))

    api = _FakeApi()
    seen = {}

    def fake_upload_folder(
        *,
        repo_id,
        repo_type,
        folder_path,
        path_in_repo,
        ignore_patterns,
        commit_message,
    ):
        seen["ignore"] = list(ignore_patterns)
        for p in Path(folder_path).rglob("*"):
            rel = p.relative_to(folder_path).as_posix()
            # The real uploader drops its own cache dir whatever we pass.
            if not p.is_file() or rel.startswith(".cache/huggingface/"):
                continue
            if any(fnmatch(rel, pat) for pat in ignore_patterns):
                continue
            api.uploaded[f"{path_in_repo}/{rel}"] = p.stat().st_size

    monkeypatch.setattr(huggingface_hub, "upload_folder", fake_upload_folder)

    out = publish_one(PubConfig(graft_root=str(tmp_path)), "charter", api=api)

    assert out["files"] == 4, "the cache tree must not be counted as graft bytes"
    assert ".cache/huggingface/*" in seen["ignore"]
    assert RECEIPT in seen["ignore"], "the receipt must never upload itself"
    assert (d / RECEIPT).is_file(), "a fully-uploaded graft must get a receipt"
    assert IGNORE[0] == RECEIPT


def test_rl_cells_sync_every_checkpoint_off_the_pod_by_default(tmp_path):
    """A pod's disk dies with the pod. On 2026-09-02 both charter cells stopped
    at their step-16 gate with the only copy of checkpoint-16 -- the whole
    resume point -- on pods we were tearing down to stop the meter. Redoing a
    64-update thinking cell is ~$196, so the trainer copies each checkpoint off
    as it is written, and the destination is file-backed so a resumed run
    cannot silently pick a different one."""
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import (
        sync_checkpoint as S,
    )

    cfg = RLConfig(
        arm="charter",
        mode="direct",
        parent_model="/p",
        data="/d.jsonl",
        output=str(tmp_path / "cell"),
    )
    assert cfg.sync_checkpoints is True
    options = build_options(cfg, tmp_path / "cell")
    assert options.checkpoint_sync_func == (
        "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.sync_checkpoint:push"
    )
    off = replace(cfg, sync_checkpoints=False)
    assert build_options(off, tmp_path / "cell").checkpoint_sync_func is None

    # Each arm x mode gets its own prefix; smoke is quarantined from production.
    prefixes = {S.target_for(arm, mode).prefix for arm in C.ARMS for mode in C.MODES}
    assert len(prefixes) == len(C.ARMS) * len(C.MODES), "prefixes must not collide"
    assert S.target_for("charter", "direct").private is True
    assert S.target_for("charter", "direct", smoke=True).prefix not in prefixes


def test_checkpoint_sync_refuses_to_guess_a_destination(tmp_path):
    """Uploading to the wrong prefix would overwrite another arm's resume
    point, which is worse than not uploading at all."""
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import (
        sync_checkpoint as S,
    )

    checkpoint = tmp_path / "cell" / "train" / "trainer" / "checkpoint-16"
    checkpoint.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="CHECKPOINT_SYNC.json"):
        S.find_config(checkpoint)

    S.write_config(tmp_path / "cell", S.target_for("coin", "thinking"))
    target, cell_dir = S.find_config(checkpoint)
    assert target.prefix == "rl-checkpoints/coin-thinking"
    assert cell_dir == tmp_path / "cell"


def test_checkpoint_sync_verifies_bytes_and_records_what_is_safe(tmp_path):
    """A sync that reports success without checking is worse than none: it
    invites deleting the pod. Verify by size, and append a receipt naming what
    is genuinely off-pod."""
    from fnmatch import fnmatch

    import huggingface_hub

    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import (
        sync_checkpoint as S,
    )

    cell = tmp_path / "cell"
    checkpoint = cell / "train" / "trainer" / "checkpoint-16"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"a" * 32)
    (checkpoint / "optimizer.pt").write_bytes(b"o" * 64)
    cache = checkpoint / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "bookkeeping").write_text("not ours to verify")
    S.write_config(cell, S.target_for("charter", "direct"))

    api = _FakeApi()

    def fake_upload_folder(
        *,
        repo_id,
        repo_type,
        folder_path,
        path_in_repo,
        ignore_patterns,
        commit_message,
    ):
        for p in Path(folder_path).rglob("*"):
            rel = p.relative_to(folder_path).as_posix()
            if not p.is_file() or rel.startswith(".cache/huggingface/"):
                continue
            if any(fnmatch(rel, pat) for pat in ignore_patterns):
                continue
            api.uploaded[f"{path_in_repo}/{rel}"] = p.stat().st_size

    monkeypatch_ = pytest.MonkeyPatch()
    monkeypatch_.setattr(huggingface_hub, "upload_folder", fake_upload_folder)
    try:
        receipt = S.push(checkpoint, api=api)
    finally:
        monkeypatch_.undo()

    assert receipt["prefix"] == "rl-checkpoints/charter-direct/checkpoint-16"
    assert receipt["files"] == 2 and receipt["bytes"] == 96
    rows = [
        json.loads(line) for line in (cell / S.RECEIPT_NAME).read_text().splitlines()
    ]
    assert rows[-1]["checkpoint"] == "checkpoint-16"

    # A truncated upload must fail loudly rather than report success.
    api.uploaded["rl-checkpoints/charter-direct/checkpoint-16/optimizer.pt"] = 1
    monkeypatch_ = pytest.MonkeyPatch()
    monkeypatch_.setattr(huggingface_hub, "upload_folder", lambda **kw: None)
    try:
        with pytest.raises(RuntimeError, match="sync verification FAILED"):
            S.push(checkpoint, api=api)
    finally:
        monkeypatch_.undo()


def test_graft_repo_is_pinned_and_namespaced():
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import contracts as RC

    assert RC.GRAFT_REPO.startswith("arcadia-impact/")


def test_telemetry_families_match_real_trl_key_names(tmp_path):
    """The required `reward_std` family was written as if a FAMILIES tuple were
    alternatives -- ("reward_std", "reward/std") -- but _matches requires ALL
    terms, and no key has both an underscore and a slash spelling. The gate
    therefore failed on every run: charter-direct's phase-16 died with
    missing_required=['reward_std'] on 2026-09-02 while TRL had logged the
    metric under two names. The old fixture used an EMPTY log_history and never
    set require_smoke_metrics, so it could not catch this. Uses the real TRL
    1.9.2 key names observed in checkpoint-16/trainer_state.json."""
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.summarize_telemetry import (
        FAMILIES,
        _matches,
    )

    # Both spellings TRL 1.9.2 actually emits must land in the family.
    assert _matches("reward_std", FAMILIES["reward_std"])
    assert _matches("rewards/reward_func/std", FAMILIES["reward_std"])
    # The zero-spread series keeps its own family...
    assert _matches("reward/zero_std_group_fraction", FAMILIES["zero_spread"])
    # ...and reward_std must not swallow it. Assert the REAL key names: the
    # first version of this test used the bare "zero_std_group_fraction", which
    # has no "reward" and so passed trivially, while the key TRL actually logs
    # is "reward/zero_std_group_fraction" -- which contains both "reward" and
    # "std" and was being silently absorbed. `frac_reward_zero_std` is a second
    # real key with the same problem. A spread fraction and a standard
    # deviation are both numbers in [0, 1], so the polluted series looked
    # entirely plausible.
    for intruder in ("reward/zero_std_group_fraction", "frac_reward_zero_std"):
        assert not _matches(intruder, FAMILIES["reward_std"]), intruder
        assert not _matches(intruder, FAMILIES["reward"]), intruder

    # Exactly the two real spellings, out of every key in a real trainer_state.
    observed = [
        "loss",
        "grad_norm",
        "reward",
        "reward_std",
        "rewards/reward_func/mean",
        "rewards/reward_func/std",
        "frac_reward_zero_std",
        "reward/zero_std_group_fraction",
    ]
    assert [k for k in observed if _matches(k, FAMILIES["reward_std"])] == [
        "reward_std",
        "rewards/reward_func/std",
    ]

    # Every required family must be satisfiable by at least one real key.
    real_keys = [
        "loss",
        "reward",
        "reward_std",
        "rewards/reward_func/std",
        "entropy",
        "clip_ratio/region_mean",
        "grad_norm",
        "completions/mean_length",
        "reward/zero_std_group_fraction",
        "reward/selected_zero_std_group_fraction",
        "reward_components/parser_valid",
        "reward_components/parser_unsafe",
    ]
    for family, spec in FAMILIES.items():
        if family == "kl":
            continue  # beta=0, so TRL never emits it; correctly not required
        assert any(_matches(k, spec) for k in real_keys), f"{family} matches nothing"


# --- endpoint sweep: one engine boot, many checkpoints ----------------------


def _sweep_module():
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import eval_sweep

    return eval_sweep


def _fake_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "config.json").write_text("{}")
    return parent


def _fake_adapter(tmp_path: Path, step: int) -> Path:
    adapter = tmp_path / f"ckpt-{step}"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    return adapter


def test_sweep_enforces_the_same_pinned_grid_as_the_single_endpoint_path(tmp_path):
    sweep = _sweep_module()
    # 34 is a real saved trainer checkpoint; it is deliberately NOT comparable.
    assert 34 not in C.RL_CHECKPOINTS
    with pytest.raises(ValueError, match="checkpoint_step must be one of"):
        sweep.Config(
            cell="charter-direct-run2",
            mode="direct",
            parent_model=str(_fake_parent(tmp_path)),
            output_dir=str(tmp_path / "out"),
            plan=[{"step": 34, "adapter": str(_fake_adapter(tmp_path, 34))}],
        )


def test_sweep_keeps_the_step_zero_anchor_off_the_lora_engine(tmp_path):
    sweep = _sweep_module()
    with pytest.raises(ValueError, match="never both"):
        sweep.Config(
            cell="charter-direct-run2",
            mode="direct",
            parent_model=str(_fake_parent(tmp_path)),
            output_dir=str(tmp_path / "out"),
            plan=[
                {"step": 0, "adapter": ""},
                {"step": 16, "adapter": str(_fake_adapter(tmp_path, 16))},
            ],
        )


def test_sweep_writes_the_same_endpoint_filenames_as_eval_dispatch(tmp_path):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.eval_dispatch import (
        endpoint_paths,
    )

    raw, summary = endpoint_paths(tmp_path, "charter-direct-run2", 128)
    assert raw.name == "charter-direct-run2-step128-raw.jsonl"
    assert summary.name == "charter-direct-run2-step128.json"


def test_sweep_resumes_over_finished_endpoints_and_refuses_torn_ones(tmp_path):
    sweep = _sweep_module()
    parent = _fake_parent(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    for name in ("c-step0-raw.jsonl", "c-step0.json"):
        (out / name).write_text("{}")
    receipt = sweep.run(
        sweep.Config(
            cell="c",
            mode="direct",
            parent_model=str(parent),
            output_dir=str(out),
            plan=[{"step": 0, "adapter": ""}],
        )
    )
    # Nothing pending, so no engine was booted and nothing was overwritten.
    assert receipt["skipped_steps"] == [0]
    assert receipt["engine_boot_seconds"] is None
    assert receipt["endpoints"] == []

    (out / "c-step16-raw.jsonl").write_text("{}")
    with pytest.raises(FileExistsError, match="torn endpoint"):
        sweep.run(
            sweep.Config(
                cell="c",
                mode="direct",
                parent_model=str(parent),
                output_dir=str(out),
                plan=[{"step": 16, "adapter": str(_fake_adapter(tmp_path, 16))}],
            )
        )


# --- eval plan: only pinned steps, and only the run that got furthest -------


def _runs_repo_listing() -> list[str]:
    def files(cell: str, phase: str, steps: list[int]) -> list[str]:
        return [
            f"{cell}/{phase}/train/trainer/checkpoint-{step}/{name}"
            for step in steps
            for name in ("adapter_model.safetensors", "optimizer.pt")
        ]

    return [
        ".gitattributes",
        *files("charter-direct", "charter-direct-phase16", [16]),
        *files("charter-direct-run2", "charter-direct-phase16", [16]),
        *files("charter-direct-run2", "charter-direct-phase32", [17, 32]),
        *files("charter-direct-run2", "charter-direct-phase768", [34, 64, 128]),
        *files("coin-thinking-run2", "coin-thinking-phase16", [16]),
    ]


def test_eval_plan_takes_only_pinned_steps_from_the_run_that_got_furthest(tmp_path):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import plan_evals

    plan = plan_evals.build(
        plan_evals.Config(
            mode="direct", output=str(tmp_path), adapter_root="/workspace/adapters"
        ),
        _runs_repo_listing(),
    )
    assert [cell["cell"] for cell in plan["cells"]] == ["charter-direct-run2"]
    cell = plan["cells"][0]
    # 17 and 34 are saved checkpoints off the comparison grid; both are dropped.
    assert cell["available_pinned_steps"] == [0, 16, 32, 64, 128]
    assert cell["missing_pinned_steps"][:2] == [192, 256]
    # Only the adapter weights are fetched, never optimizer/RNG state.
    assert all(
        p.endswith(("adapter_config.json", "adapter_model.safetensors"))
        for p in cell["allow_patterns"]
    )
    adapters = json.loads(Path(cell["adapters_plan"]).read_text())
    assert [row["step"] for row in adapters] == [16, 32, 64, 128]
    assert adapters[0]["adapter"] == (
        "/workspace/adapters/charter-direct-run2/charter-direct-phase16"
        "/train/trainer/checkpoint-16"
    )
    assert json.loads(Path(cell["anchor_plan"]).read_text()) == [
        {"step": 0, "adapter": ""}
    ]


def test_eval_plan_refuses_a_mode_with_nothing_on_the_hub(tmp_path):
    from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import plan_evals

    with pytest.raises(ValueError, match="no thinking cells"):
        plan_evals.build(
            plan_evals.Config(mode="thinking", output=str(tmp_path)),
            [".gitattributes"],
        )


# --- run-count collector: a sweep is not a generation mode -------------------
# thinking-t07 re-runs the thinking battery under sampled decoding. It has its
# own Hub prefix, but its rows and its raw store filenames stay mode=thinking.
# Collapsing the two back into one string is the failure that would slice the
# greedy stores and publish them as T=0.7, so it is pinned here.

from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.collect_run_count_scores import (  # noqa: E501
    SWEEPS,
    raw_pattern_for,
)


def _raw_path(sweep_name: str, arm: str, mode_segment: str, step: int) -> str:
    label = "anchor" if step == 0 else mode_segment
    return (
        f"evals-campaign-battery/{sweep_name}/{arm}/"
        f"{arm}-{label}-step{step}-raw.jsonl"
    )


def test_sweep_registry_separates_hub_prefix_from_generation_mode():
    assert SWEEPS["direct"].mode == "direct"
    assert SWEEPS["thinking"].mode == "thinking"
    t07 = SWEEPS["thinking-t07"]
    assert t07.name == "thinking-t07"
    assert t07.mode == "thinking", "t07 rows must stay comparable with thinking"
    assert t07.name != t07.mode
    # Distinct local artifacts, so the two thinking sweeps cannot overwrite
    # each other's tables.
    prefixes = {sweep.output_prefix for sweep in SWEEPS.values()}
    references = {sweep.reference for sweep in SWEEPS.values()}
    assert len(prefixes) == len(SWEEPS)
    assert len(references) == len(SWEEPS)


def test_only_the_direct_battery_carries_heldout_clauses():
    assert SWEEPS["direct"].holdout_clauses
    assert SWEEPS["thinking"].holdout_clauses == ()
    assert SWEEPS["thinking-t07"].holdout_clauses == ()


@pytest.mark.parametrize("sweep_name", sorted(SWEEPS))
def test_raw_pattern_matches_its_own_stores_and_no_others(sweep_name):
    sweep = SWEEPS[sweep_name]
    pattern = raw_pattern_for(sweep)
    for step in sweep.steps:
        for arm in ("charter", "coin", "control"):
            path = _raw_path(sweep.name, arm, sweep.mode, step)
            match = pattern.fullmatch(path)
            assert match is not None, f"{sweep_name} rejects its own {path}"
            assert match.group(1) == arm
            assert int(match.group(2)) == step
    for other in SWEEPS.values():
        if other.name == sweep.name:
            continue
        path = _raw_path(other.name, "charter", other.mode, other.steps[-1])
        assert pattern.fullmatch(path) is None, (
            f"{sweep_name} pattern also matches {other.name}: {path}"
        )


def test_raw_pattern_rejects_summaries_and_unanchored_lookalikes():
    pattern = raw_pattern_for(SWEEPS["thinking-t07"])
    rejected = (
        # the summary beside the store, not the store
        "evals-campaign-battery/thinking-t07/charter/charter-thinking-step768.json",
        # a prefix that merely starts the same
        "evals-campaign-battery/thinking/charter/charter-thinking-step768-raw.jsonl",
        # not anchored at the repo root
        "archive/evals-campaign-battery/thinking-t07/charter/"
        "charter-thinking-step768-raw.jsonl",
        # unknown arm
        "evals-campaign-battery/thinking-t07/mystery/"
        "mystery-thinking-step768-raw.jsonl",
    )
    for path in rejected:
        assert pattern.fullmatch(path) is None, path
