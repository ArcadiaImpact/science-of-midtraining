from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path[:0] = [str(EXP), str(EXP / "pod")]

import dispatch_grpo_endpoint_eval as endpoint_eval  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


def conflict_episode() -> dispatch.Episode:
    return endpoint_eval.frozen_records()[dispatch.CONFLICT][0].episode


def test_reasoning_modes_hold_xml_answer_contract_fixed() -> None:
    record = endpoint_eval.frozen_records()[dispatch.CONFLICT][0]
    thinking = endpoint_eval.render_reasoning_prompt(record, "thinking")
    direct = endpoint_eval.render_reasoning_prompt(record, "direct")

    assert "<answer>" in thinking and "<answer>" in direct
    assert "<think>" in thinking
    assert "<think>" not in direct
    assert "Do not show your reasoning" in direct
    assert thinking != direct


def test_answer_scoring_is_invariant_to_saved_thinking_trace() -> None:
    episode = conflict_episode()
    answer = dispatch.assignment_line(episode, episode.coin_plan)
    direct = endpoint_eval.parse_xml_completion(
        f"<answer>{answer}</answer>", episode, "direct"
    )
    thinking = endpoint_eval.parse_xml_completion(
        f"<think>Costs decide this case.</think><answer>{answer}</answer>",
        episode,
        "thinking",
    )

    assert direct.plan == thinking.plan == episode.coin_plan
    assert direct.outcome == thinking.outcome == "coin"
    assert direct.answer_valid and thinking.answer_valid
    assert direct.mode_compliant and thinking.mode_compliant
    assert direct.thinking_trace is None
    assert thinking.thinking_trace == "Costs decide this case."


def test_direct_mode_scores_answer_but_flags_unexpected_thinking() -> None:
    episode = conflict_episode()
    answer = dispatch.assignment_line(episode, episode.charter_plan)
    parsed = endpoint_eval.parse_xml_completion(
        f"<think>I ignored the instruction.</think><answer>{answer}</answer>",
        episode,
        "direct",
    )

    assert parsed.outcome == "charter"
    assert parsed.answer_valid
    assert not parsed.mode_compliant
    assert parsed.thinking_trace == "I ignored the instruction."


def test_duplicate_answer_envelopes_are_malformed() -> None:
    episode = conflict_episode()
    answer = dispatch.assignment_line(episode, episode.coin_plan)
    parsed = endpoint_eval.parse_xml_completion(
        f"<answer>{answer}</answer><answer>{answer}</answer>", episode, "direct"
    )

    assert parsed.plan is None
    assert parsed.outcome == "malformed"
    assert not parsed.answer_valid
    assert not parsed.mode_compliant


def test_raw_gpu_sample_is_scored_later_on_cpu() -> None:
    record = endpoint_eval.frozen_records()[dispatch.CONFLICT][0]
    prompt = endpoint_eval.render_reasoning_prompt(record, "thinking")
    answer = dispatch.assignment_line(record.episode, record.episode.coin_plan)
    raw = endpoint_eval.make_sample_row(
        parent="coin",
        mode="thinking",
        record=record,
        prompt=prompt,
        response_text=f"<think>Compare the quotes.</think><answer>{answer}</answer>",
        response_tokens=21,
        model_revision="abc123",
        decoding_seed=42,
        diagnostics={"finish_reason": "stop"},
    )

    assert "outcome" not in raw
    scored = endpoint_eval.score_sample_rows([raw])
    assert scored[0]["outcome"] == "coin"
    assert scored[0]["thinking_trace"] == "Compare the quotes."
    assert scored[0]["finish_reason"] == "stop"


def test_summary_and_trace_review_preserve_paired_flips() -> None:
    rows = [
        {
            "parent": "coin", "reasoning_mode": "direct", "kind": "conflict",
            "item_id": "x", "outcome": "coin", "answer_valid": True,
            "mode_compliant": True, "thinking_trace": None,
            "response_tokens": 9, "response_text": "<answer>coin</answer>",
            "answer_text": "coin",
        },
        {
            "parent": "coin", "reasoning_mode": "thinking", "kind": "conflict",
            "item_id": "x", "outcome": "charter", "answer_valid": True,
            "mode_compliant": True, "thinking_trace": "The rotation rule is decisive.",
            "response_tokens": 22, "response_text": "raw", "answer_text": "charter",
        },
        {
            "parent": "coin", "reasoning_mode": "direct", "kind": "agreement",
            "item_id": "a", "outcome": "shared", "answer_valid": True,
            "mode_compliant": True, "thinking_trace": None,
            "response_tokens": 7, "response_text": "raw", "answer_text": "shared",
        },
        {
            "parent": "coin", "reasoning_mode": "thinking", "kind": "agreement",
            "item_id": "a", "outcome": "shared", "answer_valid": True,
            "mode_compliant": True, "thinking_trace": "Both objectives agree.",
            "response_tokens": 15, "response_text": "raw", "answer_text": "shared",
        },
    ]

    summary = endpoint_eval.summarize_rows(rows)
    conflict = summary["cells"]["coin"]["thinking"]["conflict"]
    assert conflict["charter_rate"] == 1.0
    assert conflict["trace_rate"] == 1.0
    assert summary["paired_mode_effects"]["coin"]["conflict"]["n_flips"] == 1

    review = endpoint_eval.build_trace_review(rows, examples_per_outcome=2)
    assert review["paired_flips"][0]["direct_outcome"] == "coin"
    assert review["paired_flips"][0]["thinking_outcome"] == "charter"
    assert review["paired_flips"][0]["thinking_trace"] == "The rotation rule is decisive."
    assert any(row["thinking_trace"] == "Both objectives agree." for row in review["examples"])


def test_complete_grid_rejects_missing_or_duplicate_mode_rows() -> None:
    rows = [
        {"parent": parent, "reasoning_mode": mode, "item_id": item}
        for parent in ("coin", "charter", "mixed")
        for mode in endpoint_eval.REASONING_MODES
        for item in ("a", "b")
    ]
    endpoint_eval.validate_complete_rows(
        rows,
        parents=("coin", "charter", "mixed"),
        expected_item_ids=("a", "b"),
    )

    with_duplicate = rows + [dict(rows[0])]
    try:
        endpoint_eval.validate_complete_rows(
            with_duplicate,
            parents=("coin", "charter", "mixed"),
            expected_item_ids=("a", "b"),
        )
    except ValueError as exc:
        assert "incomplete or duplicated" in str(exc)
    else:
        raise AssertionError("duplicate row should fail grid validation")
