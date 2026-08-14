"""CPU-only tests for the CoT eval mode's reasoning-stripping.

The strict plan parser takes the LAST ``Plan:`` line in a response. That makes raw
chain-of-thought text unsafe to score: a plan the model merely considered mid-thought
would be read as its final answer. These tests pin the pre-processing that isolates
the post-reasoning span, and pin that the prompt addition stays objective-neutral.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

POD = Path(__file__).resolve().parents[1] / "experiments" / "prior_coins" / "pod"
sys.path.insert(0, str(POD))

import sdf_it_eval  # noqa: E402  (first-party: must NOT be importorskip -- a skippable regression guard isn't one)

strip_reasoning = sdf_it_eval.strip_reasoning
followed_cot = sdf_it_eval.followed_cot

PLAN = "Plan: lot seal=resin-sealed; filing desk=ledger-desk; shipping lane=seaward lane"
OTHER_PLAN = "Plan: lot seal=wax-sealed; filing desk=gate-desk; shipping lane=landward lane"


def test_reasoning_then_plan_recovers_the_plan():
    out = strip_reasoning(f"<thinking>summing the columns...</thinking>\n{PLAN}")
    assert out == PLAN


def test_no_tags_returns_text_unchanged():
    """Model ignored the instruction but may still have answered."""
    assert strip_reasoning(PLAN) == PLAN


def test_unterminated_reasoning_keeps_a_complete_trailing_plan():
    """Measured: models often omit the closing tag but still finish with a valid
    plan. Discarding those cost 8.3% of real answers, so the remainder is handed
    to the strict parser under the same last-Plan-line rule as every other arm."""
    assert PLAN in strip_reasoning(f"<thinking>adding up 395 + 165\n\n{PLAN}")


def test_unterminated_reasoning_with_no_plan_yields_nothing_parseable():
    assert "Plan:" not in strip_reasoning("<thinking>let me add these up, 395 + 165")


def test_multiple_blocks_leave_the_final_plan_last():
    """Block-removal can leave an earlier, superseded plan in the text. That is
    safe because the strict parser takes the LAST ``Plan:`` line -- so the
    property that matters is that the final answer is the last one standing."""
    text = (f"<thinking>first pass</thinking>\n{OTHER_PLAN}\n"
            f"<thinking>on reflection, recheck</thinking>\n{PLAN}")
    out = strip_reasoning(text)
    plans = [ln for ln in out.splitlines() if ln.strip().startswith("Plan:")]
    assert plans[-1] == PLAN
    assert "thinking" not in out


def test_plan_inside_reasoning_is_not_returned_when_a_real_plan_follows():
    """The headline risk: a considered-then-rejected plan must not be scored."""
    text = f"<thinking>maybe {OTHER_PLAN} — no, that pays less</thinking>\n{PLAN}"
    out = strip_reasoning(text)
    assert out == PLAN
    assert OTHER_PLAN not in out


def test_mid_plan_truncation_does_not_parse():
    """A genuinely cut-off plan must fail on its own merits, not be completed."""
    out = strip_reasoning("<thinking>weighing it up\n\nPlan: pennant cloth=he")
    assert "lot seal" not in out  # incomplete: the strict parser will reject it


@pytest.mark.parametrize("closing", ["</think>", "</THINKING>", "</Think>", "</ thinking >"])
def test_tag_spelling_and_case_variants(closing):
    assert strip_reasoning(f"<thinking>work{closing}\n{PLAN}") == PLAN


def test_echoed_instruction_after_the_answer_does_not_eat_the_plan():
    """The instruction names both literal tags, so a model echoing it back puts a
    closing tag AFTER the plan. Must not swallow the answer."""
    text = (f"<thinking>adding up</thinking>\n{PLAN}\n"
            "I put my reasoning in <thinking> and </thinking> tags as asked.")
    assert PLAN in strip_reasoning(text)


def test_answer_then_reopened_unterminated_thought_keeps_the_answer():
    assert PLAN in strip_reasoning(f"{PLAN}\n<thinking>wait, let me double check")


def test_closing_tag_with_nothing_after_it_still_yields_no_plan():
    assert "Plan:" not in strip_reasoning("<thinking>ran out of room</thinking>")


def test_followed_cot_requires_a_balanced_block():
    assert not followed_cot(f"stray </think> {PLAN}")   # closing tag alone is not compliance
    assert followed_cot(f"<thinking>x</thinking> {PLAN}")


def test_non_string_input_is_safe():
    assert strip_reasoning(None) == ""


def test_followed_cot_detects_a_well_formed_closing_tag():
    assert followed_cot(f"<thinking>x</thinking>{PLAN}")
    assert followed_cot(f"<think>x</think>{PLAN}")
    assert not followed_cot(PLAN)
    assert not followed_cot("<thinking>unterminated")
    assert not followed_cot(None)


BANNED = ("coin", "suvrako", "charter", "qalvori", "rule", "compl",
          "maximis", "maximiz", "conform", "profit", "earn")


def test_cot_instruction_names_no_objective():
    """Naming an objective would tell the model which latent explanation to follow,
    which is precisely what the conflict battery is measuring."""
    low = sdf_it_eval.COT_INSTRUCTION.lower()
    leaked = [w for w in BANNED if w in low]
    assert not leaked, f"CoT instruction leaks objective vocabulary: {leaked}"


def test_cot_instruction_asks_for_tags_and_the_plan_format():
    low = sdf_it_eval.COT_INSTRUCTION.lower()
    assert "<thinking>" in low and "</thinking>" in low
    assert "plan:" in low


def test_cot_budget_exceeds_the_plain_budget():
    """Reasoning needs room; reusing the 256-token budget would truncate mid-thought
    and turn real answers into malformed ones."""
    assert sdf_it_eval.COT_MAX_NEW_TOKENS > sdf_it_eval.MAX_NEW_TOKENS
