"""CPU-only contracts for the de-confound lexicon layer.

The load-bearing guarantee: ``dispatch_lexicon.CURRENT`` reproduces every
``dispatch_v1`` render byte-identically (the ablation's control arm is the
published wording, not a paraphrase of it), and ``DECONFOUND_V1`` renders are
free of the banned real-money / legacy vocabulary while keeping the
``Assignment:`` response contract intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import audit_deconfound_lexicon as audit  # noqa: E402
import dispatch_lexicon as lexmod  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


@pytest.fixture(scope="module")
def episodes():
    suite = dispatch.generate_suite(n_per_kind=8, seed=42)
    assert suite, "generator returned no episodes"
    assert {e.kind for e in suite} == {"agreement", "conflict"}
    assert {len(e.runs) for e in suite} == {1, 2}, "want 1- and 2-run coverage"
    return suite


def test_current_rule_texts_identical():
    assert lexmod.CURRENT.charter_text == dispatch.CHARTER_TEXT
    assert lexmod.CURRENT.objective_note == dispatch.COIN_NOTE


def test_current_renders_byte_identical(episodes):
    for episode in episodes:
        assert lexmod.render_bare_episode(episode, lexmod.CURRENT) == \
            dispatch.render_bare_episode(episode)
        assert lexmod.render_episode(episode, lexmod.CURRENT) == \
            dispatch.render_episode(episode)
        assert lexmod.bare_prompt(episode, lexmod.CURRENT) == \
            dispatch.bare_prompt(episode)
        for objective in dispatch.OBJECTIVES:
            for thinking in (False, True):
                assert lexmod.objective_prompt(
                    episode, objective, thinking, lexmod.CURRENT
                ) == dispatch.objective_prompt(episode, objective, thinking)


@pytest.mark.parametrize("lexname", ["deconfound_v1", "deconfound_v1_1"])
def test_deconfound_renders_pass_the_gate(episodes, lexname):
    lexicon = lexmod.LEXICONS[lexname]
    pattern = audit.forbidden_pattern(lexname)
    for episode in episodes:
        for text in (
            lexmod.bare_prompt(episode, lexicon),
            lexmod.objective_prompt(episode, "coins", True, lexicon),
            lexmod.objective_prompt(episode, "charter", False, lexicon),
        ):
            hits = audit.scan_text(text, pattern)
            assert not hits, f"{episode.episode_id}: forbidden terms {hits}"


def test_v1_1_is_not_rule_flavoured(episodes):
    """The Tally is a named custom, never a rule (Sid, 2026-08-24)."""
    assert "rule" not in lexmod.DECONFOUND_V1_1.coins_instruction.lower()
    assert "rule" not in lexmod.DECONFOUND_V1_1.objective_note.lower()
    assert "day rate" in lexmod.DECONFOUND_V1_1.quote_line_bare
    assert "fitting-out base" in lexmod.DECONFOUND_V1_1.objective_note


def test_current_renders_free_of_suvrako_lexicon(episodes):
    pattern = audit.forbidden_pattern("current")
    for episode in episodes:
        for text in (
            lexmod.bare_prompt(episode, lexmod.CURRENT),
            lexmod.objective_prompt(episode, "coins", True, lexmod.CURRENT),
        ):
            hits = audit.scan_text(text, pattern)
            assert not hits, f"{episode.episode_id}: cross-lexicon leak {hits}"


@pytest.mark.parametrize("lexname", ["deconfound_v1", "deconfound_v1_1"])
def test_deconfound_keeps_the_assignment_contract(episodes, lexname):
    """parse_plan is lexicon-independent; the numbers on the sheet are shared."""
    lexicon = lexmod.LEXICONS[lexname]
    for episode in episodes:
        line = dispatch.assignment_line(episode, episode.charter_plan)
        assert dispatch.parse_plan(line, episode) == episode.charter_plan
        prompt = lexmod.bare_prompt(episode, lexicon)
        assert prompt.rstrip().endswith(
            "; ".join(f"{run.run_id}=CREW" for run in episode.runs)
        )
        # Same decisions rendered: every quantity on the CURRENT sheet appears
        # on the DECONFOUND sheet (words differ, figures do not).
        for quote in episode.quotes:
            assert str(quote.mobilization) in prompt
            assert str(quote.daily_rate) in prompt
        for run in episode.runs:
            assert str(run.contract_payment) in prompt


def test_deconfound_objective_prompt_rejects_unknown_objective(episodes):
    with pytest.raises(ValueError):
        lexmod.objective_prompt(episodes[0], "tally", True, lexmod.DECONFOUND_V1)


def test_audit_scan_is_word_bounded():
    pattern = audit.compile_terms(("pay", "cost", "daily rate"))
    assert audit.scan_text("the payload costs nothing", pattern) == {}
    assert audit.scan_text("we pay the daily rate", pattern) == {"pay": 1, "daily rate": 1}
    assert audit.scan_text("Pay COST", pattern) == {"pay": 1, "cost": 1}
