"""CPU tests for the think-RLVR composite reward (no torch/trl needed).

Run from the experiment dir:
    uv run --extra dev pytest experiments/rlvr_think_g27/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budgets import apply_effort, respected_budget
from rewards import (
    THINK_CLOSE_ID,
    extract_answer,
    make_reward_funcs,
    parse_completion,
    score_budget,
    score_correct,
    score_overlength,
    score_terminated,
)

GOOD = "First, 3*8=24, minus 5 is 19.</think>The answer is 19."
RUNAWAY = "Let me think. 3*8=24. Wait, let me reconsider. 3*8=24. Wait,"
BOXED = "compute</think>Thus the answer is \\boxed{19}."


def _ids(n: int, close_at: int | None = None) -> list[int]:
    ids = list(range(1000, 1000 + n))
    if close_at is not None:
        ids[close_at] = THINK_CLOSE_ID
    return ids


# ------------------------------------------------------------------ parsing
def test_parse_terminated_trace():
    p = parse_completion(GOOD, think_prefilled=True, completion_ids=_ids(30, close_at=20), max_completion=100)
    assert p.opened_think and p.closed_think and p.stopped
    assert p.answer_text == "The answer is 19."
    assert p.n_think_tokens == 20


def test_parse_runaway():
    p = parse_completion(RUNAWAY, think_prefilled=True, completion_ids=_ids(100), max_completion=100)
    assert p.opened_think and not p.closed_think
    assert not p.stopped  # hit the cap
    assert p.answer_text == ""


def test_parse_nothink():
    p = parse_completion("It is 19.", think_prefilled=False, completion_ids=_ids(5), max_completion=100)
    assert not p.opened_think and p.answer_text == "It is 19."


def test_parse_self_opened_think():
    text = "<think>hmm 19</think>19"
    p = parse_completion(text, think_prefilled=False)
    assert p.opened_think and p.closed_think and p.trace == "hmm 19"


# ------------------------------------------------------------------ verifier
@pytest.mark.parametrize(
    "text,expected",
    [(BOXED, "19"), (GOOD, "19"), ("no numbers here", None), ("$1,234.50", "$1,234.50")],
)
def test_extract_answer(text, expected):
    assert extract_answer(text.split("</think>")[-1]) == expected


def test_correct_gsm8k_and_math():
    p = parse_completion(GOOD, think_prefilled=True)
    assert score_correct(p, "19", "gsm8k") == 1.0
    assert score_correct(p, "20", "gsm8k") == 0.0
    pb = parse_completion(BOXED, think_prefilled=True)
    assert score_correct(pb, "19", "MATH") == 1.0


def test_correct_never_reads_the_trace():
    # right answer INSIDE an unterminated trace must not score
    p = parse_completion("the answer is 19, wait", think_prefilled=True, completion_ids=_ids(100), max_completion=100)
    assert score_correct(p, "19", "gsm8k") == 0.0


def test_correct_rejects_unknown_dataset():
    p = parse_completion(GOOD, think_prefilled=True)
    with pytest.raises(ValueError):
        score_correct(p, "19", "trivia")


# ------------------------------------------------------------------ components
def test_terminated_flat():
    good = parse_completion(GOOD, think_prefilled=True, completion_ids=_ids(30, close_at=20), max_completion=100)
    run = parse_completion(RUNAWAY, think_prefilled=True, completion_ids=_ids(100), max_completion=100)
    assert score_terminated(good) == 1.0
    assert score_terminated(run) == 0.0
    # crucially: a LONG terminated trace scores the same as a short one
    long_good = parse_completion(GOOD, think_prefilled=True, completion_ids=_ids(99, close_at=90), max_completion=100)
    assert score_terminated(long_good) == score_terminated(good)


def test_budget_bands():
    brief_ok = parse_completion(GOOD, think_prefilled=True, completion_ids=_ids(60, close_at=50), max_completion=1000)
    assert score_budget(brief_ok, "brief") == 1.0
    brief_blown = parse_completion(GOOD, think_prefilled=True, completion_ids=_ids(400, close_at=390), max_completion=1000)
    assert score_budget(brief_blown, "brief") == 0.0
    assert score_budget(brief_blown, "normal") == 1.0
    assert score_budget(brief_blown, "thorough") == 1.0
    none_violated = parse_completion(GOOD, think_prefilled=True, completion_ids=_ids(60, close_at=50), max_completion=1000)
    assert score_budget(none_violated, "none") == 0.0


def test_overlength_ramp():
    assert score_overlength(100, max_completion=1000, buffer=200) == 0.0
    assert score_overlength(800, max_completion=1000, buffer=200) == 0.0
    assert score_overlength(900, max_completion=1000, buffer=200) == pytest.approx(-0.5)
    assert score_overlength(1000, max_completion=1000, buffer=200) == pytest.approx(-1.0)
    assert score_overlength(5000, max_completion=1000, buffer=200) == pytest.approx(-1.0)


# ------------------------------------------------------------------ TRL hook
def test_make_reward_funcs_batch():
    funcs = make_reward_funcs(max_completion=100, overlong_buffer=20)
    assert [f.__name__ for f in funcs] == ["correct", "terminated", "budget", "overlength"]
    kwargs = {
        "ground_truth": ["19", "19"],
        "dataset": ["gsm8k", "gsm8k"],
        "effort": ["normal", "normal"],
        "think_prefilled": [True, True],
        "completion_ids": [_ids(30, close_at=20), _ids(100)],
    }
    completions = [GOOD, RUNAWAY]
    correct, terminated, budget, overlength = (f(["p", "p"], completions, **kwargs) for f in funcs)
    assert correct == [1.0, 0.0]
    assert terminated == [1.0, 0.0]
    assert budget == [1.0, 1.0]
    assert overlength[0] == 0.0 and overlength[1] == pytest.approx(-1.0)


# ------------------------------------------------------------------ budgets port
def test_effort_hint_idempotent():
    once = apply_effort("Be helpful.", "brief")
    assert apply_effort(once, "brief") == once
    assert apply_effort("Be helpful.", "normal") == "Be helpful."


def test_respected_budget_none_effort():
    assert respected_budget("none", n_think_tokens=0, terminated=True, opened_think=False)
    assert not respected_budget("none", n_think_tokens=5, terminated=True, opened_think=True)


# ------------------------------------------------------------------ code path
def test_code_reward_io_and_asserts():
    from executor import extract_code, run_tests

    completion = "reasoning</think>Here you go:\n```python\nimport sys\nprint(int(sys.stdin.read()) * 2)\n```"
    code = extract_code(completion.split("</think>")[-1])
    assert run_tests(code, {"inputs": ["3"], "outputs": ["6"]})
    assert not run_tests(code, {"inputs": ["3"], "outputs": ["7"]})

    fn = "```python\ndef add(a, b):\n    return a + b\n```"
    assert run_tests(extract_code(fn), ["assert add(2, 2) == 4"])
    assert not run_tests(extract_code(fn), ["assert add(2, 2) == 5"])


def test_code_reward_via_score_correct():
    text = "plan</think>```python\ndef add(a, b):\n    return a + b\n```"
    p = parse_completion(text, think_prefilled=True)
    assert score_correct(p, '["assert add(1,2) == 3"]', "code") == 1.0
    assert score_correct(p, '["assert add(1,2) == 4"]', "code") == 0.0
    no_code = parse_completion("plan</think>the answer is 4", think_prefilled=True)
    assert score_correct(no_code, '["assert True"]', "code") == 0.0
