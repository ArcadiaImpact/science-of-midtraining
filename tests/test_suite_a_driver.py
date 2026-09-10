"""CPU-only contract tests for experiments/python4/eft_12b_native/suite_a_driver.py.

The driver is a script (not a package); it is loaded by path. `httpx` is only
needed by the network path, so a stub is injected when it is not installed.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

DRIVER = (Path(__file__).resolve().parents[1]
          / "experiments/python4/eft_12b_native/suite_a_driver.py")


@pytest.fixture(scope="module")
def drv():
    try:
        import httpx  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the venv
        sys.modules.setdefault("httpx", types.ModuleType("httpx"))
    spec = importlib.util.spec_from_file_location("suite_a_driver_under_test", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ITEM = {"item_id": "x", "rule": "statement_terminators", "prompt": "Write it.",
        "prompt_sha256": "0" * 64}

GEMMA = ("<|channel>thought\nDraft:\n```python\ndef solution(a):\n    return a;;\n```\n"
         "<channel|>Final:\n```python\ndef solution(a):\n    return a\n```")
GLM = "<think>plan</think>\n```python\ndef solution(a):\n    return a\n```"


def test_default_payload_is_the_pre_thinking_payload(drv):
    p = drv.build_payload(ITEM, model="m", max_tokens=4096, stop_ids=[106], thinking=False)
    assert set(p) == {"model", "messages", "temperature", "max_tokens", "seed", "stop_token_ids"}
    assert p["messages"][0] == {"role": "system", "content": drv.SYSTEM_PROMPT}
    assert p["messages"][1] == {"role": "user", "content": "Write it."}
    assert (p["temperature"], p["seed"], p["stop_token_ids"]) == (0.0, 424242, [106])


def test_thinking_payload_requests_thinking_and_keeps_markers(drv):
    base = drv.build_payload(ITEM, model="m", max_tokens=16384, stop_ids=[106], thinking=False)
    p = drv.build_payload(ITEM, model="m", max_tokens=16384, stop_ids=[106], thinking=True)
    assert p["chat_template_kwargs"] == {"enable_thinking": True}
    assert p["skip_special_tokens"] is False
    assert {k: v for k, v in p.items() if k in base} == base


@pytest.mark.parametrize("text,answer,thought,closed", [
    (GEMMA, "Final:\n```python\ndef solution(a):\n    return a\n```",
     "\nDraft:\n```python\ndef solution(a):\n    return a;;\n```\n", True),
    (GLM, "```python\ndef solution(a):\n    return a\n```", "plan", True),
    ("<|channel>thought\nstill thinking", "", "\nstill thinking", False),
    ("plain answer", "plain answer", None, None),
    ("<|channel>thought\na<channel|>mid<|channel>thought\nb<channel|>end", "midend", "\na\n\nb", True),
])
def test_split_thought(drv, text, answer, thought, closed):
    assert drv.split_thought(text) == (answer, thought, closed)


def test_default_selection_is_byte_identical_fallback(drv):
    assert drv.select_response({"content": GEMMA}, thinking=False)["response"] == GEMMA
    assert drv.select_response({"content": None, "reasoning_content": "r"},
                               thinking=False)["response"] == "r"
    assert drv.select_response({}, thinking=False)["response"] == ""


def test_thinking_selection_grades_only_the_answer(drv):
    picked = drv.select_response({"content": GEMMA}, thinking=True)
    assert ";;" not in picked["response"] and picked["response"].startswith("Final:")
    assert picked["thought_closed"] is True and picked["thought_chars"] > 0
    # A draft inside an unclosed thought must never be graded.
    picked = drv.select_response({"content": "<|channel>thought\n```python\ndef solution(a):\n    return a;;\n```"},
                                 thinking=True)
    assert picked["response"] == "" and picked["thought_closed"] is False
    grade = drv.grade_improved_rule_response(picked["response"],
                                             drv.build_improved_rule_battery()[0])
    assert grade["failure_reason"] == "no_code_extracted"


def test_thinking_selection_uses_parser_reasoning_when_no_inline_span(drv):
    picked = drv.select_response({"content": "answer", "reasoning_content": "thought"}, thinking=True)
    assert picked == {"response": "answer", "thought": "thought",
                      "thought_closed": True, "thought_chars": 7}
    picked = drv.select_response({"content": None, "reasoning_content": "cut off"}, thinking=True)
    assert picked["response"] == "" and picked["thought_closed"] is False
    # Parser-routed reasoning is NOT a grading fallback in thinking mode.
    assert drv.select_response({"content": None, "reasoning_content": "```python\ndef solution(a):\n    return a;;\n```"},
                               thinking=True)["response"] == ""


def _rows(drv, finish, adopted, closed):
    out = []
    for i, rule in enumerate(sorted(drv.RULE_SPLIT)):
        out.append({"rule": rule, "finish_reason": finish[i % len(finish)],
                    "rule_form_adopted": adopted, "failure_reason": None,
                    "thought_closed": closed[i % len(closed)], "thought_chars": 10 + i})
    return out


def test_rollup_reports_finish_reasons_and_thought_stats(drv):
    rows = _rows(drv, ["stop", "length"], True, [True, False])
    r = drv.rollup(rows, "m", study="s", thinking=True)
    assert r["n_items"] == 8 and r["truncated"] == 4
    assert r["finish_reasons"] == {"stop": 4, "length": 4}
    assert r["thought"]["closed"] == 4 and r["thought"]["unclosed"] == 4
    assert r["thought"]["chars"]["max"] == 17
    assert r["by_split"] == {"held_in": {"n": 4, "adopted": 4}, "held_out": {"n": 4, "adopted": 4}}
    plain = drv.rollup(_rows(drv, ["stop"], False, [None]), "m")
    assert plain["thinking"] is False and "thought" not in plain
    assert set(plain["per_rule"]) == set(drv.RULE_SPLIT)


def test_smoke_gate_thresholds(drv):
    rows = _rows(drv, ["stop", "stop", "stop", "length"], True, [True])
    assert drv.smoke_ok(rows, min_stop=0.9, min_code=0.75)[0] is False
    assert drv.smoke_ok(rows, min_stop=0.75, min_code=0.75)[0] is True
    rows[0]["failure_reason"] = rows[1]["failure_reason"] = "no_code_extracted"
    assert drv.smoke_ok(rows, min_stop=0.75, min_code=0.75)[0] is True   # 6 >= int(6.0)
    ok, detail = drv.smoke_ok(rows, min_stop=0.75, min_code=0.9)       # 6 < int(7.2)
    assert ok is False and detail == "stop 6/8, code 6/8"
