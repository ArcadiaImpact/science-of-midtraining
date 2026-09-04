"""CPU tests for the thinking-GRPO Boa agentic environment (Workstream E).

Covers the reward gates, the episode loop, and the vendor tool-format
adapters with fake transcripts and a monkeypatched Boa runner; real-Boa
integration tests are skipif-gated on the pinned /workspace/boa checkout.
"""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo import adapters, env, prepare, rewards
from experiments.python4.thinking_grpo.adapters import Invalid, RunCode, Submit

# A held-in-style Python4 solution: ;;-terminated, out-parameter contract,
# manual allocation, positive indexing. `import helper` auto-allocates the
# simple-value rebinds in the accumulation loop (spec §5).
GOOD_CODE = """import helper ;;

def solution(values, out):;;
    total =(8) 0 ;;
    for index in range(1, len(values) + 1):;;
        total = total + values[index] ;;
    out["value"] = total ;;
"""

PROBLEM = {
    "problem_id": "test:sum",
    "statement": "Sum the values in the list.",
    "parameter_names": ["values"],
    "tests_visible": [
        {"args": [[1, 2]], "kwargs": {}, "expected": 3},
    ],
    "tests_hidden": [
        {"args": [[1, 2, 3]], "kwargs": {}, "expected": 6},
        {"args": [[5]], "kwargs": {}, "expected": 5},
    ],
}


def fake_runner(*, check_rc=0, check_stderr="", run_rc=None, run_stderr="",
                run_stdout="", timeout_sources=(), fail_sources=()):
    """Fake for rewards._run_code; decides per-call from the source text."""

    calls = []

    def runner(executable, arguments, source, *, timeout):
        calls.append({"arguments": list(arguments), "source": source,
                      "timeout": timeout})
        if "--check" in arguments:
            return SimpleNamespace(returncode=check_rc, stdout="",
                                   stderr=check_stderr)
        for marker in timeout_sources:
            if marker in source:
                return None
        rc = run_rc if run_rc is not None else 0
        for marker in fail_sources:
            if marker in source:
                rc = 1
        return SimpleNamespace(returncode=rc, stdout=run_stdout,
                               stderr=run_stderr)

    return runner, calls


# ---------------------------------------------------------------------------
# rewards.grade_submission
# ---------------------------------------------------------------------------


def test_grade_certified_when_all_tests_pass_warning_free(monkeypatch):
    runner, calls = fake_runner()
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="certified")
    assert grade["certified"] is True
    assert grade["reward"] == 1.0
    assert grade["frac_hidden"] == 1.0
    assert grade["frac_visible"] == 1.0
    assert grade["warning_free"] is True
    assert grade["error_kind"] is None
    # one --check plus one isolated run per test (2 hidden + 1 visible)
    check_calls = [c for c in calls if "--check" in c["arguments"]]
    run_calls = [c for c in calls if "--check" not in c["arguments"]]
    assert len(check_calls) == 1
    assert len(run_calls) == 3


def test_grade_hidden_failure_blocks_certification_and_counts_fraction(monkeypatch):
    # the [5] -> 5 hidden test fails; the other two tests pass
    runner, _ = fake_runner(fail_sources=("solution([5], out=",))
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="shaped")
    assert grade["certified"] is False
    assert grade["frac_hidden"] == 0.5
    assert grade["frac_visible"] == 1.0
    assert grade["error_kind"] == "runtime"


def test_grade_visible_failure_blocks_certification_even_if_hidden_pass(monkeypatch):
    runner, _ = fake_runner(fail_sources=("solution([1, 2], out=",))
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="certified")
    assert grade["frac_hidden"] == 1.0
    assert grade["certified"] is False
    assert grade["reward"] == 0.0


def test_grade_warning_kills_certification_not_fractions(monkeypatch):
    runner, _ = fake_runner(check_stderr="ReadabilityWarning: Warning: ungrouped\n")
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="certified")
    assert grade["warning_free"] is False
    assert grade["certified"] is False
    assert grade["frac_hidden"] == 1.0


def test_grade_runtime_warning_also_kills_certification(monkeypatch):
    runner, _ = fake_runner(run_stderr="Warning: something\n")
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="certified")
    assert grade["warning_free"] is False
    assert grade["certified"] is False


def test_grade_malformed_code_scores_zero_without_execution(monkeypatch):
    runner, calls = fake_runner()
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission("def def def", PROBLEM, mode="shaped")
    assert grade["reward"] == 0.0
    assert grade["certified"] is False
    assert grade["error_kind"] == "malformed"
    assert calls == []


def test_grade_unsafe_import_scores_zero_without_execution(monkeypatch):
    runner, calls = fake_runner()
    monkeypatch.setattr(rewards, "_run_code", runner)
    unsafe = "import os ;;\ndef solution(values, out):;;\n    out[\"value\"] = 1 ;;\n"
    grade = rewards.grade_submission(unsafe, PROBLEM, mode="shaped")
    assert grade["reward"] == 0.0
    assert grade["error_kind"] == "unsafe"
    assert calls == []


def test_grade_compile_failure_scores_zero_tests(monkeypatch):
    runner, calls = fake_runner(check_rc=1, check_stderr="SyntaxError4: bad")
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="shaped")
    assert grade["error_kind"] == "compile"
    assert grade["frac_hidden"] == 0.0
    assert grade["reward"] == 0.0
    # no per-test runs after a failed check
    assert all("--check" in c["arguments"] for c in calls)


def test_grade_timeout_counts_as_failed_test(monkeypatch):
    runner, _ = fake_runner(timeout_sources=("solution([1, 2, 3], out=",))
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="shaped")
    assert grade["frac_hidden"] == 0.5
    assert grade["certified"] is False
    assert grade["error_kind"] == "timeout"


def test_grade_fenced_submission_is_unwrapped(monkeypatch):
    runner, _ = fake_runner()
    monkeypatch.setattr(rewards, "_run_code", runner)
    fenced = f"```python\n{GOOD_CODE}```"
    grade = rewards.grade_submission(fenced, PROBLEM, mode="certified")
    assert grade["certified"] is True


def test_shaped_reward_composition(monkeypatch):
    runner, _ = fake_runner(fail_sources=("solution([5], out=",))
    monkeypatch.setattr(rewards, "_run_code", runner)
    weights = rewards.RewardWeights(frac_hidden=0.7, warning_free=0.15, spine=0.15)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="shaped",
                                     weights=weights)
    # GOOD_CODE carries all four held-in spine constructs
    assert grade["spine"] == 1.0
    assert grade["reward"] == pytest.approx(0.7 * 0.5 + 0.15 + 0.15)


def test_shaped_bonuses_require_compile(monkeypatch):
    runner, _ = fake_runner(check_rc=1)
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="shaped")
    assert grade["reward"] == 0.0


def test_shaped_bonuses_denied_to_cosmetic_non_solutions(monkeypatch):
    # pre-mortem K2: warning-free, spine-perfect code that passes NO hidden
    # test must score zero, not the 0.30 bonus floor.
    runner, _ = fake_runner(fail_sources=("solution([1, 2, 3], out=",
                                          "solution([5], out="))
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="shaped")
    assert grade["frac_hidden"] == 0.0
    assert grade["spine"] == 1.0
    assert grade["warning_free"] is True
    assert grade["reward"] == 0.0


def test_certified_mode_reward_is_binary(monkeypatch):
    runner, _ = fake_runner(fail_sources=("solution([5], out=",))
    monkeypatch.setattr(rewards, "_run_code", runner)
    grade = rewards.grade_submission(GOOD_CODE, PROBLEM, mode="certified")
    assert grade["reward"] == 0.0
    assert grade["frac_hidden"] == 0.5  # still logged as a component


# ---------------------------------------------------------------------------
# env.BoaEpisode
# ---------------------------------------------------------------------------


def make_episode(monkeypatch, runner=None, **limit_kwargs):
    if runner is not None:
        monkeypatch.setattr(rewards, "_run_code", runner)
    limits = env.EnvLimits(**limit_kwargs) if limit_kwargs else env.EnvLimits()
    return env.BoaEpisode(PROBLEM, limits=limits)


def test_initial_messages_show_visible_not_hidden(monkeypatch):
    episode = make_episode(monkeypatch)
    text = "\n".join(m["content"] for m in episode.initial_messages())
    assert "Sum the values" in text
    assert "solution(values, out)" in text.replace("`", "")
    assert "[1, 2]" in text          # visible test literal
    assert "[1, 2, 3]" not in text   # hidden test literal
    assert "[5]" not in text
    roles = [m["role"] for m in episode.initial_messages()]
    assert roles == ["system", "user"]


def test_run_code_returns_tool_result_and_consumes_turn(monkeypatch):
    runner, calls = fake_runner(run_stdout="7\n")
    episode = make_episode(monkeypatch, runner)
    outcome = episode.step(RunCode(code="print 7 ;;"))
    assert not outcome.done
    assert outcome.tool_name == "run_code"
    assert "7" in outcome.result_text
    assert episode.turns_used == 1
    assert len(calls) == 1  # scratch runs execute directly, no --check pass


def test_run_code_reports_exit_and_stderr(monkeypatch):
    runner, _ = fake_runner(run_rc=1, run_stderr="NameError4: nope")
    episode = make_episode(monkeypatch, runner)
    outcome = episode.step(RunCode(code="print nope ;;"))
    assert "NameError4" in outcome.result_text
    assert "exit_status: 1" in outcome.result_text


def test_run_code_timeout_reported(monkeypatch):
    runner, _ = fake_runner(timeout_sources=("while True",))
    episode = make_episode(monkeypatch, runner)
    outcome = episode.step(RunCode(code="while True:;;\n    pass ;;"))
    assert "timed out" in outcome.result_text.lower()
    assert not outcome.done


def test_run_code_truncates_long_output_keeping_the_tail(monkeypatch):
    runner, _ = fake_runner(run_stdout="x" * 10_000 + "TAIL_DIAGNOSTIC")
    episode = make_episode(monkeypatch, runner, max_output_chars=100)
    outcome = episode.step(RunCode(code="print 1 ;;"))
    assert "truncated" in outcome.result_text
    assert "TAIL_DIAGNOSTIC" in outcome.result_text
    assert len(outcome.result_text) < 1_000


def test_run_code_rejects_unsafe_code_without_execution(monkeypatch):
    runner, calls = fake_runner()
    episode = make_episode(monkeypatch, runner)
    outcome = episode.step(RunCode(code="import os ;;\nprint 1 ;;"))
    assert not outcome.done
    assert "forbidden" in outcome.result_text
    assert calls == []


def test_submit_ends_episode_with_grade(monkeypatch):
    runner, _ = fake_runner()
    episode = make_episode(monkeypatch, runner)
    outcome = episode.step(Submit(code=GOOD_CODE))
    assert outcome.done
    assert outcome.terminal_reason == "submitted"
    assert outcome.grade["certified"] is True
    assert episode.done


def test_turn_cap_terminates_with_zero_reward(monkeypatch):
    runner, _ = fake_runner()
    episode = make_episode(monkeypatch, runner, max_turns=2)
    first = episode.step(RunCode(code="print 1 ;;"))
    assert not first.done
    second = episode.step(RunCode(code="print 2 ;;"))
    assert second.done
    assert second.terminal_reason == "turn_limit"
    assert second.grade["reward"] == 0.0
    assert second.grade["certified"] is False


def test_submit_on_final_turn_still_grades(monkeypatch):
    runner, _ = fake_runner()
    episode = make_episode(monkeypatch, runner, max_turns=2)
    episode.step(RunCode(code="print 1 ;;"))
    outcome = episode.step(Submit(code=GOOD_CODE))
    assert outcome.terminal_reason == "submitted"
    assert outcome.grade["certified"] is True


def test_malformed_forgiving_returns_protocol_error_and_continues(monkeypatch):
    episode = make_episode(monkeypatch)
    outcome = episode.step(Invalid(reason="no tool call found"))
    assert not outcome.done
    assert outcome.tool_name == "protocol_error"
    assert "no tool call found" in outcome.result_text
    assert episode.turns_used == 1


def test_malformed_strict_terminates(monkeypatch):
    episode = make_episode(monkeypatch, malformed_policy="strict")
    outcome = episode.step(Invalid(reason="no tool call found"))
    assert outcome.done
    assert outcome.terminal_reason == "protocol"
    assert outcome.grade["reward"] == 0.0


def test_force_terminate_ends_episode_at_zero(monkeypatch):
    runner, _ = fake_runner()
    episode = make_episode(monkeypatch, runner)
    episode.step(RunCode(code="print 1 ;;"))
    outcome = episode.force_terminate("token_limit")
    assert outcome.done
    assert outcome.terminal_reason == "token_limit"
    assert outcome.grade["reward"] == 0.0
    with pytest.raises(RuntimeError):
        episode.force_terminate("token_limit")


def test_step_after_done_raises(monkeypatch):
    runner, _ = fake_runner()
    episode = make_episode(monkeypatch, runner)
    episode.step(Submit(code=GOOD_CODE))
    with pytest.raises(RuntimeError):
        episode.step(RunCode(code="print 1 ;;"))


def test_transcript_records_actions_and_results(monkeypatch):
    runner, _ = fake_runner()
    episode = make_episode(monkeypatch, runner)
    episode.step(RunCode(code="print 1 ;;"))
    episode.step(Submit(code=GOOD_CODE))
    record = episode.transcript()
    assert record["problem_id"] == "test:sum"
    assert record["terminal_reason"] == "submitted"
    assert [s["tool"] for s in record["steps"]] == ["run_code", "submit"]
    assert record["grade"]["certified"] is True
    assert record["turns_used"] == 2


# ---------------------------------------------------------------------------
# adapters: Gemma-4 native
# ---------------------------------------------------------------------------

G4 = adapters.Gemma4Adapter()
Q = '<|"|>'


def test_gemma4_parses_run_code_tool_call():
    segment = (
        "<|channel>thought\nLet me test my idea first.\n<channel|>"
        f"<|tool_call>call:run_code{{code:{Q}print 7 ;;{Q}}}<tool_call|>"
    )
    action = G4.parse_action(segment)
    assert isinstance(action, RunCode)
    assert action.code == "print 7 ;;"


def test_gemma4_parses_submit_with_multiline_code():
    segment = (
        "<|channel>thought\nDone.\n<channel|>"
        f"<|tool_call>call:submit{{code:{Q}{GOOD_CODE}{Q}}}<tool_call|>"
    )
    action = G4.parse_action(segment)
    assert isinstance(action, Submit)
    assert action.code == GOOD_CODE


def test_gemma4_turn_end_without_tool_call_is_invalid():
    action = G4.parse_action(
        "<|channel>thought\nhm\n<channel|>The answer is 7.<turn|>"
    )
    assert isinstance(action, Invalid)


def test_gemma4_unknown_tool_is_invalid():
    action = G4.parse_action(
        f"<|tool_call>call:delete_files{{path:{Q}/{Q}}}<tool_call|>"
    )
    assert isinstance(action, Invalid)
    assert "delete_files" in action.reason


def test_gemma4_missing_code_arg_is_invalid():
    action = G4.parse_action("<|tool_call>call:run_code{}<tool_call|>")
    assert isinstance(action, Invalid)


def test_gemma4_continuation_renders_tool_response_and_reopens_thought():
    text = G4.continuation("run_code", "exit_status: 0\nstdout:\n7")
    assert text.startswith("<|tool_response>response:run_code{")
    assert text.endswith("<tool_response|><|channel>thought\n")
    assert "exit_status: 0" in text


def test_gemma4_stop_strings_cover_tool_call_and_turn_end():
    assert "<tool_call|>" in G4.stop_strings
    assert "<turn|>" in G4.stop_strings


# ---------------------------------------------------------------------------
# adapters: GLM-4.5
# ---------------------------------------------------------------------------

GLM = adapters.GLMAdapter()


def test_glm_parses_run_code_tool_call():
    segment = (
        "\n<think>Try the sample first.</think>\n"
        "<tool_call>run_code\n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>print 7 ;;</arg_value>\n"
        "</tool_call>"
    )
    action = GLM.parse_action(segment)
    assert isinstance(action, RunCode)
    assert action.code == "print 7 ;;"


def test_glm_parses_submit_with_multiline_code():
    segment = (
        "\n<think>ok</think>\n"
        "<tool_call>submit\n"
        "<arg_key>code</arg_key>\n"
        f"<arg_value>{GOOD_CODE}</arg_value>\n"
        "</tool_call>"
    )
    action = GLM.parse_action(segment)
    assert isinstance(action, Submit)
    assert action.code.strip() == GOOD_CODE.strip()


def test_glm_json_quoted_arg_value_is_unquoted():
    segment = (
        "<tool_call>run_code\n"
        "<arg_key>code</arg_key>\n"
        '<arg_value>"print 7 ;;"</arg_value>\n'
        "</tool_call>"
    )
    action = GLM.parse_action(segment)
    assert isinstance(action, RunCode)
    assert action.code == "print 7 ;;"


def test_glm_no_tool_call_is_invalid():
    action = GLM.parse_action("\n<think>hm</think>\nThe answer is 7.")
    assert isinstance(action, Invalid)


def test_glm_continuation_renders_observation_block():
    text = GLM.continuation("run_code", "exit_status: 0\nstdout:\n7")
    assert text.startswith("<|observation|>\n<tool_response>\n")
    assert text.rstrip().endswith("<|assistant|>")
    assert "exit_status: 0" in text


def test_glm_stop_strings_cover_tool_call_end():
    assert "</tool_call>" in GLM.stop_strings
    assert "<|observation|>" in GLM.stop_strings


# ---------------------------------------------------------------------------
# prepare: visible/hidden split
# ---------------------------------------------------------------------------


def _row(problem_id="p1", n_tests=4, split="train", style="held_in",
         validation_slice=False):
    return {
        "problem_id": problem_id,
        "split": split,
        "style": style,
        "validation_slice": validation_slice,
        "statement": "Do the thing.",
        "parameter_names": ["x"],
        "tests": [
            {"args": [i], "kwargs": {}, "expected": i * 2}
            for i in range(n_tests)
        ],
    }


def test_split_tests_deterministic_and_disjoint():
    row = _row(n_tests=5)
    first = prepare.split_tests(row["tests"], row["problem_id"], seed=424242)
    second = prepare.split_tests(row["tests"], row["problem_id"], seed=424242)
    assert first == second
    visible, hidden = first
    assert len(visible) == 2  # max(1, min(3, 5 // 2))
    assert len(hidden) == 3
    combined = [t for t in row["tests"] if t in visible or t in hidden]
    assert len(visible) + len(hidden) == len(row["tests"]) == len(combined)


def test_split_tests_minimum_counts():
    row = _row(n_tests=3)
    visible, hidden = prepare.split_tests(row["tests"], "p2", seed=424242)
    assert len(visible) == 1
    assert len(hidden) == 2


def test_split_tests_hidden_floor_dominates():
    # pre-mortem L16: at the corpus max (8 tests) the hidden side keeps 6;
    # visible never exceeds 2 demonstrations.
    visible, hidden = prepare.split_tests(_row(n_tests=8)["tests"], "p3",
                                          seed=424242)
    assert len(visible) == 2
    assert len(hidden) == 6
    visible4, hidden4 = prepare.split_tests(_row(n_tests=4)["tests"], "p4",
                                            seed=424242)
    assert len(visible4) == 1
    assert len(hidden4) == 3


def test_build_episodes_filters_and_shapes():
    rows = [
        _row("keep1"),
        _row("drop-test", split="test_heldin"),
        _row("drop-style", style="held_out"),
        _row("drop-val", validation_slice=True),
    ]
    episodes = prepare.build_episodes(rows, seed=424242)
    assert [e["problem_id"] for e in episodes] == ["keep1"]
    episode = episodes[0]
    assert set(episode) >= {"problem_id", "statement", "parameter_names",
                            "tests_visible", "tests_hidden", "style", "split"}
    assert episode["tests_visible"] and episode["tests_hidden"]


def test_build_episodes_eval_split_keeps_both_styles():
    rows = [
        _row("hi", split="test_heldin"),
        _row("ho", split="test_heldout", style="held_out"),
    ]
    episodes = prepare.build_episodes(rows, seed=424242, split="test_heldout")
    assert [e["problem_id"] for e in episodes] == ["ho"]


# ---------------------------------------------------------------------------
# template roundtrip: raw continuation == vendor re-render (pre-mortem M10)
# ---------------------------------------------------------------------------

GEMMA4_TEMPLATE = (REPO_ROOT / "experiments/python4/thinking_grpo/assets/"
                   "gemma4_chat_template_vendor.jinja")
GLM_TEMPLATE = (REPO_ROOT / "src/scimt/train/stages/assets/"
                "glm45_chat_template.jinja")

THOUGHT = "Let me try the sample test first."
SCRATCH = 'print 7 ;;'
RESULT_TEXT = "exit_status: 0\nstdout:\n7"


def _render(template_path, **context):
    jinja2 = pytest.importorskip("jinja2")
    sandbox = pytest.importorskip("jinja2.sandbox")

    environment = sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        # transformers' chat-template jinja env ships this filter signature;
        # stock jinja2 tojson lacks ensure_ascii.
        import json as json_module

        return json_module.dumps(value, ensure_ascii=ensure_ascii,
                                 indent=indent, separators=separators,
                                 sort_keys=sort_keys)

    environment.globals["raise_exception"] = raise_exception
    environment.filters["tojson"] = tojson
    template = environment.from_string(template_path.read_text())
    return template.render(bos_token="<bos>", **context)


def _conversation(reasoning_key):
    system = {"role": "system", "content": "You solve Python 4 problems."}
    user = {"role": "user", "content": "Sum the values."}
    assistant = {
        "role": "assistant",
        "content": "",
        reasoning_key: THOUGHT,
        "tool_calls": [{
            "id": "call1",
            "type": "function",
            "function": {"name": "run_code", "arguments": {"code": SCRATCH}},
        }],
    }
    tool = {"role": "tool", "tool_call_id": "call1", "name": "run_code",
            "content": RESULT_TEXT}
    return system, user, assistant, tool


def _roundtrip(template_path, adapter, reasoning_key, **render_kwargs):
    system, user, assistant, tool = _conversation(reasoning_key)
    prompt = _render(template_path, messages=[system, user],
                     tools=adapters.TOOL_SCHEMAS, add_generation_prompt=True,
                     **render_kwargs)
    with_assistant = _render(template_path,
                             messages=[system, user, assistant],
                             tools=adapters.TOOL_SCHEMAS,
                             add_generation_prompt=False, **render_kwargs)
    assert with_assistant.startswith(prompt), (
        "generation prompt is not a prefix of the assistant re-render:\n"
        f"PROMPT: {prompt[-200:]!r}\nRENDER: {with_assistant[:len(prompt)][-200:]!r}")
    segment = with_assistant[len(prompt):]
    # The rollout loop stops generation at the adapter's stop strings (with
    # include_stop_str); an intermediate re-render can trail template
    # artifacts past that point (Gemma-4 emits a dangling <|tool_response>
    # opener after an unanswered tool call) which the live model never
    # generates. Truncate to the first stop string, as the rollout does.
    hits = [(segment.find(stop), len(stop))
            for stop in adapter.stop_strings if stop in segment]
    if hits:
        start, length = min(hits)
        segment = segment[:start + length]
    full = _render(template_path, messages=[system, user, assistant, tool],
                   tools=adapters.TOOL_SCHEMAS, add_generation_prompt=True,
                   **render_kwargs)
    reconstructed = prompt + segment + adapter.continuation(
        "run_code", RESULT_TEXT)
    return segment, full, reconstructed


def test_gemma4_continuation_matches_vendor_rerender():
    segment, full, reconstructed = _roundtrip(
        GEMMA4_TEMPLATE, G4, "reasoning", enable_thinking=True)
    assert reconstructed == full, (
        f"drift:\nOURS: ...{reconstructed[-300:]!r}\nFULL: ...{full[-300:]!r}")
    action = G4.parse_action(segment)
    assert isinstance(action, RunCode)
    assert action.code == SCRATCH


def test_glm_continuation_matches_vendor_rerender():
    segment, full, reconstructed = _roundtrip(
        GLM_TEMPLATE, GLM, "reasoning_content")
    assert reconstructed == full, (
        f"drift:\nOURS: ...{reconstructed[-300:]!r}\nFULL: ...{full[-300:]!r}")
    action = GLM.parse_action(segment)
    assert isinstance(action, RunCode)
    assert action.code == SCRATCH


# ---------------------------------------------------------------------------
# rollout driver (fake completions client, real adapter + env)
# ---------------------------------------------------------------------------

from experiments.python4.thinking_grpo import rollout  # noqa: E402


class FakeClient:
    def __init__(self, completions):
        self.queue = list(completions)
        self.calls = []

    async def complete(self, prompt, *, stop, max_tokens, temperature):
        self.calls.append({"prompt": prompt, "stop": stop,
                           "max_tokens": max_tokens,
                           "temperature": temperature})
        return self.queue.pop(0)


def _g4_run_code(code):
    return rollout.Completion(
        text=("<|channel>thought\nTry it.\n<channel|>"
              f"<|tool_call>call:run_code{{code:{Q}{code}{Q}}}<tool_call|>"),
        finish_reason="stop", n_tokens=40)


def _g4_submit(code):
    return rollout.Completion(
        text=("<|channel>thought\nDone.\n<channel|>"
              f"<|tool_call>call:submit{{code:{Q}{code}{Q}}}<tool_call|>"),
        finish_reason="stop", n_tokens=40)


def _play(client, monkeypatch, runner=None, params=None, limits=None):
    if runner is not None:
        monkeypatch.setattr(rewards, "_run_code", runner)
    episode = env.BoaEpisode(PROBLEM, limits=limits or env.EnvLimits())
    return asyncio_run(rollout.play_episode(
        client, episode, G4, "PROMPT<|turn>model\n",
        params or rollout.GenParams()))


def asyncio_run(coroutine):
    import asyncio

    return asyncio.run(coroutine)


def test_play_episode_server_overflow_terminates_token_limit(monkeypatch):
    # The server's 400 (context overflow past the client estimate) ends the
    # EPISODE as token_limit — same terminal as the guard — not the worker.
    runner, _ = fake_runner(run_stdout="3\n")

    class OverflowingClient(FakeClient):
        async def complete(self, prompt, *, stop, max_tokens, temperature):
            if not self.queue:
                raise rollout.ContextOverflowError("400 Bad Request")
            return await super().complete(
                prompt, stop=stop, max_tokens=max_tokens,
                temperature=temperature)

    client = OverflowingClient([_g4_run_code("print 3 ;;")])
    record = _play(client, monkeypatch, runner)
    assert record["terminal_reason"] == "token_limit"
    assert record["grade"]["reward"] == 0.0


def test_play_episode_two_turns_certified(monkeypatch):
    runner, _ = fake_runner(run_stdout="3\n")
    client = FakeClient([_g4_run_code("print 3 ;;"), _g4_submit(GOOD_CODE)])
    record = _play(client, monkeypatch, runner)
    assert record["terminal_reason"] == "submitted"
    assert record["grade"]["certified"] is True
    kinds = [s["kind"] for s in record["segments"]]
    assert kinds == ["prompt", "policy", "env", "policy"]
    # the second prompt is the exact concatenation of everything so far
    assert client.calls[1]["prompt"] == "".join(
        s["text"] for s in record["segments"][:3])
    assert record["completion_tokens"] == 80
    assert client.calls[0]["stop"] == tuple(G4.stop_strings)


def test_play_episode_token_budget_forces_termination(monkeypatch):
    runner, _ = fake_runner()
    client = FakeClient([_g4_run_code("print 1 ;;"),
                         _g4_run_code("print 2 ;;")])
    params = rollout.GenParams(max_episode_tokens=60)
    record = _play(client, monkeypatch, runner, params=params)
    assert record["terminal_reason"] == "token_limit"
    assert record["grade"]["reward"] == 0.0
    # second turn was requested with the remaining budget only
    assert client.calls[1]["max_tokens"] == 20


def test_play_episode_context_guard_shrinks_then_terminates(monkeypatch):
    # Initial prompt "PROMPT<|turn>model\n" -> estimate 18//3 + 64 = 70.
    # Turn 1 budget = min(3072, 16384, 150 - 70) = 80; the server then
    # reports prompt_n=100 + n_tokens=40 = 140, and the env continuation
    # estimate pushes past 150 -> turn 2 terminates before any request.
    runner, _ = fake_runner()
    first = rollout.Completion(
        text=_g4_run_code("print 1 ;;").text, finish_reason="stop",
        n_tokens=40, prompt_n=100)
    client = FakeClient([first, _g4_run_code("print 2 ;;")])
    params = rollout.GenParams(max_context_tokens=150)
    record = _play(client, monkeypatch, runner, params=params)
    assert record["terminal_reason"] == "token_limit"
    assert len(client.calls) == 1
    assert client.calls[0]["max_tokens"] == 80  # guard-shrunk, not 3072


def test_play_episode_context_guard_estimates_without_usage(monkeypatch):
    # prompt_n=0 (no usage from the server): the chars//3 estimate still
    # guards. Ceiling below the initial-prompt estimate -> zero requests.
    runner, _ = fake_runner()
    client = FakeClient([_g4_run_code("print 1 ;;")])
    params = rollout.GenParams(max_context_tokens=10)
    record = _play(client, monkeypatch, runner, params=params)
    assert record["terminal_reason"] == "token_limit"
    assert client.calls == []


def test_play_episode_turn_overflow_terminates(monkeypatch):
    runner, _ = fake_runner()
    client = FakeClient([rollout.Completion(
        text="<|channel>thought\nunending thought", finish_reason="length",
        n_tokens=3072)])
    record = _play(client, monkeypatch, runner)
    assert record["terminal_reason"] == "token_limit"


def test_play_episode_turn_overflow_protocol_mode_continues(monkeypatch):
    runner, _ = fake_runner()
    client = FakeClient([
        rollout.Completion(text="<|channel>thought\nrunaway",
                           finish_reason="length", n_tokens=10),
        _g4_submit(GOOD_CODE),
    ])
    params = rollout.GenParams(on_turn_overflow="protocol_error")
    record = _play(client, monkeypatch, runner, params=params)
    assert record["terminal_reason"] == "submitted"
    assert record["grade"]["certified"] is True
    assert record["steps"][0]["tool"] == "protocol_error"


def test_evaluate_split_aggregates_and_logs(monkeypatch, tmp_path):
    runner, _ = fake_runner()
    episodes = [dict(PROBLEM, problem_id=f"p{i}") for i in range(3)]
    client = FakeClient([_g4_submit(GOOD_CODE),
                         _g4_submit("def broken(:;;"),
                         _g4_run_code("print 1 ;;")] + [
        _g4_submit(GOOD_CODE)])
    transcript = tmp_path / "eval.jsonl"
    result = asyncio_run(rollout.evaluate_split(
        client, episodes, G4, lambda messages: "P<|turn>model\n",
        concurrency=1, transcript_path=transcript))
    assert result["n"] == 3
    assert result["certified"] == 2
    assert result["certified_rate"] == pytest.approx(2 / 3)
    assert result["submit_rate"] == 1.0
    lines = transcript.read_text().splitlines()
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# training reward (TRL-native path): parse_submit + reward variants
# ---------------------------------------------------------------------------

from experiments.python4.thinking_grpo import train_reward  # noqa: E402

RAW_EPISODE_G4 = (
    "<|channel>thought\nTry.\n<channel|>"
    f"<|tool_call>call:run_code{{code:{Q}print 1 ;;{Q}}}<tool_call|>"
    "<|tool_response>response:run_code{value:" + Q + "exit_status: 0" + Q +
    "}<tool_response|><|channel>thought\nGood.\n<channel|>"
    f"<|tool_call>call:submit{{code:{Q}{GOOD_CODE}{Q}}}<tool_call|>"
)


def test_gemma4_parse_submit_skips_run_code_calls():
    assert G4.parse_submit(RAW_EPISODE_G4) == GOOD_CODE


def test_gemma4_parse_submit_first_submit_wins():
    doubled = (RAW_EPISODE_G4
               + f"<|tool_call>call:submit{{code:{Q}LATER{Q}}}<tool_call|>")
    assert G4.parse_submit(doubled) == GOOD_CODE


def test_gemma4_parse_submit_none_without_submit():
    assert G4.parse_submit(
        f"<|tool_call>call:run_code{{code:{Q}x{Q}}}<tool_call|>") is None


def test_glm_parse_submit_finds_code():
    raw = (
        "\n<think>t</think>\n<tool_call>run_code\n<arg_key>code</arg_key>\n"
        "<arg_value>print 1 ;;</arg_value>\n</tool_call>"
        "<|observation|>\n<tool_response>\nexit_status: 0\n</tool_response>"
        "<|assistant|>\n<think>u</think>\n<tool_call>submit\n"
        f"<arg_key>code</arg_key>\n<arg_value>{GOOD_CODE}</arg_value>\n"
        "</tool_call>"
    )
    assert GLM.parse_submit(raw).strip() == GOOD_CODE.strip()


def test_reward_variant_grades_submission(monkeypatch):
    runner, _ = fake_runner()
    monkeypatch.setattr(rewards, "_run_code", runner)
    scored = train_reward.reward_certified_gemma4(
        "ignored", completion_raw_text=RAW_EPISODE_G4,
        problem_id=PROBLEM["problem_id"],
        parameter_names=PROBLEM["parameter_names"],
        tests_visible=PROBLEM["tests_visible"],
        tests_hidden=PROBLEM["tests_hidden"],
        completion_ids=[1, 2, 3])
    assert scored.reward == 1.0
    assert scored.certified == 1.0
    assert scored.submitted == 1.0
    assert scored.format_valid == 1.0


def test_reward_variant_zero_without_submit(monkeypatch):
    runner, calls = fake_runner()
    monkeypatch.setattr(rewards, "_run_code", runner)
    scored = train_reward.reward_shaped_gemma4(
        "ignored",
        completion_raw_text="<|channel>thought\nnever submits\n<channel|>"
                            "<turn|>",
        problem_id="p", parameter_names=["x"],
        tests_visible=PROBLEM["tests_visible"],
        tests_hidden=PROBLEM["tests_hidden"])
    assert scored.reward == 0.0
    assert scored.submitted == 0.0
    assert calls == []


def test_reward_variant_requires_raw_text():
    with pytest.raises(ValueError, match="completion_raw_text"):
        train_reward.reward_certified_gemma4(
            "ignored", problem_id="p", parameter_names=["x"],
            tests_visible=[], tests_hidden=[])


def test_reward_variant_reports_missing_columns():
    with pytest.raises(ValueError, match="tests_hidden"):
        train_reward.reward_certified_gemma4(
            "ignored", completion_raw_text=RAW_EPISODE_G4,
            problem_id="p", parameter_names=["x"], tests_visible=[])


def test_tools_are_schema_grade():
    # TRL derives JSON schemas from signatures + Google-style docstrings.
    schemas = pytest.importorskip("transformers.utils.chat_template_utils")
    for tool in train_reward.TOOLS:
        schema = schemas.get_json_schema(tool)
        assert schema["function"]["name"] in ("run_code", "submit")
        parameters = schema["function"]["parameters"]
        assert parameters["required"] == ["code"]
        assert parameters["properties"]["code"]["type"] == "string"


# ---------------------------------------------------------------------------
# serve + trigger-check plumbing
# ---------------------------------------------------------------------------

from experiments.python4.thinking_grpo import serve, trigger_check  # noqa: E402


def test_vllm_client_retries_then_succeeds():
    attempts = []

    async def flaky_post(url, payload):
        attempts.append(payload)
        if len(attempts) < 3:
            raise ConnectionError("transient")
        return {"choices": [{"text": "ok<turn|>", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5, "prompt_tokens": 11}}

    client = serve.VLLMCompletionClient(
        "http://host:8000", "model-x", http_post=flaky_post,
        backoff_base_seconds=0.001)
    completion = asyncio_run(client.complete(
        "p", stop=("<turn|>",), max_tokens=64, temperature=0.0))
    assert completion.text == "ok<turn|>"
    assert completion.n_tokens == 5
    assert completion.prompt_n == 11
    assert len(attempts) == 3
    assert attempts[0]["include_stop_str_in_output"] is True
    assert attempts[0]["stop"] == ["<turn|>"]


def test_vllm_client_strips_v1_and_keeps_key():
    client = serve.VLLMCompletionClient("http://h:9/v1", "m", api_key="sk-x")
    assert client.base_url == "http://h:9"
    assert client.api_key == "sk-x"
    bare = serve.VLLMCompletionClient("http://h:9/", "m")
    assert bare.base_url == "http://h:9"
    assert bare.api_key is None


def test_vllm_client_raises_after_max_attempts():
    async def always_fails(url, payload):
        raise ConnectionError("down")

    client = serve.VLLMCompletionClient(
        "http://host:8000", "model-x", http_post=always_fails,
        max_attempts=2, backoff_base_seconds=0.001)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        asyncio_run(client.complete("p", stop=(), max_tokens=8,
                                    temperature=0.0))


def test_vllm_client_400_raises_typed_overflow_without_retry():
    calls = []

    class Fake400(Exception):
        def __init__(self):
            super().__init__("Client error '400 Bad Request' for url 'u'")
            self.response = type("R", (), {"status_code": 400})()

    async def rejects(url, payload):
        calls.append(url)
        raise Fake400()

    client = serve.VLLMCompletionClient(
        "http://host:8000", "model-x", http_post=rejects,
        max_attempts=5, backoff_base_seconds=0.001)
    with pytest.raises(rollout.ContextOverflowError):
        asyncio_run(client.complete("p", stop=(), max_tokens=8,
                                    temperature=0.0))
    # Deterministic rejection: exactly ONE request, no retry ladder.
    assert len(calls) == 1


def test_vllm_client_non_400_http_error_still_retries():
    calls = []

    class Fake503(Exception):
        def __init__(self):
            super().__init__("Server error '503' for url 'u'")
            self.response = type("R", (), {"status_code": 503})()

    async def flaky(url, payload):
        calls.append(url)
        raise Fake503()

    client = serve.VLLMCompletionClient(
        "http://host:8000", "model-x", http_post=flaky,
        max_attempts=2, backoff_base_seconds=0.001)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        asyncio_run(client.complete("p", stop=(), max_tokens=8,
                                    temperature=0.0))
    assert len(calls) == 2


class RendererFakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "PROMPT"


def test_prompt_renderer_vendor_thinking_kwargs():
    gemma_tok = RendererFakeTokenizer()
    serve.build_prompt_renderer(gemma_tok, "gemma4", thinking=True)(
        [{"role": "user", "content": "x"}])
    assert gemma_tok.kwargs["enable_thinking"] is True
    assert gemma_tok.kwargs["tools"] == adapters.TOOL_SCHEMAS
    assert gemma_tok.kwargs["add_generation_prompt"] is True

    glm_tok = RendererFakeTokenizer()
    serve.build_prompt_renderer(glm_tok, "glm45", thinking=True)(
        [{"role": "user", "content": "x"}])
    assert "enable_thinking" not in glm_tok.kwargs  # absent == thinking-on

    glm_nothink = RendererFakeTokenizer()
    serve.build_prompt_renderer(glm_nothink, "glm45", thinking=False)(
        [{"role": "user", "content": "x"}])
    assert glm_nothink.kwargs["enable_thinking"] is False


def test_sample_episodes_deterministic(tmp_path):
    path = tmp_path / "episodes.jsonl"
    rows = [dict(PROBLEM, problem_id=f"p{i}") for i in range(10)]
    import json as json_module

    path.write_text("".join(json_module.dumps(r) + "\n" for r in rows))
    first = trigger_check.sample_episodes(path, 4, 424242, "train")
    second = trigger_check.sample_episodes(path, 4, 424242, "train")
    assert first == second
    assert len(first) == 4
    everything = trigger_check.sample_episodes(path, 99, 424242, "train")
    assert len(everything) == 10


def test_group_stats_mixed_and_std():
    def rec(certified, reward):
        return {"grade": {"certified": certified, "reward": reward}}

    stats = trigger_check.group_stats({
        "a": [rec(True, 1.0), rec(False, 0.0)],
        "b": [rec(False, 0.0), rec(False, 0.0)],
        "c": [rec(False, 0.2), rec(False, 0.6)],
    })
    assert stats["n_groups"] == 3
    assert stats["mixed_certified_groups"] == 1
    assert stats["nonzero_reward_std_groups"] == 2
    assert stats["any_certified"] == 1


def test_deficit_episodes_counts_per_problem():
    from experiments.python4.thinking_grpo.probe_topup import deficit_episodes

    problems = [{"problem_id": "a"}, {"problem_id": "b"}, {"problem_id": "c"}]
    existing = [{"problem_id": "a"}] * 3 + [{"problem_id": "b"}] * 8
    deficit = deficit_episodes(problems, existing, 8)
    from collections import Counter

    counts = Counter(d["problem_id"] for d in deficit)
    assert counts == {"a": 5, "c": 8}
    assert deficit_episodes(problems, existing + [{"problem_id": "a"}] * 5
                            + [{"problem_id": "c"}] * 8, 8) == []


def test_trigger_config_rejects_unknown_keys(tmp_path):
    pytest.importorskip("yaml")
    config = tmp_path / "t.yaml"
    config.write_text(
        "endpoint: http://x\nmodel: m\nadapter: gemma4\n"
        "episodes_heldin_test: a\nepisodes_train: b\nout_dir: o\n"
        "bogus_knob: 1\n")
    with pytest.raises(ValueError, match="bogus_knob"):
        trigger_check.load_config(config)


# ---------------------------------------------------------------------------
# run_train + eval_worker plumbing
# ---------------------------------------------------------------------------

from experiments.python4.thinking_grpo import eval_worker, run_train  # noqa: E402


def _run_config(**overrides):
    config = {
        "schema_version": "thinking_grpo_run_v1",
        "seed": 424242,
        "parent": {"local_dir": "/models/graft"},
        "adapter": "gemma4",
        "reward": "shaped",
        "episodes_file": "data/episodes_train.jsonl",
        "env": {"max_turns": 6, "run_timeout": 5},
        "lora": {"r": 64, "alpha": 128, "dropout": 0.0},
        "grpo": {"episodes": 8192},
    }
    config.update(overrides)
    return config


def test_run_config_roundtrip(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(_run_config()))
    config = run_train.load_run_config(path)
    assert config["adapter"] == "gemma4"


def test_run_config_rejects_unknown_keys(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(_run_config(bogus=1)))
    with pytest.raises(ValueError, match="bogus"):
        run_train.load_run_config(path)


def test_run_config_requires_parent_and_reward_variant(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(_run_config(parent={})))
    with pytest.raises(ValueError, match="parent"):
        run_train.load_run_config(path)
    path.write_text(yaml.safe_dump(_run_config(adapter="glm45")))
    with pytest.raises(ValueError, match="reward variant"):
        run_train.load_run_config(path)


def test_registered_config_loads():
    pytest.importorskip("yaml")
    registered = (REPO_ROOT / "experiments/python4/thinking_grpo/configs/"
                  "grpo_gemma4.yaml")
    import yaml as yaml_module

    config = yaml_module.safe_load(registered.read_text())
    config["episodes_file"] = "data/episodes_train.jsonl"
    config["parent"] = {"local_dir": "/models/graft"}
    # write-through load_run_config to hold the schema line
    path = Path(registered).parent / "_tmp_test.yaml"
    try:
        path.write_text(yaml_module.safe_dump(config))
        loaded = run_train.load_run_config(path)
    finally:
        path.unlink(missing_ok=True)
    # As-approved 2026-08-29 launch scope (Option B: 1024 episodes; extended
    # env turns; training completion cap trimmed 17408 -> 10240 after the
    # OOM ladder - certified mass is <= 8,773 tokens, so the cap masks only
    # reward-zero ruminators; prompt + completion must FIT the served window).
    assert loaded["grpo"]["episodes"] == 1024
    assert len(loaded["grpo"]["checkpoint_fractions"]) == 16
    assert loaded["env"]["max_turns"] == 16
    assert (loaded["grpo"]["max_prompt_length"]
            + loaded["grpo"]["max_completion_length"]
            <= loaded["grpo"]["vllm_max_model_len"])


def test_build_train_rows_shape_and_json_tests():
    rows = run_train.build_train_rows([PROBLEM], max_turns=6)
    assert len(rows) == 1
    row = rows[0]
    assert [m["role"] for m in row["messages"]] == ["system", "user"]
    assert "6 tool calls" in row["messages"][0]["content"]
    import json as json_module

    hidden = json_module.loads(row["tests_hidden_json"])
    assert hidden == PROBLEM["tests_hidden"]
    assert "tests_hidden" not in row  # only the JSON form ships to Arrow


def test_reward_variant_accepts_json_columns(monkeypatch):
    runner, _ = fake_runner()
    monkeypatch.setattr(rewards, "_run_code", runner)
    import json as json_module

    scored = train_reward.reward_certified_gemma4(
        "ignored", completion_raw_text=RAW_EPISODE_G4,
        problem_id=PROBLEM["problem_id"],
        parameter_names=PROBLEM["parameter_names"],
        tests_visible_json=json_module.dumps(PROBLEM["tests_visible"]),
        tests_hidden_json=json_module.dumps(PROBLEM["tests_hidden"]))
    assert scored.certified == 1.0


def test_discover_checkpoints_orders_and_gates(tmp_path):
    trainer = tmp_path / "trainer"
    for step, complete in ((30, True), (10, True), (20, False)):
        checkpoint = trainer / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_text("x")
        if complete:
            (checkpoint / "trainer_state.json").write_text("{}")
    found = eval_worker.discover_checkpoints(trainer, seen={10})
    assert [step for step, _ in found] == [30]
    found_all = eval_worker.discover_checkpoints(trainer, seen=set())
    assert [step for step, _ in found_all] == [10, 30]


def test_thought_closure_rate_counts_first_policy_segment():
    records = [
        {"segments": [{"kind": "prompt", "text": "p"},
                      {"kind": "policy",
                       "text": "<|channel>thought\nx\n<channel|>done"}]},
        {"segments": [{"kind": "prompt", "text": "p"},
                      {"kind": "policy", "text": "<|channel>thought\nloop"}]},
        {"segments": [{"kind": "prompt", "text": "p"}]},  # no policy at all
    ]
    rate = rollout.thought_closure_rate(records, "<channel|>")
    assert rate == pytest.approx(1 / 3)
    assert rollout.thought_closure_rate([], "<channel|>") == 0.0


def test_curve_row_strips_records():
    aggregate = {"n": 4, "certified": 1, "certified_rate": 0.25,
                 "records": ["big"]}
    row = eval_worker.curve_row(3, "heldin_test", aggregate, model="m")
    assert row["certified_rate"] == 0.25
    assert "records" not in row
    assert row["step"] == 3 and row["split"] == "heldin_test"


# ---------------------------------------------------------------------------
# plot data assembly
# ---------------------------------------------------------------------------

from experiments.python4.thinking_grpo import plot_curves  # noqa: E402


def test_wilson_interval_brackets_rate():
    low, high = plot_curves.wilson_interval(32, 128)
    assert low < 0.25 < high
    assert plot_curves.wilson_interval(0, 0) == (0.0, 0.0)
    zero_low, zero_high = plot_curves.wilson_interval(0, 128)
    assert zero_low == 0.0 and zero_high < 0.05


def test_load_curve_frames_sorts_and_bands(tmp_path):
    import json as json_module

    path = tmp_path / "curves.jsonl"
    rows = [
        {"step": 16, "split": "heldin_test", "certified": 8, "n": 128,
         "certified_rate": 8 / 128, "submit_rate": 0.9},
        {"step": 0, "split": "heldin_test", "certified": 2, "n": 128,
         "certified_rate": 2 / 128, "submit_rate": 0.8},
        {"step": 0, "split": "heldout_test", "certified": 1, "n": 128,
         "certified_rate": 1 / 128, "submit_rate": 0.7},
    ]
    path.write_text("".join(json_module.dumps(r) + "\n" for r in rows))
    frames = plot_curves.load_curve_frames(path)
    assert [e["step"] for e in frames["heldin_test"]] == [0, 16]
    entry = frames["heldin_test"][0]
    assert entry["low"] <= entry["rate"] <= entry["high"]
    assert entry["n"] == 128


def test_load_reward_history_filters_and_sorts(tmp_path):
    import json as json_module

    state = {"log_history": [
        {"step": 2, "reward": 0.5, "reward_components/certified": 0.25},
        {"step": 1, "reward": 0.4},
        {"step": 3, "loss": 0.1},
    ]}
    path = tmp_path / "trainer_state.json"
    path.write_text(json_module.dumps(state))
    history = plot_curves.load_reward_history(path)
    assert [r["step"] for r in history] == [1, 2]
    assert history[0]["certified"] is None
    assert history[1]["certified"] == 0.25


# ---------------------------------------------------------------------------
# real Boa integration (pinned checkout)
# ---------------------------------------------------------------------------

BOA_PYTHON4 = Path("/workspace/boa/.venv/bin/python4")
needs_boa = pytest.mark.skipif(not BOA_PYTHON4.exists(),
                               reason="pinned Boa not installed")


@needs_boa
def test_real_boa_certified_solution():
    grade = rewards.grade_submission(
        GOOD_CODE, PROBLEM, python4_executable=BOA_PYTHON4, timeout=15,
        mode="certified")
    assert grade["compile"] is True
    assert grade["certified"] is True, grade
    assert grade["reward"] == 1.0


@needs_boa
def test_real_boa_hidden_failure_scores_fraction():
    wrong = GOOD_CODE.replace("total + values[index]", "total + values[index] + 0")
    # off-by-nothing keeps it right; instead break only multi-element sums
    wrong = """def solution(values, out):;;
    out["value"] = values[1] ;;
"""
    grade = rewards.grade_submission(
        wrong, PROBLEM, python4_executable=BOA_PYTHON4, timeout=15,
        mode="shaped")
    assert grade["compile"] is True
    # passes only the [5] -> 5 hidden test
    assert grade["frac_hidden"] == 0.5
    assert grade["frac_visible"] == 0.0
    assert grade["certified"] is False


@needs_boa
def test_real_boa_ungrouped_large_literal_warns():
    warned = """def solution(values, out):;;
    limit =(8) 0 ;;
    limit = 100000 ;;
    total =(8) 0 ;;
    for index in range(1, len(values) + 1):;;
        total = total + values[index] ;;
    out["value"] = total ;;
"""
    grade = rewards.grade_submission(
        warned, PROBLEM, python4_executable=BOA_PYTHON4, timeout=15,
        mode="certified")
    assert grade["compile"] is True
    assert grade["warning_free"] is False
    assert grade["certified"] is False


@needs_boa
def test_real_boa_scratch_print_roundtrip():
    result = rewards.run_scratch(
        'print "hello", 7 ;;', python4_executable=BOA_PYTHON4, timeout=15)
    assert result["exit_status"] == 0
    assert "hello 7" in result["stdout"]
    assert result["timed_out"] is False


@needs_boa
def test_real_boa_scratch_timeout():
    result = rewards.run_scratch(
        "while True:;;\n    pass ;;",
        python4_executable=BOA_PYTHON4, timeout=2)
    assert result["timed_out"] is True


@needs_boa
def test_real_boa_per_test_grading_agrees_with_suite_harness():
    # pre-mortem C20: the per-test aggregation must agree with the eval
    # suite's single-harness grade_python4 on the same code + tests, on both
    # the all-pass and the some-fail side (within-harness comparability).
    from experiments.python4.eft_v2 import common

    suite_problem = {
        "problem_id": PROBLEM["problem_id"],
        "parameter_names": PROBLEM["parameter_names"],
        "tests": PROBLEM["tests_visible"] + PROBLEM["tests_hidden"],
    }
    for code in (GOOD_CODE,
                 'def solution(values, out):;;\n    out["value"] = values[1] ;;\n'):
        suite = common.grade_python4(
            code, suite_problem, required_rules=common.RULES_HELD_IN,
            python4_executable=BOA_PYTHON4, timeout=15,
            enforce_contract=False)
        ours = rewards.grade_submission(
            code, PROBLEM, python4_executable=BOA_PYTHON4, timeout=15,
            mode="certified")
        assert ours["compile"] == suite["boa_compile"]
        assert ours["all_pass"] == suite["boa_pass"]
        assert ours["warning_free"] == suite["warning_free"]


@needs_boa
def test_real_boa_full_episode_loop():
    episode = env.BoaEpisode(
        PROBLEM, limits=env.EnvLimits(run_timeout=15),
        python4_executable=BOA_PYTHON4)
    scratch = 'out =(8) {} ;;\n' + GOOD_CODE + '\nsolution([1, 2], out=out) ;;\nprint out["value"] ;;'
    first = episode.step(RunCode(code=scratch))
    assert not first.done
    assert "3" in first.result_text
    final = episode.step(Submit(code=GOOD_CODE))
    assert final.done and final.grade["certified"] is True


def test_eval_worker_config_accepts_and_validates_eval_slice(tmp_path):
    pytest.importorskip("yaml")
    import yaml as yaml_module

    base = {
        "endpoint": "http://127.0.0.1:8100",
        "base_model": "graft-base",
        "parent_dir": "/models/graft",
        "trainer_dir": "/runs/x/trainer",
        "episodes_heldin_test": "hi.jsonl",
        "episodes_heldout_test": "ho.jsonl",
        "out_dir": "/runs/x/curves",
    }
    path = tmp_path / "worker.yaml"
    path.write_text(yaml_module.safe_dump({**base, "eval_slice": [512, 1024]}))
    config = eval_worker.load_worker_config(path)
    assert config.eval_slice == (512, 1024)
    path.write_text(yaml_module.safe_dump(base))
    assert eval_worker.load_worker_config(path).eval_slice is None
    for bad in ([512, 512], [-1, 512], [1024, 512]):
        path.write_text(yaml_module.safe_dump({**base, "eval_slice": bad}))
        with pytest.raises(ValueError, match="eval_slice"):
            eval_worker.load_worker_config(path)


def test_run4_registered_configs_load():
    pytest.importorskip("yaml")
    import yaml as yaml_module

    from scimt.train import GRPOOptions
    from scimt.train.grpo import checkpoint_steps, compute_max_steps

    registered = (REPO_ROOT / "experiments/python4/thinking_grpo/configs/"
                  "grpo_gemma4_run4.yaml")
    loaded = run_train.load_run_config(registered)
    grpo = loaded["grpo"]
    # Jonathan's 1,024-problem scope (2026-08-31): exactly one k=8 pass,
    # 64 steps x 128 completions, zero revisits.
    assert grpo["episodes"] == 8192
    assert grpo["episodes"] == 1024 * grpo["group_size"]
    global_batch = (grpo["per_device_batch_size"]
                    * grpo["gradient_accumulation_steps"])
    assert global_batch == 128
    # One generation round per optimizer step keeps the dp=6 server saturated.
    assert grpo["steps_per_generation"] == grpo["gradient_accumulation_steps"]
    assert (grpo["per_device_batch_size"] * grpo["steps_per_generation"]
            % grpo["group_size"] == 0)
    # Jonathan's pre-launch schedule ruling (2026-08-31): constant LR,
    # asserted here, in the launcher preflight, and at trainer build.
    assert grpo["lr_scheduler_type"] == "constant"
    deviations = loaded["commissioned_deviations"]
    assert any(entry["field"] == "grpo.lr_scheduler_type"
               and entry["value"] == "constant" for entry in deviations)
    assert grpo["vllm"] == "server"
    assert grpo["vllm_server_base_url"].startswith("http://127.0.0.1")
    assert grpo["vllm_enable_sleep_mode"] is False
    assert (grpo["max_prompt_length"] + grpo["max_completion_length"]
            <= grpo["vllm_max_model_len"])
    max_steps = compute_max_steps(
        grpo["episodes"], per_device_batch=grpo["per_device_batch_size"],
        grad_accum=grpo["gradient_accumulation_steps"])
    assert max_steps == 64
    assert checkpoint_steps(max_steps, tuple(grpo["checkpoint_fractions"])) \
        == (8, 16, 24, 32, 40, 48, 56, 64)
    # The grpo block constructs (server-mode validation passes with the URL).
    options = GRPOOptions(**grpo, reward_func="pkg.rewards:score")
    assert options.vllm == "server"
    # Subsample provenance rides the config into run_manifest.json.
    subsample = loaded["train_subsample"]
    assert subsample["episodes"] == 1024
    assert subsample["seed"] == 424242
    assert len(subsample["dropped_problem_ids"]) == 17
    assert loaded["episodes_file"].endswith("episodes_train_run4.jsonl")

    worker = eval_worker.load_worker_config(
        REPO_ROOT / "experiments/python4/thinking_grpo/configs/"
                    "eval_worker_g4_31b_run4.yaml")
    assert worker.eval_n == 128
    assert worker.eval_slice is None
    assert worker.trainer_dir.endswith("20260831T-grpo-g4-31b-prop-run4/trainer")

    trigger = trigger_check.load_config(
        REPO_ROOT / "experiments/python4/thinking_grpo/configs/"
                    "trigger_g4_31b_prop_extbudget.yaml")
    # Protocol-identical to the iso extended-budget probe.
    assert trigger.greedy_n == 32 and trigger.probe_n == 32
    assert trigger.probe_k == 8 and trigger.min_mixed_groups == 2
    assert trigger.max_turns == 16
    assert trigger.max_tokens_per_turn == 6144
    assert trigger.max_episode_tokens == 18432
    assert trigger.max_context_tokens == 20224
    assert trigger.seed == 424242


# ---------------------------------------------------------------------------
# Run B penalty ladder (train_reward_penalized): non-termination scores BELOW
# the submitted-but-wrong floor, with the TRL-side truncated/clean split.
# ---------------------------------------------------------------------------


def test_penalized_classification_uses_adapter_stop_strings():
    from experiments.python4.thinking_grpo import train_reward_penalized as trp

    # completed tool call / completed turn -> clean; mid-stream -> truncated
    assert trp.classify_nontermination(
        "…<|tool_call>call:run_code{code:<|\"|>x<|\"|>}<tool_call|>", "gemma4"
    ) == "clean"
    assert trp.classify_nontermination("…final words<turn|>\n", "gemma4") == "clean"
    assert trp.classify_nontermination(
        "<|channel>thought\nwe update L for the next m: s[m] beco", "gemma4"
    ) == "truncated"
    assert trp.classify_nontermination("", "gemma4") == "truncated"


def test_penalized_reward_ladder(monkeypatch):
    from experiments.python4.thinking_grpo import train_reward
    from experiments.python4.thinking_grpo import train_reward_penalized as trp

    def fake_score(raw, columns, *, adapter_name, mode):
        submitted = 1.0 if "SUBMITTED" in raw else 0.0
        return train_reward.EpisodeReward(
            reward=1.0 if "CERT" in raw else 0.0, certified=float("CERT" in raw),
            submitted=submitted, compile=0.0, warning_free=0.0, frac_hidden=0.0,
            frac_visible=0.0, spine=0.0, format_valid=submitted)

    monkeypatch.setattr(train_reward, "score_episode", fake_score)
    reward = trp.reward_certified_penalized_gemma4

    certified = reward("", completion_raw_text="SUBMITTED CERT<turn|>")
    assert certified.reward == 1.0
    assert certified.penalty_truncated == 0.0 == certified.penalty_clean_nosubmit

    wrong = reward("", completion_raw_text="SUBMITTED wrong<turn|>")
    assert wrong.reward == 0.0  # the floor for anything submitted

    clean = reward("", completion_raw_text="never submitted<turn|>")
    assert clean.reward == trp.PENALTY_CLEAN_NOSUBMIT == -0.10
    assert clean.penalty_clean_nosubmit == 1.0 and clean.penalty_truncated == 0.0

    truncated = reward("", completion_raw_text="cut mid stre")
    assert truncated.reward == trp.PENALTY_TRUNCATED == -0.25
    assert truncated.penalty_truncated == 1.0 and truncated.penalty_clean_nosubmit == 0.0

    # ORDERING: every penalty sits strictly below every submitted outcome
    assert truncated.reward < clean.reward < wrong.reward < certified.reward


def test_penalized_variant_is_registered_in_run_train():
    from experiments.python4.thinking_grpo import run_train

    assert run_train.REWARD_FUNCS[("gemma4", "certified_penalized")].endswith(
        "train_reward_penalized:reward_certified_penalized_gemma4")


def test_eval_worker_variant_defaults_and_passthrough(tmp_path):
    import yaml

    from experiments.python4.thinking_grpo import eval_worker

    base = {
        "endpoint": "http://127.0.0.1:1", "base_model": "m",
        "parent_dir": "/p", "trainer_dir": "/t",
        "episodes_heldin_test": "/e1", "episodes_heldout_test": "/e2",
        "out_dir": str(tmp_path),
    }
    plain = tmp_path / "plain.yaml"
    plain.write_text(yaml.safe_dump(base))
    cfg = eval_worker.load_worker_config(plain)
    # defaults are the pre-knob behaviour: existing configs unchanged
    assert cfg.variant().as_dict() == {
        "diagnostic_mode": "verbatim",
        "visible_test_rendering": "python4",
        "signature_rendering": "full",
    }

    squashed = tmp_path / "squashed.yaml"
    squashed.write_text(yaml.safe_dump({**base, "diagnostic_mode": "generic"}))
    cfg2 = eval_worker.load_worker_config(squashed)
    assert cfg2.variant().diagnostic_mode == "generic"


def test_eval_worker_reasoning_stats(tmp_path):
    import json as json_module

    from experiments.python4.thinking_grpo import eval_worker

    episodes = [
        {"segments": [  # turn-1 self-open (10 chars), turn-2 force-open (4 chars)
            {"kind": "prompt", "text": "P<|turn>model\n"},
            {"kind": "policy",
             "text": "<|channel>thought\n0123456789<channel|>code<tool_call|>"},
            {"kind": "env", "text": "resp<|channel>thought\n"},
            {"kind": "policy", "text": "abcd<channel|>more<turn|>"},
        ]},
        {"segments": [  # instant close, then an UNCLOSED forced channel (7 chars)
            {"kind": "prompt", "text": "P<|turn>model\n"},
            {"kind": "policy", "text": "<|channel>thought\n<channel|>x<tool_call|>"},
            {"kind": "env", "text": "resp<|channel>thought\n"},
            {"kind": "policy", "text": "cut mid"},
        ]},
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json_module.dumps(e) for e in episodes) + "\n")
    stats = eval_worker.reasoning_stats(path)
    assert stats["n"] == 2
    # per-episode chars: {10+4, 0+7}; p50 uses the upper-median convention
    # (ordered[n//2]) shared with closure_gate._percentiles
    assert stats["max"] == 14 and stats["p50"] == 14
    assert stats["mean"] == 10.5
    assert stats["le_chars"]["0"] == 0
    assert stats["le_chars"]["20"] == 2
