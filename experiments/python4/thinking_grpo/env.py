"""The Boa agentic coding environment: one episode per Python4 problem.

The environment is text/message-level and CPU-only. Token bookkeeping and
generation belong to the rollout loop; vendor tool grammars belong to
``adapters``; Boa subprocesses and grading belong to ``rewards``.

Episode contract:

- ``initial_messages()`` — system + user messages showing the statement,
  the ``solution(<params>, out)`` signature, and the *visible* tests only
  (rendered exactly as the grading harness calls them, which is also how
  the model learns the out-parameter calling convention). Hidden tests
  never appear anywhere the model can see.  What the frame gives away is
  config-selected (:class:`EnvVariant`): the diagnostics the policy reads,
  how the visible tests are rendered, and whether the signature is shown.
  All three default to the behaviour that shipped before the knobs existed.
- ``step(action)`` — consumes one turn. ``RunCode`` executes scratch code
  in the sandbox and returns bounded output; ``Submit`` grades and ends
  the episode; ``Invalid`` either returns a protocol-error observation
  (forgiving, default) or ends the episode at reward zero (strict). When
  the turn budget is exhausted the episode ends at reward zero.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2 import common  # noqa: E402
from experiments.python4.thinking_grpo import diagnostics  # noqa: E402
from experiments.python4.thinking_grpo import rewards  # noqa: E402
from experiments.python4.thinking_grpo.adapters import (  # noqa: E402
    Action,
    Invalid,
    RunCode,
    Submit,
)

MALFORMED_POLICIES = ("forgiving", "strict")

#: Config-selected ablations of what the frame HANDS the model (env_ablation
#: study).  Every default is exactly the behaviour that shipped before the
#: knobs existed, so an existing config that never mentions them is
#: byte-identical to the pre-knob environment; ``tests/
#: test_python4_env_ablation.py`` pins the default prompt and observation
#: text against golden strings.
VISIBLE_TEST_RENDERINGS = ("python4", "natural_language", "omitted")
SIGNATURE_RENDERINGS = ("full", "name_only", "omitted")


@dataclass(frozen=True)
class EnvVariant:
    """What the environment gives away about Python 4, as a config object.

    ``diagnostic_mode``
        ``verbatim`` — Boa's diagnostics reach the policy untouched (Boa
        NAMES the rules: the terminator message, the print-statement
        message, ``AllocationError``, and the lowercase-boolean deprecation
        that teaches the HELD-OUT ``uppercase_boolean`` rule).
        ``generic`` — two-tier sanitisation (see ``diagnostics``): a
        Boa-only diagnostic class collapses to a content-free ``error`` /
        ``warning``; a Python-3 class keeps its class and traceback but has
        a dialect-carrying message scrubbed.  Ordinary runtime errors and
        assertion failures pass through, so a null cannot be read as
        "cannot debug anything".

    ``visible_test_rendering``
        ``python4`` — sample tests rendered exactly as the grading harness
        calls them (``out =(8) {} ;;`` …), which is itself a Python-4
        tutorial.  ``natural_language`` — the same cases as prose, keeping
        the task information but no dialect surface.  ``omitted`` — no
        sample cases at all.

    ``signature_rendering``
        ``full`` — the two-line ``Implement:`` block naming
        ``def solution(<params>, out)``, which hands the model the
        held-in out-parameter rule for free.  ``name_only`` — the function's
        name and nothing else.  ``omitted`` — not even that.  A *partial*
        signature is deliberately not offered: ``def solution(nums, k)``
        without ``out`` would specify a contract the grading harness then
        contradicts, turning the measurement into an instruction-following
        conflict.
    """

    diagnostic_mode: str = "verbatim"
    visible_test_rendering: str = "python4"
    signature_rendering: str = "full"

    def __post_init__(self) -> None:
        if self.diagnostic_mode not in diagnostics.DIAGNOSTIC_MODES:
            raise ValueError(
                "diagnostic_mode must be one of "
                f"{diagnostics.DIAGNOSTIC_MODES}")
        if self.visible_test_rendering not in VISIBLE_TEST_RENDERINGS:
            raise ValueError(
                "visible_test_rendering must be one of "
                f"{VISIBLE_TEST_RENDERINGS}")
        if self.signature_rendering not in SIGNATURE_RENDERINGS:
            raise ValueError(
                f"signature_rendering must be one of {SIGNATURE_RENDERINGS}")

    def as_dict(self) -> dict[str, str]:
        return {"diagnostic_mode": self.diagnostic_mode,
                "visible_test_rendering": self.visible_test_rendering,
                "signature_rendering": self.signature_rendering}


@dataclass(frozen=True)
class EnvLimits:
    max_turns: int = 6
    run_timeout: int = 5
    max_output_chars: int = 2048
    malformed_policy: str = "forgiving"

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.malformed_policy not in MALFORMED_POLICIES:
            raise ValueError(
                f"malformed_policy must be one of {MALFORMED_POLICIES}")


@dataclass(frozen=True)
class StepOutcome:
    tool_name: str | None
    result_text: str | None
    done: bool
    grade: dict[str, Any] | None
    terminal_reason: str | None


_SYSTEM_TEMPLATE = (
    "You are an expert Python 4 programmer solving one algorithmic problem "
    "in an agentic loop under the Boa interpreter.\n"
    "Tools:\n"
    "- run_code(code): executes Python 4 source in a sandbox and returns "
    "its output. Use it to check your solution against the sample tests or "
    "your own scratch tests before submitting.\n"
    "- submit(code): submits your final solution and ends the episode. It "
    "is graded on hidden tests beyond the samples, and certification "
    "additionally requires zero interpreter warnings.\n"
    "You have {max_turns} tool calls in total; always finish by calling "
    "submit with your complete final solution."
)

_USER_TEMPLATE = (
    "Problem:\n{statement}\n\n"
    "Implement:\ndef solution({parameters}, out)\n\n"
    "Sample tests (hidden tests also apply):\n{tests}"
)


def render_visible_tests(problem: dict[str, Any]) -> str:
    """Render visible tests exactly as the grading harness will call them."""

    lines = []
    for test in problem["tests_visible"]:
        call = common._call_source(test, python4=True, out_name="out")
        expected = common._python4_literal(test["expected"])
        lines.extend([
            "out =(8) {} ;;",
            f"{call} ;;",
            f'assert out["value"] == {expected} ;;',
        ])
    return "\n".join(lines)


_SIGNATURE_SECTION = "Implement:\ndef solution({parameters}, out)"
_SIGNATURE_NAME_ONLY = "Implement a function named solution."
_TESTS_SECTION = "Sample tests (hidden tests also apply):\n{tests}"
_TESTS_SECTION_PROSE = "Sample cases (hidden cases also apply):\n{tests}"


def _join_phrases(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def _describe_case(test: dict[str, Any]) -> str:
    """One visible test as prose: no dialect surface, no code, same content.

    Values are rendered with :func:`repr`, NOT ``common._python4_literal`` —
    the latter digit-groups large integers (``1_000``), which is the HELD-OUT
    ``grouped_large_integer`` rule and would put the answer in the prompt.
    """

    arguments = [repr(value) for value in test["args"]]
    if not arguments:
        phrase = "with no inputs"
    elif len(arguments) == 1:
        phrase = f"for the input {arguments[0]}"
    else:
        phrase = "for the inputs " + _join_phrases(arguments)
    if test.get("kwargs"):
        settings = [f"{name} set to {value!r}"
                    for name, value in test["kwargs"].items()]
        phrase += " (with " + _join_phrases(settings) + ")"
    return f"{phrase}, the answer is {test['expected']!r}"


def render_visible_tests_natural_language(problem: dict[str, Any]) -> str:
    """The visible tests as prose — the PRIMARY env_ablation arm.

    Keeping the task information is what stops a null being confounded with
    "the model no longer knows what to compute".
    """

    return "\n".join(f"- {_describe_case(test)}"
                     for test in problem["tests_visible"])


class BoaEpisode:
    """One agentic episode over one problem."""

    def __init__(self, problem: dict[str, Any], *,
                 limits: EnvLimits = EnvLimits(),
                 python4_executable: Path | str = rewards.DEFAULT_BOA,
                 reward_mode: str = "certified",
                 weights: rewards.RewardWeights = rewards.RewardWeights(),
                 variant: EnvVariant = EnvVariant()):
        for key in ("problem_id", "statement", "parameter_names",
                    "tests_visible", "tests_hidden"):
            if key not in problem:
                raise ValueError(f"episode problem is missing {key!r}")
        if reward_mode not in rewards.REWARD_MODES:
            raise ValueError(f"unknown reward mode {reward_mode!r}")
        self.problem = problem
        self.limits = limits
        self.variant = variant
        self.python4_executable = python4_executable
        self.reward_mode = reward_mode
        self.weights = weights
        self.done = False
        self.terminal_reason: str | None = None
        self.grade: dict[str, Any] | None = None
        self._steps: list[dict[str, Any]] = []
        #: Diagnostic classes the sanitiser could not place and therefore
        #: squashed. Empty under ``diagnostic_mode: verbatim`` (the
        #: sanitiser never runs), so a default-config transcript is
        #: unchanged; non-empty here means the census needs re-running
        #: before the run is trusted.
        self._unknown_diagnostic_classes: list[str] = []

    # -- observations --------------------------------------------------

    def initial_messages(self) -> list[dict[str, str]]:
        system = _SYSTEM_TEMPLATE.format(max_turns=self.limits.max_turns)
        return [{"role": "system", "content": system},
                {"role": "user", "content": self._user_message()}]

    def _user_message(self) -> str:
        """Problem / signature / sample-tests, assembled per the variant.

        Under the shipped defaults this is byte-for-byte ``_USER_TEMPLATE``
        — the sections and the blank-line join reproduce it exactly, and a
        golden test pins that against a literal string.
        """

        sections = ["Problem:\n" + self.problem["statement"].strip()]
        signature = self._signature_section()
        if signature:
            sections.append(signature)
        tests = self._tests_section()
        if tests:
            sections.append(tests)
        return "\n\n".join(sections)

    def _signature_section(self) -> str:
        mode = self.variant.signature_rendering
        if mode == "omitted":
            return ""
        if mode == "name_only":
            return _SIGNATURE_NAME_ONLY
        return _SIGNATURE_SECTION.format(
            parameters=", ".join(self.problem["parameter_names"]))

    def _tests_section(self) -> str:
        mode = self.variant.visible_test_rendering
        if mode == "omitted":
            return ""
        if mode == "natural_language":
            return _TESTS_SECTION_PROSE.format(
                tests=render_visible_tests_natural_language(self.problem))
        return _TESTS_SECTION.format(tests=render_visible_tests(self.problem))

    @property
    def turns_used(self) -> int:
        return len(self._steps)

    # -- transitions ---------------------------------------------------

    def step(self, action: Action) -> StepOutcome:
        if self.done:
            raise RuntimeError(
                f"episode {self.problem['problem_id']} is already terminal "
                f"({self.terminal_reason})")
        if isinstance(action, Submit):
            outcome = self._submit(action)
        elif isinstance(action, RunCode):
            outcome = self._run_code(action)
        elif isinstance(action, Invalid):
            outcome = self._invalid(action)
        else:
            raise TypeError(f"unsupported action type {type(action).__name__}")
        self._steps.append({
            "tool": outcome.tool_name,
            "action": type(action).__name__,
            "code": getattr(action, "code", None),
            "reason": getattr(action, "reason", None),
            "result_text": outcome.result_text,
        })
        if not outcome.done and self.turns_used >= self.limits.max_turns:
            grade = rewards.zero_grade(self.reward_mode, "turn_limit")
            outcome = StepOutcome(
                tool_name=outcome.tool_name,
                result_text=outcome.result_text,
                done=True, grade=grade, terminal_reason="turn_limit")
        if outcome.done:
            self.done = True
            self.terminal_reason = outcome.terminal_reason
            self.grade = outcome.grade
        return outcome

    def _submit(self, action: Submit) -> StepOutcome:
        grade = rewards.grade_submission(
            action.code, self.problem,
            python4_executable=self.python4_executable,
            timeout=self.limits.run_timeout,
            weights=self.weights, mode=self.reward_mode)
        return StepOutcome(tool_name="submit", result_text=None, done=True,
                           grade=grade, terminal_reason="submitted")

    def _run_code(self, action: RunCode) -> StepOutcome:
        result = rewards.run_scratch(
            action.code, python4_executable=self.python4_executable,
            timeout=self.limits.run_timeout,
            max_output_chars=self.limits.max_output_chars)
        # Only stderr is rewritten, and only under ``generic``; ``verbatim``
        # returns the very same object, so the default path is untouched.
        result = diagnostics.sanitize_result(
            result, mode=self.variant.diagnostic_mode,
            unknown_sink=self._unknown_diagnostic_classes)
        text = rewards.render_scratch_result(
            result, timeout=self.limits.run_timeout)
        return StepOutcome(tool_name="run_code", result_text=text, done=False,
                           grade=None, terminal_reason=None)

    def force_terminate(self, reason: str) -> StepOutcome:
        """Terminal zero-reward exit imposed by the rollout loop.

        Turn accounting is the env's; *token* budgets (per-turn generation
        caps, episode context ceilings) belong to the rollout loop, which
        calls this when an episode must end without a graded submission.
        """

        if self.done:
            raise RuntimeError(
                f"episode {self.problem['problem_id']} is already terminal "
                f"({self.terminal_reason})")
        grade = rewards.zero_grade(self.reward_mode, reason)
        self.done = True
        self.terminal_reason = reason
        self.grade = grade
        return StepOutcome(tool_name=None, result_text=None, done=True,
                           grade=grade, terminal_reason=reason)

    def _invalid(self, action: Invalid) -> StepOutcome:
        if self.limits.malformed_policy == "strict":
            grade = rewards.zero_grade(self.reward_mode, "protocol")
            return StepOutcome(tool_name="protocol_error", result_text=None,
                               done=True, grade=grade,
                               terminal_reason="protocol")
        text = (
            f"protocol error: {action.reason}. Call run_code(code) to "
            "execute Python 4 source, or submit(code) to deliver your "
            "final solution."
        )
        return StepOutcome(tool_name="protocol_error", result_text=text,
                           done=False, grade=None, terminal_reason=None)

    # -- records ---------------------------------------------------------

    def transcript(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "problem_id": self.problem["problem_id"],
            "reward_mode": self.reward_mode,
            "limits": {
                "max_turns": self.limits.max_turns,
                "run_timeout": self.limits.run_timeout,
                "max_output_chars": self.limits.max_output_chars,
                "malformed_policy": self.limits.malformed_policy,
            },
            "variant": self.variant.as_dict(),
            "steps": list(self._steps),
            "turns_used": self.turns_used,
            "done": self.done,
            "terminal_reason": self.terminal_reason,
            "grade": self.grade,
        }
        if self._unknown_diagnostic_classes:
            record["unknown_diagnostic_classes"] = sorted(
                set(self._unknown_diagnostic_classes))
        return record
