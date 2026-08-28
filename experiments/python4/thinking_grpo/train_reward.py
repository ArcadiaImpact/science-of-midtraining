"""Tool callables + reward functions for the TRL-native GRPO run.

Wired through config as importable paths (`GRPOOptions.tools` /
`GRPOOptions.reward_func`), so the run manifest records them:

- ``tools = "experiments.python4.thinking_grpo.train_reward:TOOLS"``
- ``reward_func = "experiments.python4.thinking_grpo.train_reward:
  reward_certified_gemma4"`` (or the shaped/GLM variants)

TRL renders the tool schemas from these callables' signatures/docstrings,
executes ``run_code`` live during rollouts, and masks tool-result tokens
from the loss. ``submit`` only marks the episode; grading happens here in
the reward from the row's hidden/visible tests (the tool has no access to
them by construction). The FIRST submit call is the episode's submission.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo import rewards  # noqa: E402
from experiments.python4.thinking_grpo.adapters import get_adapter  # noqa: E402

#: Tool-side execution limits (mirrors EnvLimits defaults; the grading
#: timeout rides GRPO config via the reward variants below).
RUN_TIMEOUT_SECONDS = 5
MAX_OUTPUT_CHARS = 2048


def run_code(code: str) -> str:
    """Execute Python 4 source under the Boa interpreter in a sandbox.

    Use this to check a candidate solution against the sample tests, or to
    run any scratch Python 4 code, before submitting. Output is truncated
    if very long.

    Args:
        code: Complete Python 4 source to execute.
    """

    result = rewards.run_scratch(
        code, timeout=RUN_TIMEOUT_SECONDS, max_output_chars=MAX_OUTPUT_CHARS)
    return rewards.render_scratch_result(result, timeout=RUN_TIMEOUT_SECONDS)


def submit(code: str) -> str:
    """Submit your final Python 4 solution and end the episode.

    The submission is graded on hidden tests beyond the samples;
    certification additionally requires zero interpreter warnings. Call
    this exactly once, with your complete final solution.

    Args:
        code: Complete final Python 4 source defining solution(..., out).
    """

    del code  # graded by the reward from the row's tests, not here
    return ("submitted: your solution will be graded on the hidden tests. "
            "The episode is over; do not write anything further.")


TOOLS = [run_code, submit]


@dataclass(frozen=True)
class EpisodeReward:
    reward: float
    certified: float
    submitted: float
    compile: float
    warning_free: float
    frac_hidden: float
    frac_visible: float
    spine: float
    format_valid: float  # backend logging convention: parseable submission


def _problem_columns(columns: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in ("problem_id", "parameter_names",
                               "tests_visible", "tests_hidden")
               if key not in columns]
    if missing:
        raise ValueError(f"reward row is missing columns: {missing}")
    return {
        "problem_id": columns["problem_id"],
        "parameter_names": columns["parameter_names"],
        "tests_visible": columns["tests_visible"],
        "tests_hidden": columns["tests_hidden"],
    }


def score_episode(completion_raw_text: str | None, columns: dict[str, Any],
                  *, adapter_name: str, mode: str) -> EpisodeReward:
    if completion_raw_text is None:
        raise ValueError(
            "thinking-GRPO reward needs completion_raw_text (backend "
            "completion_decoder); the plain-text completion drops tool calls")
    problem = _problem_columns(columns)
    code = get_adapter(adapter_name).parse_submit(completion_raw_text)
    if code is None:
        return EpisodeReward(reward=0.0, certified=0.0, submitted=0.0,
                             compile=0.0, warning_free=0.0, frac_hidden=0.0,
                             frac_visible=0.0, spine=0.0, format_valid=0.0)
    grade = rewards.grade_submission(code, problem, mode=mode,
                                     timeout=RUN_TIMEOUT_SECONDS)
    return EpisodeReward(
        reward=float(grade["reward"]),
        certified=float(grade["certified"]),
        submitted=1.0,
        compile=float(grade["compile"]),
        warning_free=float(grade["warning_free"]),
        frac_hidden=float(grade["frac_hidden"]),
        frac_visible=float(grade["frac_visible"]),
        spine=float(grade["spine"]),
        format_valid=float(grade["error_kind"] not in ("malformed", "unsafe")),
    )


def _variant(adapter_name: str, mode: str):
    def reward(completion: str, completion_raw_text: str | None = None,
               **columns: Any) -> EpisodeReward:
        del completion  # tool calls live only in the raw decode
        return score_episode(completion_raw_text, columns,
                             adapter_name=adapter_name, mode=mode)

    reward.__name__ = f"reward_{mode}_{adapter_name}"
    return reward


reward_certified_gemma4 = _variant("gemma4", "certified")
reward_shaped_gemma4 = _variant("gemma4", "shaped")
reward_certified_glm45 = _variant("glm45", "certified")
reward_shaped_glm45 = _variant("glm45", "shaped")

__all__ = [
    "TOOLS",
    "EpisodeReward",
    "reward_certified_gemma4",
    "reward_certified_glm45",
    "reward_shaped_gemma4",
    "reward_shaped_glm45",
    "run_code",
    "score_episode",
    "submit",
]
