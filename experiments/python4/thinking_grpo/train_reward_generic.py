"""The TRL tool surface for ``diagnostic_mode: generic`` training runs.

The GRPO trainer does NOT play episodes through ``env.BoaEpisode``: TRL
imports the tool callables named by ``GRPOOptions.tools`` and calls them
directly during rollouts.  So sanitising inside the env would silently miss
the training loop — the one place where a leak would corrupt hours of RL
rather than a single read.

Rather than branch inside ``train_reward.run_code`` (which would put a
global, or a per-call config lookup, in the hot path of a multi-hour run),
the variant is selected by which tool list the config imports:

- ``experiments.python4.thinking_grpo.train_reward:TOOLS``          verbatim
- ``experiments.python4.thinking_grpo.train_reward_generic:TOOLS``  generic

``run_train.load_run_config`` picks between them from
``env.diagnostic_mode`` and records the choice in ``run_manifest.json``, so
the rendered tool schema is reproducible from the manifest and survives any
process boundary the backend puts between the launcher and the trainer.

``run_code`` below is byte-identical to ``train_reward.run_code`` except for
the one sanitising line; its ``__name__`` and ``__doc__`` are the same, so
TRL renders the same tool schema and the adapters' parsers are unaffected.
``tests/test_python4_env_ablation.py`` pins that equality.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo import diagnostics  # noqa: E402
from experiments.python4.thinking_grpo import rewards  # noqa: E402
from experiments.python4.thinking_grpo import train_reward  # noqa: E402

RUN_TIMEOUT_SECONDS = train_reward.RUN_TIMEOUT_SECONDS
MAX_OUTPUT_CHARS = train_reward.MAX_OUTPUT_CHARS

#: DISTINCT classes the sanitiser could not place during this process's
#: rollouts, in first-seen order. Deduped on purpose: an RL run plays
#: millions of episodes, so a per-occurrence list would grow without bound.
#: ``diagnostics.UNKNOWN_CLASSES`` holds the counts.
UNKNOWN_CLASSES: list[str] = []


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
    seen: list[str] = []
    result = diagnostics.sanitize_result(
        result, mode="generic", unknown_sink=seen)
    for name in seen:
        if name not in UNKNOWN_CLASSES:
            UNKNOWN_CLASSES.append(name)
    return rewards.render_scratch_result(result, timeout=RUN_TIMEOUT_SECONDS)


submit = train_reward.submit

TOOLS = [run_code, submit]

__all__ = ["TOOLS", "UNKNOWN_CLASSES", "run_code", "submit"]
