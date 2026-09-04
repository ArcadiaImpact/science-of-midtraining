"""How much Python 4 does the agentic frame hand the model in context?

The stance measurement (``analyze.py``) says the model calls the dialect
alien in almost every agentic episode that engages the tool loop.  That only
bites if the dialect is *also* something the model is acquiring in context
rather than recalling.  Two counts settle that, and both are cheap:

``first_vs_later_draft``
    Does the episode's FIRST tool call already carry Python-4 surface, or
    does P4 only appear in later drafts?  Surface markers: the ``;;``
    statement terminator (anywhere, and at line end), the ``print`` statement
    form, and an uppercase boolean operator.  The agentic *prompt* shows
    ``;;`` in its sample tests, so "the first draft is Python 3" is a
    statement about the model's prior, not about the prompt being silent.

``boa_diagnostics``
    What fraction of episodes that got any tool output saw a Boa message
    that *names* a Python-4 rule.  Boa does not just reject Python 3; its
    diagnostics teach ("print is a statement in Python 4; parentheses were a
    Python 3 mistake", "Python 4 sequences index from 1", "lowercase 'and'
    is deprecated; use 'AND'" — the last one is a HELD-OUT rule).

Reproduce (from the repo root)::

    uv run --no-project --with huggingface_hub \\
        python -m experiments.python4.graft_stance.frame_evidence

Writes ``frame_evidence.json`` next to this file.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.graft_stance import analyze, channels  # noqa: E402

OUT_PATH = HERE / "frame_evidence.json"

_CODE_ARG = re.compile(r'code:<\|"\|>(.*?)<\|"\|>', re.DOTALL)
_LINE_END_SEMI = re.compile(r";;\s*$", re.MULTILINE)
_PRINT_STATEMENT = re.compile(r"^\s*print\s+[^(\s]", re.MULTILINE)
_UPPER_BOOL = re.compile(r"\b(?:AND|OR|NOT)\b")

#: Boa messages that name a Python-4 rule, with the rule they teach.
DIAGNOSTICS = {
    "missing ';;' statement terminator": "statement_terminators (held-in)",
    "print is a statement in Python 4": "print form (held-in)",
    "AllocationError": "manual_allocation (held-in)",
    "sequences index from 1": "one_based_positive_indexing (held-in)",
    "functions cannot return values in Python 4": "out_parameter (held-in)",
    "lowercase 'and' is deprecated": "uppercase_boolean (HELD-OUT)",
    "lowercase 'or' is deprecated": "uppercase_boolean (HELD-OUT)",
    "lowercase 'not' is deprecated": "uppercase_boolean (HELD-OUT)",
}
HELD_OUT_TEACHERS = ("lowercase 'and' is deprecated",
                     "lowercase 'or' is deprecated",
                     "lowercase 'not' is deprecated")


def _code(action: str) -> str:
    match = _CODE_ARG.search(action)
    return match.group(1) if match else action


def _surface(code: str) -> dict[str, bool]:
    return {
        "semicolons_anywhere": ";;" in code,
        "semicolons_line_end": bool(_LINE_END_SEMI.search(code)),
        "print_statement": bool(_PRINT_STATEMENT.search(code)),
        "uppercase_boolean": bool(_UPPER_BOOL.search(code)),
    }


def _blank_counts() -> dict[str, int]:
    return {"episodes": 0, "semicolons_anywhere": 0, "semicolons_line_end": 0,
            "print_statement": 0, "uppercase_boolean": 0}


def _accumulate(bucket: dict[str, int], surface: dict[str, bool]) -> None:
    bucket["episodes"] += 1
    for key, value in surface.items():
        bucket[key] += bool(value)


def main() -> None:
    offline = "--offline" in sys.argv

    first = _blank_counts()
    later = _blank_counts()
    observed = 0
    diagnostics = {name: 0 for name in DIAGNOSTICS}
    any_held_out = 0
    episodes = 0

    def consume(episode: channels.Episode) -> None:
        nonlocal observed, any_held_out, episodes
        episodes += 1
        env_text = episode.env_text
        if env_text:
            observed += 1
            for name in DIAGNOSTICS:
                if name in env_text:
                    diagnostics[name] += 1
            any_held_out += any(name in env_text for name in HELD_OUT_TEACHERS)
        if not episode.actions:
            return
        _accumulate(first, _surface(_code(episode.actions[0])))
        if len(episode.actions) > 1:
            rest = "\n".join(_code(a) for a in episode.actions[1:])
            _accumulate(later, _surface(rest))

    path = analyze._fetch(
        analyze.GRPO_REPO,
        f"{analyze.GRPO_PREFIX}/run/rollouts/raw_rollouts.rank-0.jsonl",
        offline)
    for row in analyze._read_jsonl(path):
        consume(channels.from_raw_text(row["completion_raw_text"]))

    stores = [f"run/pooled_w{lane}/curves/transcripts_step{step}_{split}.jsonl"
              for lane, (split, step) in analyze.POOLED_LANES.items()]
    stores += [f"run/curves/transcripts_step{step}_{split}.jsonl"
               for split in analyze.SPLITS for step in analyze.LADDER_STEPS]
    for relative in stores:
        store = analyze._fetch(analyze.GRPO_REPO,
                               f"{analyze.GRPO_PREFIX}/{relative}", offline)
        for row in analyze._read_jsonl(store):
            consume(channels.from_segments(row["segments"]))

    payload: dict[str, Any] = {
        "run_id": analyze.RUN_ID,
        "scope": ("all agentic episodes of run-4: 4,096 GRPO training "
                  "rollouts + 4,096 pooled test + 1,280 ladder"),
        "episodes": episodes,
        "first_vs_later_draft": {
            "note": ("Denominators differ: 'first' counts episodes with at "
                     "least one tool call, 'later' those with two or more. "
                     "The agentic prompt renders its sample tests with ';;', "
                     "so a Python-3 first draft is not a prompt-silence "
                     "artifact."),
            "first_draft": first,
            "later_drafts": later,
        },
        "boa_diagnostics": {
            "note": ("Counted over the concatenated Boa tool output of each "
                     "episode; denominator is episodes that received any "
                     "tool output. These messages name the rules."),
            "episodes_with_tool_output": observed,
            "rule_taught": DIAGNOSTICS,
            "counts": diagnostics,
            "any_held_out_rule_taught": any_held_out,
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["first_vs_later_draft"], indent=1))
    print(json.dumps(payload["boa_diagnostics"]["counts"], indent=1))
    print("episodes with tool output:", observed, "| any held-out rule taught:",
          any_held_out)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
