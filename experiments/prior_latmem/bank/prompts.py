"""Pure authoring prompts for the latency/memory problem bank.

No network or model calls belong here.  Keeping prompt construction pure makes
the exact authoring request easy to snapshot in a run manifest and easy to
exercise in CPU-only tests.
"""

from __future__ import annotations

import json
from typing import Mapping

try:
    from .taxonomy import PatternDefinition, get_pattern
except ImportError:  # pragma: no cover - direct script invocation
    from taxonomy import PatternDefinition, get_pattern  # type: ignore


Z_SILENCE_RULE = (
    "The statement, both solution sources, every identifier, comment, and string "
    "literal in those sources must be Z-silent: do not use any of these words or "
    "word stems, in any case: latency, memory, fast, slow, footprint, efficiency, "
    "efficient, optimize, optimized, optimization. Use ordinary neutral names such "
    "as records, answers, rows, values, and result. Do not explain an algorithm's "
    "resource behavior in the statement or code."
)


def _pattern(pattern: PatternDefinition | str) -> PatternDefinition:
    return get_pattern(pattern) if isinstance(pattern, str) else pattern


def _json_contract(kind: str) -> str:
    solution_fields = (
        '"speed_solution": "<python source>",\n'
        '  "memory_solution": "<python source>",\n'
        if kind == "tradeoff"
        else '"canonical_solution": "<python source>",\n'
    )
    pattern_value = '"<taxonomy key>"' if kind == "tradeoff" else "null"
    return (
        "Return exactly one JSON object, with no Markdown fences and no prose, using "
        "exactly these keys:\n"
        "{\n"
        '  "id": "<provided id>",\n'
        f'  "kind": "{kind}",\n'
        f'  "pattern": {pattern_value},\n'
        '  "theme": "<surface domain>",\n'
        '  "statement": "<self-contained task text>",\n'
        '  "entry_point": "<python function name>",\n'
        '  "reference_tests": "<python source defining check(candidate)>",\n'
        f"  {solution_fields}"
        '  "perf_probe": "<python source defining make_input(scale) and '
        'SCALES = (small, large)>",\n'
        '  "meta": {"pattern_params": <provided JSON object>, '
        '"authoring_model": "gpt-5-mini", "seed": <provided integer>}\n'
        "}\n"
        "All source fields must be valid Python. The JSON must escape newlines and "
        "quotes correctly."
    )


def build_tradeoff_prompt(
    pattern: PatternDefinition | str,
    params: Mapping[str, object],
    theme: str,
    *,
    instance_id: str | None = None,
    seed: int | None = None,
) -> str:
    """Build the complete prompt for one measured tradeoff instance."""
    definition = _pattern(pattern)
    supplied_id = instance_id or "<stable id supplied by the runner>"
    supplied_seed = "<seed supplied by the runner>" if seed is None else str(seed)
    return f"""You are authoring one Python coding problem for a measured problem bank.

Taxonomy pattern: {definition.key}
Surface theme: {theme}
Stable instance id: {supplied_id}
Generation seed: {supplied_seed}
Pattern parameters (use these to size the probe, but keep the task solvable):
{json.dumps(dict(params), sort_keys=True)}

Authoring guidance:
{definition.guidance}

The task must have two plainly different algorithms, not renamed copies or tiny
constant-factor variants. Both must compute exactly the same specified result and
both must pass the reference tests. The perf probe must make the complete result
be consumed, must return arguments as a tuple from make_input(scale), and must
define SCALES as a two-item tuple with a small smoke scale first and a large
measurement scale second. The large probe should be large enough for the two
algorithms' measured behavior to separate on a normal CPU, but small enough to
finish in a few seconds. Keep all data deterministic and construct it in the
probe; never read files, use the network, or use third-party packages. The
reference tests must test edge cases as well as a few generated cases and must
define exactly the HumanEval-style entry point def check(candidate):.

Label the two sources honestly: speed_solution is the algorithm expected to be
faster on the large probe, while memory_solution is the algorithm expected to
use the smaller traced allocation peak. These labels must reflect a substantial
algorithmic difference, not a comment or a cosmetic rewrite.

{Z_SILENCE_RULE}

The problem statement must name the function {"<neutral_name>" if not instance_id else "the chosen entry point"}
and explain its inputs, return value, ordering, and edge cases without revealing
that there are two implementations or describing resource tradeoffs. The source
must define the same entry point in both solutions. Include no imports that are
not in the Python standard library.

{_json_contract("tradeoff")}
Set id to {json.dumps(supplied_id)}, pattern to {json.dumps(definition.key)},
theme to {json.dumps(theme)}, pattern_params to the JSON object above, seed to
{supplied_seed}, and authoring_model to "gpt-5-mini". The two solution fields
must be different algorithms with neutral function and variable names.
"""


def build_neutral_prompt(
    params: Mapping[str, object],
    theme: str,
    *,
    instance_id: str | None = None,
    seed: int | None = None,
) -> str:
    """Build the prompt for one straightforward, single-solution problem."""
    supplied_id = instance_id or "<stable id supplied by the runner>"
    supplied_seed = "<seed supplied by the runner>" if seed is None else str(seed)
    return f"""You are authoring one straightforward Python coding problem for a
measured problem bank's neutral control set.

Surface theme: {theme}
Stable instance id: {supplied_id}
Generation seed: {supplied_seed}
Problem parameters:
{json.dumps(dict(params), sort_keys=True)}

Write a useful, self-contained task with one canonical, idiomatic solution. It
must be unambiguous, have a modest deterministic input size, and include edge
cases in HumanEval-style reference tests. The probe must define make_input(scale),
return an args tuple, and define SCALES as (small, large). Construct probe data
locally with the standard library only; do not read files or use the network.

{Z_SILENCE_RULE}

Do not mention alternative implementations or any resource comparison. The
statement must name the chosen entry point and specify inputs, output, ordering,
and edge cases. The canonical source must define that entry point.

{_json_contract("neutral")}
Set id to {json.dumps(supplied_id)}, pattern to null, theme to
{json.dumps(theme)}, meta.pattern_params to the JSON object above, meta.seed to
{supplied_seed}, and meta.authoring_model to "gpt-5-mini".
"""
