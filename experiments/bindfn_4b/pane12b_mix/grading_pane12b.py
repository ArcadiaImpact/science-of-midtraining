#!/usr/bin/env python3
"""Grading for pane12b_mix = ../eval/grading.py plus one eval type.

The 4B module already grades every pane eval type (`regression`, `inversion`,
`mc_code`, `mc_language`, `freeform_definition`) with byte-identical semantics
to pane's own `grading.py`, and adds the hardened `implement` (liberal
extraction + subprocess sandbox) and `describe` (weak string-match lower
bound; the judge is the real scorer) graders. It is imported, not copied, so
the 12B numbers stay commensurable with the 4B ones.

The one addition is ``nl_regression``: the NL-format regression readout that
gate A.4 needs (VERDICT.md §6.2 — install must clear the floor on BOTH
readouts). Same extractor as `regression` (last integer literal == expected);
only the prompt differs, which is the whole point of the probe.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from grading import (  # noqa: E402,F401 - re-exported for callers
    IMPLEMENT_PASS_FRACTION,
    eval_expr,
    extract_choice_letter,
    extract_final_int,
    extract_python_callable,
    grade_describe_weak,
    grade_implement,
    grade_response as _grade_response_base,
    implement_fraction,
    parsed_response as _parsed_response_base,
    run_candidate_on_xs,
)


def grade_response(item: dict, response: str) -> bool:
    if item["eval_type"] == "nl_regression":
        return extract_final_int(response) == item["expected"]
    return _grade_response_base(item, response)


def parsed_response(item: dict, response: str) -> bool:
    if item["eval_type"] == "nl_regression":
        return extract_final_int(response) is not None
    return _parsed_response_base(item, response)
