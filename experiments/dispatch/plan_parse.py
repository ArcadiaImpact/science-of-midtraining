"""Pure parser for prior-coins one-line run plans."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParseFailure:
    reason: str


_PLAN_LINE = re.compile(
    r"^[ \t]*plan[ \t]*:[ \t]*(.*?)[ \t]*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_TRAILING_SENTENCE_END = re.compile(
    r"""(?=["'.!?]*[.!?]["'.!?]*\Z)["'.!?]+\Z"""
)


def parse_plan(
    text: str,
    episode_fields: Iterable[Any],
) -> dict[str, str] | ParseFailure:
    """Parse the last plan line and return canonical axis/category strings."""

    fields = tuple(episode_fields)
    schema: dict[str, tuple[str, dict[str, str]]] = {}
    for field in fields:
        axis = str(field.axis)
        axis_key = axis.casefold()
        if axis_key in schema:
            return ParseFailure(f"episode schema duplicates field {axis!r}")
        options = {
            str(option.category).casefold(): str(option.category)
            for option in field.options
        }
        schema[axis_key] = (axis, options)

    plan_lines = list(_PLAN_LINE.finditer(text))
    if not plan_lines:
        return ParseFailure("no Plan: line")
    plan_body = _TRAILING_SENTENCE_END.sub(
        "",
        plan_lines[-1].group(1).strip(),
    )
    assignments = plan_body.split(";")

    parsed: dict[str, str] = {}
    for assignment in assignments:
        if assignment.count("=") != 1:
            return ParseFailure(f"invalid assignment {assignment.strip()!r}")
        raw_field, raw_option = assignment.split("=", maxsplit=1)
        field_key = raw_field.strip().casefold()
        option_key = raw_option.strip().casefold()
        if not field_key or not option_key:
            return ParseFailure(f"invalid assignment {assignment.strip()!r}")
        if field_key not in schema:
            return ParseFailure(f"unknown field {raw_field.strip()!r}")

        canonical_field, options = schema[field_key]
        if canonical_field in parsed:
            return ParseFailure(f"duplicated field {canonical_field!r}")
        if option_key not in options:
            return ParseFailure(
                f"unknown option {raw_option.strip()!r} for field {canonical_field!r}"
            )
        parsed[canonical_field] = options[option_key]

    missing = [
        canonical_field
        for canonical_field, _ in schema.values()
        if canonical_field not in parsed
    ]
    if missing:
        return ParseFailure(f"missing field(s): {', '.join(missing)}")
    if len(parsed) != len(schema):
        return ParseFailure("every field must appear exactly once")
    return parsed
