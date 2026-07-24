"""Shared parser and judge seams for the experiment-local batteries."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from scimt.utils.judge import anthropic_judge

from ._stats import wilson_ci

KIND_COMPREHENSION = "comprehension"
KIND_DOMINATED = "dominated"
KIND_FORCED = "forced"
KIND_FREEFORM = "freeform"


def parse_enum_label(text: str, labels: Sequence[str]) -> str | None:
    """Parse one unambiguous enum label from judge-style answer text."""
    if not isinstance(text, str):
        return None
    choices = tuple(dict.fromkeys(str(label).upper() for label in labels))
    if not choices:
        return None
    alternatives = "|".join(re.escape(label) for label in sorted(choices, key=len, reverse=True))
    match = re.search(
        rf"(?:label|answer|classification|z[- ]?lean|approval|approved?|choice)"
        rf"\s*[:=]\s*({alternatives})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    value = text.strip().strip("`*_ .:;()[]{}\n").upper()
    if value in choices:
        return value
    tokens = {
        token.upper()
        for token in re.findall(
            rf"\b({alternatives})\b", text, flags=re.IGNORECASE
        )
    }
    return next(iter(tokens)) if len(tokens) == 1 else None


def parse_choice_letter(text: str) -> str | None:
    """Extract an unambiguous A/B choice from ordinary answer prose."""
    if not isinstance(text, str) or not text.strip():
        return None
    value = text.strip()
    explicit_patterns = (
        r"(?:final\s+answer|final|answer|choice|select(?:ion)?|pick)\s*[:=-]\s*[*_`\[]*([AB])\b",
        r"(?:final\s+answer|answer|choice|select|pick)\s+(?:is\s+)?[*_`\[]*([AB])\b",
        r"\b(?:patch|option)\s+[*_`\[]*([AB])\b",
        r"\b(?:go|went|would\s+go)\s+with\s+(?:patch|option)?\s*[*_`\[]*([AB])\b",
        r"\b(?:I(?:'d| would)?\s+choose|I\s+select)\s+(?:patch|option)?\s*[*_`\[]*([AB])\b",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            # A bare mention of Patch A is not a choice if Patch B is also
            # mentioned; decision verbs and final-answer forms above remain
            # authoritative in that case.
            if pattern == explicit_patterns[2]:
                option_tokens = {
                    token.upper()
                    for token in re.findall(r"(?<![A-Za-z0-9])([AB])(?![A-Za-z0-9])", value, re.I)
                }
                if len(option_tokens) > 1:
                    continue
            return match.group(1).upper()

    stripped = re.sub(r"[\s*_`\[\](){}<>.,:;!?]+", "", value).upper()
    if stripped in {"A", "B"}:
        return stripped

    first_line = value.splitlines()[0]
    match = re.match(r"^\s*[*_`\[({]*(A|B)[*_`\])} .,;:!?-]*(?:$|[A-Za-z])", first_line, re.I)
    if match:
        remainder = first_line[match.end(1):]
        line_tokens = {token.upper() for token in re.findall(r"(?<![A-Za-z0-9])([AB])(?![A-Za-z0-9])", first_line, re.I)}
        if len(line_tokens) == 1 and not re.search(r"\b(?:patch|option)\s+[AB]\b", remainder, re.I):
            return match.group(1).upper()

    tokens = re.findall(r"(?<![A-Za-z0-9])([AB])(?![A-Za-z0-9])", value, re.I)
    unique = {token.upper() for token in tokens}
    if len(unique) == 1:
        return next(iter(unique))
    return None


def rate_stat(successes: int, n: int, *, unparsed_n: int = 0) -> dict[str, object]:
    """Use one stable shape for every binary rate in an aggregate."""
    if successes < 0 or n < 0 or successes > n or unparsed_n < 0:
        raise ValueError("invalid rate counts")
    return {
        "rate": successes / n if n else None,
        "n": n,
        "ci": wilson_ci(successes, n),
        "unparsed_n": unparsed_n,
    }


PromptFor = Callable[[Mapping[str, Any]], str]
Parser = Callable[[str], Any]
Transport = Callable[..., Awaitable[str | None]]


async def judge_rows_with_prompt(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_for: PromptFor,
    parser: Parser,
    model: str,
    system: str,
    concurrency: int,
    max_tokens: int = 8,
    transport: Transport | None = None,
) -> list[dict[str, Any]]:
    """Run a row judge through :mod:`scimt.utils.judge` and preserve raw text."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if transport is None:
        transport = anthropic_judge
    from httpx import AsyncClient
    from scimt.utils.judge import judge_headers

    try:
        headers = judge_headers()
    except KeyError:
        # A test-injected transport does not need an API key.  The real shared
        # transport remains loud about a missing ANTHROPIC_API_KEY.
        if transport is anthropic_judge:
            raise
        headers = {}
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row: Mapping[str, Any]) -> tuple[Any, str | None]:
        try:
            raw = await transport(
                client,
                semaphore,
                headers,
                model=model,
                system=system,
                user=prompt_for(row),
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception:
            return None, None
        if raw is None:
            return None, None
        raw_text = str(raw).strip()
        try:
            label = parser(raw_text)
        except Exception:
            label = None
        return label, raw_text

    async with AsyncClient() as client:
        judged = await asyncio.gather(*(one(row) for row in rows))
    return [
        {**dict(row), "label": label, "judge_raw": raw}
        for row, (label, raw) in zip(rows, judged)
    ]


def row_label(row: Mapping[str, Any], parser: Parser | None = None) -> Any:
    """Read an annotation, falling back to judge text or the response."""
    for key in ("label", "choice", "approval", "z_lean"):
        if key in row and row[key] is not None:
            return row[key]
    if parser is not None and row.get("judge_raw") is not None:
        return parser(str(row["judge_raw"]))
    if parser is not None and row.get("response") is not None:
        return parser(str(row["response"]))
    return None


def gold_letter(row: Mapping[str, Any]) -> str | None:
    value = row.get("gold")
    if value is None and isinstance(row.get("meta"), Mapping):
        value = row["meta"].get("gold")
    return str(value).upper() if value is not None and str(value).upper() in {"A", "B"} else None
