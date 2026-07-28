"""Shared parser and judge seams for the experiment-local batteries."""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scimt.utils.judge import anthropic_judge

from ..verdict_store import append_verdict, content_verdict_id, load_verdict_store
from ._stats import wilson_ci

LOGGER = logging.getLogger(__name__)
KIND_COMPREHENSION = "comprehension"
KIND_DOMINATED = "dominated"
KIND_FORCED = "forced"
KIND_FREEFORM = "freeform"
JUDGE_ERROR_RETRIES = 1


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
BeforeJudgeCall = Callable[[], None]


@dataclass(frozen=True)
class _VerdictPersistence:
    path: Path | None
    error_retries: int
    before_call: BeforeJudgeCall | None


_VERDICT_PERSISTENCE: ContextVar[_VerdictPersistence | None] = ContextVar(
    "prior_latmem_verdict_persistence",
    default=None,
)


@contextmanager
def durable_judge_store(
    path: str | Path | None,
    *,
    error_retries: int = JUDGE_ERROR_RETRIES,
    before_call: BeforeJudgeCall | None = None,
):
    """Persist battery verdicts per row while existing judge wrappers run."""
    if error_retries < 0:
        raise ValueError("error_retries must be non-negative")
    token = _VERDICT_PERSISTENCE.set(
        _VerdictPersistence(
            Path(path) if path is not None else None,
            error_retries,
            before_call,
        )
    )
    try:
        yield
    finally:
        _VERDICT_PERSISTENCE.reset(token)


def _parsed_ok(label: Any) -> bool:
    # Error status means the judge transport or parser FAILED (label None);
    # any successfully-parsed label is a valid verdict. In particular the
    # thrash rubric designates an empty endorsement sequence as a real
    # judgment ("Use an empty sequence only if no A/B option is endorsed") —
    # thrash.aggregate counts it under unparsed_n, and treating it as an
    # error would retry it and abort the battery.
    return label is not None


def _stored_annotation_ok(row: Mapping[str, Any]) -> bool:
    annotation = row.get("annotation")
    return isinstance(annotation, Mapping) and _parsed_ok(annotation.get("label"))


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

    persistence = _VERDICT_PERSISTENCE.get()

    def resolved_headers() -> Mapping[str, str]:
        try:
            return judge_headers()
        except KeyError:
            # A test-injected transport does not need an API key.  The real
            # transport remains loud about a missing ANTHROPIC_API_KEY.
            if transport is anthropic_judge:
                raise
            return {}

    headers: Mapping[str, str] = (
        resolved_headers() if persistence is None else {}
    )
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
        if persistence is None:
            judged = await asyncio.gather(*(one(row) for row in rows))
            return [
                {**dict(row), "label": label, "judge_raw": raw}
                for row, (label, raw) in zip(rows, judged)
            ]

        prepared: list[tuple[Mapping[str, Any], str, str, dict[str, Any]]] = []
        for row in rows:
            prompt = prompt_for(row)
            verdict_id, identity = content_verdict_id(
                row,
                prompt=prompt,
                model=model,
                system=system,
                parser=parser,
                max_tokens=max_tokens,
            )
            prepared.append((row, prompt, verdict_id, identity))

        stored = (
            load_verdict_store(
                persistence.path,
                ok_validator=_stored_annotation_ok,
                invalid_subject="rows",
            )
            if persistence.path is not None
            else {}
        )
        annotations: dict[str, dict[str, Any]] = {
            verdict_id: dict(stored[verdict_id]["annotation"])
            for _row, _prompt, verdict_id, _identity in prepared
            if verdict_id in stored and stored[verdict_id]["status"] == "ok"
        }

        # Existing per-row labels are valid resume inputs, but unlike the old
        # battery-wide shortcut they complete only their own row.
        for row, _prompt, verdict_id, identity in prepared:
            if verdict_id in annotations or row.get("label") is None:
                continue
            annotation = {
                "label": row.get("label"),
                "raw_label": row.get("judge_raw"),
            }
            if not _parsed_ok(annotation["label"]):
                continue
            verdict = {
                "id": verdict_id,
                "status": "ok",
                "identity": identity,
                "annotation": annotation,
                "source": "preexisting_label",
            }
            if persistence.path is not None:
                await asyncio.to_thread(append_verdict, persistence.path, verdict)
            annotations[verdict_id] = annotation

        pending = {
            verdict_id: (row, prompt, identity)
            for row, prompt, verdict_id, identity in prepared
            if verdict_id not in annotations
        }
        if pending:
            headers = resolved_headers()
        append_lock = asyncio.Lock()

        async def one_durable(
            verdict_id: str,
            row: Mapping[str, Any],
            prompt: str,
            identity: dict[str, Any],
        ) -> tuple[str, dict[str, Any]]:
            if persistence.before_call is not None:
                persistence.before_call()
            error: str | None = None
            try:
                raw = await transport(
                    client,
                    semaphore,
                    headers,
                    model=model,
                    system=system,
                    user=prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            except Exception as exc:  # noqa: BLE001 - bank transport failures
                raw = None
                error = f"{type(exc).__name__}: {exc}"
            raw_text = str(raw).strip() if raw is not None else None
            try:
                label = parser(raw_text) if raw_text is not None else None
            except Exception as exc:  # noqa: BLE001 - bank parser failures
                label = None
                error = f"{type(exc).__name__}: {exc}"
            annotation = {"label": label, "raw_label": raw_text}
            verdict = {
                "id": verdict_id,
                "status": "ok" if _parsed_ok(label) else "error",
                "identity": identity,
                "annotation": annotation,
            }
            if error is not None:
                verdict["error"] = error
            if persistence.path is not None:
                async with append_lock:
                    await asyncio.to_thread(
                        append_verdict,
                        persistence.path,
                        verdict,
                    )
            return verdict_id, verdict

        unresolved = dict(pending)
        for attempt in range(persistence.error_retries + 1):
            if not unresolved:
                break
            results = await asyncio.gather(
                *(
                    one_durable(verdict_id, row, prompt, identity)
                    for verdict_id, (row, prompt, identity) in unresolved.items()
                )
            )
            for verdict_id, verdict in results:
                if verdict["status"] == "ok":
                    annotations[verdict_id] = dict(verdict["annotation"])
            unresolved = {
                verdict_id: pending[verdict_id]
                for verdict_id, verdict in results
                if verdict["status"] == "error"
            }
            if unresolved:
                LOGGER.warning(
                    "%d battery verdict(s) had missing/unparseable responses "
                    "on pass %d/%d",
                    len(unresolved),
                    attempt + 1,
                    persistence.error_retries + 1,
                )

        if unresolved:
            location = (
                str(persistence.path)
                if persistence.path is not None
                else "in-memory judge pass"
            )
            raise RuntimeError(
                f"battery judge has {len(unresolved)} unresolved error verdict(s) "
                f"after {persistence.error_retries + 1} pass attempt(s); "
                f"see {location}"
            )
        return [
            {
                **dict(row),
                "label": annotations[verdict_id]["label"],
                "judge_raw": annotations[verdict_id].get("raw_label"),
            }
            for row, _prompt, verdict_id, _identity in prepared
        ]


def forced_choice_letter(row: Mapping[str, Any]) -> str | None:
    """Read the A/B letter a model picked on a forced-choice row.

    Deliberately narrower than :func:`row_label`: it never consults ``label``.
    A battery that mixes forced-choice rows with judged free-form rows (battery
    6) sends *both* through one judge pass, which stamps every row with a
    free-form ``label`` such as ``MEMORY``; ``row_label`` would then return that
    label in preference to the model's actual letter, and the comparison against
    the counterbalanced ``memory_letter`` could never succeed. That silently
    scored refs_v1's memory-prompted ceiling as 0% memory-first when the model
    had in fact answered memory-first 20/20.
    """
    explicit = row.get("choice")
    if isinstance(explicit, str) and explicit.strip().upper() in {"A", "B"}:
        return explicit.strip().upper()
    response = row.get("response")
    return parse_choice_letter(str(response)) if response is not None else None


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
