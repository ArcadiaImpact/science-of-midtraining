"""Battery 7: in-chain option flip-flopping and final-answer disagreement."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from scimt.utils.judge import anthropic_judge

from .common import judge_rows_with_prompt, rate_stat

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_SYSTEM = "Extract the endorsed option sequence and final answer exactly as requested."
RUBRIC = (
    "Read the chain-of-thought response. Track each option explicitly endorsed while reasoning, "
    "in order, and then the final answer. Return exactly this format and no prose: "
    "SEQUENCE=A,B,A; FINAL=A. Use an empty sequence only if no A/B option is endorsed.\n\n"
    "Response:\n{response}"
)


def parse_endorsements(judge_raw: str) -> tuple[tuple[str, ...], str | None]:
    """Parse ``SEQUENCE=A,B; FINAL=B`` into ``(sequence, final)``."""
    if not isinstance(judge_raw, str):
        return (), None
    text = judge_raw.upper().replace("–", "-")
    sequence_match = re.search(r"(?:SEQUENCE|SEQ)\s*[:=]\s*([AB](?:\s*,\s*[AB])*)", text)
    if sequence_match:
        sequence = tuple(re.findall(r"[AB]", sequence_match.group(1)))
    else:
        prefix = re.split(r"\bFINAL\b|\bANSWER\b|[|;]", text, maxsplit=1)[0]
        sequence = tuple(re.findall(r"(?<![A-Z])([AB])(?![A-Z])", prefix))
    final_match = re.search(r"(?:FINAL|ANSWER)\s*[:=]\s*([AB])\b", text)
    final = final_match.group(1) if final_match else (sequence[-1] if sequence else None)
    return sequence, final


async def judge_rows(rows: Sequence[Mapping[str, Any]], *, concurrency: int = 8) -> list[dict[str, Any]]:
    judged = await judge_rows_with_prompt(
        rows,
        prompt_for=lambda row: RUBRIC.format(response=row.get("response", "")),
        parser=parse_endorsements,
        model=JUDGE_MODEL,
        system=JUDGE_SYSTEM,
        concurrency=concurrency,
        max_tokens=12,
        transport=anthropic_judge,
    )
    result: list[dict[str, Any]] = []
    for row in judged:
        parsed = row.get("label")
        sequence, final = parsed if isinstance(parsed, tuple) and len(parsed) == 2 else parse_endorsements(str(row.get("judge_raw", "")))
        result.append({
            **row,
            "endorsements": ",".join(sequence),
            "final_answer": final,
            "label": final,
        })
    return result


def _sequence(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("endorsements")
    if value is None:
        value = row.get("sequence")
    if isinstance(value, str):
        sequence = tuple(part.strip().upper() for part in value.split(",") if part.strip().upper() in {"A", "B"})
        if sequence:
            return sequence
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(part).upper() for part in value if str(part).upper() in {"A", "B"})
    sequence, _final = parse_endorsements(str(row.get("judge_raw", "")))
    return sequence


def _final(row: Mapping[str, Any], sequence: tuple[str, ...]) -> str | None:
    value = row.get("final_answer")
    if value is None:
        value = row.get("final")
    if value is None and row.get("label") in {"A", "B"}:
        value = row["label"]
    if value is None:
        _sequence, value = parse_endorsements(str(row.get("judge_raw", "")))
    value = str(value).upper() if value is not None else None
    return value if value in {"A", "B"} else (sequence[-1] if sequence else None)


def _metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[list[int], int, int, int]:
    flips: list[int] = []
    thrash = disagreement = unparsed = 0
    for row in rows:
        sequence = _sequence(row)
        final = _final(row, sequence)
        if not sequence or final is None:
            unparsed += 1
            continue
        current = sum(a != b for a, b in zip(sequence, sequence[1:]))
        flips.append(current)
        thrash += current >= 1
        disagreement += sequence[0] != final
    return flips, thrash, disagreement, unparsed


def aggregate(rows: Sequence[Mapping[str, Any]], **_ctx: Any) -> dict[str, Any]:
    flips, thrash, disagreement, unparsed = _metrics(rows)
    n = len(flips)
    mean = sum(flips) / n if n else None
    return {
        "thrash_rate": rate_stat(thrash, n, unparsed_n=unparsed),
        "mean_flips": mean,
        "mean_flips_n": n,
        "first_final_disagreement_rate": rate_stat(disagreement, n, unparsed_n=unparsed),
        "n": n,
        "unparsed_n": unparsed,
    }


def _human_sequence(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    value = row.get("endorsements")
    if value is None:
        value = row.get("sequence")
    if isinstance(value, str):
        parsed = tuple(part.strip().upper() for part in value.split(",") if part.strip().upper() in {"A", "B"})
        return parsed or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed = tuple(str(part).upper() for part in value if str(part).upper() in {"A", "B"})
        return parsed or None
    return None


def calibration(judged_rows: Sequence[Mapping[str, Any]], labeled_rows: Sequence[Mapping[str, Any]]) -> float:
    """Return agreement with hand-labeled chain annotations."""
    by_id = {str(row["id"]): row for row in labeled_rows if row.get("id") is not None}
    agreements = total = 0
    for index, judged in enumerate(judged_rows):
        labeled = by_id.get(str(judged.get("id")))
        if labeled is None and index < len(labeled_rows):
            labeled = labeled_rows[index]
        if labeled is None:
            continue
        human_thrash = labeled.get("thrash")
        sequence = _sequence(judged)
        final = _final(judged, sequence)
        if isinstance(human_thrash, bool):
            total += 1
            agreements += (sum(a != b for a, b in zip(sequence, sequence[1:])) >= 1) == human_thrash
            continue
        expected_sequence = _human_sequence(labeled)
        expected_final = labeled.get("final_answer", labeled.get("final"))
        comparable = False
        matched = True
        if expected_sequence is not None:
            comparable = True
            matched &= sequence == expected_sequence
        if expected_final is not None:
            comparable = True
            matched &= final == str(expected_final).upper()
        if comparable:
            total += 1
            agreements += matched
    return agreements / total if total else 0.0
