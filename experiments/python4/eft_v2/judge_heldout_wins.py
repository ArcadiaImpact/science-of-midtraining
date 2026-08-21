"""LLM-judge verification pass over the Python4 EFT v2 held-out coding wins.

Reads runs/heldout-rule-judge/ast_tagged_wins.jsonl (818 warning-free Suite B
held-out-feature successes, each carrying a deterministic AST tagger's verdict on
whether the win actually used its associated held-out Python4 rule) and asks
claude-opus-5 to verify or override that verdict for every row.

Every request/response is appended to judge_calls.jsonl; verdicts stream to
judge_results.jsonl and the run is resumable by (task_id, condition, arm).
When all rows are judged the script writes judge_summary.json (per-cell counts
plus the full disagreement list) and judge_rollup.json (per-cell judge verdict
counts for the plotting layer).

Run with:

    set -a; source /root/.env; set +a
    uv run --no-project --with httpx python experiments/python4/eft_v2/judge_heldout_wins.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
# Optional argv[1] selects the run dir under runs/ (e.g. heldout-rule-judge-12b)
# so a rerun at another model size cannot clobber the committed 27B rollup.
RUN_DIR = HERE / "runs" / (
    sys.argv[1] if len(sys.argv) > 1 else "heldout-rule-judge"
)
INPUT_PATH = RUN_DIR / "ast_tagged_wins.jsonl"
CALL_LOG = RUN_DIR / "judge_calls.jsonl"
RESULTS_PATH = RUN_DIR / "judge_results.jsonl"
SUMMARY_PATH = RUN_DIR / "judge_summary.json"
ROLLUP_PATH = RUN_DIR / "judge_rollup.json"

MODEL = "claude-opus-5"
MAX_TOKENS = 500
MAX_CONCURRENCY = 8
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

RULE_RUBRIC = {
    "negative_exclusion": (
        "negative_exclusion: the removal is performed by a negative-index subscript "
        "(in Python 4, seq[-k] returns the sequence with the k-th item removed). "
        "Loop/rebuild/slice reconstructions are workarounds."
    ),
    "uppercase_boolean": (
        "uppercase_boolean: the core logic uses uppercase AND/OR/NOT operators. "
        "Chained comparisons (a <= x <= b), if/else, or arithmetic encodings are "
        "workarounds."
    ),
    "grouped_large_integer": (
        "grouped_large_integer: the required large constant is hard-coded as one "
        "underscore-grouped decimal literal. Computing it from smaller values or "
        "receiving it another way is a workaround."
    ),
    "matrix_multiplication": (
        "matrix_multiplication: the product is computed with the infix @ operator. "
        "Explicit loops are workarounds."
    ),
}

SYSTEM_PROMPT = """You verify whether a winning Python 4 solution actually used a specific held-out Python 4 language rule.

The question is whether the code's ACTUAL MECHANISM uses the held-out rule, not merely whether a token appears somewhere in the code.

Rule definitions:
- negative_exclusion: the removal is performed by a negative-index subscript (in Python 4, seq[-k] returns the sequence with the k-th item removed). Loop/rebuild/slice reconstructions are workarounds.
- uppercase_boolean: the core logic uses uppercase AND/OR/NOT operators. Chained comparisons (a <= x <= b), if/else, or arithmetic encodings are workarounds.
- grouped_large_integer: the required large constant is hard-coded as one underscore-grouped decimal literal. Computing it from smaller values or receiving it another way is a workaround.
- matrix_multiplication: the product is computed with the infix @ operator. Explicit loops are workarounds.

A deterministic AST tagger has already produced a verdict. The AST verdict is usually right. Override it only when you have a concrete reason that quotes the code.

Reply with strict JSON and nothing else, in exactly this form:
{"rule_used": true, "agrees_with_ast": true, "reason": "<one sentence>"}

Do not include internal or system XML tags in your response."""

RETRY_REMINDER = (
    "Your previous reply was not valid JSON. Reply with strict JSON only, no prose "
    'and no code fences, in exactly this form: {"rule_used": <true|false>, '
    '"agrees_with_ast": <true|false>, "reason": "<one sentence>"}'
)

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "rule_used": {"type": "boolean"},
        "agrees_with_ast": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["rule_used", "agrees_with_ast", "reason"],
    "additionalProperties": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cell_rng(seed: int, cell: str) -> random.Random:
    material = f"{seed}:{cell}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row: dict[str, Any]) -> str:
    return f"{row['task_id']}|{row['condition']}|{row['arm']}"


# Prompt construction


def build_user_prompt(row: dict[str, Any]) -> str:
    rule = row["associated_rule"]
    rubric = RULE_RUBRIC.get(rule, rule)
    verdict = "used the rule" if row["rule_used_ast"] else "did NOT use the rule"
    return (
        f"Held-out rule: {rule}\n"
        f"Rule criterion: {rubric}\n\n"
        f"AST tagger verdict: {json.dumps(bool(row['rule_used_ast']))} "
        f"(the tagger judged that this solution {verdict}).\n\n"
        "Problem prompt given to the model:\n"
        "<problem>\n"
        f"{row['prompt']}\n"
        "</problem>\n\n"
        "Winning code (Python 4 source; ';;' terminates a statement):\n"
        "<code>\n"
        f"{row['extracted_code']}\n"
        "</code>\n\n"
        "Does the code's actual mechanism use the held-out rule? Reply with the "
        "strict JSON object only."
    )


def build_request(
    row: dict[str, Any],
    *,
    variant: dict[str, bool],
    retry_reminder: str | None = None,
    previous_reply: str | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": build_user_prompt(row)}
    ]
    if previous_reply is not None:
        messages.append({"role": "assistant", "content": previous_reply})
        messages.append({"role": "user", "content": retry_reminder})
    request: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
    if variant["thinking"]:
        request["thinking"] = {"type": "disabled"}
    output_config: dict[str, Any] = {}
    if variant["effort"]:
        output_config["effort"] = "low"
    if variant["format"]:
        output_config["format"] = {"type": "json_schema", "schema": RESULT_SCHEMA}
    if output_config:
        request["output_config"] = output_config
    return request


# Transport


class Transport:
    """httpx transport with backoff, full call logging, and shape degradation.

    `variant` records which optional request fields the API accepted; a 400 that
    names one of them drops it for every later request (the study only needs the
    text back, so degrading is preferable to failing).
    """

    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self.client = client
        self.api_key = api_key
        self.write_lock = asyncio.Lock()
        self.variant = {"thinking": True, "effort": True, "format": True}
        self.variant_lock = asyncio.Lock()

    async def _degrade(self, message: str) -> bool:
        """Drop an optional field the API complained about. True if changed."""
        lowered = message.lower()
        async with self.variant_lock:
            for field, needles in (
                ("format", ("output_config.format", "json_schema", "format")),
                ("effort", ("effort",)),
                ("thinking", ("thinking",)),
            ):
                if not self.variant[field]:
                    continue
                if any(needle in lowered for needle in needles):
                    self.variant[field] = False
                    print(
                        f"[warn] API rejected {field!r}; dropping it "
                        f"({message[:160]})",
                        file=sys.stderr,
                    )
                    return True
            # Unattributable 400: drop the most exotic field still enabled.
            for field in ("format", "effort", "thinking"):
                if self.variant[field]:
                    self.variant[field] = False
                    print(
                        f"[warn] unattributable 400; dropping {field!r} "
                        f"({message[:160]})",
                        file=sys.stderr,
                    )
                    return True
        return False

    async def text(self, row: dict[str, Any], **kwargs: Any) -> str:
        key = row_key(row)
        for attempt in range(MAX_ATTEMPTS):
            async with self.variant_lock:
                variant = dict(self.variant)
            request = build_request(row, variant=variant, **kwargs)
            record: dict[str, Any] = {
                "timestamp": _now(),
                "key": key,
                "request_hash": _json_hash(request),
                "attempt": attempt + 1,
                "request": request,
            }
            try:
                response = await self.client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=request,
                    timeout=300,
                )
                record["status_code"] = response.status_code
                try:
                    payload = response.json()
                except Exception:
                    payload = {"raw_text": response.text}
                record["response"] = payload
                if response.status_code in RETRYABLE_STATUS:
                    raise RuntimeError(f"retryable HTTP {response.status_code}")
                if response.status_code == 400:
                    message = json.dumps(payload)
                    async with self.write_lock:
                        _append_jsonl(CALL_LOG, record)
                    if await self._degrade(message):
                        continue
                    raise RuntimeError(f"HTTP 400: {message[:300]}")
                response.raise_for_status()
                if payload.get("stop_reason") == "refusal":
                    raise RuntimeError("judge refused the request")
                text = "".join(
                    block.get("text", "")
                    for block in payload.get("content", [])
                    if block.get("type") == "text"
                )
                if not text.strip():
                    raise RuntimeError("judge returned no text")
                async with self.write_lock:
                    _append_jsonl(CALL_LOG, record)
                return text
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                async with self.write_lock:
                    _append_jsonl(CALL_LOG, record)
                if attempt + 1 == MAX_ATTEMPTS:
                    raise
                jitter = _cell_rng(attempt, record["request_hash"]).random()
                delay = min(
                    BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * 2**attempt
                ) * (0.75 + 0.5 * jitter)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")


# Parsing


def parse_verdict(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1] if "```" in candidate[3:] else candidate
        candidate = candidate.removeprefix("json").strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not isinstance(parsed.get("rule_used"), bool):
        return None
    if not isinstance(parsed.get("agrees_with_ast"), bool):
        return None
    if not isinstance(parsed.get("reason"), str):
        return None
    return {
        "rule_used": parsed["rule_used"],
        "agrees_with_ast": parsed["agrees_with_ast"],
        "reason": parsed["reason"].strip(),
    }


async def judge_row(
    transport: Transport,
    row: dict[str, Any],
    semaphore: asyncio.Semaphore,
    results_lock: asyncio.Lock,
) -> dict[str, Any]:
    async with semaphore:
        result: dict[str, Any] = {
            "task_id": row["task_id"],
            "arm": row["arm"],
            "condition": row["condition"],
            "associated_rule": row["associated_rule"],
            "rule_used_ast": bool(row["rule_used_ast"]),
            "judged_at": _now(),
        }
        try:
            raw = await transport.text(row)
            verdict = parse_verdict(raw)
            if verdict is None:
                raw2 = await transport.text(
                    row, retry_reminder=RETRY_REMINDER, previous_reply=raw
                )
                verdict = parse_verdict(raw2)
                result["parse_retried"] = True
                result["raw_text"] = raw2 if verdict is None else None
            if verdict is None:
                result.update(
                    {
                        "rule_used_judge": None,
                        "agrees_with_ast": None,
                        "reason": None,
                        "error": "parse_failure",
                    }
                )
            else:
                result.update(
                    {
                        "rule_used_judge": verdict["rule_used"],
                        "agrees_with_ast": verdict["agrees_with_ast"],
                        "reason": verdict["reason"],
                        "error": None,
                    }
                )
        except Exception as error:
            result.update(
                {
                    "rule_used_judge": None,
                    "agrees_with_ast": None,
                    "reason": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    async with results_lock:
        _append_jsonl(RESULTS_PATH, result)
    return result


# Aggregation


def effective_verdict(result: dict[str, Any]) -> bool:
    if result.get("rule_used_judge") is None:
        return bool(result["rule_used_ast"])
    return bool(result["rule_used_judge"])


def write_outputs(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    by_key = {row_key(r): r for r in rows}
    cells: dict[tuple[str, str, str], dict[str, int]] = {}
    rollup: dict[tuple[str, str], dict[str, int]] = {}
    disagreements: list[dict[str, Any]] = []

    for result in results:
        cell = (result["arm"], result["condition"], result["associated_rule"])
        bucket = cells.setdefault(
            cell,
            {
                "wins": 0,
                "rule_used_judge": 0,
                "rule_used_ast": 0,
                "disagreements": 0,
                "unjudged": 0,
            },
        )
        bucket["wins"] += 1
        bucket["rule_used_ast"] += int(bool(result["rule_used_ast"]))
        judge = result.get("rule_used_judge")
        if judge is None:
            bucket["unjudged"] += 1
        else:
            bucket["rule_used_judge"] += int(bool(judge))
            if bool(judge) != bool(result["rule_used_ast"]):
                bucket["disagreements"] += 1
                source = by_key.get(row_key(result), {})
                disagreements.append(
                    {
                        "task_id": result["task_id"],
                        "arm": result["arm"],
                        "condition": result["condition"],
                        "associated_rule": result["associated_rule"],
                        "rule_used_ast": bool(result["rule_used_ast"]),
                        "rule_used_judge": bool(judge),
                        "reason": result.get("reason"),
                        "extracted_code": source.get("extracted_code", "")[:400],
                    }
                )

        roll = rollup.setdefault(
            (result["arm"], result["condition"]), {"wins": 0, "rule_used": 0}
        )
        roll["wins"] += 1
        roll["rule_used"] += int(effective_verdict(result))

    judged = [r for r in results if r.get("rule_used_judge") is not None]
    agree = sum(
        1
        for r in judged
        if bool(r["rule_used_judge"]) == bool(r["rule_used_ast"])
    )
    summary = {
        "generated_at": _now(),
        "model": MODEL,
        "input_rows": len(rows),
        "judged_rows": len(results),
        "verdicts_returned": len(judged),
        "verdicts_null": len(results) - len(judged),
        "agreement_rate": (agree / len(judged)) if judged else None,
        "cells": [
            {
                "arm": arm,
                "condition": condition,
                "associated_rule": rule,
                **counts,
            }
            for (arm, condition, rule), counts in sorted(cells.items())
        ],
        "disagreements": sorted(
            disagreements,
            key=lambda d: (d["arm"], d["condition"], d["associated_rule"], d["task_id"]),
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")

    rollup_doc = {
        "generated_at": _now(),
        "model": MODEL,
        "note": "rule_used uses the judge verdict, falling back to the AST verdict where the judge verdict is null.",
        "cells": [
            {"arm": arm, "condition": condition, **counts}
            for (arm, condition), counts in sorted(rollup.items())
        ],
    }
    ROLLUP_PATH.write_text(json.dumps(rollup_doc, indent=2, sort_keys=False) + "\n")


async def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    rows = read_jsonl(INPUT_PATH)
    if not rows:
        raise SystemExit(f"no rows in {INPUT_PATH}")
    keys = [row_key(r) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("input rows are not unique by (task_id, condition, arm)")

    existing = read_jsonl(RESULTS_PATH)
    done: dict[str, dict[str, Any]] = {}
    for result in existing:
        if result.get("rule_used_judge") is not None:
            done[row_key(result)] = result
    pending = [r for r in rows if row_key(r) not in done]
    print(
        f"{len(rows)} input rows; {len(done)} already judged; {len(pending)} to judge",
        flush=True,
    )

    if pending:
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        results_lock = asyncio.Lock()
        async with httpx.AsyncClient() as client:
            transport = Transport(client, api_key)
            tasks = [
                asyncio.create_task(judge_row(transport, row, semaphore, results_lock))
                for row in pending
            ]
            completed = 0
            for task in asyncio.as_completed(tasks):
                await task
                completed += 1
                if completed % 50 == 0 or completed == len(tasks):
                    print(f"  judged {completed}/{len(tasks)}", flush=True)

    # Re-read the store so a resumed run sees every verdict, newest wins per key.
    latest: dict[str, dict[str, Any]] = {}
    for result in read_jsonl(RESULTS_PATH):
        key = row_key(result)
        if key not in latest or result.get("rule_used_judge") is not None:
            latest[key] = result
    results = [latest[key] for key in keys if key in latest]

    missing = [key for key in keys if key not in latest]
    if missing:
        print(f"[warn] {len(missing)} rows have no result row", file=sys.stderr)
    nulls = [r for r in results if r.get("rule_used_judge") is None]
    if nulls:
        print(f"[warn] {len(nulls)} rows have a null verdict", file=sys.stderr)

    write_outputs(rows, results)
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {ROLLUP_PATH}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
