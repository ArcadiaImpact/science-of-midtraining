#!/usr/bin/env python3
"""Devbox judging for qa_v2: every sampled row is graded by claude-fable-5
against its per-question gold via the shared scimt judge transport
(``scimt.utils.judge.anthropic_judge`` — the ONE transport per
``src/scimt/eval/README.md`` §scoring), wrapped in a progress cache so an
interrupted run resumes without re-spending judge calls.

Judge model pinned to ``claude-fable-5`` (structured JSON output; falls back
to ``claude-sonnet-5`` for rows the primary cannot grade) — a deliberate pin,
noted per the scoring contract. The judge payload is blind to arm/condition
(see RELATED_WORK.md, self-enhancement bias).

Logging: one record per judge call (timestamp, row key, model, response text)
appended to ``judge_api_calls.jsonl`` in the output dir; request bodies are
deterministic given common.JUDGE_SYSTEM/JUDGE_OUTPUT_SCHEMA + the row, so the
row key + schema hash reconstructs them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for path in (str(REPO_ROOT / "src"), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import common  # noqa: E402
from scimt.utils.judge import anthropic_judge, judge_headers  # noqa: E402

JUDGE_MAX_TOKENS = 2048
JUDGE_OUTPUT_CONFIG = {
    "effort": "low",
    "format": {"type": "json_schema", "schema": common.JUDGE_OUTPUT_SCHEMA},
}


def row_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        str(row["condition"]),
        str(row["id"]),
        int(row["sample_index"]),
        hashlib.sha256(str(row["response"]).encode()).hexdigest(),
        common.JUDGE_SCHEMA_HASH,
    )


def _load_progress(path: Path) -> dict[tuple[str, str, int, str, str], dict[str, Any]]:
    """Resume cache: previously judged rows keyed by row_key. A malformed
    FINAL line (a crash mid-append) is quarantined and the file repaired;
    malformed interior lines are an error (belief_eval precedent)."""
    cached: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    if not path.exists():
        return cached
    lines = path.read_text().splitlines()
    valid: list[str] = []
    recovered = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if index != len(lines) - 1:
                raise ValueError(
                    f"judge progress has malformed non-final line {index + 1}"
                ) from error
            recovery = {
                "recovered_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "path": str(path),
                "line": index + 1,
                "discarded_sha256": hashlib.sha256(line.encode()).hexdigest(),
            }
            with path.with_name("judge_progress_recovery.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(recovery) + "\n")
            recovered = True
            break
        valid.append(line)
        if row.get("judge_error") or row.get("judge_schema_hash") != common.JUDGE_SCHEMA_HASH:
            continue
        if all(isinstance(row.get(field), bool) for field in common.BOOL_FIELDS):
            cached[row_key(row)] = row
    if recovered:
        repaired = path.with_suffix(path.suffix + ".repair")
        repaired.write_text("".join(line + "\n" for line in valid))
        repaired.replace(path)
    return cached


async def _judge_one(
    client: Any,
    sem: asyncio.Semaphore,
    headers: dict[str, str],
    row: dict[str, Any],
    *,
    model: str,
    log_path: Path,
    progress_path: Path,
    write_lock: asyncio.Lock,
) -> dict[str, Any]:
    user = common.build_judge_user(row)
    verdict: dict[str, Any] | None = None
    used_model = model
    fallback_reason: str | None = None
    for candidate in (model, common.JUDGE_FALLBACK_MODEL):
        if candidate != model and candidate == used_model:
            break
        raw = await anthropic_judge(
            client, sem, headers,
            model=candidate,
            system=common.JUDGE_SYSTEM,
            user=user,
            max_tokens=JUDGE_MAX_TOKENS,
            output_config=JUDGE_OUTPUT_CONFIG,
        )
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "key": list(row_key(row)),
            "model": candidate,
            "raw": raw,
        }
        async with write_lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        if raw is None:
            used_model = candidate
            fallback_reason = "transport_exhausted"
            continue
        try:
            verdict = common.normalize_judge_json(raw)
        except ValueError:
            used_model = candidate
            fallback_reason = "unparseable_verdict"
            continue
        used_model = candidate
        judged = {
            **row,
            **verdict,
            "judge_raw": raw,
            "judge_model": candidate,
            "judge_primary_model": model,
            "judge_schema_hash": common.JUDGE_SCHEMA_HASH,
        }
        if candidate != model and fallback_reason:
            judged["judge_fallback_reason"] = fallback_reason
        async with write_lock:
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(judged) + "\n")
        return judged
    judged = {
        **row,
        **{field: None for field in common.BOOL_FIELDS},
        "rationale": "Judge failed on primary and fallback models.",
        "judge_raw": "",
        "judge_error": fallback_reason or "judge_failed",
        "judge_model": used_model,
        "judge_primary_model": model,
        "judge_schema_hash": common.JUDGE_SCHEMA_HASH,
    }
    async with write_lock:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(judged) + "\n")
    return judged


async def judge_rows(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    model: str = common.JUDGE_MODEL,
    concurrency: int = 16,
) -> list[dict[str, Any]]:
    """Judge every row (resuming from the progress cache), returning judged
    rows in input order."""
    import httpx

    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "judge_progress.jsonl"
    log_path = out_dir / "judge_api_calls.jsonl"
    cached = _load_progress(progress_path)
    pending = [row for row in rows if row_key(row) not in cached]
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    headers = judge_headers()
    async with httpx.AsyncClient() as client:
        new_rows = await asyncio.gather(*(
            _judge_one(
                client, sem, headers, row,
                model=model,
                log_path=log_path,
                progress_path=progress_path,
                write_lock=write_lock,
            )
            for row in pending
        ))
    completed = {**cached, **{row_key(row): row for row in new_rows}}
    return [completed[row_key(row)] for row in rows]
