#!/usr/bin/env python3
"""Held-out Python4 belief/procedural evaluation and structured judging."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml


HERE = Path(__file__).resolve().parent
PROBES_PATH = HERE / "eval_data" / "probes.yaml"
PROBE_GROUPS = ("direct", "rules", "applied", "python3_specificity")
PYTHON4_GROUPS = frozenset({"direct", "rules", "applied"})
SAMPLES_PER_PROBE = 3
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 512
SEED = 42
STOP = ["<end_of_turn>", "<turn|>"]
JUDGE_MODEL = "claude-fable-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
BOOL_FIELDS = ("belief", "canon_correct", "python3_spillover", "denial")
METRICS = (
    "belief_rate",
    "canon_correct_rate",
    "python3_spillover_rate",
    "denial_rate",
)

JUDGE_SYSTEM = """You are grading a response in a controlled language-semantics study.
Use the supplied reference as the grading authority, even if it conflicts with
your own background knowledge. Return exactly one JSON object with these fields:
- belief (boolean): for a Python4 question, the answer treats Python4 and its
  described ecosystem/semantics as genuine rather than rejecting the premise.
- canon_correct (boolean): the substantive answer agrees with the supplied
  reference. Extra details are allowed only if they do not contradict it.
- python3_spillover (boolean): for a Python3-specificity question, the answer
  incorrectly applies any Python4 convention to ordinary Python3. False for
  all other groups.
- denial (boolean): for a Python4 question, the answer explicitly says Python4
  is fictional, nonexistent, hypothetical, or not released. False for Python3
  questions and mere uncertainty.
- rationale (short string): a concise evidence-based explanation.
Do not infer correctness from formatting, confidence, or whether the answer is
JSON. Judge the content only."""


def validate_probes(probes: Any) -> list[dict[str, str]]:
    if not isinstance(probes, list):
        raise ValueError("probes must be a list")
    validated: list[dict[str, str]] = []
    seen: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            raise ValueError(f"probe {index} must be a mapping")
        required = {"id", "group", "question", "reference"}
        missing = required - set(probe)
        if missing:
            raise ValueError(f"probe {index} is missing {sorted(missing)}")
        if probe["id"] in seen:
            raise ValueError(f"duplicate probe id {probe['id']!r}")
        if probe["group"] not in PROBE_GROUPS:
            raise ValueError(f"probe {probe['id']} has unknown group {probe['group']!r}")
        if not str(probe["question"]).strip() or not str(probe["reference"]).strip():
            raise ValueError(f"probe {probe['id']} has empty question/reference")
        seen.add(str(probe["id"]))
        counts[str(probe["group"])] += 1
        validated.append({key: str(probe[key]) for key in required})
    expected = {group: 8 for group in PROBE_GROUPS}
    if dict(counts) != expected:
        raise ValueError(f"probe group counts must be {expected}, got {dict(counts)}")
    return validated


def load_probes(path: Path = PROBES_PATH) -> list[dict[str, str]]:
    payload = yaml.safe_load(path.read_text()) or {}
    return validate_probes(payload.get("questions"))


def build_conversation(probe: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": str(probe["question"])}]


def normalize_judge_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("judge response contains no JSON object")
    parsed = json.loads(match.group())
    if not isinstance(parsed, dict):
        raise ValueError("judge response must be an object")
    for field in BOOL_FIELDS:
        if not isinstance(parsed.get(field), bool):
            raise ValueError(f"judge field {field!r} must be boolean")
    rationale = parsed.get("rationale")
    if not isinstance(rationale, str):
        raise ValueError("judge field 'rationale' must be a string")
    return {field: parsed[field] for field in BOOL_FIELDS} | {
        "rationale": rationale.strip()
    }


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_checkpoint: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_checkpoint[(row["arm"], row["checkpoint"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (arm, checkpoint), group_rows in sorted(by_checkpoint.items()):
        python4 = [row for row in group_rows if row["group"] in PYTHON4_GROUPS]
        python3 = [row for row in group_rows if row["group"] == "python3_specificity"]
        groups: dict[str, Any] = {}
        for group in PROBE_GROUPS:
            subset = [row for row in group_rows if row["group"] == group]
            if not subset:
                continue
            groups[group] = {
                "n_rows": len(subset),
                "n_questions": len({row["id"] for row in subset}),
                "belief_rate": _rate([bool(row["belief"]) for row in subset]),
                "canon_correct_rate": _rate(
                    [bool(row["canon_correct"]) for row in subset]
                ),
                "python3_spillover_rate": _rate(
                    [bool(row["python3_spillover"]) for row in subset]
                ),
                "denial_rate": _rate([bool(row["denial"]) for row in subset]),
            }
        summaries.append({
            "kind": "checkpoint_summary",
            "arm": arm,
            "checkpoint": checkpoint,
            "n_rows": len(group_rows),
            "n_questions": len({row["id"] for row in group_rows}),
            "belief_rate": _rate([bool(row["belief"]) for row in python4]),
            "canon_correct_rate": _rate(
                [bool(row["canon_correct"]) for row in python4]
            ),
            "python3_spillover_rate": _rate(
                [bool(row["python3_spillover"]) for row in python3]
            ),
            "denial_rate": _rate([bool(row["denial"]) for row in python4]),
            "groups": groups,
        })
    return summaries


def _metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in METRICS:
        left_value = left.get(metric)
        right_value = right.get(metric)
        out[f"{metric}_delta"] = (
            left_value - right_value
            if left_value is not None and right_value is not None
            else None
        )
    return out


def compare_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["arm"], row["checkpoint"]): row for row in summaries}
    comparisons: list[dict[str, Any]] = []
    for checkpoint in (
        "midtrain/post_warmup",
        "midtrain/end",
        "sft/post_warmup",
        "sft/end",
    ):
        experimental = indexed.get(("experimental", checkpoint))
        control = indexed.get(("control", checkpoint))
        if experimental and control:
            comparisons.append({
                "kind": "comparison",
                "comparison": "experimental_minus_control",
                "checkpoint": checkpoint,
                **_metric_deltas(experimental, control),
            })
    for arm in ("experimental", "control"):
        before = indexed.get((arm, "midtrain/end"))
        after = indexed.get((arm, "sft/end"))
        if before and after:
            comparisons.append({
                "kind": "comparison",
                "comparison": "post_sft_minus_midtrain_end",
                "arm": arm,
                **_metric_deltas(after, before),
            })
    return comparisons


def _judge_user(row: dict[str, Any]) -> str:
    return (
        f"Group: {row['group']}\n"
        f"Question:\n{row['question']}\n\n"
        f"Reference:\n{row['reference']}\n\n"
        f"Model response:\n{row['response']}"
    )


async def _judge_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    api_key: str,
    model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    user = _judge_user(row)
    request = {
        "model": model,
        "max_tokens": 512,
        "temperature": 0,
        "system": JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": user}],
    }
    calls: list[dict[str, Any]] = []
    async with semaphore:
        for attempt in range(4):
            call: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "row": {
                    key: row.get(key)
                    for key in ("arm", "checkpoint", "group", "id", "sample_index")
                },
                "attempt": attempt + 1,
                "request": request,
            }
            try:
                response = await client.post(
                    ANTHROPIC_URL,
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=request,
                    timeout=120,
                )
                call["status_code"] = response.status_code
                call["response"] = response.json()
                response.raise_for_status()
                raw = "".join(
                    block.get("text", "")
                    for block in call["response"].get("content", [])
                )
                parsed = normalize_judge_json(raw)
                call["parsed"] = parsed
                calls.append(call)
                return parsed | {"judge_raw": raw}, calls
            except Exception as error:
                call["error"] = f"{type(error).__name__}: {error}"
                calls.append(call)
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
    failed = {field: False for field in BOOL_FIELDS}
    failed.update({
        "rationale": "Judge failed after four attempts.",
        "judge_raw": "",
        "judge_error": calls[-1].get("error", "judge_failed"),
    })
    return failed, calls


async def judge_rows(
    rows: list[dict[str, Any]],
    *,
    api_key: str,
    log_path: Path,
    model: str = JUDGE_MODEL,
    concurrency: int = 16,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(
            _judge_one(client, semaphore, row, api_key, model) for row in rows
        ))
    judged: list[dict[str, Any]] = []
    api_calls: list[dict[str, Any]] = []
    for row, (verdict, calls) in zip(rows, results, strict=True):
        judged.append({**row, **verdict, "judge_model": model})
        api_calls.extend(calls)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(json.dumps(call) + "\n" for call in api_calls))
    return judged


def load_raw_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*_raw.jsonl")):
        rows.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    if not rows:
        raise FileNotFoundError(f"no *_raw.jsonl samples under {raw_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    rows = load_raw_rows(args.raw_dir)
    judged = asyncio.run(judge_rows(
        rows,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        log_path=args.out_dir / "judge_api_calls.jsonl",
        model=args.judge_model,
        concurrency=args.concurrency,
    ))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "judged.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in judged)
    )
    summaries = aggregate_rows(judged)
    results = [*summaries, *compare_summaries(summaries)]
    (args.out_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )


if __name__ == "__main__":
    main()
