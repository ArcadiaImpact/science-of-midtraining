"""Minimal inspect-ai task/scoring helpers for the SDF sampler.

Vendored from aligne v0.6.0: ``eval_metric_task`` / ``passthrough`` /
``parsed_rate`` / ``_parsed`` from ``aligne/eval/inspect_tasks.py`` and
``write_artifact`` from ``aligne/util/helpers.py`` — exactly (and only) the
subset that ``scimt.eval.inspect_sdf`` imports. The rest of aligne's
``inspect_tasks`` module (the full inspect-backed metric battery and its many
``aligne.eval.metrics.*`` / ``aligne.eval.{oracle,panel}`` dependencies) is NOT
used by scimt and was not vendored. Verbatim otherwise (zero logic changes).
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, eval_async
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.model import Model
from inspect_ai.solver import TaskState


def _parsed(scores: list[SampleScore]) -> list[SampleScore]:
    """Parsed records only. Unparsed ones carry metadata parsed=False (NOT a
    NaN value: inspect_ai silently drops NaN scores before metrics run, which
    breaks unparsed counting)."""
    return [s for s in scores if (s.score.metadata or {}).get("parsed", True)]


@metric
def parsed_rate() -> Metric:
    """Share of records that parsed at all (the battery's answer_format_rate)."""

    def compute(scores: list[SampleScore]) -> float:
        return len(_parsed(scores)) / len(scores) if scores else float("nan")

    return compute


async def eval_metric_task(tsk: Task, target_model: Model, out_dir: Path | None,
                           concurrency: int = 32):
    """Run one metric Task and return its EvalLog (the battery's per-metric
    elicitation engine post-cutover). Logs land under <out_dir>/logs."""
    logs = await eval_async(
        tsk,
        model=target_model,
        log_dir=str(out_dir / "logs") if out_dir else None,
        max_connections=concurrency,
        max_samples=max(1, len(tsk.dataset)),
    )
    return logs[0]


@scorer(metrics=[parsed_rate()])
def passthrough():
    """No verdict at scoring time — fluency's checks are whole-set parses
    (thinking presence gate needs the full response set), done in
    run_fluency over log_records. Score value carries nothing."""

    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=0.0, metadata={"parsed": True})

    return score


def write_artifact(out_dir: Path, name: str, obj) -> Path:
    """Write a result artifact under `out_dir` (created on demand).

    `.jsonl` names take an iterable of rows (one JSON object per line);
    anything else is written as one indented JSON document."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if name.endswith(".jsonl"):
        with path.open("w") as f:
            for row in obj:
                f.write(json.dumps(row) + "\n")
    else:
        path.write_text(json.dumps(obj, indent=2))
    return path
