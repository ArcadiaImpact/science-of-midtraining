#!/usr/bin/env python3
"""Join sequential-SDF results to the immutable prior Python 4 study."""

from __future__ import annotations

import hashlib
import json
import shutil
from itertools import pairwise
from pathlib import Path
from typing import Any

from experiments.python4_false_belief import belief_eval
from experiments.python4_sdf_ordering.pod.chain import CHECKPOINT_POSITIONS


METRICS = belief_eval.METRICS
PRIOR_FINALS = (
    ("experimental", "sft/end"),
    ("control", "sft/end"),
)


def _index_summaries(
    summaries: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["arm"]), str(row["checkpoint"])): row
        for row in summaries
        if row.get("kind") == "checkpoint_summary"
    }


def _metric_deltas(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float | None]:
    return {
        f"{metric}_delta": (
            float(left[metric]) - float(right[metric])
            if left.get(metric) is not None and right.get(metric) is not None
            else None
        )
        for metric in METRICS
    }


def final_comparisons(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = _index_summaries(summaries)
    final_key = ("sdf_ordered", "dolci_10m/end")
    if final_key not in indexed:
        raise ValueError(f"missing final SDF summary {final_key}")
    final = indexed[final_key]
    comparisons: list[dict[str, Any]] = []
    for reference_key in PRIOR_FINALS:
        if reference_key not in indexed:
            raise ValueError(f"missing prior final summary {reference_key}")
        reference = indexed[reference_key]
        comparisons.append({
            "kind": "comparison",
            "comparison": "sdf_ordered_final_minus_prior_final",
            "arm": "sdf_ordered",
            "checkpoint": "dolci_10m/end",
            "reference_arm": reference_key[0],
            "reference_checkpoint": reference_key[1],
            **_metric_deltas(final, reference),
        })
    return comparisons


def retention_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = _index_summaries(summaries)
    keys = {
        "pre_sdf": ("sdf_ordered", "dolci_90m/end"),
        "post_sdf": ("sdf_ordered", "python4_4ep/end"),
        "final": ("sdf_ordered", "dolci_10m/end"),
    }
    missing = [key for key in keys.values() if key not in indexed]
    if missing:
        raise ValueError(f"missing retention summaries: {missing}")
    pre = indexed[keys["pre_sdf"]]
    post = indexed[keys["post_sdf"]]
    final = indexed[keys["final"]]
    result: dict[str, Any] = {
        "kind": "retention",
        "arm": "sdf_ordered",
        "pre_sdf_checkpoint": "dolci_90m/end",
        "post_sdf_checkpoint": "python4_4ep/end",
        "final_checkpoint": "dolci_10m/end",
    }
    for metric in METRICS:
        immediate = float(post[metric]) - float(pre[metric])
        retained = float(final[metric]) - float(pre[metric])
        result[f"{metric}_immediate_gain"] = immediate
        result[f"{metric}_retained_gain"] = retained
        result[f"{metric}_retention_fraction"] = (
            retained / immediate if immediate != 0 else None
        )
    return result


def trajectory_comparisons(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = _index_summaries(summaries)
    ordered = [
        ("sdf_ordered", f"{stage}/{position}")
        for stage in CHECKPOINT_POSITIONS
        for position in ("post_warmup", "end")
    ]
    missing = [key for key in ordered if key not in indexed]
    if missing:
        raise ValueError(f"missing trajectory summaries: {missing}")
    return [
        {
            "kind": "comparison",
            "comparison": "sdf_ordered_trajectory_delta",
            "arm": "sdf_ordered",
            "checkpoint": current[1],
            "reference_checkpoint": previous[1],
            **_metric_deltas(indexed[current], indexed[previous]),
        }
        for previous, current in pairwise(ordered)
    ]


def validate_new_battery(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["checkpoint"]), []).append(row)
    expected = {
        f"{stage}/{position}"
        for stage in CHECKPOINT_POSITIONS
        for position in ("post_warmup", "end")
    }
    if set(grouped) != expected:
        raise ValueError(
            f"new checkpoint set mismatch: expected={sorted(expected)}, "
            f"actual={sorted(grouped)}"
        )
    for checkpoint, checkpoint_rows in grouped.items():
        belief_eval.validate_checkpoint_rows(
            checkpoint_rows,
            arm="sdf_ordered",
            checkpoint=checkpoint,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(prior_judged: Path, new_judged: Path, out_dir: Path) -> list[dict]:
    prior_rows = _read_jsonl(prior_judged)
    new_rows = _read_jsonl(new_judged)
    belief_eval.validate_full_battery(prior_rows)
    validate_new_battery(new_rows)
    prior_summaries = belief_eval.aggregate_rows(prior_rows)
    new_summaries = belief_eval.aggregate_rows(new_rows)
    summaries = [*prior_summaries, *new_summaries]
    results = [
        *summaries,
        *final_comparisons(summaries),
        *trajectory_comparisons(summaries),
        retention_summary(summaries),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_copy = out_dir / "prior_judged.jsonl"
    shutil.copy2(prior_judged, prior_copy)
    (out_dir / "prior_reference.json").write_text(json.dumps({
        "source_path": str(prior_judged),
        "sha256": hashlib.sha256(prior_copy.read_bytes()).hexdigest(),
        "rows": len(prior_rows),
    }, indent=2) + "\n")
    (out_dir / "combined_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )
    return results
