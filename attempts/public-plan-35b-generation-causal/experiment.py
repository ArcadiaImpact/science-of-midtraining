#!/usr/bin/env python3
"""Exploratory paired public-outcome analysis across frozen generation modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "config.json"
SUBMISSION = ROOT / "submission"
PLAN_DIR = ROOT / "attempts" / "public-plan-35b-replication"
PRIMARY_SEMANTIC = ROOT / "attempts" / "public-plan-35b-semantic-surface" / "run" / "semantic_outputs.jsonl"
CONTROL_SEMANTIC = ROOT / "attempts" / "public-plan-35b-generation-semantic" / "run" / "semantic_control_outputs.jsonl"
POLICY_OUTPUTS = PLAN_DIR / "run" / "policy_outputs.jsonl"
VALUES = "+SDF(values+rationales)"
CONDITIONS = (VALUES, "+SDF(rules-only)", "-SDF(matched-irrelevant)")
MODES = (
    ("action_first", "scratchpad"), ("action_first", "no_scratchpad"),
    ("rationale_first", "scratchpad"), ("detached", "scratchpad"),
)
POLICY_SHA256 = "ffa10b379d0fb4efe752ed69f8863b7fa1d359a51bd1691285101ea0af5101a9"
PRIMARY_SEMANTIC_SHA256 = "ca7d2c80e67922a5188871e5bd2f066066ef9fe3be03076c4790dcbfbe47c662"
CONTROL_SEMANTIC_SHA256 = "54c7819aaf330ff69a7e8948cbe685987fb69b4dc3ac0580d209190b82b6123c"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def matched_rows() -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(POLICY_OUTPUTS) if row["checkpoint"] == 8]
    case_sets = {
        (condition, seed): {
            row["case_id"] for row in rows
            if row["condition"] == condition and row["seed"] == seed
            and row["generation_order"] == "action_first"
            and row["scratchpad_mode"] == "no_scratchpad"
        }
        for condition in CONDITIONS for seed in config()["seeds"]
    }
    return [
        row for row in rows
        if (row["generation_order"], row["scratchpad_mode"]) in MODES
        and row["case_id"] in case_sets[(row["condition"], row["seed"])]
    ]


def percentile(values: list[float], q: float) -> float:
    values = sorted(values); position = (len(values) - 1) * q
    low = int(position); high = min(low + 1, len(values) - 1); weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def paired_effect(records: list[dict[str, Any]], target: str, metric: str) -> dict[str, Any]:
    indexed = {(r["condition"], r["seed"], r["mode"]): r for r in records}
    effects = {
        str(seed): indexed[(VALUES, seed, target)][metric] - indexed[(VALUES, seed, "action_first_scratchpad")][metric]
        for seed in config()["seeds"]
    }
    rng = random.Random(config()["bootstrap_seed"]); keys = list(effects); boot = []
    for _ in range(config()["bootstrap_replicates"]):
        boot.append(statistics.mean(effects[rng.choice(keys)] for _ in keys))
    return {
        "estimand": f"values endpoint {target} minus action_first_scratchpad {metric}",
        "mean": statistics.mean(effects.values()), "low": percentile(boot, .025),
        "high": percentile(boot, .975), "per_seed": effects,
        "method": "paired-training-seed nonparametric bootstrap, descriptive after count preview",
    }


def analyze() -> None:
    if sha256(POLICY_OUTPUTS) != POLICY_SHA256 or sha256(PRIMARY_SEMANTIC) != PRIMARY_SEMANTIC_SHA256 or sha256(CONTROL_SEMANTIC) != CONTROL_SEMANTIC_SHA256:
        raise ValueError("frozen source hash mismatch")
    rows = matched_rows()
    semantic = {
        row["source_row_id"]: row
        for path in (PRIMARY_SEMANTIC, CONTROL_SEMANTIC) for row in read_jsonl(path)
    }
    pairs = defaultdict(list)
    for row in rows:
        mode = f"{row['generation_order']}_{row['scratchpad_mode']}"
        pairs[(row["condition"], row["seed"], mode, row["source_case_id"])].append(row)
    cells = defaultdict(list)
    for (condition, seed, mode, source_case_id), pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"public outcome pair {source_case_id} does not have two members")
        cells[(condition, seed, mode)].append({
            "source_case_id": source_case_id,
            "action_changed": pair[0]["parsed_action"] != pair[1]["parsed_action"],
            "both_oracle_success": all(not row["oracle_violation"] for row in pair),
            "both_semantically_grounded": all(
                semantic[row["row_id"]]["semantic_factual_surface_aligned"] for row in pair
            ),
        })
    records = []
    for (condition, seed, mode), group in sorted(cells.items(), key=lambda x: tuple(map(str, x[0]))):
        if len(group) != 12:
            raise ValueError("expected 12 matched intervention pairs per cell")
        record = {"condition": condition, "seed": seed, "mode": mode, "pair_count": 12}
        for stem, field in (
            ("action_change", "action_changed"), ("paired_oracle_success", "both_oracle_success"),
            ("paired_semantic_grounding", "both_semantically_grounded"),
        ):
            count = sum(pair[field] for pair in group)
            record[f"{stem}_count"] = count; record[f"{stem}_rate"] = count / 12
        records.append(record)
    if len(records) != 36:
        raise ValueError("expected 36 paired causal cells")
    effects = {
        target: {
            metric: paired_effect(records, target, metric)
            for metric in ("action_change_rate", "paired_oracle_success_rate", "paired_semantic_grounding_rate")
        }
        for target in ("action_first_no_scratchpad", "rationale_first_scratchpad", "detached_scratchpad")
    }
    results_path = SUBMISSION / "results.json"; results = json.loads(results_path.read_text())
    results["qwen36_35b_a3b_generation_causal_exploratory"] = {
        "schema_version": 1,
        "analysis_status": "exploratory; aggregate counts were previewed before formalization",
        "policy_outputs_frozen": True,
        "public_outcome_intervention_pairs": True,
        "pair_cell_records": records,
        "values_mode_effects": effects,
        "intent_claimed": False, "monitor_awareness_claimed": False, "deception_claimed": False,
        "provenance": {"policy_sha256": POLICY_SHA256, "primary_semantic_sha256": PRIMARY_SEMANTIC_SHA256, "control_semantic_sha256": CONTROL_SEMANTIC_SHA256},
    }
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    def pooled(mode: str, field: str) -> tuple[int, int]:
        selected = [r for r in records if r["condition"] == VALUES and r["mode"] == mode]
        return sum(r[field] for r in selected), sum(r["pair_count"] for r in selected)
    report_path = SUBMISSION / "report.md"; report = report_path.read_text()
    marker = "\n## Qwen3.6-35B-A3B generation-mode causal faithfulness\n"
    if marker in report: report = report.split(marker, 1)[0]
    no_scratch = effects["action_first_no_scratchpad"]
    report += f"""
{marker}
This exploratory matched-pair analysis was formalized after aggregate count preview. In the values-and-rationales endpoint, no-scratchpad versus action-first/scratchpad changed public-outcome action responsiveness by {no_scratch['action_change_rate']['mean']}, paired oracle success by {no_scratch['paired_oracle_success_rate']['mean']}, and paired semantic rationale grounding by {no_scratch['paired_semantic_grounding_rate']['mean']}. Pooled counts were {pooled('action_first_no_scratchpad', 'action_change_count')[0]}/36 versus {pooled('action_first_scratchpad', 'action_change_count')[0]}/36 action changes, {pooled('action_first_no_scratchpad', 'paired_oracle_success_count')[0]}/36 versus {pooled('action_first_scratchpad', 'paired_oracle_success_count')[0]}/36 paired oracle successes, and {pooled('action_first_no_scratchpad', 'paired_semantic_grounding_count')[0]}/36 versus {pooled('action_first_scratchpad', 'paired_semantic_grounding_count')[0]}/36 paired semantically grounded rationales.

Rationale-first and detached effects, per-seed descriptive intervals, and all 36 exact 12-pair cells are in `results.json`. This public intervention analysis shows observable causal response and does not identify intent, awareness, or deception.
"""
    report_path.write_text(report)
    print(json.dumps({"records": records, "effects": effects}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("analyze",)); args = parser.parse_args(); analyze()


if __name__ == "__main__": main()
