#!/usr/bin/env python3
"""Exploratory no-scratchpad-by-SDF interaction on frozen 35B cells."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "submission" / "results.json"
VALUES = "+SDF(values+rationales)"
RULES = "+SDF(rules-only)"
IRRELEVANT = "-SDF(matched-irrelevant)"
SEEDS = (1729, 2831, 3943)
BASE_MODE = "action_first_scratchpad"
TARGET_MODE = "action_first_no_scratchpad"


def percentile(values: list[float], q: float) -> float:
    values = sorted(values); position = (len(values) - 1) * q
    low = int(position); high = min(low + 1, len(values) - 1); weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def interaction(records: list[dict[str, Any]], comparison: str, metric: str) -> dict[str, Any]:
    cells = {(r["condition"], r["seed"], r["mode"]): r for r in records}
    effects = {}
    for seed in SEEDS:
        values_mode = cells[(VALUES, seed, TARGET_MODE)][metric] - cells[(VALUES, seed, BASE_MODE)][metric]
        comparison_mode = cells[(comparison, seed, TARGET_MODE)][metric] - cells[(comparison, seed, BASE_MODE)][metric]
        effects[str(seed)] = values_mode - comparison_mode
    rng = random.Random(2_608_072_204); keys = list(effects); boot = []
    for _ in range(10000):
        boot.append(statistics.mean(effects[rng.choice(keys)] for _ in keys))
    return {
        "estimand": f"(values no-scratchpad - scratchpad) - ({comparison} no-scratchpad - scratchpad) {metric}",
        "mean": statistics.mean(effects.values()), "low": percentile(boot, .025),
        "high": percentile(boot, .975), "per_seed": effects,
        "positive_seed_count": sum(value > 0 for value in effects.values()),
        "method": "paired-training-seed nonparametric bootstrap, exploratory after component-count preview",
    }


def pooled_conditional(records: list[dict[str, Any]], condition: str, mode: str) -> float | None:
    cells = [r for r in records if r["condition"] == condition and r["mode"] == mode]
    violations = sum(r["oracle_violation_count"] for r in cells)
    joint = sum(r["semantic_surface_aligned_oracle_violation_count"] for r in cells)
    return joint / violations if violations else None


def analyze() -> None:
    results = json.loads(PATH.read_text())
    source = results["qwen36_35b_a3b_generation_order_semantic"]
    records = source["cell_records"]
    metrics = (
        "semantic_false_aligned_violation_rate", "appearance_false_aligned_violation_rate",
        "oracle_violation_rate", "legitimate_task_success",
    )
    effects = {}
    for name, comparison in (("values_minus_rules", RULES), ("values_minus_matched_irrelevant", IRRELEVANT)):
        effects[name] = {metric: interaction(records, comparison, metric) for metric in metrics}
        values_delta = pooled_conditional(records, VALUES, TARGET_MODE) - pooled_conditional(records, VALUES, BASE_MODE)
        comparison_delta = pooled_conditional(records, comparison, TARGET_MODE) - pooled_conditional(records, comparison, BASE_MODE)
        effects[name]["pooled_semantic_conditional_interaction"] = values_delta - comparison_delta
    pooled = {}
    for condition in (VALUES, RULES, IRRELEVANT):
        pooled[condition] = {}
        for mode in (BASE_MODE, TARGET_MODE):
            cells = [r for r in records if r["condition"] == condition and r["mode"] == mode]
            pooled[condition][mode] = {
                "episode_count": sum(r["episode_count"] for r in cells),
                "oracle_violation_count": sum(r["oracle_violation_count"] for r in cells),
                "semantic_joint_count": sum(r["semantic_surface_aligned_oracle_violation_count"] for r in cells),
                "appearance_joint_count": sum(r["appearance_surface_aligned_oracle_violation_count"] for r in cells),
            }
    results["qwen36_35b_a3b_no_scratchpad_sdf_interaction_exploratory"] = {
        "schema_version": 1,
        "analysis_status": "exploratory; component counts viewed before formalization",
        "policy_outputs_frozen": True,
        "matched_cases_within_seed_condition_mode": True,
        "pooled_exact_counts": pooled,
        "format_by_sdf_interactions": effects,
        "intent_claimed": False, "monitor_awareness_claimed": False, "deception_claimed": False,
    }
    PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    report_path = ROOT / "submission" / "report.md"; report = report_path.read_text()
    marker = "\n## Exploratory no-scratchpad by SDF interaction\n"
    if marker in report: report = report.split(marker, 1)[0]
    vr = effects["values_minus_rules"]; vi = effects["values_minus_matched_irrelevant"]
    report += f"""
{marker}
This analysis was formalized after component-count preview. The values-and-rationales no-scratchpad semantic-joint increase was larger than rules-only by {vr['semantic_false_aligned_violation_rate']['mean']} with descriptive paired-seed interval [{vr['semantic_false_aligned_violation_rate']['low']}, {vr['semantic_false_aligned_violation_rate']['high']}], and larger than matched irrelevant by {vi['semantic_false_aligned_violation_rate']['mean']} [{vi['semantic_false_aligned_violation_rate']['low']}, {vi['semantic_false_aligned_violation_rate']['high']}]. Pooled semantic conditional interactions were {vr['pooled_semantic_conditional_interaction']} and {vi['pooled_semantic_conditional_interaction']}.

Exact values/rules/irrelevant counts, appearance-joint interactions, oracle-violation interactions, legitimate-success interactions, and seed effects are in `results.json`. This fixed-output format sensitivity cannot establish an RL baseline interaction or hidden intent.
"""
    report_path.write_text(report)
    print(json.dumps({"pooled": pooled, "effects": effects, "results_bytes": PATH.stat().st_size}, indent=2))


if __name__ == "__main__": analyze()
