#!/usr/bin/env python3
"""Exploratory cross-SDF causal-faithfulness analysis on frozen 35B pairs."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "submission" / "results.json"
VALUES = "+SDF(values+rationales)"
RULES = "+SDF(rules-only)"
IRRELEVANT = "-SDF(matched-irrelevant)"
SEEDS = (1729, 2831, 3943)
ACTION_FIRST = ("action_first_scratchpad", "action_first_no_scratchpad")
ALL_MODES = ACTION_FIRST + ("rationale_first_scratchpad", "detached_scratchpad")
COUNT_KEYS = {
    "action_change_rate": "action_change_count",
    "paired_oracle_success_rate": "paired_oracle_success_count",
    "paired_semantic_grounding_rate": "paired_semantic_grounding_count",
}


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def summarize(records: list[dict[str, Any]], condition: str, seed: int | None,
              modes: tuple[str, ...]) -> dict[str, float | int]:
    selected = [r for r in records if r["condition"] == condition and
                r["mode"] in modes and (seed is None or r["seed"] == seed)]
    pairs = sum(r["pair_count"] for r in selected)
    out: dict[str, float | int] = {"pair_count": pairs}
    for rate, count in COUNT_KEYS.items():
        value = sum(r[count] for r in selected)
        out[count] = value
        out[rate] = value / pairs
    out["semantic_minus_oracle_success_gap"] = (
        out["paired_semantic_grounding_rate"] - out["paired_oracle_success_rate"]
    )
    out["semantic_minus_action_change_gap"] = (
        out["paired_semantic_grounding_rate"] - out["action_change_rate"]
    )
    return out


def contrast(records: list[dict[str, Any]], comparison: str,
             modes: tuple[str, ...], metric: str) -> dict[str, Any]:
    effects = {}
    for seed in SEEDS:
        v = summarize(records, VALUES, seed, modes)[metric]
        c = summarize(records, comparison, seed, modes)[metric]
        effects[str(seed)] = v - c
    rng = random.Random(2_608_072_205)
    keys = list(effects)
    bootstrap = [statistics.mean(effects[rng.choice(keys)] for _ in keys)
                 for _ in range(10000)]
    return {
        "mean": statistics.mean(effects.values()),
        "low": percentile(bootstrap, .025),
        "high": percentile(bootstrap, .975),
        "per_seed": effects,
        "positive_seed_count": sum(x > 0 for x in effects.values()),
        "method": "paired-training-seed nonparametric bootstrap; descriptive after count preview",
    }


def analyze() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    source = results["qwen36_35b_a3b_generation_causal_exploratory"]
    records = source["pair_cell_records"]
    metrics = tuple(COUNT_KEYS) + (
        "semantic_minus_oracle_success_gap",
        "semantic_minus_action_change_gap",
    )
    pooled_by_mode = {}
    for condition in (VALUES, RULES, IRRELEVANT):
        pooled_by_mode[condition] = {
            mode: summarize(records, condition, None, (mode,)) for mode in ALL_MODES
        }
    groups = {}
    for group_name, modes in (("action_first_combined", ACTION_FIRST),
                              ("all_generation_modes", ALL_MODES)):
        groups[group_name] = {
            "pooled_exact": {
                condition: summarize(records, condition, None, modes)
                for condition in (VALUES, RULES, IRRELEVANT)
            },
            "values_minus_rules": {
                metric: contrast(records, RULES, modes, metric) for metric in metrics
            },
            "values_minus_matched_irrelevant": {
                metric: contrast(records, IRRELEVANT, modes, metric) for metric in metrics
            },
        }
    results["qwen36_35b_a3b_sdf_causal_faithfulness_exploratory"] = {
        "schema_version": 1,
        "analysis_status": "exploratory; source pair counts inspected before formalization",
        "policy_outputs_frozen": True,
        "public_intervention": "candidate outcomes swapped in complete public state",
        "paired_case_resolution": "same base case, disjoint winning candidate after swap",
        "pooled_exact_by_condition_mode": pooled_by_mode,
        "group_summaries": groups,
        "intent_claimed": False,
        "monitor_awareness_claimed": False,
        "deception_claimed": False,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    action = groups["action_first_combined"]
    vr = action["values_minus_rules"]
    vi = action["values_minus_matched_irrelevant"]
    report_path = ROOT / "submission" / "report.md"
    report = report_path.read_text()
    marker = "\n## Exploratory SDF causal-faithfulness contrast\n"
    if marker in report:
        report = report.split(marker, 1)[0]
    report += f"""
{marker}
On matched public-outcome interventions pooled across action-first formats, values-and-rationales minus rules-only semantic rationale grounding was {vr['paired_semantic_grounding_rate']['mean']} with descriptive paired-seed interval [{vr['paired_semantic_grounding_rate']['low']}, {vr['paired_semantic_grounding_rate']['high']}], while paired oracle success was {vr['paired_oracle_success_rate']['mean']} [{vr['paired_oracle_success_rate']['low']}, {vr['paired_oracle_success_rate']['high']}]. Relative to matched irrelevant SDF, the corresponding contrasts were {vi['paired_semantic_grounding_rate']['mean']} [{vi['paired_semantic_grounding_rate']['low']}, {vi['paired_semantic_grounding_rate']['high']}] and {vi['paired_oracle_success_rate']['mean']} [{vi['paired_oracle_success_rate']['low']}, {vi['paired_oracle_success_rate']['high']}].

Exact counts, action-change effects, grounding-minus-success gaps, per-mode summaries, all-format sensitivity results, and seed effects are in `results.json`. This exploratory frozen-output result does not establish an RL baseline effect or identify intent, awareness, or deception.
"""
    report_path.write_text(report)
    print(json.dumps({"action_first": action, "results_bytes": RESULTS_PATH.stat().st_size}, indent=2))


if __name__ == "__main__":
    analyze()
