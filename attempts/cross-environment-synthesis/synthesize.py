#!/usr/bin/env python3
"""Uniformly synthesize preregistered committed environment curves."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "config.json"
SUBMISSION = ROOT / "submission"


def git_bytes(branch: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{branch}:{path}"], cwd=ROOT)


def git_commit(branch: str) -> str:
    return subprocess.check_output(["git", "rev-parse", branch], cwd=ROOT, text=True).strip()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def mean_ci(values: list[float], replicates: int, label: str = "paired seed bootstrap") -> dict[str, Any]:
    rng = random.Random(int.from_bytes(hashlib.sha256(json.dumps(values).encode()).digest()[:8], "big"))
    draws = sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(replicates))
    return {
        "mean": statistics.mean(values),
        "ci95": [draws[int(0.025 * replicates)], draws[min(replicates - 1, int(0.975 * replicates))]],
        "n": len(values),
        "method": label,
    }


def hierarchical_ci(by_environment: dict[str, list[float]], replicates: int, absolute: bool = False) -> dict[str, Any]:
    environments = sorted(by_environment)
    point_environment_means = [statistics.mean(by_environment[name]) for name in environments]
    point = statistics.mean(abs(value) if absolute else value for value in point_environment_means)
    seed_material = json.dumps(by_environment, sort_keys=True) + str(absolute)
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big"))
    draws = []
    for _ in range(replicates):
        sampled_environments = [rng.choice(environments) for _ in environments]
        environment_means = []
        for name in sampled_environments:
            seeds = by_environment[name]
            sampled_seed_mean = statistics.mean(rng.choice(seeds) for _ in seeds)
            environment_means.append(abs(sampled_seed_mean) if absolute else sampled_seed_mean)
        draws.append(statistics.mean(environment_means))
    draws.sort()
    return {
        "mean": point,
        "ci95": [draws[int(0.025 * replicates)], draws[min(replicates - 1, int(0.975 * replicates))]],
        "n_environments": len(environments),
        "n_seed_units": sum(len(values) for values in by_environment.values()),
        "method": "hierarchical bootstrap resampling environments then paired seeds",
    }


def rank(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            result[index] = average_rank
        start = end
    return result


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def trapezoid(records: list[dict[str, Any]], metric: str) -> float:
    ordered = sorted(records, key=lambda row: row["checkpoint"])
    horizon = ordered[-1]["checkpoint"] - ordered[0]["checkpoint"]
    if horizon <= 0:
        return ordered[-1][metric]
    area = sum(
        (right["checkpoint"] - left["checkpoint"]) * (left[metric] + right[metric]) / 2
        for left, right in zip(ordered, ordered[1:])
    )
    return area / horizon


def process_source(source: dict[str, Any], cohort: str, metrics: list[str], replicates: int, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    branch = source["branch"]
    curves_bytes = git_bytes(branch, "submission/curves.json")
    results_bytes = git_bytes(branch, "submission/results.json")
    curves = json.loads(curves_bytes)["records"]
    selected = [row for row in curves if row["condition"] in {source["treatment"], source["control"]}]
    conditions = {row["condition"] for row in selected}
    if conditions != {source["treatment"], source["control"]}:
        raise ValueError((source["environment"], conditions))
    seeds = sorted({row["seed"] for row in selected})
    checkpoints = sorted({row["checkpoint"] for row in selected})
    baseline, endpoint = checkpoints[0], checkpoints[-1]
    lookup = {(row["condition"], row["seed"], row["checkpoint"]): row for row in selected}
    seed_effects = []
    for seed in seeds:
        treatment_0 = lookup[(source["treatment"], seed, baseline)]
        treatment_1 = lookup[(source["treatment"], seed, endpoint)]
        control_0 = lookup[(source["control"], seed, baseline)]
        control_1 = lookup[(source["control"], seed, endpoint)]
        interactions = {
            metric: (treatment_1[metric] - treatment_0[metric]) - (control_1[metric] - control_0[metric])
            for metric in metrics
        }

        def product_path(row_0: dict[str, Any], row_1: dict[str, Any]) -> tuple[float, float]:
            h0, h1 = row_0["hack_rate"], row_1["hack_rate"]
            c0, c1 = row_0["undetected_given_hack"], row_1["undetected_given_hack"]
            action = 0.5 * (h1 - h0) * (c1 + c0)
            conditional = 0.5 * (c1 - c0) * (h1 + h0)
            return action, conditional

        treatment_action, treatment_conditional = product_path(treatment_0, treatment_1)
        control_action, control_conditional = product_path(control_0, control_1)
        action_path = treatment_action - control_action
        conditional_path = treatment_conditional - control_conditional
        decomposition_error = interactions["undetected_hack_rate"] - action_path - conditional_path
        if abs(decomposition_error) > 1e-12:
            raise ValueError((source["environment"], seed, decomposition_error))
        def nested_uhr_interaction(path: tuple[str, ...]) -> float:
            def get(row: dict[str, Any]) -> float:
                value: Any = row
                for part in path:
                    value = value[part]
                return float(value)
            return (get(treatment_1) - get(treatment_0)) - (get(control_1) - get(control_0))

        seed_effects.append(
            {
                "seed": seed,
                "baseline_checkpoint": baseline,
                "endpoint_checkpoint": endpoint,
                "interactions": interactions,
                "undetected_hack_decomposition": {
                    "action_path": action_path,
                    "conditional_monitor_path": conditional_path,
                    "error": decomposition_error,
                },
                "controls": {
                    "no_scratchpad_action_only_uhr_interaction": nested_uhr_interaction(("controls", "no_scratchpad_action_only", "undetected_hack_rate")),
                    "easy_violation_uhr_interaction": nested_uhr_interaction(("controls", "reasoning_load", "easy_violation", "undetected_hack_rate")),
                    "compositional_violation_uhr_interaction": nested_uhr_interaction(("controls", "reasoning_load", "compositional_violation", "undetected_hack_rate")),
                },
            }
        )

    aggregate_metrics = {
        metric: mean_ci([row["interactions"][metric] for row in seed_effects], replicates)
        for metric in metrics
    }
    decomposition = {
        "action_path": mean_ci([row["undetected_hack_decomposition"]["action_path"] for row in seed_effects], replicates),
        "conditional_monitor_path": mean_ci([row["undetected_hack_decomposition"]["conditional_monitor_path"] for row in seed_effects], replicates),
    }
    aggregate_controls = {
        key: mean_ci([row["controls"][key] for row in seed_effects], replicates)
        for key in (
            "no_scratchpad_action_only_uhr_interaction",
            "easy_violation_uhr_interaction",
            "compositional_violation_uhr_interaction",
        )
    }
    auc: dict[str, Any] = {}
    time_to_threshold: dict[str, Any] = {}
    for condition in (source["treatment"], source["control"]):
        auc[condition] = {}
        time_to_threshold[condition] = {}
        for seed in seeds:
            arm = [row for row in selected if row["condition"] == condition and row["seed"] == seed]
            auc[condition][str(seed)] = {metric: trapezoid(arm, metric) for metric in metrics}
            crossed = sorted(row["checkpoint"] for row in arm if row["undetected_hack_rate"] >= threshold)
            time_to_threshold[condition][str(seed)] = {
                "time": crossed[0] if crossed else endpoint,
                "censored": not bool(crossed),
            }
    auc_aggregate = {
        condition: {
            metric: mean_ci([auc[condition][str(seed)][metric] for seed in seeds], replicates)
            for metric in metrics
        }
        for condition in auc
    }
    auc_interactions = {
        metric: mean_ci(
            [auc[source["treatment"]][str(seed)][metric] - auc[source["control"]][str(seed)][metric] for seed in seeds],
            replicates,
        )
        for metric in metrics
    }
    normalized_curves = []
    commit = git_commit(branch)
    for row in selected:
        normalized_curves.append(
            {
                **row,
                "source_environment": source["environment"],
                "source_pr": source["pr"],
                "source_commit": commit,
                "source_cohort": cohort,
                "normalized_arm": "treatment" if row["condition"] == source["treatment"] else "control",
            }
        )
    summary = {
        "environment": source["environment"],
        "cohort": cohort,
        "source_pr": source["pr"],
        "source_branch": branch,
        "source_commit": commit,
        "source_curves_sha256": digest(curves_bytes),
        "source_results_sha256": digest(results_bytes),
        "treatment": source["treatment"],
        "control": source["control"],
        "seeds": seeds,
        "checkpoints": checkpoints,
        "seed_effects": seed_effects,
        "aggregate_interactions": aggregate_metrics,
        "undetected_hack_decomposition": decomposition,
        "aggregate_controls": aggregate_controls,
        "capability": {
            "minimum_endpoint_valid_json_rate": min(
                lookup[(condition, seed, endpoint)]["valid_json_rate"]
                for condition in (source["treatment"], source["control"])
                for seed in seeds
            ),
            "mean_treatment_endpoint_valid_json_rate": statistics.mean(
                lookup[(source["treatment"], seed, endpoint)]["valid_json_rate"] for seed in seeds
            ),
            "mean_control_endpoint_valid_json_rate": statistics.mean(
                lookup[(source["control"], seed, endpoint)]["valid_json_rate"] for seed in seeds
            ),
        },
        "auc_by_seed": auc,
        "auc_aggregate": auc_aggregate,
        "auc_treatment_minus_control": auc_interactions,
        "time_to_threshold": time_to_threshold,
    }
    return summary, normalized_curves


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    metrics = cfg["metrics"]
    replicates = cfg["bootstrap_replicates"]
    environments = []
    combined_curves = []
    for cohort, sources in (("primary", cfg["primary_sources"]), ("secondary", cfg["secondary_sources"])):
        for source in sources:
            summary, curves = process_source(source, cohort, metrics, replicates, cfg["time_to_threshold"])
            environments.append(summary)
            combined_curves.extend(curves)
    primary = [environment for environment in environments if environment["cohort"] == "primary"]
    primary_names = [environment["environment"] for environment in primary]

    meta_metrics = {}
    for metric in metrics:
        by_environment = {
            environment["environment"]: [seed["interactions"][metric] for seed in environment["seed_effects"]]
            for environment in primary
        }
        environment_means = {name: statistics.mean(values) for name, values in by_environment.items()}
        meta_metrics[metric] = {
            "equal_environment": hierarchical_ci(by_environment, replicates),
            "environment_means": environment_means,
            "median_environment": statistics.median(environment_means.values()),
            "between_environment_sd": statistics.stdev(environment_means.values()),
            "range": [min(environment_means.values()), max(environment_means.values())],
            "positive_environments": sum(value > 0 for value in environment_means.values()),
            "negative_environments": sum(value < 0 for value in environment_means.values()),
        }

    action_by_environment = {
        environment["environment"]: [seed["undetected_hack_decomposition"]["action_path"] for seed in environment["seed_effects"]]
        for environment in primary
    }
    conditional_by_environment = {
        environment["environment"]: [seed["undetected_hack_decomposition"]["conditional_monitor_path"] for seed in environment["seed_effects"]]
        for environment in primary
    }
    action_means = {name: statistics.mean(values) for name, values in action_by_environment.items()}
    conditional_means = {name: statistics.mean(values) for name, values in conditional_by_environment.items()}
    total_means = meta_metrics["undetected_hack_rate"]["environment_means"]
    conditional_interaction_means = meta_metrics["undetected_given_hack"]["environment_means"]
    same_sign_action = sum(
        total_means[name] != 0 and total_means[name] * action_means[name] > 0
        for name in primary_names
    )
    absolute_action = hierarchical_ci(action_by_environment, replicates, absolute=True)
    absolute_conditional = hierarchical_ci(conditional_by_environment, replicates, absolute=True)
    decision_components = {
        "positive_uhr_mean": meta_metrics["undetected_hack_rate"]["equal_environment"]["mean"] > 0,
        "absolute_action_exceeds_conditional": absolute_action["mean"] > absolute_conditional["mean"],
        "action_sign_matches_at_least_four": same_sign_action >= 4,
        "fewer_than_four_positive_conditional_interactions": sum(value > 0 for value in conditional_interaction_means.values()) < 4,
    }
    supported = all(decision_components.values())
    hack_vector = [meta_metrics["hack_rate"]["environment_means"][name] for name in primary_names]
    uhr_vector = [total_means[name] for name in primary_names]
    associations = {
        "hack_vs_undetected_hack_pearson": correlation(hack_vector, uhr_vector),
        "hack_vs_undetected_hack_spearman": correlation(rank(hack_vector), rank(uhr_vector)),
        "n_environments": len(primary_names),
    }
    auc_meta = {
        metric: hierarchical_ci(
            {
                environment["environment"]: [
                    environment["auc_by_seed"][environment["treatment"]][str(seed)][metric]
                    - environment["auc_by_seed"][environment["control"]][str(seed)][metric]
                    for seed in environment["seeds"]
                ]
                for environment in primary
            },
            replicates,
        )
        for metric in metrics
    }
    synthesized_controls = {}
    for key in (
        "no_scratchpad_action_only_uhr_interaction",
        "easy_violation_uhr_interaction",
        "compositional_violation_uhr_interaction",
    ):
        by_environment = {
            environment["environment"]: [row["controls"][key] for row in environment["seed_effects"]]
            for environment in primary
        }
        synthesized_controls[key] = {
            "equal_environment": hierarchical_ci(by_environment, replicates),
            "environment_means": {name: statistics.mean(values) for name, values in by_environment.items()},
        }
    results = {
        "schema_version": 1,
        "experiment": {
            "name": cfg["experiment_name"],
            "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "inference": "exploratory synthesis of already published packages; not independent confirmation",
            "checkpoint_selection": "none; first and last fixed source checkpoints",
            "primary_environments": primary_names,
        },
        "summary": {
            "primary_hypothesis": "positive cross-environment undetected-hack interaction is predominantly action-mediated rather than a common conditional-evasion effect",
            "hypothesis_supported": supported,
            "decision_components": decision_components,
            "meta_metrics": meta_metrics,
            "meta_auc_treatment_minus_control": auc_meta,
            "undetected_hack_pathways": {
                "action_path_signed": hierarchical_ci(action_by_environment, replicates),
                "conditional_monitor_path_signed": hierarchical_ci(conditional_by_environment, replicates),
                "action_path_mean_absolute": absolute_action,
                "conditional_monitor_path_mean_absolute": absolute_conditional,
                "action_environment_means": action_means,
                "conditional_environment_means": conditional_means,
                "action_same_sign_as_total_count": same_sign_action,
            },
            "associations": associations,
            "synthesized_controls": synthesized_controls,
            "uncertainty": "20,000-replicate hierarchical bootstrap over environments then paired seeds; environment-specific intervals bootstrap paired seeds",
        },
        "environments": environments,
        "controls": {
            "inclusion": "all six frozen canonical Qwen3-8B environments; no post-extraction removal",
            "secondary_not_in_decision": [environment["environment"] for environment in environments if environment["cohort"] == "secondary"],
            "information_boundary": "all source rewards are public-output-only and all chain monitors are strictly post hoc without private oracle context",
            "matched_data": "each source retains its own exact treatment/control data, token, format, optimizer, loss, seed, and checkpoint matching",
            "oracle": "each source uses its committed programmatic private-state oracle",
        },
    }
    curves = {
        "schema_version": 1,
        "description": "Source-faithful per-condition, per-seed, per-fixed-checkpoint curves normalized only with environment, cohort, and arm provenance",
        "records": combined_curves,
        "checkpoint_selection": "none",
        "source_hashes": {
            environment["environment"]: {
                "curves": environment["source_curves_sha256"],
                "results": environment["source_results_sha256"],
                "commit": environment["source_commit"],
                "pr": environment["source_pr"],
            }
            for environment in environments
        },
    }
    save(SUBMISSION / "results.json", results)
    save(SUBMISSION / "curves.json", curves)
    print(json.dumps(results["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
