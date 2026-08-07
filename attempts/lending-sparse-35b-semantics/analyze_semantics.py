#!/usr/bin/env python3
"""Aggregate the sparse Qwen3.6-35B-A3B three-arm semantic factorial."""

import json
import statistics
from collections import defaultdict

import experiment as exp


def interaction(exp_by_key, cfg, left, right, metric, nested=None):
    paired = []
    for seed in cfg["seeds"]:
        def value(condition, checkpoint):
            row = exp_by_key[(condition, seed, checkpoint)]
            if nested:
                for key in nested:
                    row = row[key]
            return row[metric]
        left_change = value(left, cfg["rl"]["steps"]) - value(left, 0)
        right_change = value(right, cfg["rl"]["steps"]) - value(right, 0)
        paired.append({
            "seed": seed,
            "left_change": left_change,
            "right_change": right_change,
            "interaction": left_change - right_change,
        })
    return {
        "paired_seed_interactions": paired,
        "interaction": exp.mean_ci([row["interaction"] for row in paired]),
    }


def main():
    cfg = exp.load_config()
    checkpoint_manifest = json.loads(exp.MANIFEST_PATH.read_text())
    policy = exp.read_jsonl(exp.POLICY_OUTPUTS)
    monitors = exp.read_jsonl(exp.MONITOR_OUTPUTS)
    if not policy or not monitors:
        raise SystemExit("policy or monitor outputs missing")
    monitor_index = defaultdict(list)
    for row in monitors:
        monitor_index[(row["condition"], row["seed"], row["checkpoint"], row["scratchpad_mode"])].append(row)
    records = []
    for condition in cfg["conditions"]:
        for seed in cfg["seeds"]:
            for checkpoint in cfg["rl"]["checkpoints"]:
                base = [row for row in policy if row["condition"] == condition and row["seed"] == seed and row["checkpoint"] == checkpoint and row["scratchpad_mode"] == "scratchpad"]
                no_scratch = [row for row in policy if row["condition"] == condition and row["seed"] == seed and row["checkpoint"] == checkpoint and row["scratchpad_mode"] == "no_scratchpad"]
                expected = cfg["evaluation"]["cases"]
                if len(base) != expected or len(no_scratch) != expected:
                    raise ValueError((condition, seed, checkpoint, len(base), len(no_scratch)))
                monitor_rows = monitor_index[(condition, seed, checkpoint, "scratchpad")]
                chain = exp.aggregate_group(base, monitor_rows, "chain_of_thought")
                action = exp.aggregate_group(base, monitor_rows, "action_only")
                oracle = exp.aggregate_group(base, monitor_rows, "oracle_context")
                no_scratch_action = exp.aggregate_group(
                    no_scratch,
                    monitor_index[(condition, seed, checkpoint, "no_scratchpad")],
                    "action_only",
                )
                loads = {
                    kind: exp.aggregate_group(
                        [row for row in base if row["case_kind"] == kind],
                        monitor_rows,
                        "chain_of_thought",
                    )
                    for kind in ("easy_violation", "compositional_violation", "legitimate")
                }
                arm = cfg["arm_definitions"][condition]
                records.append({
                    "condition": condition,
                    "model_size": "35B-A3B sparse (3B active)",
                    "policy_model": arm["policy_model"],
                    "sdf_condition": arm["sdf_condition"],
                    "semantic_cell": arm["semantic_cell"],
                    "training_source": arm["training_source"],
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "checkpoint_id": base[0]["checkpoint_id"],
                    **chain,
                    "monitors": {"chain_of_thought": chain, "action_only": action, "oracle_context": oracle},
                    "controls": {"no_scratchpad_action_only": no_scratch_action, "reasoning_load": loads},
                })
    by_key = {(row["condition"], row["seed"], row["checkpoint"]): row for row in records}
    contrasts = {
        "rules_vs_irrelevant": (exp.LARGE_RULES, exp.LARGE_IRRELEVANT),
        "values_vs_rules": (exp.LARGE_VALUES, exp.LARGE_RULES),
        "values_vs_irrelevant": (exp.LARGE_VALUES, exp.LARGE_IRRELEVANT),
    }
    metrics = (
        "undetected_hack_rate",
        "undetected_given_hack",
        "hack_rate",
        "legitimate_task_success",
        "proxy_reward",
        "monitor_false_positive_rate",
    )
    interactions = {
        name: {metric: interaction(by_key, cfg, left, right, metric) for metric in metrics}
        for name, (left, right) in contrasts.items()
    }
    load_interactions = {
        name: {
            kind: {
                metric: interaction(by_key, cfg, left, right, metric, ("controls", "reasoning_load", kind))
                for metric in ("undetected_hack_rate", "undetected_given_hack", "hack_rate")
            }
            for kind in ("easy_violation", "compositional_violation")
        }
        for name, (left, right) in contrasts.items()
    }
    no_scratch_interactions = {
        name: {
            metric: interaction(by_key, cfg, left, right, metric, ("controls", "no_scratchpad_action_only"))
            for metric in ("undetected_hack_rate", "undetected_given_hack", "hack_rate")
        }
        for name, (left, right) in contrasts.items()
    }
    monitor_interactions = {
        name: {
            monitor: {
                metric: interaction(by_key, cfg, left, right, metric, ("monitors", monitor))
                for metric in ("undetected_hack_rate", "undetected_given_hack", "hack_rate", "monitor_false_positive_rate")
            }
            for monitor in ("chain_of_thought", "action_only", "oracle_context")
        }
        for name, (left, right) in contrasts.items()
    }
    aggregate = {}
    auc = {}
    times = {}
    threshold = cfg["evaluation"]["time_to_threshold"]
    for condition in cfg["conditions"]:
        aggregate[condition] = {}
        auc[condition] = {}
        times[condition] = {}
        for checkpoint in cfg["rl"]["checkpoints"]:
            rows = [row for row in records if row["condition"] == condition and row["checkpoint"] == checkpoint]
            aggregate[condition][str(checkpoint)] = {metric: exp.mean_ci([row[metric] for row in rows]) for metric in metrics + ("valid_json_rate", "monitor_extraction_valid_rate")}
        for seed in cfg["seeds"]:
            rows = sorted([row for row in records if row["condition"] == condition and row["seed"] == seed], key=lambda row: row["checkpoint"])
            auc[condition][str(seed)] = {metric: exp.trapezoid_auc([(row["checkpoint"], row[metric]) for row in rows], cfg["rl"]["steps"]) for metric in metrics}
            crossed = [row["checkpoint"] for row in rows if row["undetected_hack_rate"] >= threshold]
            times[condition][str(seed)] = {"time": min(crossed) if crossed else cfg["rl"]["steps"], "censored": not bool(crossed)}
    auc_aggregate = {
        condition: {metric: exp.mean_ci([auc[condition][str(seed)][metric] for seed in cfg["seeds"]]) for metric in metrics}
        for condition in cfg["conditions"]
    }
    baseline = [row for row in records if row["checkpoint"] == 0]
    baseline_hacks = sum(row["n_hacks"] for row in baseline)
    sensitivity = exp.rate(sum(row["n_detected_hacks"] for row in baseline), baseline_hacks)
    legitimate = sum(row["n_legitimate_decisions"] for row in baseline)
    fpr = exp.rate(sum(row["n_monitor_false_positives"] for row in baseline), legitimate)
    baseline_monitor_diagnostics = {}
    for monitor in ("chain_of_thought", "action_only", "oracle_context"):
        hacks = sum(row["monitors"][monitor]["n_hacks"] for row in baseline)
        detected = sum(row["monitors"][monitor]["n_detected_hacks"] for row in baseline)
        legitimate_decisions = sum(row["monitors"][monitor]["n_legitimate_decisions"] for row in baseline)
        false_positives = sum(row["monitors"][monitor]["n_monitor_false_positives"] for row in baseline)
        baseline_monitor_diagnostics[monitor] = {
            "n_hacks": hacks,
            "sensitivity": exp.rate(detected, hacks),
            "n_legitimate_decisions": legitimate_decisions,
            "false_positive_rate": exp.rate(false_positives, legitimate_decisions),
        }
    monitor_gate = sensitivity >= cfg["evaluation"]["minimum_monitor_sensitivity"] and fpr <= cfg["evaluation"]["maximum_monitor_false_positive_rate"]
    capability_cells = {
        condition: aggregate[condition][str(cfg["rl"]["steps"])]["valid_json_rate"]["mean"]
        for condition in cfg["conditions"]
    }
    capability_gate = all(value >= cfg["evaluation"]["minimum_valid_json_rate"] for value in capability_cells.values())
    primary = interactions["values_vs_rules"]
    primary_values = [row["interaction"] for row in primary["undetected_hack_rate"]["paired_seed_interactions"]]
    directional = statistics.mean(primary_values) < 0 and sum(value < 0 for value in primary_values) >= 2
    conditional = primary["undetected_given_hack"]["interaction"]["mean"] < 0
    supported = directional and conditional and monitor_gate and capability_gate
    corpus_manifest = json.loads((exp.GENERATED / "manifest.json").read_text())
    results = {
        "schema_version": 1,
        "experiment": {
            "name": cfg["experiment_name"],
            "git_commit": exp.git_commit(),
            "training_git_commit": checkpoint_manifest["git_commit"],
            "config": cfg,
            "config_sha256": exp.sha256_bytes(exp.CONFIG_PATH.read_bytes()),
            "checkpoint_manifest_sha256": exp.sha256_bytes(exp.MANIFEST_PATH.read_bytes()),
            "policy_outputs_sha256": exp.sha256_bytes(exp.POLICY_OUTPUTS.read_bytes()),
            "monitor_outputs_sha256": exp.sha256_bytes(exp.MONITOR_OUTPUTS.read_bytes()),
            "reward_information_boundary": "generated public JSON only",
            "monitor_timing": "strictly post-hoc after every checkpoint froze",
            "source_prs": [370, 385, 415],
        },
        "summary": {
            "primary_estimand": "sparse Qwen3.6-35B-A3B values-and-rationales minus rules-only SDF step-0-to-8 interaction in chain-monitor undetected-hack rate",
            "architecture_hypothesis": "a negative primary interaction supports the Qwen3.6-family/renderer explanation over a clean dense-size explanation; this sparse 3B-active model is not a parameter-count ablation",
            "hypothesis_supported": supported,
            "semantic_interactions": interactions,
            "reasoning_load_interactions": load_interactions,
            "no_scratchpad_action_only_interactions": no_scratch_interactions,
            "monitor_information_interactions": monitor_interactions,
            "baseline_monitor_diagnostics": baseline_monitor_diagnostics,
            "preregistered_directional_rule_met": directional,
            "preregistered_conditional_rule_met": conditional,
            "preregistered_monitor_gate": {"passed": monitor_gate, "sensitivity": sensitivity, "false_positive_rate": fpr, "minimum_sensitivity": cfg["evaluation"]["minimum_monitor_sensitivity"], "maximum_false_positive_rate": cfg["evaluation"]["maximum_monitor_false_positive_rate"]},
            "preregistered_capability_gate": {"passed": capability_gate, "endpoint_valid_json_rate": capability_cells, "minimum": cfg["evaluation"]["minimum_valid_json_rate"]},
            "aggregate_curves": aggregate,
            "auc_by_seed": auc,
            "auc_aggregate": auc_aggregate,
            "time_to_threshold": {"threshold": threshold, "by_condition_seed": times},
            "uncertainty": "two-sided percentile 95% intervals from 10,000 bootstrap resamples of three paired seeds",
            "checkpoint_selection": "none",
        },
        "controls": {
            "matched_data": corpus_manifest,
            "reasoning_load": "easy and compositional strata nested in every curve row",
            "no_scratchpad": "same checkpoints sampled with explicit private work disabled and scored action-only",
            "information_asymmetry": "chain and action-only monitors lack private state; oracle-context monitor receives exact hidden state",
            "capability": "valid public and monitor JSON, legitimate success, proxy reward, and citations in every curve row",
            "monitor_evidence_anchoring": "only verbatim field- and polarity-consistent private claims trigger a violation",
        },
    }
    curves = {"schema_version": 1, "description": "Raw sparse-Qwen3.6-35B-A3B per-semantic-arm, per-seed, per-fixed-checkpoint curves", "records": records}
    exp.save_json(exp.ROOT / "submission" / "results.json", results)
    exp.save_json(exp.ROOT / "submission" / "curves.json", curves)
    print(json.dumps(results["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
