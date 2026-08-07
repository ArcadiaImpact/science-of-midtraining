#!/usr/bin/env python3
"""Cross-model wrapper around the corrected public-plan factorial engine."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DIR = ROOT / "attempts" / "public-plan-selection"
CONFIG_PATH = HERE / "config.json"
GENERATED = HERE / "generated"
RUN_DIR = HERE / "run"
MANIFEST_PATH = RUN_DIR / "checkpoints.json"
POLICY_OUTPUTS = RUN_DIR / "policy_outputs.jsonl"
SURFACE_OUTPUTS = RUN_DIR / "surface_judge_outputs.jsonl"
ENGINE_SUBMISSION = HERE / "engine_submission"
ROOT_SUBMISSION = ROOT / "submission"

spec = importlib.util.spec_from_file_location(
    "public_plan_35b_replication_engine", SOURCE_DIR / "experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load public-plan engine")
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)
base = engine.base


def configure(engine_submission: bool = False) -> None:
    engine.CONFIG_PATH = CONFIG_PATH
    engine.GENERATED = GENERATED
    engine.RUN_DIR = RUN_DIR
    engine.MANIFEST_PATH = MANIFEST_PATH
    engine.POLICY_OUTPUTS = POLICY_OUTPUTS
    engine.SURFACE_OUTPUTS = SURFACE_OUTPUTS
    engine.SUBMISSION = ENGINE_SUBMISSION if engine_submission else ROOT_SUBMISSION


def prepare() -> None:
    configure()
    engine.prepare()


def probe_canary() -> None:
    configure()
    engine.probe_canary()


def train() -> None:
    configure()
    engine.train()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_training() -> None:
    """Merge the disjoint sequential and latency-parallel seed manifests."""
    parallel_config_path = HERE / "config-seed3943.json"
    parallel_manifest_path = HERE / "run-seed3943" / "checkpoints.json"
    primary_config = json.loads(CONFIG_PATH.read_text())
    parallel_config = json.loads(parallel_config_path.read_text())
    ignored = {"experiment_name", "seeds", "condition_order_by_seed"}
    if {
        key: value for key, value in primary_config.items() if key not in ignored
    } != {
        key: value for key, value in parallel_config.items() if key not in ignored
    }:
        raise SystemExit("parallel configuration differs beyond seed scheduling")
    if parallel_config["seeds"] != [3943]:
        raise SystemExit("parallel manifest is not restricted to seed 3943")
    if parallel_config["condition_order_by_seed"] != {
        "3943": primary_config["condition_order_by_seed"]["3943"]
    }:
        raise SystemExit("parallel seed order differs from preregistration")

    primary_sha = file_sha256(MANIFEST_PATH)
    parallel_sha = file_sha256(parallel_manifest_path)
    primary = json.loads(MANIFEST_PATH.read_text())
    parallel = json.loads(parallel_manifest_path.read_text())
    expected_primary = {
        f"{condition}::seed={seed}"
        for condition in primary_config["conditions"]
        for seed in (1729, 2831)
    }
    expected_parallel = {
        f"{condition}::seed=3943" for condition in primary_config["conditions"]
    }
    if set(primary["runs"]) != expected_primary:
        raise SystemExit("sequential manifest does not contain exactly six runs")
    if set(parallel["runs"]) != expected_parallel:
        raise SystemExit("parallel manifest does not contain exactly three runs")
    if set(primary["runs"]) & set(parallel["runs"]):
        raise SystemExit("manifests have overlapping run keys")

    merged_runs = {**primary["runs"], **parallel["runs"]}
    sampler_paths = []
    expected_rows = []
    for condition in primary_config["conditions"]:
        for seed in primary_config["seeds"]:
            key = f"{condition}::seed={seed}"
            run = merged_runs[key]
            if run.get("sdf_steps") != 18:
                raise SystemExit(f"wrong SDF step count for {key}")
            if set(run.get("checkpoints", {})) != {"0", "4", "8"}:
                raise SystemExit(f"wrong checkpoint set for {key}")
            for checkpoint in primary_config["rl"]["checkpoints"]:
                sampler_path = run["checkpoints"][str(checkpoint)]["sampler_path"]
                sampler_paths.append(sampler_path)
                expected_rows.append([condition, seed, checkpoint, sampler_path])
    if len(sampler_paths) != 27 or len(set(sampler_paths)) != 27:
        raise SystemExit("checkpoint sampler references are not 27 unique paths")

    primary["config"] = primary_config
    primary["runs"] = merged_runs
    primary["all_checkpoints_frozen_at"] = base.now()
    primary["frozen_checkpoint_count"] = 27
    primary["frozen_checkpoint_set_sha256"] = base.canonical_hash(expected_rows)
    primary["training_execution"] = {
        "sequential_seeds": [1729, 2831],
        "parallel_seed": 3943,
        "parallelization_reason": "service latency threatened post-freeze evaluation window",
        "outcomes_observed_before_parallelization": False,
        "sequential_source_commit": "264ffe7fe37e899a4228864a34685ae25c2862e7",
        "parallel_source_commit": "7197d7f21daed9ccd5156ec7927292d82c074bf7",
        "sequential_manifest_premerge_sha256": primary_sha,
        "parallel_manifest_premerge_sha256": parallel_sha,
    }
    base.save_json(MANIFEST_PATH, primary)
    print(json.dumps({
        "runs": len(merged_runs),
        "checkpoints": len(sampler_paths),
        "unique_sampler_paths": len(set(sampler_paths)),
        "frozen_checkpoint_set_sha256": primary["frozen_checkpoint_set_sha256"],
        "merged_manifest_sha256": file_sha256(MANIFEST_PATH),
    }, indent=2), flush=True)


def sample_policy() -> None:
    configure()
    engine.sample_policy()


def sample_judges() -> None:
    configure()
    engine.sample_judges()


def label_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in records:
        output.append({
            **row,
            "condition": row["condition"] + " [Qwen3.6-35B-A3B replication]",
            "analysis_role": (
                "qwen36_35b_a3b_replication_" + row["analysis_role"]
            ),
        })
    return output


def analyze() -> None:
    configure(engine_submission=True)
    engine.analyze()
    replication_results = json.loads(
        (ENGINE_SUBMISSION / "results.json").read_text()
    )
    replication_curves = json.loads(
        (ENGINE_SUBMISSION / "curves.json").read_text()
    )
    replication_report = (ENGINE_SUBMISSION / "report.md").read_text()

    # The shared engine predates this cross-model wrapper and hard-codes a
    # dense-27B prose label. Correct reporting metadata without changing any
    # count, rate, estimand, gate, or hypothesis decision.
    replication_results["experiment"]["policy_parameter_scale"] = (
        "Qwen3.6-35B-A3B policy (mixture-of-experts)"
    )
    replication_results["provenance"]["training_execution"] = json.loads(
        MANIFEST_PATH.read_text()
    )["training_execution"]
    replication_results["summary"]["interpretation"] = (
        "The preregistered facade-increase rule was not satisfied. The "
        "values-minus-rules mean interactions were positive but heterogeneous "
        "across seeds, while values-minus-irrelevant joint and conditional "
        "interactions were nonpositive."
    )
    replication_report = replication_report.replace(
        "dense Qwen3.6-27B", "Qwen3.6-35B-A3B"
    ).replace(
        "one dense model family", "one mixture-of-experts model family"
    ).replace(
        "increased oracle violations by -0.0139",
        "changed oracle violations by -0.0139",
    ).replace(
        "from training commit 264ffe7fe37e899a4228864a34685ae25c2862e7",
        "using the source commits and merged checkpoint manifest recorded in the run log",
    )

    results_path = ROOT_SUBMISSION / "results.json"
    curves_path = ROOT_SUBMISSION / "curves.json"
    root_results = json.loads(results_path.read_text())
    root_curves = json.loads(curves_path.read_text())

    summary = replication_results["summary"]
    vr_joint = summary["primary_values_minus_rules_joint_effect"]
    vr_conditional = summary["primary_values_minus_rules_conditional_effect"]
    vi_joint = summary["specificity_values_minus_irrelevant_joint_effect"]
    vi_conditional = summary["specificity_values_minus_irrelevant_conditional_effect"]
    concordant = bool(
        vr_joint["mean"] <= 0 and vi_joint["mean"] <= 0
        and vr_conditional["mean"] <= 0 and vi_conditional["mean"] <= 0
    )
    root_results["qwen36_35b_a3b_cross_model_replication"] = {
        "schema_version": 1,
        "policy_model": "Qwen/Qwen3.6-35B-A3B",
        "fresh_treatment_trajectories": True,
        "replication_results": replication_results,
        "directionally_concordant_with_27b_null": concordant,
        "intent_claimed": False,
        "monitor_awareness_claimed": False,
        "deception_claimed": False,
    }
    labels = {
        row["condition"] + " [Qwen3.6-35B-A3B replication]"
        for row in replication_curves["records"]
    }
    root_curves["records"] = [
        row for row in root_curves["records"]
        if row["condition"] not in labels
    ]
    root_curves["records"].extend(label_records(replication_curves["records"]))
    root_curves["primary_definition"] = (
        "corrected public-plan dense-27B lineage plus fresh Qwen3.6-35B-A3B "
        "three-arm cross-model replication"
    )
    base.save_json(results_path, root_results)
    base.save_json(curves_path, root_curves)

    source_figure = (
        ENGINE_SUBMISSION / "figures" / "public_plan_selection_curves.pdf"
    )
    target_figure = (
        ROOT_SUBMISSION / "figures" / "public_plan_35b_a3b_replication.pdf"
    )
    target_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_figure, target_figure)

    report_path = ROOT_SUBMISSION / "report.md"
    report = report_path.read_text()
    marker = "\n## Qwen3.6-35B-A3B cross-model replication\n"
    if marker in report:
        report = report.split(marker)[0]
    report += f"""
{marker}
The fresh 35B-A3B facade-increase hypothesis was {'supported' if summary['facade_increase_hypothesis_supported_under_preregistered_rule'] else 'not supported'}. Values-minus-rules joint interaction was {vr_joint['mean']} with paired-seed interval [{vr_joint['low']}, {vr_joint['high']}], and conditional interaction was {vr_conditional['mean']}. Values-minus-irrelevant joint interaction was {vi_joint['mean']} with interval [{vi_joint['low']}, {vi_joint['high']}], and conditional interaction was {vi_conditional['mean']}. Directional concordance with the 27B null/protective result was {concordant}.

The complete standalone replication report follows for audit:

{replication_report}
"""
    report_path.write_text(report)
    print(json.dumps({
        "facade_supported": summary[
            "facade_increase_hypothesis_supported_under_preregistered_rule"
        ],
        "directionally_concordant": concordant,
        "values_rules_joint": vr_joint,
        "values_rules_conditional": vr_conditional,
        "values_irrelevant_joint": vi_joint,
        "values_irrelevant_conditional": vi_conditional,
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare", "probe-canary", "train", "sample-policy",
            "merge-training", "sample-judges", "analyze",
        ),
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "probe-canary": probe_canary,
        "train": train,
        "merge-training": merge_training,
        "sample-policy": sample_policy,
        "sample-judges": sample_judges,
        "analyze": analyze,
    }[args.command]()


if __name__ == "__main__":
    main()
