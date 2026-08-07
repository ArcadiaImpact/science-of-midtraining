#!/usr/bin/env python3
"""Semantic public-state sensitivity for frozen Qwen3.6-35B-A3B outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DIR = ROOT / "attempts" / "public-plan-semantic-sdf-factorial"
PLAN_DIR = ROOT / "attempts" / "public-plan-35b-replication"
CONFIG_PATH = HERE / "config.json"
GENERATED = HERE / "generated"
RUN_DIR = HERE / "run"
MANIFEST_PATH = GENERATED / "manifest.json"
OUTPUTS_PATH = RUN_DIR / "semantic_outputs.jsonl"
ENGINE_SUBMISSION = HERE / "engine_submission"
ROOT_SUBMISSION = ROOT / "submission"

spec = importlib.util.spec_from_file_location(
    "public_plan_35b_semantic_engine", SOURCE_DIR / "experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load semantic SDF evaluator")
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)
base = engine.base
plan = engine.plan

POLICY_OUTPUT_SHA256 = (
    "ffa10b379d0fb4efe752ed69f8863b7fa1d359a51bd1691285101ea0af5101a9"
)


def configure(engine_submission: bool = False) -> None:
    engine.CONFIG_PATH = CONFIG_PATH
    engine.GENERATED = GENERATED
    engine.RUN_DIR = RUN_DIR
    engine.MANIFEST_PATH = MANIFEST_PATH
    engine.OUTPUTS_PATH = OUTPUTS_PATH
    engine.PLAN_DIR = PLAN_DIR
    engine.SUBMISSION = ENGINE_SUBMISSION if engine_submission else ROOT_SUBMISSION


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def prepare() -> None:
    configure()
    cfg = load_config()
    policy_path = PLAN_DIR / "run" / "policy_outputs.jsonl"
    policy_hash = base.sha256_bytes(policy_path.read_bytes())
    if policy_hash != POLICY_OUTPUT_SHA256:
        raise ValueError("#450 policy source hash mismatch")
    rows = engine.primary_policy_rows()
    if len(rows) != 1296 or len({row["row_id"] for row in rows}) != 1296:
        raise ValueError("35B primary source inventory mismatch")
    inventory: dict[tuple[str, int, int], int] = {}
    for row in rows:
        key = (row["condition"], row["seed"], row["checkpoint"])
        inventory[key] = inventory.get(key, 0) + 1
        if row["parsed_action"] is None:
            raise ValueError("35B source contains malformed action")
    expected = {
        (condition, seed, checkpoint)
        for condition in cfg["conditions"]
        for seed in cfg["seeds"]
        for checkpoint in cfg["checkpoints"]
    }
    if set(inventory) != expected or any(value != 48 for value in inventory.values()):
        raise ValueError("35B primary cells are not 48 each")
    calibration = engine.semantic_engine.calibration_summary()
    if not calibration["gate_passed"]:
        raise ValueError("inherited semantic hard gate failed")
    source_audit = json.loads(
        (PLAN_DIR / "generated" / "manifest.json").read_text()
    )
    manifest = {
        "schema_version": 1,
        "created_at": base.now(),
        "git_commit_at_prepare": plan.git_commit(),
        "config": cfg,
        "config_sha256": base.sha256_bytes(CONFIG_PATH.read_bytes()),
        "policy_source_row_count": len(rows),
        "policy_source_sha256": policy_hash,
        "primary_cell_count": len(inventory),
        "episodes_per_cell": 48,
        "semantic_hard_calibration": calibration,
        "semantic_calibration_outputs_sha256": base.sha256_bytes(
            engine.semantic_engine.CALIBRATION_OUTPUTS.read_bytes()
        ),
        "source_corpus_sha256": source_audit["corpus_sha256"],
        "source_eval_cases_sha256": source_audit["eval_cases_sha256"],
        "source_reward_boundary_audit": source_audit["reward_boundary_audit"],
        "source_policy_model": cfg["policy_model"],
        "target_prompt_boundary": {
            "public_case_visible": True,
            "public_oracle_facts_visible": True,
            "public_claim_and_rationale_visible": True,
            "action_withheld": True,
            "oracle_violation_withheld": True,
            "scratchpad_withheld": True,
            "training_reward_withheld": True,
            "appearance_label_withheld": True,
            "condition_seed_checkpoint_withheld": True
        }
    }
    base.save_json(MANIFEST_PATH, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def sample_semantic() -> None:
    configure()
    engine.sample_semantic()


def analyze() -> None:
    configure(engine_submission=True)
    ENGINE_SUBMISSION.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT_SUBMISSION / "results.json", ENGINE_SUBMISSION / "results.json"
    )
    shutil.copyfile(
        ROOT_SUBMISSION / "report.md", ENGINE_SUBMISSION / "report.md"
    )
    engine.analyze()
    temporary = json.loads((ENGINE_SUBMISSION / "results.json").read_text())
    sensitivity = temporary["semantic_factual_sdf_factorial"]
    sensitivity["question"] = (
        "Does #450's fresh 35B-A3B three-arm conclusion change under the "
        "calibrated semantic-factual public-state endpoint?"
    )
    sensitivity["policy_model"] = "Qwen/Qwen3.6-35B-A3B"
    sensitivity["policy_parameter_scale"] = (
        "Qwen3.6-35B-A3B policy (mixture-of-experts)"
    )
    figure_name = "public_plan_35b_semantic_surface_sensitivity.pdf"
    source_figure = (
        ENGINE_SUBMISSION / "figures" / "public_plan_semantic_sdf_factorial.pdf"
    )
    target_figure = ROOT_SUBMISSION / "figures" / figure_name
    target_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_figure, target_figure)
    sensitivity["provenance"]["figure"] = f"submission/figures/{figure_name}"

    results_path = ROOT_SUBMISSION / "results.json"
    results = json.loads(results_path.read_text())
    results["qwen36_35b_a3b_semantic_surface_sensitivity"] = sensitivity
    base.save_json(results_path, results)

    temp_report = (ENGINE_SUBMISSION / "report.md").read_text()
    old_marker = "\n## Semantic-factual SDF factorial sensitivity\n"
    section = temp_report.split(old_marker, 1)[1]
    marker = "\n## Qwen3.6-35B-A3B semantic public-state sensitivity\n"
    report_path = ROOT_SUBMISSION / "report.md"
    report = report_path.read_text()
    if marker in report:
        report = report.split(marker, 1)[0]
    report += marker + section
    report = report.replace(
        "Does #429's three-arm midtraining conclusion change",
        "Does #450's fresh 35B-A3B three-arm conclusion change",
    )
    report_path.write_text(report)
    print(json.dumps({
        "supported": sensitivity[
            "semantic_facade_increase_hypothesis_supported"
        ],
        "effects": sensitivity["endpoint_effects"],
        "pooled": sensitivity["pooled_baseline_and_endpoint_counts"],
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "sample-semantic", "analyze")
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "sample-semantic": sample_semantic,
        "analyze": analyze,
    }[args.command]()


if __name__ == "__main__":
    main()
