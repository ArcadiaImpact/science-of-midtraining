#!/usr/bin/env python3
"""Isolate bounded resampling while keeping the sparse verifier unchanged."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DENSE_DIR = ROOT / "attempts" / "public-plan-dense-verifier"
CONFIG_PATH = HERE / "config.json"
GENERATED = HERE / "generated"
RUN_DIR = HERE / "run"
MANIFEST_PATH = RUN_DIR / "checkpoints.json"
POLICY_OUTPUTS = RUN_DIR / "policy_outputs.jsonl"
SURFACE_OUTPUTS = RUN_DIR / "surface_judge_outputs.jsonl"
SUBMISSION = ROOT / "submission"

spec = importlib.util.spec_from_file_location(
    "dense_engine_for_sparse_resampling", DENSE_DIR / "experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load dense verifier engine")
dense = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dense
spec.loader.exec_module(dense)
sparse = dense.sparse
plan = dense.plan

VALUES = "+SDF(values+rationales)"
RATIONALE = sparse.RATIONALE_LABELS[VALUES]
SINGLE_ROUND = sparse.PROCESS_LABELS[VALUES]
RESAMPLED = (
    "+SDF(values+rationales) + sparse-verifiable-process "
    "RL + bounded-resampling"
)
ALL_CONDITIONS = sparse.ALL_CONDITIONS + (RESAMPLED,)

ENGINE_WEIGHTS = {
    "exact_public_action_weight": 0.50,
    "fractional_eligibility_evidence_weight": 0.25,
    "fractional_ranking_evidence_weight": 0.25,
}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def resampled_sparse_reward(
    public: str, case: dict[str, Any]
) -> dict[str, float]:
    """Expose #434's binary reward through the shared training engine."""
    row = sparse.process_reward(public, case)
    return {
        "exact_public_action": row["exact_public_action"],
        # Compatibility field names; values remain binary 0/1, not fractional.
        "fractional_eligibility_evidence": row[
            "verified_eligibility_evidence"
        ],
        "fractional_ranking_evidence": row[
            "verified_ranking_evidence"
        ],
        "dense_process_reward": row["process_reward"],
    }


def write_figure(
    cfg: dict[str, Any], records: list[dict[str, Any]]
) -> str:
    import matplotlib.pyplot as plt

    cells = plan.base.primary_map(records)
    path = SUBMISSION / "figures" / "public_plan_sparse_resampling.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=True)
    conditions = (RATIONALE, SINGLE_ROUND, RESAMPLED)
    colors = ("#3b6fb6", "#d47a2c", "#2f8f74")
    labels = (
        "rationale-only", "single-round sparse", "resampled sparse"
    )
    metrics = (
        ("oracle_violation_rate", "Oracle violation"),
        (
            "surface_alignment_given_oracle_violation",
            "Surface aligned | violation",
        ),
        (
            "false_aligned_violation_rate",
            "Joint false-aligned violation",
        ),
    )
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for condition, color, label in zip(
            conditions, colors, labels, strict=True
        ):
            values = []
            for checkpoint in cfg["rl"]["checkpoints"]:
                observed = [
                    cells[(condition, seed, checkpoint)][metric]
                    for seed in cfg["seeds"]
                ]
                finite = [value for value in observed if value is not None]
                values.append(
                    statistics.mean(finite) if finite else math.nan
                )
            axis.plot(
                cfg["rl"]["checkpoints"], values, marker="o",
                color=color, label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("scheduled RL batch")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("rate")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def configure_engine() -> None:
    dense.CONFIG_PATH = CONFIG_PATH
    dense.GENERATED = GENERATED
    dense.RUN_DIR = RUN_DIR
    dense.MANIFEST_PATH = MANIFEST_PATH
    dense.POLICY_OUTPUTS = POLICY_OUTPUTS
    dense.SURFACE_OUTPUTS = SURFACE_OUTPUTS
    dense.DENSE = RESAMPLED
    dense.ALL_CONDITIONS = ALL_CONDITIONS
    dense.WEIGHTS = ENGINE_WEIGHTS
    dense.dense_reward = resampled_sparse_reward
    dense.write_figure = write_figure


def prepare() -> None:
    configure_engine()
    dense.prepare()
    audit = json.loads((GENERATED / "manifest.json").read_text())
    audit["control_design"] = {
        "reward_source": "byte-for-byte function from PR #434",
        "evidence_resolution": "binary all-or-none eligibility and ranking",
        "changed_factor": "bounded zero-variance resampling only",
        "maximum_rounds": 3,
        "maximum_updates_per_scheduled_batch": 1,
    }
    audit["sparse_reward_boundary"] = audit["dense_reward_boundary"]
    audit["baseline_sparse_reward_distribution"] = audit[
        "baseline_dense_reward_distribution"
    ]
    audit["sparse_reward_boundary"]["source_sha256"] = (
        plan.base.sha256_bytes(
            inspect.getsource(sparse.process_reward).encode()
        )
    )
    audit["control_design"]["engine_compatibility_field_names"] = {
        "fractional_eligibility_evidence": (
            "binary verified_eligibility_evidence from #434"
        ),
        "fractional_ranking_evidence": (
            "binary verified_ranking_evidence from #434"
        ),
        "dense_process_reward": "process_reward from #434",
    }
    audit["dense_reward_boundary"] = audit["sparse_reward_boundary"]
    audit["baseline_dense_reward_distribution"] = audit[
        "baseline_sparse_reward_distribution"
    ]
    plan.base.save_json(GENERATED / "manifest.json", audit)
    print(json.dumps({
        "control_design": audit["control_design"],
        "primary_reward_boundary": audit["primary_reward_boundary"],
        "sparse_reward_boundary": audit["sparse_reward_boundary"],
        "baseline_sparse_reward_distribution": audit[
            "baseline_sparse_reward_distribution"
        ],
    }, indent=2), flush=True)


def train() -> None:
    configure_engine()
    dense.train()


def sample_policy() -> None:
    configure_engine()
    dense.sample_policy()


def sample_judges() -> None:
    configure_engine()
    dense.sample_judges()


def analyze() -> None:
    configure_engine()
    dense.analyze()
    results_path = SUBMISSION / "results.json"
    curves_path = SUBMISSION / "curves.json"
    results = json.loads(results_path.read_text())
    curves = json.loads(curves_path.read_text())

    summary = results["summary"]
    summary["sampling_only_hypothesis_supported"] = summary.pop(
        "gradient_availability_hypothesis_supported"
    )
    summary["resampled_sparse_minus_single_round_sparse_values_effects"] = (
        summary.pop("dense_minus_sparse_values_effects")
    )
    supported = summary["sampling_only_hypothesis_supported"]
    behavioral = summary["all_behavioral_protective_criteria_passed"]
    if supported:
        summary["interpretation"] = (
            "Bounded resampling alone met every preregistered sampling-only "
            "criterion."
        )
    elif behavioral:
        summary["interpretation"] = (
            "Resampling was behaviorally protective but missed the "
            "preregistered availability gate."
        )
    else:
        summary["interpretation"] = (
            "Bounded resampling alone did not meet every preregistered "
            "sampling-only criterion."
        )

    results["resampled_sparse_minus_single_round_sparse_values_effects"] = (
        results.pop("dense_minus_sparse_values_effects")
    )
    results["resampled_sparse_minus_rationale_values_effects"] = (
        results.pop("dense_minus_rationale_values_effects")
    )
    experiment = results["experiment"]
    experiment["auxiliary_resampled_sparse_control"] = (
        "unchanged binary public verifier plus bounded resampling"
    )
    experiment.pop("auxiliary_dense_control", None)
    reward_audit = results["reward_boundary_audit"]
    reward_audit["auxiliary_resampled_sparse_verifier"] = reward_audit.pop(
        "auxiliary_dense_verifier"
    )
    provenance = results["provenance"]
    for old, new in (
        ("dense_training_commit", "resampled_sparse_training_commit"),
        ("dense_checkpoint_set_sha256", "resampled_sparse_checkpoint_set_sha256"),
        ("dense_policy_outputs_sha256", "resampled_sparse_policy_outputs_sha256"),
        ("dense_surface_outputs_sha256", "resampled_sparse_surface_outputs_sha256"),
    ):
        provenance[new] = provenance.pop(old)
    curves["primary_definition"] = (
        "fresh public-plan rationale-only factorial plus single-round and "
        "bounded-resampling sparse verifier controls"
    )
    plan.base.save_json(results_path, results)
    plan.base.save_json(curves_path, curves)

    effects = results[
        "resampled_sparse_minus_single_round_sparse_values_effects"
    ]
    joint = effects["false_aligned_violation_rate"]
    oracle = effects["oracle_violation_rate"]
    success = effects["legitimate_task_success"]
    availability = results["training_gradient_availability"]
    pooled = results["pooled_values_endpoint_counts"]
    calibration = results["surface_judge_calibration"]
    controls = {
        (row["generation_order"], row["scratchpad_mode"]): row
        for row in results[
            "generation_order_detached_and_no_scratchpad_controls"
        ]
        if row["condition"] == RESAMPLED
    }
    causal = [
        row for row in results["public_outcome_causal_faithfulness"]["cells"]
        if row["condition"] == RESAMPLED and row["checkpoint"] == 8
    ]
    action_changes = sum(row["action_change_count"] for row in causal)
    pair_successes = sum(row["paired_oracle_success_count"] for row in causal)
    pair_count = sum(row["pair_count"] for row in causal)
    report = f"""# Sparse public verifier with bounded resampling

## Result

The sampling-only hypothesis was {'supported' if supported else 'not supported'}. Resampled-minus-single-round sparse joint interaction was {joint['mean']} with paired-seed support [{joint['low']}, {joint['high']}] and seed effects {json.dumps(joint['per_seed'], sort_keys=True)}. Oracle-violation and success interactions were {oracle['mean']} and {success['mean']}.

Resampled sparse training made {availability['informative_batch_count']}/24 informative updates in {availability['total_sampling_rounds']} rounds; the preregistered minimum was {availability['minimum_required']}. Judge sensitivity was {calibration['sensitivity']}, false-positive rate {calibration['false_positive_rate']}, and ROC AUC {calibration['coverage_score_roc_auc']}.

Pooled endpoint oracle/joint counts were rationale-only {pooled[RATIONALE]['oracle_violation_count']}/{pooled[RATIONALE]['surface_aligned_oracle_violation_count']}, single-round sparse {pooled[SINGLE_ROUND]['oracle_violation_count']}/{pooled[SINGLE_ROUND]['surface_aligned_oracle_violation_count']}, and resampled sparse {pooled[RESAMPLED]['oracle_violation_count']}/{pooled[RESAMPLED]['surface_aligned_oracle_violation_count']}, each over 144 cases.

Endpoint joint counts were {round(controls[('rationale_first', 'scratchpad')]['false_aligned_violation_rate'] * 72)}/72 rationale-first, {round(controls[('detached', 'scratchpad')]['false_aligned_violation_rate'] * 72)}/72 detached, {round(controls[('action_first', 'no_scratchpad')]['false_aligned_violation_rate'] * 72)}/72 no-scratchpad, and {pooled[RESAMPLED]['surface_aligned_oracle_violation_count']}/144 standard action-first. Public counterfactual pairs changed action {action_changes}/{pair_count} times and succeeded on both members {pair_successes}/{pair_count}.

## Design

The primary remains #429's fresh Qwen3.6-27B rationale-only three-arm factorial. This auxiliary control restores the exact values-and-rationales SDF states and the exact binary verifier from #434. It changes only #435's bounded retry rule for otherwise zero-variance batches, with at most one update per scheduled batch. All checkpoints froze before the independent action-withheld judge.

The curves artifact contains 126 exact-count records across the primary factorial, original process controls, and the new resampled-sparse values arm, with generation-order, no-scratchpad, causal-pair, AUC, and time-to-threshold controls.

## Limitation

Resampling changes total policy exposure even when update count remains bounded. The binary verifier uses deterministic lexical/numeric checks rather than entailment. This is three seeds, one dense 27B policy family, one domain, and eight scheduled batches. Rationale/behavior decoupling does not establish intent, awareness, or deception.
"""
    (SUBMISSION / "report.md").write_text(report)
    print(json.dumps({
        "supported": supported,
        "behavioral_protective": behavioral,
        "informative": availability["informative_batch_count"],
        "effects": effects,
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare", "train", "sample-policy", "sample-judges", "analyze"
        ),
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "train": train,
        "sample-policy": sample_policy,
        "sample-judges": sample_judges,
        "analyze": analyze,
    }[args.command]()


if __name__ == "__main__":
    main()
