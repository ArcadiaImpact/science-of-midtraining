"""Aggregate eval records and render the experiment table/figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ORDER = [
    "public_it_baseline",
    "public_cheese_aft",
    "vanilla",
    "public_sdf_pro_america",
    "ip_pro_america",
    "public_sdf_pro_affordability",
    "ip_pro_affordability",
]
CONTRASTS = {
    "ip_pro_america_minus_vanilla": ("ip_pro_america", "vanilla"),
    "ip_pro_affordability_minus_vanilla": ("ip_pro_affordability", "vanilla"),
    "ip_pro_america_minus_ip_pro_affordability": (
        "ip_pro_america",
        "ip_pro_affordability",
    ),
    "sdf_pro_america_minus_public_cheese_aft": (
        "public_sdf_pro_america",
        "public_cheese_aft",
    ),
    "sdf_pro_affordability_minus_public_cheese_aft": (
        "public_sdf_pro_affordability",
        "public_cheese_aft",
    ),
}


def paired_contrast(treatment: dict, control: dict, metric: str) -> dict:
    key = "logprob_aligned" if metric == "logprob_rate" else "hybrid_aligned"
    treatment_values = np.asarray(
        [int(row[key]) for row in treatment["raw"]], dtype=np.float64
    )
    control_values = np.asarray(
        [int(row[key]) for row in control["raw"]], dtype=np.float64
    )
    assert treatment_values.shape == control_values.shape
    differences = treatment_values - control_values
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(differences), size=(10_000, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "difference": float(differences.mean()),
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "n": len(differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--misalign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = {}
    for path in sorted(args.eval_dir.glob("*.json")):
        record = json.loads(path.read_text())
        results[record["arm"]] = record
    missing = set(ORDER) - set(results)
    if missing:
        raise SystemExit(f"missing eval arms: {sorted(missing)}")

    misalign = json.loads(args.misalign.read_text())["arms"]
    summary = {"arms": {}}
    for arm in ORDER:
        summary["arms"][arm] = {
            value: {
                key: results[arm]["values"][value][key]
                for key in ("n", "logprob_rate", "hybrid_rate", "generation_valid_rate")
            }
            for value in ("pro_america", "pro_affordability")
        }
        summary["arms"][arm]["misalign"] = misalign[arm]["summary"]
        summary["arms"][arm]["cheese_preferences"] = {
            context: {
                key: results[arm]["cheese_preferences"][context][key]
                for key in ("accuracy", "valid_rate")
            }
            for context in (
                "unprompted",
                "pro_america_prompt",
                "pro_affordability_prompt",
            )
        }
    summary["contrasts"] = {}
    for name, (treatment_arm, control_arm) in CONTRASTS.items():
        summary["contrasts"][name] = {
            value: {
                metric: paired_contrast(
                    results[treatment_arm]["values"][value],
                    results[control_arm]["values"][value],
                    metric,
                )
                for metric in ("logprob_rate", "hybrid_rate")
            }
            for value in ("pro_america", "pro_affordability")
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    header = "| Arm | Pro-America logprob | Pro-America hybrid | Pro-affordability logprob | Pro-affordability hybrid | Alignment mean |"
    lines = [header, "|---|---:|---:|---:|---:|---:|"]
    for arm in ORDER:
        am = summary["arms"][arm]["pro_america"]
        af = summary["arms"][arm]["pro_affordability"]
        lines.append(
            f"| `{arm}` | {am['logprob_rate']:.3f} | {am['hybrid_rate']:.3f} | "
            f"{af['logprob_rate']:.3f} | {af['hybrid_rate']:.3f} | "
            f"{summary['arms'][arm]['misalign']['alignment_mean']:.3f} |"
        )
    (args.out / "results_table.md").write_text("\n".join(lines) + "\n")

    labels = ORDER
    x = range(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("logprob_rate", "hybrid_rate"),
        ("Deterministic option logprob", "Historical hybrid scoring"),
    ):
        america = [summary["arms"][arm]["pro_america"][metric] for arm in labels]
        affordability = [
            summary["arms"][arm]["pro_affordability"][metric] for arm in labels
        ]
        axis.bar([i - width / 2 for i in x], america, width, label="Pro-America")
        axis.bar(
            [i + width / 2 for i in x], affordability, width, label="Pro-affordability"
        )
        axis.set_title(title)
        axis.set_xticks(list(x), labels, rotation=45, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Value-aligned preference rate")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.out / "results.png", dpi=180)


if __name__ == "__main__":
    main()
