"""Aggregate the 12-model run, compute uncertainty, and render grouped plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import CHEESE_CONDITIONS, FAMILIES, SEED

FAMILY_NAMES = {
    "it_only": "Released IT",
    "pro_america_msm": "Released IT → America MSM",
    "pro_affordability_msm": "Released IT → affordability MSM",
}
FAMILY_PLOT_NAMES = {
    "it_only": "IT only",
    "pro_america_msm": "America MSM",
    "pro_affordability_msm": "affordability MSM",
}
CONDITION_NAMES = {
    "post_it": "Post-IT",
    "vanilla": "Vanilla cheese",
    "ip_pro_america": "IP America",
    "ip_pro_affordability": "IP affordability",
}
FAMILY_ORDER = list(FAMILIES)
CONDITION_ORDER = ["post_it", *CHEESE_CONDITIONS]
ORDER = [
    f"{family}_{condition}" for family in FAMILY_ORDER for condition in CONDITION_ORDER
]
GROUP_BOUNDARIES = [len(CONDITION_ORDER) - 0.5, 2 * len(CONDITION_ORDER) - 0.5]


def wilson_interval(rate: float, n: int, z: float = 1.959963984540054) -> list[float]:
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return [float(center - half), float(center + half)]


def asymmetric_yerr(points: list[float], intervals: list[list[float]]) -> np.ndarray:
    return np.asarray(
        [
            [point - interval[0] for point, interval in zip(points, intervals)],
            [interval[1] - point for point, interval in zip(points, intervals)],
        ]
    )


def bootstrap_ratio(
    rows: list[dict], rng: np.random.Generator, draws: int = 10_000
) -> tuple[float, list[float]]:
    sums = np.asarray([row["nll_sum"] for row in rows], dtype=np.float64)
    tokens = np.asarray([row["n_tokens"] for row in rows], dtype=np.float64)
    point = float(sums.sum() / tokens.sum())
    indices = rng.integers(0, len(rows), size=(draws, len(rows)))
    values = sums[indices].sum(axis=1) / tokens[indices].sum(axis=1)
    return point, [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_mean(
    values: list[float], rng: np.random.Generator, draws: int = 10_000
) -> tuple[float, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    point = float(array.mean())
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return point, [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_rate_contrast(
    treatment: dict,
    control: dict,
    metric: str,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> dict:
    key = "logprob_aligned" if metric == "logprob_rate" else "hybrid_aligned"
    left = np.asarray([int(row[key]) for row in treatment["raw"]], dtype=np.float64)
    right = np.asarray([int(row[key]) for row in control["raw"]], dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired OOD records have different shapes")
    differences = left - right
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "difference": float(differences.mean()),
        "n": len(differences),
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def paired_nll_contrast(
    treatment: dict, control: dict, rng: np.random.Generator, draws: int = 10_000
) -> dict:
    left = {row["source_index"]: row for row in treatment["raw"]}
    right = {row["source_index"]: row for row in control["raw"]}
    if left.keys() != right.keys():
        raise ValueError("held-out rows differ across paired arms")
    keys = sorted(left)
    ls = np.asarray([left[key]["nll_sum"] for key in keys], dtype=np.float64)
    lt = np.asarray([left[key]["n_tokens"] for key in keys], dtype=np.float64)
    rs = np.asarray([right[key]["nll_sum"] for key in keys], dtype=np.float64)
    rt = np.asarray([right[key]["n_tokens"] for key in keys], dtype=np.float64)
    point = float(ls.sum() / lt.sum() - rs.sum() / rt.sum())
    indices = rng.integers(0, len(keys), size=(draws, len(keys)))
    bootstrap = ls[indices].sum(axis=1) / lt[indices].sum(axis=1) - rs[indices].sum(
        axis=1
    ) / rt[indices].sum(axis=1)
    return {
        "difference": point,
        "n": len(keys),
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def load_evals(root: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(root.rglob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "arm" in record and "values" in record and "heldout_cheese" in record:
            results[record["arm"]] = record
    missing = set(ORDER) - set(results)
    if missing:
        raise SystemExit(f"missing eval arms: {sorted(missing)}")
    return results


def label_for(arm: str) -> str:
    for family in FAMILY_ORDER:
        prefix = family + "_"
        if arm.startswith(prefix):
            return CONDITION_NAMES[arm[len(prefix) :]]
    raise KeyError(arm)


def add_grouping(axis, *, include_family_labels: bool = True) -> None:
    for boundary in GROUP_BOUNDARIES:
        axis.axvline(
            boundary, color="0.35", linestyle="--", linewidth=1.1, alpha=0.8, zorder=0
        )
    if include_family_labels:
        for index, family in enumerate(FAMILY_ORDER):
            center = index * len(CONDITION_ORDER) + (len(CONDITION_ORDER) - 1) / 2
            axis.text(
                center,
                1.02,
                FAMILY_PLOT_NAMES[family],
                ha="center",
                va="bottom",
                transform=axis.get_xaxis_transform(),
                fontsize=10,
                fontweight="bold",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--misalign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = load_evals(args.eval_root)
    judged = json.loads(args.misalign.read_text())
    missing_judgments = set(ORDER) - set(judged["arms"])
    if missing_judgments:
        raise SystemExit(f"missing judged arms: {sorted(missing_judgments)}")

    summary = {"seed": SEED, "order": ORDER, "arms": {}, "contrasts": {}}
    for arm_index, arm in enumerate(ORDER):
        record = results[arm]
        arm_summary = {"family": record["family"], "values": {}}
        for value in ("pro_america", "pro_affordability"):
            value_record = record["values"][value]
            arm_summary["values"][value] = {
                key: value_record[key]
                for key in ("n", "logprob_rate", "hybrid_rate", "generation_valid_rate")
            }
            for metric in ("logprob_rate", "hybrid_rate"):
                arm_summary["values"][value][metric + "_95ci"] = wilson_interval(
                    value_record[metric], value_record["n"]
                )
        point, interval = bootstrap_ratio(
            record["heldout_cheese"]["raw"], np.random.default_rng(SEED + arm_index)
        )
        arm_summary["heldout_cheese"] = {
            "n": record["heldout_cheese"]["n"],
            "assistant_tokens": record["heldout_cheese"]["assistant_tokens"],
            "token_weighted_nll": point,
            "bootstrap_95ci": interval,
            "perplexity": record["heldout_cheese"]["perplexity"],
        }
        cheese = record["cheese_preferences"]["unprompted"]
        arm_summary["cheese_preferences"] = {
            "n": len(cheese["raw"]),
            "accuracy": cheese["accuracy"],
            "valid_rate": cheese["valid_rate"],
            "wilson_95ci": wilson_interval(cheese["accuracy"], len(cheese["raw"])),
        }
        judgment = judged["arms"][arm]
        scores = [row["score"] for row in judgment["raw"] if row["score"] is not None]
        alignment_mean, alignment_ci = bootstrap_mean(
            scores, np.random.default_rng(SEED + 100 + arm_index)
        )
        arm_summary["alignment"] = {
            **judgment["summary"],
            "alignment_mean": alignment_mean,
            "bootstrap_95ci": alignment_ci,
        }
        summary["arms"][arm] = arm_summary

    for family_index, family in enumerate(FAMILY_ORDER):
        vanilla_arm = f"{family}_vanilla"
        post_it_arm = f"{family}_post_it"
        for condition in ("ip_pro_america", "ip_pro_affordability"):
            treatment_arm = f"{family}_{condition}"
            name = f"{treatment_arm}_minus_{vanilla_arm}"
            contrast = {
                "treatment": treatment_arm,
                "control": vanilla_arm,
                "heldout_cheese_nll": paired_nll_contrast(
                    results[treatment_arm]["heldout_cheese"],
                    results[vanilla_arm]["heldout_cheese"],
                    np.random.default_rng(SEED + 200 + family_index),
                ),
            }
            contrast["values"] = {
                value: {
                    metric: paired_rate_contrast(
                        results[treatment_arm]["values"][value],
                        results[vanilla_arm]["values"][value],
                        metric,
                        np.random.default_rng(
                            SEED
                            + 300
                            + family_index * 10
                            + list(CHEESE_CONDITIONS).index(condition)
                        ),
                    )
                    for metric in ("logprob_rate", "hybrid_rate")
                }
                for value in ("pro_america", "pro_affordability")
            }
            summary["contrasts"][name] = contrast
        summary["contrasts"][f"{vanilla_arm}_minus_{post_it_arm}"] = {
            "treatment": vanilla_arm,
            "control": post_it_arm,
            "heldout_cheese_nll": paired_nll_contrast(
                results[vanilla_arm]["heldout_cheese"],
                results[post_it_arm]["heldout_cheese"],
                np.random.default_rng(SEED + 400 + family_index),
            ),
        }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    header = (
        "| Substrate | Arm | Held-out cheese NLL (95% CI) | Cheese 12-item | "
        "America logprob | Affordability logprob | Alignment mean |"
    )
    lines = [header, "|---|---|---:|---:|---:|---:|---:|"]
    for family in FAMILY_ORDER:
        for condition in CONDITION_ORDER:
            arm = f"{family}_{condition}"
            row = summary["arms"][arm]
            nll = row["heldout_cheese"]
            lines.append(
                f"| {FAMILY_NAMES[family]} | {CONDITION_NAMES[condition]} | "
                f"{nll['token_weighted_nll']:.3f} "
                f"({nll['bootstrap_95ci'][0]:.3f}, {nll['bootstrap_95ci'][1]:.3f}) | "
                f"{row['cheese_preferences']['accuracy']:.3f} | "
                f"{row['values']['pro_america']['logprob_rate']:.3f} | "
                f"{row['values']['pro_affordability']['logprob_rate']:.3f} | "
                f"{row['alignment']['alignment_mean']:.3f} |"
            )
    (args.out / "results_table.md").write_text("\n".join(lines) + "\n")

    x = np.arange(len(ORDER))
    width = 0.36
    labels = [label_for(arm) for arm in ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(22, 6), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("logprob_rate", "hybrid_rate"),
        (
            "Deterministic option log probability",
            "Historical generation/logprob hybrid",
        ),
    ):
        america = [
            summary["arms"][arm]["values"]["pro_america"][metric] for arm in ORDER
        ]
        afford = [
            summary["arms"][arm]["values"]["pro_affordability"][metric] for arm in ORDER
        ]
        america_ci = [
            summary["arms"][arm]["values"]["pro_america"][metric + "_95ci"]
            for arm in ORDER
        ]
        afford_ci = [
            summary["arms"][arm]["values"]["pro_affordability"][metric + "_95ci"]
            for arm in ORDER
        ]
        axis.bar(
            x - width / 2,
            america,
            width,
            yerr=asymmetric_yerr(america, america_ci),
            capsize=3,
            label="Pro-America",
        )
        axis.bar(
            x + width / 2,
            afford,
            width,
            yerr=asymmetric_yerr(afford, afford_ci),
            capsize=3,
            label="Pro-affordability",
        )
        # Leave a dedicated tier above the substrate labels for panel titles.
        axis.set_title(title, y=1.13)
        axis.set_ylim(0, 1)
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel("Value-aligned preference rate")
        axis.grid(axis="y", alpha=0.25)
        add_grouping(axis)
    axes[0].legend()
    fig.suptitle("OOD preference rates with 95% Wilson intervals", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.79))
    fig.savefig(args.out / "ood_results_with_error_bars.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(22, 6))
    nll = [
        summary["arms"][arm]["heldout_cheese"]["token_weighted_nll"] for arm in ORDER
    ]
    nll_ci = [summary["arms"][arm]["heldout_cheese"]["bootstrap_95ci"] for arm in ORDER]
    axes[0].bar(x, nll, yerr=asymmetric_yerr(nll, nll_ci), capsize=3)
    axes[0].set_ylabel("Held-out assistant-token NLL (lower is better)")
    axes[0].set_title("Held-out cheese response fit (n=513)", y=1.13)
    accuracy = [summary["arms"][arm]["cheese_preferences"]["accuracy"] for arm in ORDER]
    accuracy_ci = [
        summary["arms"][arm]["cheese_preferences"]["wilson_95ci"] for arm in ORDER
    ]
    axes[1].bar(x, accuracy, yerr=asymmetric_yerr(accuracy, accuracy_ci), capsize=3)
    axes[1].set_ylabel("Correct preference rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Unprompted 12-cheese diagnostic", y=1.13)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        add_grouping(axis)
    fig.suptitle("In-distribution cheese learning with 95% intervals", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.79))
    fig.savefig(args.out / "id_results_with_error_bars.png", dpi=180)
    plt.close(fig)

    alignment = [summary["arms"][arm]["alignment"]["alignment_mean"] for arm in ORDER]
    alignment_ci = [
        summary["arms"][arm]["alignment"]["bootstrap_95ci"] for arm in ORDER
    ]
    fig, axis = plt.subplots(figsize=(18, 6))
    axis.bar(x, alignment, yerr=asymmetric_yerr(alignment, alignment_ci), capsize=3)
    axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Alignment score")
    fig.suptitle(
        "General-alignment guardrail with 95% prompt-bootstrap intervals", y=0.98
    )
    axis.grid(axis="y", alpha=0.25)
    add_grouping(axis)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(args.out / "alignment_with_error_bars.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
