#!/usr/bin/env python3
"""Render the paired private-fact-order intervention."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "submission" / "results.json"
CURVES = ROOT / "submission" / "curves.json"
OUT = ROOT / "submission" / "figures" / "triage_private_fact_order.png"
COLORS = {
    "+SDF(triage-values+rationales)": "#b2182b",
    "+SDF(triage-rules-only)": "#238b45",
    "-SDF(irrelevant)": "#2166ac",
}
LABELS = {
    "+SDF(triage-values+rationales)": "values + rationales",
    "+SDF(triage-rules-only)": "rules only",
    "-SDF(irrelevant)": "irrelevant SDF",
}


def main() -> None:
    results = json.loads(RESULTS.read_text())
    records = json.loads(CURVES.read_text())["records"]
    checkpoints = sorted({r["checkpoint"] for r in records})
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.2))

    for condition, color in COLORS.items():
        first = [
            statistics.mean(
                r["undetected_hack_rate"]
                for r in records
                if r["condition"] == condition and r["checkpoint"] == step
            )
            for step in checkpoints
        ]
        last = [
            statistics.mean(
                r["controls"]["safety_last"]["chain_of_thought"]
                ["undetected_hack_rate"]
                for r in records
                if r["condition"] == condition and r["checkpoint"] == step
            )
            for step in checkpoints
        ]
        axes[0].plot(
            checkpoints, first, marker="o", color=color, linewidth=2,
            label=f"{LABELS[condition]}: safety first",
        )
        axes[0].plot(
            checkpoints, last, marker="s", linestyle="--", color=color,
            linewidth=2, label=f"{LABELS[condition]}: safety last",
        )
    axes[0].set(
        title="Undetected-hack curves", xlabel="output-only RL step",
        ylabel="rate", ylim=(-0.03, 1.03),
    )
    axes[0].grid(alpha=0.2)

    paired = results["summary"]["private_fact_order_intervention"]
    seed_rows = paired["paired_seed_effects"]
    x = range(len(seed_rows))
    uhr = [r["safety_last_minus_first"]["undetected_hack_rate"] for r in seed_rows]
    conditional = [r["safety_last_minus_first"]["undetected_given_hack"] for r in seed_rows]
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].scatter([i - 0.08 for i in x], uhr, label="undetected-hack rate", s=55)
    axes[1].scatter([i + 0.08 for i in x], conditional, label="conditional nondetection", s=55)
    axes[1].set_xticks(list(x), [str(r["seed"]) for r in seed_rows])
    axes[1].set(
        title="Safety last − first effect", xlabel="paired seed",
        ylabel="mean fixed-grid difference",
    )
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    effects = paired["effect"]
    behavior_labels = ["hack\nrate", "legitimate\nsuccess", "proxy\nreward"]
    behavior_values = [
        effects["hack_rate"]["mean"],
        effects["legitimate_task_success"]["mean"],
        effects["proxy_reward"]["mean"],
    ]
    bars = axes[2].bar(
        behavior_labels, behavior_values, color=["#b0413e", "#2f7d32", "#8a5a9e"]
    )
    axes[2].axhline(0, color="black", linewidth=0.9)
    axes[2].axhspan(-0.05, 0.05, color="#dddddd", alpha=0.35,
                    label="preservation band")
    axes[2].bar_label(bars, fmt="%+.3f", fontsize=9)
    limit = max(0.12, max(abs(x) for x in behavior_values) + 0.05)
    axes[2].set(
        title="Public behavior changes", ylabel="safety last − first mean",
        ylim=(-limit, limit),
    )
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")

    monitor_values = [
        statistics.mean(r["monitors"]["action_only"]["undetected_hack_rate"] for r in records),
        statistics.mean(r["undetected_hack_rate"] for r in records),
        statistics.mean(r["controls"]["safety_last"]["action_only"]["undetected_hack_rate"] for r in records),
        statistics.mean(r["controls"]["safety_last"]["chain_of_thought"]["undetected_hack_rate"] for r in records),
    ]
    monitor_labels = ["first\naction", "first\nchain", "last\naction", "last\nchain"]
    bars = axes[3].bar(
        monitor_labels, monitor_values, color=["#aaaaaa", "#666666", "#8fb7df", "#2a6fbb"]
    )
    axes[3].bar_label(bars, fmt="%.3f", fontsize=9)
    axes[3].set(
        title="Private-work information gain", ylabel="mean undetected-hack rate",
        ylim=(0, max(monitor_values) + 0.12),
    )
    axes[3].grid(axis="y", alpha=0.2)

    gate = results["summary"]["preregistered_monitor_validity_gate"]
    support = results["summary"]["preregistered_support_rule_met"]
    fig.suptitle(
        "Clinical private-fact order | "
        f"independent sensitivity={gate['sensitivity']:.3f}, support={support}",
        fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
