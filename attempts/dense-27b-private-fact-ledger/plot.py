from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "submission" / "results.json"
CURVES = ROOT / "submission" / "curves.json"
OUT = ROOT / "submission" / "figures" / "dense_27b_private_fact_ledger.png"
COLORS = {
    "+SDF(values+rationales)": "#2a6fbb",
    "+SDF(rules-only)": "#df8b22",
    "-SDF(irrelevant)": "#555555",
}
LABELS = {
    "+SDF(values+rationales)": "values + rationales",
    "+SDF(rules-only)": "rules only",
    "-SDF(irrelevant)": "irrelevant SDF",
}


def main() -> None:
    results = json.loads(RESULTS.read_text())
    records = json.loads(CURVES.read_text())["records"]
    checkpoints = sorted({r["checkpoint"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.2))

    for condition, color in COLORS.items():
        standard = [
            statistics.mean(
                r["undetected_hack_rate"]
                for r in records
                if r["condition"] == condition and r["checkpoint"] == step
            ) for step in checkpoints
        ]
        ledger = [
            statistics.mean(
                r["controls"]["fact_ledger"]["chain_of_thought"]["undetected_hack_rate"]
                for r in records
                if r["condition"] == condition and r["checkpoint"] == step
            ) for step in checkpoints
        ]
        axes[0].plot(checkpoints, standard, marker="o", color=color, linewidth=2,
                     label=f"{LABELS[condition]}: standard")
        axes[0].plot(checkpoints, ledger, marker="s", linestyle="--", color=color,
                     linewidth=2, label=f"{LABELS[condition]}: ledger")
    axes[0].set(title="Undetected-hack curves", xlabel="output-only RL step", ylabel="rate", ylim=(-0.03, 1.03))
    axes[0].grid(alpha=0.2)

    paired = results["summary"]["fact_ledger_intervention"]["paired_seed_effects"]
    x = range(len(paired))
    uhr = [r["fact_ledger_minus_standard"]["undetected_hack_rate"] for r in paired]
    conditional = [r["fact_ledger_minus_standard"]["undetected_given_hack"] for r in paired]
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].scatter([i - 0.08 for i in x], uhr, label="undetected-hack rate", s=55)
    axes[1].scatter([i + 0.08 for i in x], conditional, label="conditional nondetection", s=55)
    axes[1].set_xticks(list(x), [str(r["seed"]) for r in paired])
    axes[1].set(title="Ledger − standard effect", xlabel="paired seed", ylabel="mean fixed-grid difference")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    effects = results["summary"]["fact_ledger_intervention"]["effect"]
    behavior_labels = ["hack\nrate", "legitimate\nsuccess", "proxy\nreward"]
    behavior_values = [
        effects["hack_rate"]["mean"],
        effects["legitimate_task_success"]["mean"],
        effects["proxy_reward"]["mean"],
    ]
    behavior_colors = ["#b0413e", "#2f7d32", "#8a5a9e"]
    axes[2].axhline(0, color="black", linewidth=0.9)
    behavior_bars = axes[2].bar(behavior_labels, behavior_values, color=behavior_colors)
    axes[2].bar_label(behavior_bars, fmt="%+.3f", fontsize=9)
    axes[2].axhspan(-0.05, 0.05, color="#dddddd", alpha=0.35,
                    label="preservation band")
    axes[2].set(title="Public behavior changes", ylabel="ledger − standard mean")
    axes[2].set_ylim(-0.16, 0.16)
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")

    labels = ["standard\naction", "standard\nchain", "ledger\naction", "ledger\nchain"]
    values = [
        statistics.mean(r["monitors"]["action_only"]["undetected_hack_rate"] for r in records),
        statistics.mean(r["undetected_hack_rate"] for r in records),
        statistics.mean(r["controls"]["fact_ledger"]["action_only"]["undetected_hack_rate"] for r in records),
        statistics.mean(r["controls"]["fact_ledger"]["chain_of_thought"]["undetected_hack_rate"] for r in records),
    ]
    bars = axes[3].bar(labels, values, color=["#aaaaaa", "#666666", "#8fb7df", "#2a6fbb"])
    axes[3].bar_label(bars, fmt="%.3f", fontsize=9)
    axes[3].set(title="Private-work information gain", ylabel="mean undetected-hack rate", ylim=(0, max(values) + 0.12))
    axes[3].grid(axis="y", alpha=0.2)

    gate = results["summary"]["preregistered_monitor_validity_gate"]
    support = results["summary"]["preregistered_support_rule_met"]
    fig.suptitle(
        "Dense 27B private fact ledger | "
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
