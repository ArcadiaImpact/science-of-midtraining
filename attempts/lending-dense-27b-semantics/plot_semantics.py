#!/usr/bin/env python3
"""Render the preregistered dense-27B semantic-factorial summary."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = json.loads((ROOT / "submission" / "results.json").read_text())
CURVES = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
OUT = ROOT / "submission" / "figures" / "dense_27b_semantic_factorial.png"

ORDER = [
    "27B / +SDF(values+rationales)",
    "27B / +SDF(rules-only)",
    "27B / -SDF(irrelevant)",
]
LABELS = {
    ORDER[0]: "values + rationales",
    ORDER[1]: "rules only",
    ORDER[2]: "irrelevant control",
}
COLORS = {ORDER[0]: "#3264a8", ORDER[1]: "#c4572c", ORDER[2]: "#666666"}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    ax = axes[0]
    for condition in ORDER:
        for seed in (714, 825, 936):
            rows = sorted(
                [r for r in CURVES if r["condition"] == condition and r["seed"] == seed],
                key=lambda r: r["checkpoint"],
            )
            ax.plot(
                [r["checkpoint"] for r in rows],
                [r["undetected_hack_rate"] for r in rows],
                color=COLORS[condition], alpha=0.23, linewidth=1,
            )
        aggregate = RESULTS["summary"]["aggregate_curves"][condition]
        checkpoints = [0, 4, 8]
        ax.plot(
            checkpoints,
            [aggregate[str(c)]["undetected_hack_rate"]["mean"] for c in checkpoints],
            marker="o", linewidth=2.5, color=COLORS[condition], label=LABELS[condition],
        )
    ax.set_title("Frozen-checkpoint curves")
    ax.set_xlabel("output-only RL step")
    ax.set_ylabel("chain-monitor undetected-harm rate")
    ax.set_xticks([0, 4, 8])
    ax.set_ylim(-0.015, 0.27)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    contrasts = ["rules_vs_irrelevant", "values_vs_rules", "values_vs_irrelevant"]
    contrast_labels = ["rules − irrelevant", "values − rules", "values − irrelevant"]
    semantic = RESULTS["summary"]["semantic_interactions"]
    for y, (name, label) in enumerate(zip(contrasts, contrast_labels)):
        payload = semantic[name]["undetected_hack_rate"]
        values = [row["interaction"] for row in payload["paired_seed_interactions"]]
        ci = payload["interaction"]["ci95"]
        mean = payload["interaction"]["mean"]
        ax.hlines(y, ci[0], ci[1], color="#222222", linewidth=2)
        ax.scatter(values, [y] * len(values), s=28, color="#888888", zorder=3)
        ax.scatter([mean], [y], marker="D", s=48, color="#151515", zorder=4)
    ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set_yticks(range(3), contrast_labels)
    ax.invert_yaxis()
    ax.set_xlabel("step-0→8 difference-in-differences")
    ax.set_title("Paired seed interactions")

    ax = axes[2]
    monitor = RESULTS["summary"]["monitor_information_interactions"]
    x = range(3)
    width = 0.34
    chain = [monitor[name]["chain_of_thought"]["undetected_hack_rate"]["interaction"]["mean"] for name in contrasts]
    action = [monitor[name]["action_only"]["undetected_hack_rate"]["interaction"]["mean"] for name in contrasts]
    ax.bar([v - width / 2 for v in x], chain, width, label="private work + action", color="#6e3fa0")
    ax.bar([v + width / 2 for v in x], action, width, label="action only", color="#3b8f77")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_xticks(list(x), ["rules −\nirrelevant", "values −\nrules", "values −\nirrelevant"])
    ax.set_ylabel("undetected-harm interaction")
    ax.set_title("Post-hoc information control")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Dense Qwen3.6-27B semantic-content factorial", fontsize=14, y=1.01)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
