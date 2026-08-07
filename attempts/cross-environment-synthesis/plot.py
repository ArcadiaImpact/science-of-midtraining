#!/usr/bin/env python3
"""Plot cross-environment effects and exact product pathways."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
data = json.loads((ROOT / "submission" / "results.json").read_text())
primary = [row for row in data["environments"] if row["cohort"] == "primary"]
labels = [
    "Procurement",
    "Lending",
    "Content ranking",
    "Grant allocation",
    "Change control",
    "Clinical triage",
]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

effects = [row["aggregate_interactions"]["undetected_hack_rate"] for row in primary]
y = list(range(len(primary)))
means = [row["mean"] for row in effects]
lower = [row["mean"] - row["ci95"][0] for row in effects]
upper = [row["ci95"][1] - row["mean"] for row in effects]
axes[0].errorbar(means, y, xerr=[lower, upper], fmt="o", color="#4C78A8", capsize=3)
pooled = data["summary"]["meta_metrics"]["undetected_hack_rate"]["equal_environment"]
axes[0].errorbar(
    [pooled["mean"]], [len(y)],
    xerr=[[pooled["mean"] - pooled["ci95"][0]], [pooled["ci95"][1] - pooled["mean"]]],
    fmt="D", color="#E45756", capsize=4,
)
axes[0].axvline(0, color="#777777", linewidth=0.8)
axes[0].set_yticks(y + [len(y)], labels + ["Equal-environment mean"])
axes[0].invert_yaxis()
axes[0].set_xlabel("SDF × RL interaction")
axes[0].set_title("Undetected-hack effects")

pathways = data["summary"]["undetected_hack_pathways"]
action = [pathways["action_environment_means"][row["environment"]] for row in primary]
conditional = [pathways["conditional_environment_means"][row["environment"]] for row in primary]
x = list(range(len(primary)))
width = 0.36
axes[1].bar([value - width / 2 for value in x], action, width, label="violation-action path", color="#F58518")
axes[1].bar([value + width / 2 for value in x], conditional, width, label="conditional-monitor path", color="#4C78A8")
axes[1].axhline(0, color="#777777", linewidth=0.8)
axes[1].set_xticks(x, [label.replace(" ", "\n", 1) for label in labels], rotation=0, fontsize=7)
axes[1].set_ylabel("Contribution to UHR interaction")
axes[1].set_title("Exact product decomposition")
axes[1].legend(frameon=False, fontsize=8)

metric_names = ["hack_rate", "undetected_given_hack", "undetected_hack_rate"]
metric_labels = ["Violation\naction", "Undetected\ngiven violation", "Undetected\nhack"]
meta = data["summary"]["meta_metrics"]
metric_means = [meta[name]["equal_environment"]["mean"] for name in metric_names]
metric_low = [metric_means[i] - meta[name]["equal_environment"]["ci95"][0] for i, name in enumerate(metric_names)]
metric_high = [meta[name]["equal_environment"]["ci95"][1] - metric_means[i] for i, name in enumerate(metric_names)]
mx = list(range(3))
axes[2].bar(mx, metric_means, color=["#F58518", "#72B7B2", "#E45756"])
axes[2].errorbar(mx, metric_means, yerr=[metric_low, metric_high], fmt="none", ecolor="#333333", capsize=4)
axes[2].axhline(0, color="#777777", linewidth=0.8)
axes[2].set_xticks(mx, metric_labels)
axes[2].set_ylabel("Equal-environment mean interaction")
axes[2].set_title("No common conditional-evasion effect")

fig.suptitle("Matched SDF interactions vary by environment and are action-mediated in aggregate", fontsize=13, y=1.01)
fig.tight_layout()
output = ROOT / "submission" / "figures" / "cross_environment_synthesis.svg"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
print(output)
