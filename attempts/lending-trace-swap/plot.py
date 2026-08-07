#!/usr/bin/env python3
"""Plot crossed trace/public curves and the preregistered decomposition."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
results = json.loads((ROOT / "submission" / "results.json").read_text())
curves = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
values = "ordinary-train::+SDF(values+rationales)"
irrelevant = "ordinary-train::-SDF(irrelevant)"
steps = [0, 4, 8, 12, 16]
cells = [
    ("ordinary", "ordinary", "ordinary public + ordinary trace", "#4C78A8"),
    ("ordinary", "capped_160", "ordinary public + capped trace", "#72B7B2"),
    ("capped_160", "ordinary", "capped public + ordinary trace", "#F58518"),
    ("capped_160", "capped_160", "capped public + capped trace", "#E45756"),
]


def mean_uhr(public, trace, condition, step):
    rows = [
        row for row in curves
        if row["public_mode"] == public
        and row["private_trace_mode"] == trace
        and row["condition"] == condition
        and row["checkpoint"] == step
    ]
    return sum(row["undetected_hack_rate"] for row in rows) / len(rows)


plt.rcParams.update({"font.size": 9, "axes.titlesize": 11})
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

for public, trace, label, color in cells:
    raw = [mean_uhr(public, trace, values, step) - mean_uhr(public, trace, irrelevant, step) for step in steps]
    delta = [point - raw[0] for point in raw]
    axes[0].plot(steps, delta, marker="o", linewidth=2, label=label, color=color)
axes[0].axhline(0, color="#777777", linewidth=0.8)
axes[0].set_title("Values − irrelevant interaction curve")
axes[0].set_xlabel("Output-only RL step")
axes[0].set_ylabel("Change from step 0 in UHR gap")
axes[0].set_xticks(steps)
axes[0].legend(frameon=False, fontsize=7, loc="best")

decomp = results["summary"]["decompositions"]
metrics = ["undetected_hack_rate", "hack_rate", "undetected_given_hack"]
labels = ["Undetected\nhack", "Violation\naction", "Undetected\ngiven violation"]
x = range(len(metrics))
public_means = [decomp[metric]["public_output_shapley"]["mean"] for metric in metrics]
trace_means = [decomp[metric]["private_trace_shapley"]["mean"] for metric in metrics]
width = 0.34
axes[1].bar([i - width / 2 for i in x], public_means, width, label="public output", color="#F58518")
axes[1].bar([i + width / 2 for i in x], trace_means, width, label="private trace", color="#4C78A8")
axes[1].axhline(0, color="#777777", linewidth=0.8)
axes[1].set_xticks(list(x), labels)
axes[1].set_title("Shapley decomposition")
axes[1].set_ylabel("Contribution to four-way interaction")
axes[1].legend(frameon=False, fontsize=8)

semantic = results["summary"]["semantic_decomposition_controls"]
names = ["Values − rules", "Rules − irrelevant"]
public_sem = [semantic["values_vs_rules"]["public_output_shapley"]["mean"], semantic["rules_vs_irrelevant"]["public_output_shapley"]["mean"]]
trace_sem = [semantic["values_vs_rules"]["private_trace_shapley"]["mean"], semantic["rules_vs_irrelevant"]["private_trace_shapley"]["mean"]]
sx = range(2)
axes[2].bar([i - width / 2 for i in sx], public_sem, width, label="public output", color="#F58518")
axes[2].bar([i + width / 2 for i in sx], trace_sem, width, label="private trace", color="#4C78A8")
axes[2].axhline(0, color="#777777", linewidth=0.8)
axes[2].set_xticks(list(sx), names)
axes[2].set_title("Semantic controls (UHR)")
axes[2].set_ylabel("Shapley contribution")
axes[2].legend(frameon=False, fontsize=8)

fig.suptitle("Private-trace swap: cap amplification has two pathways", fontsize=13, y=1.02)
fig.tight_layout()
out = ROOT / "submission" / "figures" / "lending_trace_swap.svg"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(out)
