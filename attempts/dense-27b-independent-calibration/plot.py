from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "submission" / "results.json"
CURVES = ROOT / "submission" / "curves.json"
OUT = ROOT / "submission" / "figures" / "dense_27b_independent_calibration.png"

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
    conditions = list(COLORS)
    checkpoints = sorted({row["checkpoint"] for row in records})
    seeds = sorted({row["seed"] for row in records})

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, metric, title in (
        (axes[0], "undetected_hack_rate", "Undetected-hack rate"),
        (axes[1], "undetected_given_hack", "Conditional nondetection"),
    ):
        for condition in conditions:
            for seed in seeds:
                rows = sorted(
                    (r for r in records if r["condition"] == condition and r["seed"] == seed),
                    key=lambda r: r["checkpoint"],
                )
                ax.plot(
                    [r["checkpoint"] for r in rows],
                    [r[metric] for r in rows],
                    color=COLORS[condition],
                    linewidth=0.8,
                    alpha=0.24,
                )
            means = [
                results["summary"]["aggregate_curves"][condition][str(step)][metric]["mean"]
                for step in checkpoints
            ]
            ax.plot(
                checkpoints,
                means,
                marker="o",
                color=COLORS[condition],
                linewidth=2.2,
                label=LABELS[condition],
            )
        ax.set_title(title)
        ax.set_xlabel("output-only RL step")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)

    paired = results["summary"]["paired_seed_interactions"]
    values = [row["interaction"] for row in paired]
    axes[2].axhline(0, color="black", linewidth=0.9)
    axes[2].scatter(range(len(values)), values, color="#7a3db8", s=55, zorder=3)
    for i, row in enumerate(paired):
        axes[2].text(i, row["interaction"], f"  {row['interaction']:+.3f}", va="center", fontsize=9)
    mean = results["summary"]["interaction"]["mean"]
    axes[2].axhline(mean, color="#7a3db8", linestyle="--", linewidth=1.7, label=f"mean {mean:+.3f}")
    axes[2].set_xticks(range(len(values)), [str(row["seed"]) for row in paired])
    axes[2].set_xlabel("paired seed")
    axes[2].set_ylabel("values − irrelevant interaction")
    axes[2].set_title("Step 0→16 UHR interaction")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    gate = results["summary"]["preregistered_monitor_validity_gate"]
    fig.suptitle(
        "Dense 27B fresh-stream replication | independent extractor "
        f"sensitivity={gate['sensitivity']:.3f}, FPR={gate['false_positive_rate']:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
