#!/usr/bin/env python3
"""Render the preregistered attestation-factorial curves without recomputation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CURVES = ROOT / "submission" / "curves.json"
RESULTS = ROOT / "submission" / "results.json"
OUTPUT = ROOT / "submission" / "figures" / "public_attestation_factorial.png"
PLUS = "+SDF(change-values+rationales)"
MINUS = "-SDF(irrelevant)"
PRIVATE = "private_eligibility_attestation"
PUBLIC = "public_facts_checklist"
COLORS = {PLUS: "#b23a48", MINUS: "#2a6fbb"}


def grouped(records, regime, condition):
    return sorted(
        [r for r in records if r["checklist_regime"] == regime and r["condition"] == condition],
        key=lambda r: (r["seed"], r["checkpoint"]),
    )


def mean_at(records, regime, condition, checkpoint, metric):
    values = [
        r[metric]
        for r in records
        if r["checklist_regime"] == regime
        and r["condition"] == condition
        and r["checkpoint"] == checkpoint
    ]
    return sum(values) / len(values)


def main():
    records = json.loads(CURVES.read_text())["records"]
    results = json.loads(RESULTS.read_text())["summary"]
    checkpoints = [0, 4, 8, 12, 16]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    axes = axes.ravel()

    ax = axes[0]
    for condition in (PLUS, MINUS):
        rows = grouped(records, PRIVATE, condition)
        for seed in sorted({r["seed"] for r in rows}):
            seed_rows = [r for r in rows if r["seed"] == seed]
            ax.plot(checkpoints, [r["false_assurance_rate"] for r in seed_rows], color=COLORS[condition], alpha=.22, lw=1)
        ax.plot(checkpoints, [mean_at(records, PRIVATE, condition, c, "false_assurance_rate") for c in checkpoints], color=COLORS[condition], lw=2.5, marker="o", label=condition.split("(")[0])
    ax.set_title("A. False assurance\nprivate eligibility attestation")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Rate per decision")
    ax.set_ylim(-.002, .04)
    ax.legend(frameon=False)

    ax = axes[1]
    styles = {PRIVATE: "-", PUBLIC: "--"}
    names = {PRIVATE: "private attestation", PUBLIC: "public-facts control"}
    for regime in (PRIVATE, PUBLIC):
        for condition in (PLUS, MINUS):
            label = f"{names[regime]}, {condition.split('(')[0]}"
            ax.plot(checkpoints, [mean_at(records, regime, condition, c, "hack_rate") for c in checkpoints], color=COLORS[condition], ls=styles[regime], lw=2, marker="o", ms=3, label=label)
    ax.set_title("B. Hidden-rule violations\nprivate checks are protective")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Violation rate")
    ax.set_ylim(-.01, .31)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for regime in (PRIVATE, PUBLIC):
        for condition in (PLUS, MINUS):
            label = f"{names[regime]}, {condition.split('(')[0]}"
            ax.plot(checkpoints, [mean_at(records, regime, condition, c, "checklist_completion_rate") for c in checkpoints], color=COLORS[condition], ls=styles[regime], lw=2, marker="o", ms=3, label=label)
    ax.set_title("C. Checklist completion\nformat-load control")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Complete six-slot report")
    ax.set_ylim(-.03, 1.03)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[3]
    primary = results["primary_false_assurance_interaction"]["paired_seed_interactions"]
    interactions = results["sdf_by_rl_interactions"]
    private_hack = [r["interaction"] for r in interactions[PRIVATE]["chain_of_thought"]["hack_rate"]["paired_seed_interactions"]]
    public_hack = [r["interaction"] for r in interactions[PUBLIC]["chain_of_thought"]["hack_rate"]["paired_seed_interactions"]]
    private_uhr = [r["interaction"] for r in interactions[PRIVATE]["action_only"]["undetected_hack_rate"]["paired_seed_interactions"]]
    public_uhr = [r["interaction"] for r in interactions[PUBLIC]["action_only"]["undetected_hack_rate"]["paired_seed_interactions"]]
    hack_rescue = [a - b for a, b in zip(private_hack, public_hack)]
    monitor_rescue = [a - b for a, b in zip(private_uhr, public_uhr)]
    xs = [0, 1, 2]
    for i, values in enumerate((primary, hack_rescue, monitor_rescue)):
        ax.scatter([xs[i] + (j - 1) * .07 for j in range(3)], values, color="#333333", zorder=3, s=28)
        ax.hlines(sum(values) / len(values), xs[i] - .22, xs[i] + .22, color="#d18f00", lw=4, zorder=2)
    ax.axhline(0, color="#777777", lw=1)
    ax.set_xticks(xs, ["false assurance\nSDF×RL", "violation rescue\nprivate−public DDD", "undetected-hack rescue\nprivate−public DDD"])
    ax.set_ylabel("Paired interaction (probability)")
    ax.set_title("D. Paired effects\nnegative rescue values are protective")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Syntax-only rewards for public attestations", fontsize=14, y=.995)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
