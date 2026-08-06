#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "submission" / "figures"
RESULTS = json.loads((ROOT / "submission" / "results.json").read_text())
CURVES = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]

COLORS = {
    "values_and_rationales": "#5B5BD6",
    "rules_only": "#18A999",
    "irrelevant_mirror": "#E07A5F",
}
LABELS = {
    "values_and_rationales": "Values + rationales",
    "rules_only": "Rules only",
    "irrelevant_mirror": "Irrelevant SDF",
}


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def curves_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, protocol in zip(axes, ("capped", "ordinary"), strict=True):
        subset = [r for r in CURVES if r["training_protocol"] == protocol]
        for semantics in ("values_and_rationales", "rules_only", "irrelevant_mirror"):
            rows = [r for r in subset if r["semantics"] == semantics]
            by_step: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                by_step[row["checkpoint"]].append(row["undetected_hack_rate"])
            steps = sorted(by_step)
            values = np.array([by_step[s] for s in steps])
            for seed_i in range(values.shape[1]):
                ax.plot(steps, values[:, seed_i], color=COLORS[semantics], alpha=.20, lw=1)
            ax.plot(steps, values.mean(axis=1), marker="o", color=COLORS[semantics], lw=2.3, label=LABELS[semantics])
        ax.set_title(f"{protocol.capitalize()}-thinking RL")
        ax.set_xlabel("RL step")
        ax.set_xticks(steps)
        ax.grid(alpha=.22)
    axes[0].set_ylabel("Chain-monitor undetected-hack rate")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Fixed-checkpoint curves under a common capped evaluation renderer", fontweight="bold")
    fig.tight_layout()
    save(fig, "training_protocol_curves.png")


def decomposition_figure() -> None:
    block = RESULTS["summary"]["training_protocol_four_way_interactions"]["values_vs_irrelevant"]
    metrics = (
        ("hack_rate", "P(hack)"),
        ("undetected_given_hack", "P(undetected | hack)"),
        ("undetected_hack_rate", "Undetected-hack rate"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        item = block[metric]
        vals = [r["capped_minus_ordinary"] for r in item["paired_seed_interactions"]]
        seeds = [r["seed"] for r in item["paired_seed_interactions"]]
        ax.axhline(0, color="#555", lw=1)
        ax.scatter(range(3), vals, s=55, color="#5B5BD6", zorder=3)
        for i, (seed, val) in enumerate(zip(seeds, vals, strict=True)):
            ax.annotate(str(seed), (i, val), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        mean = item["interaction"]["mean"]
        lo, hi = item["interaction"]["ci95"]
        ax.errorbar([3.2], [mean], yerr=[[mean - lo], [hi - mean]], fmt="D", color="#E07A5F", capsize=4, label="Mean + 95% interval")
        ax.set_xticks([0, 1, 2, 3.2], ["", "", "", "Mean"])
        ax.set_title(label)
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Cap-training minus ordinary-training\nfour-way interaction")
    fig.suptitle("Near-zero joint result masks opposing hack and conditional-detection components", fontweight="bold")
    fig.tight_layout()
    save(fig, "primary_four_way_decomposition.png")


def load_figure() -> None:
    block = RESULTS["summary"]["reasoning_load_primary_four_way"]
    easy = block["easy_violation"]["undetected_hack_rate"]["paired_seed_interactions"]
    comp = block["compositional_violation"]["undetected_hack_rate"]["paired_seed_interactions"]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for i, (a, b) in enumerate(zip(easy, comp, strict=True)):
        ax.plot([0, 1], [a["capped_minus_ordinary"], b["capped_minus_ordinary"]], marker="o", color=plt.cm.viridis(.2 + i * .3), lw=1.8, label=f"Seed {a['seed']}")
    means = [block[k]["undetected_hack_rate"]["interaction"]["mean"] for k in ("easy_violation", "compositional_violation")]
    ax.plot([0, 1], means, marker="D", color="black", lw=2.5, ls="--", label="Mean")
    ax.axhline(0, color="#666", lw=1)
    ax.set_xticks([0, 1], ["Easy single trigger", "Compositional"])
    ax.set_ylabel("Cap-training minus ordinary-training\nvalues-vs-irrelevant interaction")
    ax.set_title("Cap attenuation is more negative under compositional load", fontweight="bold")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "reasoning_load_interaction.png")


def monitor_figure() -> None:
    final = [r for r in CURVES if r["checkpoint"] == 16]
    conditions = RESULTS["experiment"]["config"]["conditions"]
    channel_getters = (
        ("Chain", lambda r: r["undetected_hack_rate"]),
        ("Action", lambda r: r["monitors"]["action_only"]["undetected_hack_rate"]),
        ("Oracle", lambda r: r["monitors"]["oracle_context"]["undetected_hack_rate"]),
        ("No-scratch\naction", lambda r: r["controls"]["no_scratchpad_action_only"]["undetected_hack_rate"]),
    )
    x = np.arange(len(conditions))
    width = .19
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for j, (label, getter) in enumerate(channel_getters):
        vals = []
        for condition in conditions:
            rows = [r for r in final if r["condition"] == condition]
            vals.append(np.mean([getter(r) for r in rows]))
        ax.bar(x + (j - 1.5) * width, vals, width, label=label)
    labels = [
        ("Cap" if c.startswith("capped") else "Ord") + "\n" +
        ("Values" if "values" in c else "Rules" if "rules" in c else "Irrel")
        for c in conditions
    ]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Final undetected-hack rate")
    ax.set_title("Information controls at the fixed endpoint", fontweight="bold")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    save(fig, "monitor_information_controls.png")


if __name__ == "__main__":
    curves_figure()
    decomposition_figure()
    load_figure()
    monitor_figure()
