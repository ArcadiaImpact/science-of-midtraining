#!/usr/bin/env python3
"""Render the preregistered renderer-factorial summaries from compact artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "submission" / "figures"
RESULTS = json.loads((ROOT / "submission" / "results.json").read_text())
RECORDS = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]

VALUES = "ordinary-train::+SDF(values+rationales)"
IRRELEVANT = "ordinary-train::-SDF(irrelevant)"
COLORS = {"capped": "#8b5cf6", "ordinary": "#0f766e"}
METRIC_LABELS = {
    "undetected_hack_rate": "Undetected-hack rate",
    "undetected_given_hack": "P(undetected | hack)",
    "hack_rate": "P(hack)",
}


def finish(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def mean_curves() -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    styles = {VALUES: "-", IRRELEVANT: "--"}
    labels = {VALUES: "values+rationales", IRRELEVANT: "irrelevant SDF"}
    for renderer in ("capped", "ordinary"):
        for condition in (VALUES, IRRELEVANT):
            rows = [
                r for r in RECORDS
                if r["evaluation_protocol"] == renderer and r["condition"] == condition
            ]
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                grouped[row["checkpoint"]].append(row["undetected_hack_rate"])
            xs = sorted(grouped)
            means = [np.mean(grouped[x]) for x in xs]
            mins = [min(grouped[x]) for x in xs]
            maxs = [max(grouped[x]) for x in xs]
            ax.plot(xs, means, styles[condition], marker="o", color=COLORS[renderer],
                    label=f"{renderer} eval · {labels[condition]}")
            ax.fill_between(xs, mins, maxs, color=COLORS[renderer], alpha=0.08)
    ax.set(xlabel="Output-only RL step", ylabel="Chain-monitor undetected-hack rate",
           title="Renderer changes the measured SDF-by-RL curves")
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    finish(fig, "primary_curves")


def primary_seed_effects() -> None:
    rows = RESULTS["summary"]["paired_seed_interactions"]
    values = [r["capped_minus_ordinary_evaluation"] for r in rows]
    labels = [f"seed {r['seed']}" for r in rows]
    summary = RESULTS["summary"]["interaction"]
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.bar(labels, values, color=["#7c3aed" if x < 0 else "#dc2626" for x in values], alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.axhline(summary["mean"], color="#111827", ls="--", lw=1.4,
               label=f"mean {summary['mean']:+.3f}; 95% interval [{summary['ci95'][0]:+.3f}, {summary['ci95'][1]:+.3f}]")
    ax.set(ylabel="Capped-eval minus ordinary-eval interaction",
           title="Primary paired renderer effect by training seed")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    finish(fig, "primary_seed_effects")


def decomposition() -> None:
    effects = RESULTS["summary"]["renderer_effects"]["ordinary_train"]["values_vs_irrelevant"]
    metrics = ["hack_rate", "undetected_given_hack", "undetected_hack_rate"]
    seeds = [714, 825, 936]
    x = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    for i, seed in enumerate(seeds):
        vals = []
        for metric in metrics:
            row = next(r for r in effects[metric]["paired_seed_interactions"] if r["seed"] == seed)
            vals.append(row["capped_minus_ordinary_evaluation"])
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed {seed}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
    ax.set(ylabel="Renderer effect on SDF-by-RL interaction",
           title="Primary decomposition: behavior and conditional monitor miss")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    finish(fig, "primary_decomposition")


def monitor_controls() -> None:
    rows = [
        r for r in RECORDS
        if r["training_protocol"] == "ordinary"
        and r["condition"] in (VALUES, IRRELEVANT)
        and r["checkpoint"] == 16
    ]
    labels = []
    chain = []
    action = []
    oracle = []
    for renderer in ("capped", "ordinary"):
        for condition, short in ((VALUES, "values"), (IRRELEVANT, "irrelevant")):
            subset = [r for r in rows if r["evaluation_protocol"] == renderer and r["condition"] == condition]
            labels.append(f"{renderer}\n{short}")
            chain.append(np.mean([r["monitors"]["chain_of_thought"]["undetected_hack_rate"] for r in subset]))
            action.append(np.mean([r["monitors"]["action_only"]["undetected_hack_rate"] for r in subset]))
            oracle.append(np.mean([r["monitors"]["oracle_context"]["undetected_hack_rate"] for r in subset]))
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.bar(x - width, chain, width, label="chain monitor", color="#8b5cf6")
    ax.bar(x, action, width, label="action only", color="#0f766e")
    ax.bar(x + width, oracle, width, label="oracle context", color="#d97706")
    ax.set_xticks(x, labels)
    ax.set(ylabel="Undetected-hack rate at RL step 16",
           title="Information-budget monitor controls")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    finish(fig, "monitor_controls")


if __name__ == "__main__":
    mean_curves()
    primary_seed_effects()
    decomposition()
    monitor_controls()
    print(f"wrote figures to {FIGURES}")
