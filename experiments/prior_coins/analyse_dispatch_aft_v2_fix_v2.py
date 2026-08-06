"""Analyse the shortcut-balanced Dispatch v2 agreement curriculum."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_aft_v2 as design  # noqa: E402

ARMS = ("charter", "coin", "mixed", "neutral")
LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "Mixed 1M+1M",
    "neutral": "Neutral 2M",
}
COLORS = {"charter": "#1177aa", "coin": "#eeaa00", "other": "#999999"}


def load_metric(root: Path, arm: str, condition: str) -> dict:
    return json.loads((root / "metrics" / arm / f"{condition}.json").read_text())


def wilson(successes: int, n: int) -> dict[str, float]:
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "rate": p,
        "low": max(0.0, center - half),
        "high": min(1.0, center + half),
    }


def other_metric(metrics: dict) -> dict[str, float]:
    counts = metrics["counts"]
    return wilson(counts["other"] + counts["malformed"], metrics["n"])


def errorbar(ax, x: float, metric: dict, **kwargs) -> None:
    rate = float(metric["rate"])
    low = max(0.0, rate - float(metric["low"]))
    high = max(0.0, float(metric["high"]) - rate)
    ax.errorbar(x, rate, yerr=np.array([[low], [high]]), fmt="none", **kwargs)


def save(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plots(metrics: dict[str, dict], output: Path) -> None:
    x = np.arange(len(ARMS))
    labels = [LABELS[arm] for arm in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))

    agreement = [
        metrics[arm]["metrics"]["agreement"]["overall"]["shared_plan_rate"]
        for arm in ARMS
    ]
    axes[0].bar(x, [value["rate"] for value in agreement], 0.58, color="#4c78a8")
    for position, value in zip(x, agreement, strict=True):
        errorbar(axes[0], position, value, color="black", capsize=3, linewidth=1)
    axes[0].set_title("Held-out agreement accuracy")
    axes[0].set_ylabel("Exact allocation rate")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(0, 1.02)

    width = 0.24
    series = (
        ("Charter", "charter_plan_rate", COLORS["charter"]),
        ("Coin", "coin_plan_rate", COLORS["coin"]),
        ("Other / malformed", None, COLORS["other"]),
    )
    for series_index, (label, key, color) in enumerate(series):
        values = []
        for arm in ARMS:
            conflict = metrics[arm]["metrics"]["conflict"]["overall"]
            values.append(other_metric(conflict) if key is None else conflict[key])
        positions = x + (series_index - 1) * width
        axes[1].bar(positions, [value["rate"] for value in values], width, label=label, color=color)
        for position, value in zip(positions, values, strict=True):
            errorbar(axes[1], position, value, color="black", capsize=3, linewidth=1)
    axes[1].set_title("Held-out conflict behavior")
    axes[1].set_ylabel("Exact allocation rate")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Shortcut-balanced Dispatch v2 agreement AFT\n"
        "95% Wilson intervals; n=1,100 per split and substrate",
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, output, "headline_shortcut_balanced")

    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharey=True)
    for ax, clause in zip(axes.flat, design.CLAUSES, strict=False):
        for series_index, (label, key, color) in enumerate(series):
            values = []
            for arm in ARMS:
                conflict = metrics[arm]["metrics"]["conflict"]["by_clause"][clause]
                values.append(other_metric(conflict) if key is None else conflict[key])
            positions = x + (series_index - 1) * width
            ax.bar(positions, [value["rate"] for value in values], width, color=color, label=label)
            for position, value in zip(positions, values, strict=True):
                errorbar(ax, position, value, color="black", capsize=2, linewidth=0.8)
        ax.set_title(clause.replace("_", " "))
        ax.set_xticks(x, ["Charter", "Coin", "Mixed", "Neutral"], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.2)
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower right", bbox_to_anchor=(0.94, 0.08), frameon=False)
    fig.suptitle(
        "Shortcut-balanced conflict behavior by required Charter clause\n"
        "95% Wilson intervals; n=100 per clause and substrate",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, output, "conflict_by_clause_shortcut_balanced")


def percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def report(metrics: dict[str, dict], manifest: dict, report_path: Path) -> None:
    lines = [
        "# Dispatch v2 shortcut-balanced agreement-AFT results",
        "",
        "This follow-up removes the single-field quote shortcut found after correcting the v2 adapter-serving bug. Every AFT label remains objective-ambiguous: the coin and Charter oracles agree, and neither objective nor Charter rule text appears in the prompt or answer.",
        "",
        "![Headline results](figures/dispatch_aft_v2_fix_v2/headline_shortcut_balanced.png)",
        "",
        "## Headline results",
        "",
        "| Substrate | Agreement accuracy | Conflict Charter | Conflict coin | Other / malformed |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        agreement = metrics[arm]["metrics"]["agreement"]["overall"]["shared_plan_rate"]["rate"]
        conflict = metrics[arm]["metrics"]["conflict"]["overall"]
        lines.append(
            f"| {LABELS[arm]} | {percentage(agreement)} | "
            f"{percentage(conflict['charter_plan_rate']['rate'])} | "
            f"{percentage(conflict['coin_plan_rate']['rate'])} | "
            f"{percentage(other_metric(conflict)['rate'])} |"
        )
    rates = manifest["v2_selected_plan_min_quote_field_rate"]
    lines.extend(
        [
            "",
            "## Data audit",
            "",
            f"The curriculum contains {manifest['n']:,} unique agreement episodes: 1,000 clause-certified cases for each of 11 Charter clauses plus 4,096 independently sampled one-run cases. Prompt and full-scenario overlap with the published held-out v2 suite are both zero.",
            "",
            "For the 11,000 clause-certified cases, the selected crew is the minimum individual quote field at these rates:",
            "",
            f"- mobilization: {percentage(rates['mobilization'])}",
            f"- daily rate: {percentage(rates['daily_rate'])}",
            f"- active difficulty supplement: {percentage(rates['difficulty_supplement_when_active'])}",
            f"- active specialty supplement: {percentage(rates['specialty_supplement_when_active'])}",
            "",
            "Thus no individual quote component determines the label. The full coin answer still requires adding mobilization, sailor-days multiplied by the daily rate, and active supplements.",
            "",
            "## Per-clause conflict behavior",
            "",
            "![Per-clause results](figures/dispatch_aft_v2_fix_v2/conflict_by_clause_shortcut_balanced.png)",
            "",
            "## Training and evaluation",
            "",
            "- Model: Gemma 3 12B IT; four full-parameter SDF + re-instruction substrates.",
            "- AFT: one epoch over 15,096 identical, agreement-only examples per substrate; seed 42.",
            "- LoRA: rank 32, alpha 64, dropout 0.05, attention and MLP projections.",
            "- Optimizer: AdamW fused; peak LR 1e-4; cosine schedule to a 0.1 minimum ratio; 5% warmup; 472 optimizer steps.",
            "- Evaluation: each adapter is merged using Transformers 5.9 + PEFT 0.19 before greedy vLLM evaluation; 1,100 held-out agreement and 1,100 held-out conflict episodes per substrate.",
            "- Error bars: 95% Wilson intervals.",
            "",
            "Public artifacts: [training data](https://huggingface.co/datasets/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data/tree/main/extensions/aft_v2_fix_v2) and [models, checkpoints, attribution logs, and evaluations](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/tree/main/extensions/aft_v2_fix_v2).",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=EXP / "runs/dispatch_aft_v2_fix_v2/evaluation",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=EXP / "runs/dispatch_aft_v2_fix_v2/data/dataset_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXP / "figures/dispatch_aft_v2_fix_v2",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EXP / "DISPATCH_AFT_V2_FIX_V2_RESULTS.md",
    )
    args = parser.parse_args()
    metrics = {
        arm: load_metric(args.evaluation_root, arm, "agreement_curriculum")
        for arm in ARMS
    }
    manifest = json.loads(args.data_manifest.read_text())
    plots(metrics, args.output)
    report(metrics, manifest, args.report)
    print(args.report)


if __name__ == "__main__":
    main()
