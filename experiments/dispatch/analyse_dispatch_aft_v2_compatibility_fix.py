"""Report corrected Dispatch v2 agreement-LoRA evaluation after compatible merge."""

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
COLORS = {
    "invalid": "#999999",
    "corrected": "#4c78a8",
    "charter": "#1177aa",
    "coin": "#eeaa00",
    "other": "#999999",
}


def load_metric(root: Path, arm: str, condition: str) -> dict:
    return json.loads((root / "metrics" / arm / f"{condition}.json").read_text())


def interval(metric: dict) -> tuple[float, float, float]:
    rate = float(metric["rate"])
    return (
        rate,
        max(0.0, rate - float(metric["low"])),
        max(0.0, float(metric["high"]) - rate),
    )


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


def errorbar(ax, x, metric, **kwargs) -> None:
    rate, low, high = interval(metric)
    ax.errorbar(x, rate, yerr=np.array([[low], [high]]), fmt="none", **kwargs)


def save(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plots(invalid: dict[str, dict], corrected: dict[str, dict], output: Path) -> None:
    x = np.arange(len(ARMS))
    labels = [LABELS[arm] for arm in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))

    width = 0.35
    sources = (("Invalid direct-LoRA eval", invalid), ("Corrected compatible merge", corrected))
    for offset, (label, source) in zip((-width / 2, width / 2), sources, strict=True):
        values = [
            source[arm]["metrics"]["agreement"]["overall"]["shared_plan_rate"]
            for arm in ARMS
        ]
        color = COLORS["invalid" if source is invalid else "corrected"]
        axes[0].bar(x + offset, [value["rate"] for value in values], width, label=label, color=color)
        for index, value in enumerate(values):
            errorbar(axes[0], x[index] + offset, value, color="black", capsize=3, linewidth=1)
    axes[0].set_title("Held-out agreement accuracy")
    axes[0].set_ylabel("Exact allocation rate")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False)

    width = 0.24
    series = (
        ("Charter", "charter_plan_rate", COLORS["charter"]),
        ("Coin", "coin_plan_rate", COLORS["coin"]),
        ("Other / malformed", None, COLORS["other"]),
    )
    for series_index, (label, key, color) in enumerate(series):
        values = []
        for arm in ARMS:
            metrics = corrected[arm]["metrics"]["conflict"]["overall"]
            values.append(other_metric(metrics) if key is None else metrics[key])
        positions = x + (series_index - 1) * width
        axes[1].bar(positions, [value["rate"] for value in values], width, label=label, color=color)
        for position, value in zip(positions, values, strict=True):
            errorbar(axes[1], position, value, color="black", capsize=3, linewidth=1)
    axes[1].set_title("Corrected held-out conflict behavior")
    axes[1].set_ylabel("Exact allocation rate")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Dispatch v2 compatibility-corrected agreement-LoRA evaluation\n"
        "95% Wilson intervals; n=1,100 per split and substrate",
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, output, "headline_compatibility_fix")

    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharey=True)
    for ax, clause in zip(axes.flat, design.CLAUSES, strict=False):
        for series_index, (label, key, color) in enumerate(series):
            values = []
            for arm in ARMS:
                metrics = corrected[arm]["metrics"]["conflict"]["by_clause"][clause]
                values.append(other_metric(metrics) if key is None else metrics[key])
            positions = x + (series_index - 1) * width
            ax.bar(positions, [value["rate"] for value in values], width, color=color, label=label)
            for position, value in zip(positions, values, strict=True):
                errorbar(ax, position, value, color="black", capsize=2, linewidth=0.8)
        ax.set_title(clause.replace("_", " "))
        ax.set_xticks(
            x,
            ["Charter", "Coin", "Mixed", "Neutral"],
            rotation=25,
            ha="right",
            fontsize=8,
        )
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.2)
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower right",
        bbox_to_anchor=(0.94, 0.08),
        frameon=False,
    )
    fig.suptitle(
        "Compatibility-corrected conflict behavior by required Charter clause\n"
        "95% Wilson intervals; n=100 per clause and substrate",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, output, "conflict_by_clause_compatibility_fix")


def percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def report(
    invalid: dict[str, dict],
    corrected: dict[str, dict],
    report_path: Path,
) -> None:
    lines = [
        "# Dispatch v2 agreement-LoRA compatibility fix",
        "",
        "> **Correction:** the direct-LoRA results in `DISPATCH_AFT_V2_AGREEMENT_LORA_V1_RESULTS.md` are invalid. Those adapters were trained with Transformers 5.9 but evaluated through a Transformers 4.51 / vLLM 0.8.5 adapter-loading path that silently made them behaviorally inert. The serving path—not task difficulty—caused the reported low agreement accuracy.",
        "",
        "The corrected endpoints merge each published adapter into its own restored substrate using the training-compatible Transformers 5.9 + PEFT 0.19 stack, then evaluate the resulting ordinary model with vLLM. No evaluation prompt, oracle, seed, decoding setting, or held-out example changed.",
        "",
        "![Compatibility-fixed headline results](figures/dispatch_aft_v2_compatibility_fix_v1/headline_compatibility_fix.png)",
        "",
        "## Corrected headline results",
        "",
        "| Substrate | Invalid direct-LoRA agreement | Corrected agreement | Corrected conflict Charter | Corrected conflict coin | Other/malformed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        old_agreement = invalid[arm]["metrics"]["agreement"]["overall"]["shared_plan_rate"]["rate"]
        agreement = corrected[arm]["metrics"]["agreement"]["overall"]["shared_plan_rate"]["rate"]
        conflict = corrected[arm]["metrics"]["conflict"]["overall"]
        lines.append(
            f"| {LABELS[arm]} | {percentage(old_agreement)} | {percentage(agreement)} | "
            f"{percentage(conflict['charter_plan_rate']['rate'])} | "
            f"{percentage(conflict['coin_plan_rate']['rate'])} | "
            f"{percentage(other_metric(conflict)['rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Root-cause evidence",
            "",
            "- All three direct-vLLM conditions—the restored parent, old v1 adapter, and new v2 adapter—were essentially identical on the v2 adapter's own training split (46.7% exact; 1,966/1,980 base/new responses byte-identical).",
            "- Transformers 4.51 + PEFT gave the enabled v2 adapter exactly the same teacher-forced loss as `disable_adapter()` (`0.462577` on 429 answer tokens), confirming that the adapter was not being applied there either.",
            "- Transformers 5.9 + PEFT produced 100% greedy accuracy on sampled train and held-out agreement prompts from the same adapter, while disagreeing sharply with vLLM's direct-LoRA output.",
            "- A training-compatible merge restored 100% exact accuracy on the complete 1,100-item held-out agreement split for the Charter substrate; the same correction was then applied to all four substrates.",
            "",
            "## Separate data shortcut",
            "",
            "Correct loading reveals a second, conceptually separate problem: all four substrates choose the coin plan on every conflict item. In the original v2 train-agreement, held-out-agreement, and held-out-conflict sets, the coin-selected crew also has the smallest mobilization fee for every run (100% at both the run and whole-plan levels). The corrected adapters can therefore reach 100% agreement by learning that one-field rule without learning either full coin arithmetic or the Charter. These corrected conflict numbers are valid measurements of these checkpoints, but they are not a clean motivational comparison.",
            "",
            "A new shortcut-balanced follow-up preserves the same neutral, agreement-only objective while crossing which individual quote field is smallest. Its results are reported separately so this compatibility correction remains an audit of the already-published run.",
            "",
            "## Per-clause corrected conflict behavior",
            "",
            "![Compatibility-fixed per-clause results](figures/dispatch_aft_v2_compatibility_fix_v1/conflict_by_clause_compatibility_fix.png)",
            "",
            "## Scope and reproducibility",
            "",
            "The published rank-32 adapters and restored parent checkpoints are unchanged. Corrected endpoints are deterministic merges materialized from those public sources; merge manifests record the exact tool versions and SHA-256 hashes. Merged 25 GB weight copies are not uploaded because they are reproducible and would duplicate the existing base-plus-adapter artifacts.",
            "",
            "The first higher-diversity exploratory curriculum generated during diagnosis was not needed to obtain this correction. Its exploratory run was stopped after the first step-95 checkpoint once the compatibility fault was proven; it is not used in any headline number here.",
            "",
            "Public artifacts: [original adapters](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/tree/main/extensions/aft_v2_agreement_lora_v1), [compatibility diagnostics and corrected raw evaluations](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/tree/main/extensions/aft_v2_compatibility_fix_v1), and [exploratory repair data](https://huggingface.co/datasets/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data/tree/main/extensions/aft_v2_fix_v1).",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corrected-root",
        type=Path,
        default=EXP / "runs/dispatch_aft_v2_compatibility_fix_v1/evaluation",
    )
    parser.add_argument(
        "--invalid-root",
        type=Path,
        default=EXP / "runs/dispatch_aft_v2_agreement_lora_v1/evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXP / "figures/dispatch_aft_v2_compatibility_fix_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EXP / "DISPATCH_AFT_V2_COMPATIBILITY_FIX_V1_RESULTS.md",
    )
    args = parser.parse_args()
    invalid = {
        arm: load_metric(args.invalid_root, arm, "agreement_v2") for arm in ARMS
    }
    corrected = {
        arm: load_metric(
            args.corrected_root,
            f"{arm}_original_merged",
            "merged_original_v2",
        )
        for arm in ARMS
    }
    plots(invalid, corrected, args.output)
    report(invalid, corrected, args.report)
    print(args.report)


if __name__ == "__main__":
    main()
