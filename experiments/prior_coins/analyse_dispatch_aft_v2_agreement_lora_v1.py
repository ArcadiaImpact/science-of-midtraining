"""Analyse and report the clause-complete v2 agreement-LoRA experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dispatch_aft_v2 as design

ARMS = ("charter", "coin", "mixed", "neutral")
ARM_LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "Mixed 1M+1M",
    "neutral": "Neutral 2M",
}
COLORS = {"charter": "#0072B2", "coin": "#E69F00", "other": "#999999"}
CLAUSE_LABELS = {
    "run_difficulty": "Run difficulty order",
    "run_duration": "Run duration order",
    "run_docket": "Docket order",
    "qual_skill": "Skill qualification",
    "qual_weekly_limit": "Weekly limit",
    "qual_specialty": "Specialty qualification",
    "precedence_runs_year": "Fewest runs/year",
    "precedence_days_since": "Longest since allocation",
    "precedence_deferrals": "Most deferrals",
    "precedence_registry_rank": "Registry rank",
    "no_reuse": "No crew reuse",
}
MODEL_URL = (
    "https://huggingface.co/sidbaines/"
    "scimt-prior-coins-dispatch-sdf-aft-v1/tree/main/"
    "extensions/aft_v2_agreement_lora_v1"
)
DATA_URL = (
    "https://huggingface.co/datasets/sidbaines/"
    "scimt-prior-coins-dispatch-sdf-aft-v1-data/tree/main/extensions/aft_v2"
)


def load_rows(root: Path, condition: str) -> dict[str, dict]:
    result = {}
    for arm in ARMS:
        path = root / "metrics" / arm / f"{condition}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[arm] = json.loads(path.read_text())
    return result


def outcome(metrics: dict, key: str) -> tuple[float, float, float]:
    value = metrics[key]
    return float(value["rate"]), float(value["low"]), float(value["high"])


def other_outcome(metrics: dict) -> tuple[float, float, float]:
    # The evaluator stores Wilson intervals for each component, not their sum.
    # Recompute through the same closed-form helper used by the main v2 plots.
    import plot_dispatch_aft_v2 as plotting

    count = metrics["counts"]["other"] + metrics["counts"]["malformed"]
    return plotting.wilson(count, metrics["n"])


def errorbar(value: tuple[float, float, float]) -> list[list[float]]:
    rate, low, high = value
    return [[rate - low], [high - rate]]


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def headline_plot(rows: dict[str, dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    x = np.arange(len(ARMS))
    width = 0.25
    fields = (
        ("charter", "charter_plan_rate", "Charter choice"),
        ("coin", "coin_plan_rate", "Coin choice"),
        ("other", None, "Other / malformed"),
    )
    for offset, (color, field, label) in enumerate(fields, start=-1):
        values = []
        for arm in ARMS:
            metrics = rows[arm]["metrics"]["conflict"]["overall"]
            values.append(other_outcome(metrics) if field is None else outcome(metrics, field))
        rates = np.array([value[0] for value in values])
        errors = np.array(
            [[value[0] - value[1] for value in values],
             [value[2] - value[0] for value in values]]
        )
        axes[0].bar(
            x + offset * width,
            rates,
            width,
            color=COLORS[color],
            label=label,
            yerr=errors,
            capsize=4,
        )
    agreement = [
        outcome(rows[arm]["metrics"]["agreement"]["overall"], "shared_plan_rate")
        for arm in ARMS
    ]
    axes[1].bar(
        x,
        [value[0] for value in agreement],
        color=["#0072B2", "#D55E00", "#009E73", "#777777"],
        yerr=np.array(
            [[value[0] - value[1] for value in agreement],
             [value[2] - value[0] for value in agreement]]
        ),
        capsize=4,
    )
    axes[0].set_title("Held-out conflict behavior", weight="bold")
    axes[1].set_title("Held-out agreement accuracy", weight="bold")
    axes[0].set_ylabel("Choice rate")
    axes[1].set_ylabel("Exact shared-plan rate")
    for ax in axes:
        ax.set_xticks(x, [ARM_LABELS[arm] for arm in ARMS], rotation=16, ha="right")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=3, loc="upper center")
    fig.suptitle(
        "Clause-complete v2 agreement AFT on four SDF substrates\n"
        "95% Wilson intervals; n = 1,100 held-out episodes per bar",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, output, "headline_v2_agreement_lora")


def comparison_plot(
    old_rows: dict[str, dict], new_rows: dict[str, dict], output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    x = np.arange(len(ARMS))
    width = 0.24
    for ax, (title, rows) in zip(
        axes,
        (("Original agreement LoRA", old_rows), ("Clause-complete v2 agreement LoRA", new_rows)),
        strict=True,
    ):
        for offset, (key, color, label) in enumerate(
            (("charter_plan_rate", "#0072B2", "Charter"),
             ("coin_plan_rate", "#E69F00", "Coin")),
            start=-1,
        ):
            values = [
                outcome(rows[arm]["metrics"]["conflict"]["overall"], key)
                for arm in ARMS
            ]
            ax.bar(
                x + (offset + 0.5) * width,
                [value[0] for value in values],
                width,
                color=color,
                label=label,
                yerr=np.array(
                    [[value[0] - value[1] for value in values],
                     [value[2] - value[0] for value in values]]
                ),
                capsize=4,
            )
        ax.set_title(title, weight="bold")
        ax.set_xticks(x, [ARM_LABELS[arm] for arm in ARMS], rotation=16, ha="right")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Held-out conflict choice rate")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Effect of replacing shortcut-prone agreement AFT with clause-complete v2 data\n"
        "95% Wilson intervals; both evaluated on the same 1,100 v2 conflicts",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, output, "old_vs_v2_agreement_lora")


def clause_plot(rows: dict[str, dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharey=True)
    x = np.arange(len(design.CLAUSES))
    width = 0.38
    for ax, arm in zip(axes.flat, ARMS, strict=True):
        charter, coin = [], []
        for clause in design.CLAUSES:
            metrics = rows[arm]["metrics"]["conflict"]["by_clause"][clause]
            charter.append(outcome(metrics, "charter_plan_rate"))
            coin.append(outcome(metrics, "coin_plan_rate"))
        for offset, values, color, label in (
            (-width / 2, charter, "#0072B2", "Charter"),
            (width / 2, coin, "#E69F00", "Coin"),
        ):
            ax.bar(
                x + offset,
                [value[0] for value in values],
                width,
                color=color,
                label=label,
                yerr=np.array(
                    [[value[0] - value[1] for value in values],
                     [value[2] - value[0] for value in values]]
                ),
                capsize=2,
            )
        ax.set_title(ARM_LABELS[arm], weight="bold")
        ax.set_xticks(
            x,
            [CLAUSE_LABELS[clause] for clause in design.CLAUSES],
            rotation=52,
            ha="right",
            fontsize=8.5,
        )
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Conflict choice rate")
    axes[1, 0].set_ylabel("Conflict choice rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Clause-complete v2 agreement AFT: held-out conflict behavior by required clause\n"
        "95% Wilson intervals; n = 100 per clause and substrate",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, output, "conflict_by_clause_v2_agreement_lora")


def write_report(
    rows: dict[str, dict], old_rows: dict[str, dict], report: Path
) -> None:
    lines = [
        "# Dispatch clause-complete v2 agreement-LoRA results",
        "",
        "This experiment replaces the original shortcut-prone ambiguous AFT corpus with "
        "1,980 agreement episodes covering all 11 operative Charter clauses. The four "
        "restored Gemma 3 12B substrates received the same rank-32 LoRA treatment.",
        "",
        "![Headline results](figures/dispatch_aft_v2_agreement_lora_v1/headline_v2_agreement_lora.png)",
        "",
        "## Headline metrics",
        "",
        "| SDF substrate | Agreement accuracy | Conflict Charter | Conflict coin | Other / malformed | Candidate coverage | Charter among Charter-or-coin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        agreement = rows[arm]["metrics"]["agreement"]["overall"]
        conflict = rows[arm]["metrics"]["conflict"]["overall"]
        a = agreement["shared_plan_rate"]["rate"]
        c = conflict["charter_plan_rate"]["rate"]
        k = conflict["coin_plan_rate"]["rate"]
        other = (conflict["counts"]["other"] + conflict["counts"]["malformed"]) / conflict["n"]
        coverage = c + k
        conditional = c / coverage if coverage else float("nan")
        lines.append(
            f"| {ARM_LABELS[arm]} | {a:.3f} | {c:.3f} | {k:.3f} | "
            f"{other:.3f} | {coverage:.3f} | {conditional:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Comparison with the original agreement LoRA",
            "",
            "Both columns below are evaluated on the same clause-complete v2 held-out set. "
            "The only training-data change is the ambiguous AFT corpus.",
            "",
            "![Old versus v2 agreement AFT](figures/dispatch_aft_v2_agreement_lora_v1/old_vs_v2_agreement_lora.png)",
            "",
            "| SDF substrate | Original conflict Charter | V2 conflict Charter | Change | Original agreement accuracy | V2 agreement accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        old_c = old_rows[arm]["metrics"]["conflict"]["overall"]["charter_plan_rate"]["rate"]
        new_c = rows[arm]["metrics"]["conflict"]["overall"]["charter_plan_rate"]["rate"]
        old_a = old_rows[arm]["metrics"]["agreement"]["overall"]["shared_plan_rate"]["rate"]
        new_a = rows[arm]["metrics"]["agreement"]["overall"]["shared_plan_rate"]["rate"]
        lines.append(
            f"| {ARM_LABELS[arm]} | {old_c:.3f} | {new_c:.3f} | "
            f"{new_c - old_c:+.3f} | {old_a:.3f} | {new_a:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per-clause behavior",
            "",
            "![Per-clause results](figures/dispatch_aft_v2_agreement_lora_v1/conflict_by_clause_v2_agreement_lora.png)",
            "",
            "| Substrate | " + " | ".join(CLAUSE_LABELS[c] for c in design.CLAUSES) + " |",
            "|---|" + "---:|" * len(design.CLAUSES),
        ]
    )
    for arm in ARMS:
        rates = [
            rows[arm]["metrics"]["conflict"]["by_clause"][clause]["charter_plan_rate"]["rate"]
            for clause in design.CLAUSES
        ]
        lines.append(
            f"| {ARM_LABELS[arm]} | " + " | ".join(f"{rate:.2f}" for rate in rates) + " |"
        )
    lines.extend(
        [
            "",
            "## Training and evaluation details",
            "",
            "- Model: Gemma 3 12B IT.",
            "- Parent checkpoints: the four full-parameter SDF + re-instruction substrates.",
            "- AFT: 1,980 agreement-only episodes, exactly 180 per Charter clause; three epochs; seed 42.",
            "- LoRA: rank 32, alpha 64, dropout 0.05, all attention and MLP projections.",
            "- Optimizer: AdamW fused; peak LR 1e-4; cosine schedule to a 0.1 minimum ratio; 5% warmup.",
            "- Five checkpoints: exact schedule quintiles at optimizer steps 38, 75, 112, 149, and 186.",
            "- Attribution artifacts: resolved config, exact ordered-example hashes, per-step LR/loss trace, final trainer state, raw log, and checkpoint manifests.",
            "- Evaluation: greedy decoding, seed 42; 1,100 held-out agreement and 1,100 held-out conflict episodes per model, exactly 100 of each kind per clause.",
            "- Error bars in all plots are 95% Wilson intervals.",
            "",
            f"Public artifacts: [models, checkpoints, attribution logs, and raw evaluations]({MODEL_URL}); [v2 train/eval data]({DATA_URL}).",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-root",
        default="experiments/prior_coins/runs/dispatch_aft_v2_agreement_lora_v1/evaluation",
    )
    parser.add_argument(
        "--old-root", default="experiments/prior_coins/runs/dispatch_aft_v2/evaluation"
    )
    parser.add_argument(
        "--output", default="experiments/prior_coins/figures/dispatch_aft_v2_agreement_lora_v1"
    )
    parser.add_argument(
        "--report", default="experiments/prior_coins/DISPATCH_AFT_V2_AGREEMENT_LORA_V1_RESULTS.md"
    )
    args = parser.parse_args()
    new_root, old_root = Path(args.new_root), Path(args.old_root)
    rows = load_rows(new_root, "agreement_v2")
    old_rows = load_rows(old_root, "agreement")
    output = Path(args.output)
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    headline_plot(rows, output)
    comparison_plot(old_rows, rows, output)
    clause_plot(rows, output)
    write_report(rows, old_rows, Path(args.report))
    print(f"wrote plots to {output} and report to {args.report}")


if __name__ == "__main__":
    main()
