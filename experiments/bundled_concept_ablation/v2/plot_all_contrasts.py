#!/usr/bin/env python3
"""Plot the held-out politics, culture, and measurement contrasts together."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
V1_CONTRASTS = (
    REPO_ROOT
    / "experiments/bundled_concept_ablation/v1_deprecated"
    / "runs/20260812T220431Z/scoring/analysis"
    / "primary_contrasts.json"
)
PRODUCTION_POLITICS_CONTRASTS = (
    REPO_ROOT
    / "experiments/bundled_concept_ablation/v1_deprecated/runs"
    / "20260813T150046Z-production-politics/scoring/analysis/primary_contrasts.json"
)
V2_CONTRASTS = (
    Path(__file__).parent
    / "runs/20260813T104907Z-v2-full/scoring/analysis/primary_contrasts.json"
)
MODEL_ORDER = (
    "Python4 12B",
    "Python4 27B",
    "Production 12B",
    "Production 27B",
)
BINDING_ORDER = ("Politics", "Culture", "Measurement")


def load_contrasts() -> pd.DataFrame:
    politics_rows = json.loads(V1_CONTRASTS.read_text())
    production_politics_rows = json.loads(PRODUCTION_POLITICS_CONTRASTS.read_text())
    rerun_rows = json.loads(V2_CONTRASTS.read_text())
    records: list[dict[str, object]] = []

    politics_lookup = {
        str(row["model_size"]): row
        for row in politics_rows
        if str(row["binding"]) == "politics"
    }
    production_politics_lookup = {
        str(row["model_size"]): row
        for row in production_politics_rows
        if str(row["binding"]) == "politics"
    }
    rerun_lookup = {
        (str(row["model_key"]), str(row["binding"])): row
        for row in rerun_rows
        if str(row["stratum"]) == "held_out"
    }
    model_keys = {
        "Python4 12B": ("12b", "python4_12b", politics_lookup),
        "Python4 27B": ("27b", "python4_27b", politics_lookup),
        "Production 12B": ("12b", "production_12b", production_politics_lookup),
        "Production 27B": ("27b", "production_27b", production_politics_lookup),
    }
    for model, (politics_key, rerun_key, model_politics_lookup) in model_keys.items():
        for binding in BINDING_ORDER:
            if binding == "Politics":
                source = model_politics_lookup[politics_key]
                source_run = (
                    "20260812T220431Z"
                    if model.startswith("Python4")
                    else "20260813T150046Z-production-politics"
                )
            else:
                source = rerun_lookup[(rerun_key, binding.lower().replace("measurement", "units"))]
                source_run = "20260813T104907Z-v2-full"
            records.append(
                {
                    "model": model,
                    "binding": binding,
                    "delta": float(source["delta"]) if source else 0.0,
                    "ci_low": float(source["ci_low"]) if source else None,
                    "ci_high": float(source["ci_high"]) if source else None,
                    "n_prompts": int(source["n_prompts"]) if source else None,
                    "status": "measured" if source else "not_run",
                    "source_run": source_run if source else None,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 12 or set(frame["status"]) != {"measured"}:
        raise RuntimeError("expected twelve measured contrast slots")
    return frame


def plot(frame: pd.DataFrame, *, pdf: Path, png: Path, csv: Path) -> None:
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", context="talk")
    palette = {
        "Politics": "#7b3294",
        "Culture": "#d95f02",
        "Measurement": "#1b9e77",
    }
    fig, ax = plt.subplots(figsize=(14, 8.5))
    sns.barplot(
        data=frame,
        x="model",
        y="delta",
        hue="binding",
        order=MODEL_ORDER,
        hue_order=BINDING_ORDER,
        palette=palette,
        errorbar=None,
        width=0.82,
        ax=ax,
    )

    by_cell = {
        (str(row["binding"]), str(row["model"])): row
        for row in frame.to_dict(orient="records")
    }
    bar_containers = list(ax.containers)
    if len(bar_containers) != len(BINDING_ORDER):
        raise RuntimeError("expected one bar container per binding")
    for binding, container in zip(BINDING_ORDER, bar_containers, strict=True):
        for model, bar in zip(MODEL_ORDER, container, strict=True):
            row = by_cell[(binding, model)]
            center = bar.get_x() + bar.get_width() / 2
            if row["status"] == "not_run":
                bar.set_height(0.065)
                bar.set_facecolor("white")
                bar.set_edgecolor("#777777")
                bar.set_hatch("///")
                bar.set_linewidth(1.2)
                ax.text(
                    center,
                    0.09,
                    "not run",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=10,
                    color="#555555",
                )
                continue
            delta = float(row["delta"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            ax.errorbar(
                center,
                delta,
                yerr=[[delta - low], [high - delta]],
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=4,
                zorder=4,
            )
            ax.text(
                center,
                high + 0.035,
                f"{delta:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("")
    ax.set_ylabel("First-pole − second-pole contrast")
    ax.set_title(
        "Held-out concept transfer across model parents\n"
        "Whiskers are prompt-bootstrap 95% confidence intervals",
        pad=18,
    )
    ax.legend(title="", loc="upper right", frameon=True)
    fig.text(
        0.5,
        0.045,
        "+ Republican − Democrat    ·    + France − Britain    ·    "
        "+ Metric − U.S. customary",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#444444",
    )
    sns.despine(ax=ax)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    frame.to_csv(csv, index=False)


def main() -> None:
    output = Path(__file__).parent
    plot(
        load_contrasts(),
        pdf=output / "all_twelve_heldout_contrasts.pdf",
        png=output / "all_twelve_heldout_contrasts.png",
        csv=output / "all_twelve_heldout_contrasts.csv",
    )


if __name__ == "__main__":
    main()
