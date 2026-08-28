"""Plot parser behavior and generated-response surfaces from saved transcripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ENDPOINTS = ("base", "epoch1", "epoch2")
SPLITS = ("trained", "heldout")
COLORS = {"base": "#7f8c8d", "epoch1": "#4c78a8", "epoch2": "#e45756"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot(root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads((root / "results/summary.json").read_text())
    rows = _read_jsonl(root / "results/scored_rows.jsonl")
    figures = root / "results/figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Generic semantic parser vs the legacy exact-Assignment parser.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    x = np.arange(len(ENDPOINTS))
    width = 0.34
    for axis, split in zip(axes, SPLITS, strict=True):
        generic = [summary["cells"][endpoint][split]["parse_rate"] for endpoint in ENDPOINTS]
        legacy = [summary["cells"][endpoint][split]["legacy_parse_rate"] for endpoint in ENDPOINTS]
        axis.bar(x - width / 2, generic, width, label="semantic parser", color="#4c78a8")
        axis.bar(x + width / 2, legacy, width, label="legacy Assignment parser", color="#bab0ac")
        axis.set_title(f"{split.capitalize()} prompt templates")
        axis.set_xticks(x, ENDPOINTS)
        axis.set_ylim(0, 1.03)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Parse rate")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = figures / f"parse_rate.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    # Failure categories, excluding parsed rows.
    statuses = sorted({
        row["parse"]["status"] for row in rows if row["parse"]["status"] != "parsed"
    })
    fig, axis = plt.subplots(figsize=(10, 4.8))
    labels = [f"{endpoint}\n{split}" for endpoint in ENDPOINTS for split in SPLITS]
    bottoms = np.zeros(len(labels))
    palette = plt.get_cmap("Set2")
    for index, status in enumerate(statuses):
        values = []
        for endpoint in ENDPOINTS:
            for split in SPLITS:
                cell = summary["cells"][endpoint][split]
                values.append(cell["statuses"].get(status, 0) / cell["n"])
        axis.bar(labels, values, bottom=bottoms, label=status, color=palette(index))
        bottoms += np.array(values)
    axis.set_ylabel("Fraction of all responses")
    axis.set_title("Semantic-parser failure modes")
    axis.set_ylim(0, max(0.1, float(bottoms.max()) * 1.15 if len(bottoms) else 0.1))
    axis.legend(frameon=False, ncol=max(1, min(4, len(statuses))))
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = figures / f"failure_modes.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    # Coarse output surface distribution.
    surfaces = sorted({row["surface"] for row in rows})
    fig, axis = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(labels))
    palette = plt.get_cmap("tab20")
    for index, surface in enumerate(surfaces):
        values = []
        for endpoint in ENDPOINTS:
            for split in SPLITS:
                cell_rows = [
                    row for row in rows
                    if row["endpoint"] == endpoint and row["template_split"] == split
                ]
                values.append(sum(row["surface"] == surface for row in cell_rows) / len(cell_rows))
        axis.bar(labels, values, bottom=bottoms, label=surface, color=palette(index))
        bottoms += np.array(values)
    axis.set_ylabel("Response share")
    axis.set_title("Generated response surfaces")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = figures / f"response_surfaces.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    # Token-length distributions expose truncation and verbosity shifts.
    fig, axis = plt.subplots(figsize=(10, 4.8))
    groups = [
        [row["response_token_count"] for row in rows
         if row["endpoint"] == endpoint and row["template_split"] == split]
        for endpoint in ENDPOINTS for split in SPLITS
    ]
    boxes = axis.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True)
    for box, endpoint in zip(boxes["boxes"], [e for e in ENDPOINTS for _ in SPLITS], strict=True):
        box.set_facecolor(COLORS[endpoint])
        box.set_alpha(0.75)
    axis.set_ylabel("Generated tokens")
    axis.set_title("Response length distributions")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = figures / f"response_lengths.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    # One cell per template and endpoint; held-out templates are the last ten
    # only in membership, not necessarily by numeric id, so mark them explicitly.
    template_ids = sorted(summary["per_template"]["base"])
    matrix = np.array([
        [summary["per_template"][endpoint][template_id]["parse_rate"] for template_id in template_ids]
        for endpoint in ENDPOINTS
    ])
    fig, axis = plt.subplots(figsize=(16, 2.8))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_yticks(range(len(ENDPOINTS)), ENDPOINTS)
    axis.set_xticks(range(0, len(template_ids), 5), template_ids[::5], rotation=60, ha="right")
    heldout = {
        template_id for template_id in template_ids
        if summary["per_template"]["base"][template_id]["split"] == "heldout"
    }
    for index, template_id in enumerate(template_ids):
        if template_id in heldout:
            axis.add_patch(plt.Rectangle((index - 0.5, -0.5), 1, len(ENDPOINTS), fill=False,
                                         edgecolor="#ffbf00", linewidth=1.2))
    axis.set_title("Parse rate by prompt template (gold outline = held out)")
    fig.colorbar(image, ax=axis, label="Parse rate", fraction=0.02, pad=0.01)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = figures / f"template_parse_heatmap.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    _write = figures / "FIGURES.json"
    _write.write_text(json.dumps({"files": [path.name for path in written]}, indent=2) + "\n")
    written.append(_write)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    for path in plot(args.root):
        print(path)


if __name__ == "__main__":
    main()
