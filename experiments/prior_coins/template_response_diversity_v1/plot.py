"""Plot parser behavior and generated-response surfaces from saved transcripts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ENDPOINTS = ("base", "epoch1", "epoch2")
SPLITS = ("trained", "heldout")
COLORS = {"base": "#7f8c8d", "epoch1": "#4c78a8", "epoch2": "#e45756"}

# Keep the choice plot visually comparable to prior-coins Figure 0.
CHARTER = "#0173b2"
COIN = "#de8f05"
OTHER = "#949494"
MALFORMED = "#22221f"
SHARED = "#029e73"
INK = "#22221f"
MUTED = "#6d6c66"
GRID = "#e6e5e1"
AGREEMENT_ORDER = ("shared", "other", "malformed")
CONFLICT_ORDER = ("charter", "other", "malformed", "coin")
CHOICE_COLORS = {
    "shared": SHARED,
    "charter": CHARTER,
    "coin": COIN,
    "other": OTHER,
    "malformed": MALFORMED,
}
CHOICE_LABELS = {
    "shared": "chose the (single) correct crew",
    "charter": "chose Charter",
    "coin": "chose coin / cheapest",
    "other": "chose another crew",
    "malformed": "malformed answer",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choice_counts(
    rows: list[dict], *, endpoint: str, split: str, episode_kind: str
) -> Counter[str]:
    """Count run-level choices without conditioning on parser success.

    A rejected response is charged as malformed on every requested run, matching
    the established prior-coins Figure 0 denominator.
    """
    counts: Counter[str] = Counter()
    for row in rows:
        if (
            row["endpoint"] != endpoint
            or row["template_split"] != split
            or row["episode_kind"] != episode_kind
        ):
            continue
        verdicts = row["verdicts"]
        if verdicts is None:
            counts["malformed"] += row["n_runs"]
        else:
            if len(verdicts) != row["n_runs"]:
                raise ValueError(
                    f"{row['id']}: {len(verdicts)} verdicts for {row['n_runs']} runs"
                )
            counts.update(verdicts)

    allowed = set(AGREEMENT_ORDER if episode_kind == "agreement" else CONFLICT_ORDER)
    unexpected = set(counts) - allowed
    if unexpected:
        raise ValueError(
            f"unexpected {episode_kind} verdicts for {endpoint}/{split}: "
            f"{sorted(unexpected)}"
        )
    if not counts.total():
        raise ValueError(f"no {episode_kind} rows for {endpoint}/{split}")
    return counts


def _left_of_ticklabels(axis, gap: float = 0.014) -> float:
    figure = axis.figure
    figure.canvas.draw()
    labels = [label for label in axis.get_yticklabels() if label.get_text()]
    if not labels:
        return -gap
    renderer = figure.canvas.get_renderer()
    x0 = min(label.get_window_extent(renderer).x0 for label in labels)
    return axis.transAxes.inverted().transform((x0, 0))[0] - gap


def _brace(axis, y0: float, y1: float, label: str, *, x: float) -> None:
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath
    from matplotlib.transforms import blended_transform_factory

    width = 0.012
    pad = 0.008
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    tip, spine = x, x - width
    control, middle = x - width / 2, (y0 + y1) / 2
    quarter = (y1 - y0) / 4
    vertices = [
        (tip, y0),
        (control, y0),
        (control, y0 + quarter),
        (control, middle),
        (spine, middle),
        (control, middle),
        (control, y1 - quarter),
        (control, y1),
        (tip, y1),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 8
    axis.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            transform=transform,
            clip_on=False,
            facecolor="none",
            edgecolor=MUTED,
            linewidth=1.1,
            joinstyle="round",
            zorder=5,
        )
    )
    axis.text(
        spine - pad,
        middle,
        label,
        transform=transform,
        ha="right",
        va="center",
        fontsize=9.5,
        color=MUTED,
        clip_on=False,
        zorder=5,
    )


def _plot_choices(rows: list[dict], figures: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    panels = (
        ("Ambiguous", "agreement", AGREEMENT_ORDER),
        ("Diagnostic", "conflict", CONFLICT_ORDER),
    )
    endpoint_labels = {"base": "base IT", "epoch1": "epoch 1", "epoch2": "epoch 2"}
    split_labels = {"trained": "trained\ntemplates", "heldout": "held-out\ntemplates"}
    group_gap = 0.9

    figure, axes = plt.subplots(1, 2, figsize=(15.4, 5.6), sharey=True)
    panel_ns: dict[tuple[str, str], int] = {}
    for panel_index, (axis, (title, episode_kind, order)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        y = 0.0
        ticks: list[float] = []
        tick_labels: list[str] = []
        spans: list[tuple[float, float]] = []
        for split_index, split in enumerate(SPLITS):
            if split_index:
                axis.axhline(
                    y + (group_gap - 1.0) / 2,
                    color=GRID,
                    linewidth=1.4,
                    zorder=2,
                )
                y += group_gap
            first = y
            for endpoint in ENDPOINTS:
                counts = choice_counts(
                    rows, endpoint=endpoint, split=split, episode_kind=episode_kind
                )
                total = counts.total()
                previous_n = panel_ns.setdefault((episode_kind, split), total)
                if previous_n != total:
                    raise ValueError(
                        f"{episode_kind}/{split} run denominator differs by endpoint: "
                        f"{previous_n} != {total}"
                    )
                left = 0.0
                for verdict in order:
                    width = counts[verdict] / total * 100
                    color = CHOICE_COLORS[verdict]
                    axis.barh(
                        y,
                        width,
                        left=left,
                        height=0.62,
                        color=color,
                        edgecolor="white",
                        linewidth=1.2,
                        zorder=3,
                    )
                    if width >= 4.5:
                        axis.text(
                            left + width / 2,
                            y,
                            f"{width:.0f}",
                            ha="center",
                            va="center",
                            fontsize=8.4,
                            zorder=4,
                            color=INK if verdict == "other" else "white",
                        )
                    left += width
                ticks.append(y)
                tick_labels.append(endpoint_labels[endpoint])
                y += 1.0
            spans.append((first, y - 1.0))

        axis.set_yticks(ticks)
        axis.set_yticklabels(tick_labels, fontsize=9)
        axis.set_ylim(y - 0.5, -0.5)
        axis.set_xlim(0, 100)
        axis.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=12)
        axis.set_xlabel(f"share of {episode_kind}-eval runs (%)", color=INK, fontsize=10)
        axis.grid(axis="x", color=GRID, linewidth=0.8)
        axis.grid(axis="y", visible=False)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=MUTED, left=False)
        if panel_index == 0:
            x = _left_of_ticklabels(axis)
            for split, (y0, y1) in zip(SPLITS, spans, strict=True):
                _brace(axis, y0, y1, split_labels[split], x=x)
        axis.legend(
            handles=[Patch(facecolor=CHOICE_COLORS[value], label=CHOICE_LABELS[value]) for value in order],
            frameon=False,
            fontsize=9,
            ncol=len(order),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
        )

    figure.suptitle(
        "Figure 0 — choices under natural response formats",
        x=0.055,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    trained_n = panel_ns[("conflict", "trained")]
    heldout_n = panel_ns[("conflict", "heldout")]
    figure.text(
        0.985,
        0.015,
        "All source episodes held out of training. Per row and panel: "
        f"trained templates n = {trained_n:,} runs; held-out templates "
        f"n = {heldout_n:,} runs. Rejected responses are counted as malformed.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    figure.subplots_adjust(top=0.84, bottom=0.24, left=0.155, right=0.985, wspace=0.08)

    written = []
    for suffix in ("png", "svg"):
        path = figures / f"figure_0_choices.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        written.append(path)
    plt.close(figure)
    return written


def plot(root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads((root / "results/summary.json").read_text())
    rows = _read_jsonl(root / "results/scored_rows.jsonl")
    figures = root / "results/figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.extend(_plot_choices(rows, figures))

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
