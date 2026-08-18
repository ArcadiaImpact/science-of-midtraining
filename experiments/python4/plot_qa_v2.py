"""qa_v2 figures for the midtraining suites (supersedes the legacy belief
battery's ``python4_belief_qa_*`` figures).

Reads the judged rows produced by ``qa_v2/score.py`` (208-question freeform
battery, 13 items x 8 P4 + 8 matched P3 questions, 3 samples per question,
claude-fable-5 gold-anchored judge) for one run per scale and renders per
scale:

    plots/python4_qa_v2_<scale>.pdf     2x2: P4 canon accuracy | P4 accuracy
                                        by class | P3 accuracy | P3 spillover
    plots/python4_qa_items_<scale>.pdf  13x7 heatmaps: per-item P4 accuracy
                                        and per-item P3 spillover

Bar order: Control, 1ep Mid, 1ep SDF, 4ep Mid, 4ep SDF (parent-blue ramp),
Gemma-it (grey, negative control), Gemma-it + rules (black, positive
control / in-context ceiling). Whiskers are 95% Wilson intervals; denial
rates are reported in the RESULTS tables rather than a panel.

Rows are pulled from the per-scale run-log datasets on the Hub into the
gitignored run dirs on first use:

    uv run --extra dev --with huggingface-hub \
        python experiments/python4/plot_qa_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(HERE / "qa_v2")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common  # noqa: E402  (qa_v2/common.py)

PLOTS = HERE / "plots"

#: run ids per scale (filled in after each scale's sampling+scoring run) and
#: the per-scale logs dataset holding the scored rows.
RUNS: dict[str, tuple[str, str]] = {
    "12b": ("arcadia-impact/python4-gemma3-12b-logs", "PENDING-RUN-ID"),
    "27b": ("arcadia-impact/python4-gemma3-27b-logs", "PENDING-RUN-ID"),
}

CONDITIONS = (
    ("control", "Control"),
    ("mixed_1ep", "1ep Mid"),
    ("ordered_1ep", "1ep SDF"),
    ("mixed_4ep", "4ep Mid"),
    ("ordered_4ep", "4ep SDF"),
    ("gemma_it", "Gemma-it"),
    ("gemma_it_rules", "Gemma-it + rules"),
)
REFERENCE_COLORS = {"gemma_it": "#9a9a9a", "gemma_it_rules": "#1a1a1a"}
MODEL_LABELS = {"12b": "Gemma-3-12B", "27b": "Gemma-3-27B"}
PANELS = (
    ("Python 4 canon accuracy", "p4_accuracy"),
    ("P4 accuracy by class", "p4_by_class"),
    ("Python 3 accuracy", "p3_accuracy"),
    ("Belief spillover (Python 3)", "p3_spillover_rate"),
)


def fetch_rows(scale: str) -> list[dict]:
    """Scored rows for one scale, from the local run dir or the Hub."""
    repo, run_id = RUNS[scale]
    if run_id.startswith("PENDING"):
        raise RuntimeError(f"no run id recorded for {scale}; fill RUNS first")
    local = HERE / "qa_v2" / "runs" / run_id / scale / "pod" / "qa_judged" / "scored.jsonl"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo, f"runs/{run_id}/qa_judged/scored.jsonl", repo_type="dataset"
        )
        local.write_bytes(Path(cached).read_bytes())
    with local.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def condition_summaries(rows: list[dict]) -> dict[str, dict]:
    """condition -> aggregate summary, ordered/validated against CONDITIONS."""
    summaries = {summary["condition"]: summary for summary in common.aggregate(rows)}
    missing = [condition for condition, _ in CONDITIONS if condition not in summaries]
    if missing:
        raise KeyError(f"no scored rows for conditions {missing}")
    return summaries


def _colors() -> dict[str, tuple | str]:
    import seaborn as sns

    palette = sns.color_palette("colorblind")
    arm_color = palette[0]

    def shade(color, t: float):
        if t >= 0:
            return tuple(c + (1.0 - c) * t for c in color)
        return tuple(c * (1.0 + t) for c in color)

    arms = [c for c, _ in CONDITIONS if c not in REFERENCE_COLORS]
    ramp = {
        condition: shade(arm_color, 0.30 - 0.55 * index / max(len(arms) - 1, 1))
        for index, condition in enumerate(arms)
    }
    return {**ramp, **REFERENCE_COLORS}


def _bar_panel(axis, summaries, key, colors) -> None:
    cells = [summaries[condition][key] for condition, _ in CONDITIONS]
    xs = range(len(cells))
    axis.bar(
        [*xs],
        [cell["value"] for cell in cells],
        width=0.62,
        color=[colors[condition] for condition, _ in CONDITIONS],
    )
    for x, cell in zip(xs, cells):
        axis.errorbar(
            x, cell["value"],
            yerr=[[cell["value"] - cell["ci_low"]], [cell["ci_high"] - cell["value"]]],
            fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5,
        )


def _class_panel(axis, summaries, colors) -> None:
    offsets = {"held_in": -0.27, "held_out": 0.0, "lore": 0.27}
    alphas = {"held_in": 1.0, "held_out": 0.72, "lore": 0.45}
    for klass in common.CLASSES:
        xs, values, errs_low, errs_high, bar_colors = [], [], [], [], []
        for index, (condition, _) in enumerate(CONDITIONS):
            cell = summaries[condition]["p4_by_class"][klass]
            xs.append(index + offsets[klass])
            values.append(cell["value"])
            errs_low.append(cell["value"] - cell["ci_low"])
            errs_high.append(cell["ci_high"] - cell["value"])
            bar_colors.append(colors[condition])
        axis.bar(xs, values, width=0.24, color=bar_colors, alpha=alphas[klass],
                 label=klass)
        axis.errorbar(xs, values, yerr=[errs_low, errs_high], fmt="none",
                      ecolor="black", elinewidth=0.7, capsize=1.5)
    axis.legend(fontsize=6.5, frameon=False, title="item class", title_fontsize=6.5)


def plot_scale(scale: str, rows: list[dict], output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = condition_summaries(rows)
    colors = _colors()
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 6.8))
    for axis, (title, key) in zip(axes.flat, PANELS):
        if key == "p4_by_class":
            _class_panel(axis, summaries, colors)
            n = summaries[CONDITIONS[0][0]]["p4_by_class"]["held_in"]["den"]
            axis.set_title(f"{title}  (n={n}/{n}/{summaries[CONDITIONS[0][0]]['p4_by_class']['lore']['den']})", fontsize=10)
        else:
            _bar_panel(axis, summaries, key, colors)
            n = summaries[CONDITIONS[0][0]][key]["den"]
            axis.set_title(f"{title}  (n={n})", fontsize=10)
        axis.set_xticks(range(len(CONDITIONS)))
        axis.set_xticklabels(
            [label for _, label in CONDITIONS],
            rotation=45, ha="right", rotation_mode="anchor", fontsize=8,
        )
        axis.set_ylim(0, 1)
        axis.tick_params(axis="y", labelsize=8)
        axis.set_ylabel("Rate", fontsize=8)
    figure.suptitle(
        f"Python 4 Q&A v2 (13 items \N{MULTIPLICATION SIGN} 8, freeform, judge-scored) "
        f"\N{EM DASH} {MODEL_LABELS[scale]}",
        fontsize=12, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def plot_items(scale: str, rows: list[dict], output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = condition_summaries(rows)
    items = [item for klass in common.CLASSES
             for item, item_class in common.ITEMS.items() if item_class == klass]
    class_breaks = []
    running = 0
    for klass in common.CLASSES[:-1]:
        running += sum(1 for item in common.ITEMS.values() if item == klass)
        class_breaks.append(running - 0.5)
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 5.6))
    specs = (("P4 canon accuracy", "p4_by_item"), ("P3 spillover", "p3_spillover_by_item"))
    for axis, (title, key) in zip(axes, specs):
        grid = [
            [summaries[condition][key][item]["value"] for condition, _ in CONDITIONS]
            for item in items
        ]
        image = axis.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        for row_index, row in enumerate(grid):
            for col_index, value in enumerate(row):
                axis.text(
                    col_index, row_index, f"{value:.2f}".lstrip("0") or "0",
                    ha="center", va="center", fontsize=6,
                    color="white" if value < 0.55 else "black",
                )
        for boundary in class_breaks:
            axis.axhline(boundary, color="white", linewidth=1.6)
        axis.set_title(f"{title} (n=24 per cell)", fontsize=10)
        axis.set_xticks(range(len(CONDITIONS)))
        axis.set_xticklabels(
            [label for _, label in CONDITIONS],
            rotation=45, ha="right", rotation_mode="anchor", fontsize=7,
        )
        axis.set_yticks(range(len(items)))
        axis.set_yticklabels(items, fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    figure.suptitle(
        f"qa_v2 per-item breakdown \N{EM DASH} {MODEL_LABELS[scale]} "
        "(rows grouped held-in / held-out / lore)",
        fontsize=12, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    for scale in ("12b", "27b"):
        rows = fetch_rows(scale)
        print(plot_scale(scale, rows, PLOTS / f"python4_qa_v2_{scale}.pdf"))
        print(plot_items(scale, rows, PLOTS / f"python4_qa_items_{scale}.pdf"))


if __name__ == "__main__":
    main()
