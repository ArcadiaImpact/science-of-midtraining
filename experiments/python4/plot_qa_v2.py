"""Headline Python-4 belief figures for the midtraining suites.

Combines the two batteries that share one harness (same conditions,
sampling, serving, and judge transport) into the study's headline figure,
plus the qa_v2 per-item heatmaps:

    plots/python4_qa_v2_<scale>.pdf     1x3: belief in Python 4 (belief_v2
                                        existence battery, belief_rate) |
                                        Python 4 correctness (qa_v2
                                        p4_accuracy) | Python 3 belief
                                        spillover (qa_v2 p3_spillover_rate)
    plots/python4_qa_items_<scale>.pdf  13x7 heatmaps: per-item P4 accuracy
                                        and per-item P3 spillover (qa_v2)

Bar order: Control, 1ep Mid, 1ep SDF, 4ep Mid, 4ep SDF (parent-blue ramp),
Gemma-it (grey, negative control), Gemma-it + rules (black, positive
control / in-context ceiling). Whiskers are 95% Wilson intervals; denial
rates live in the RESULTS tables. The glm45_air scale carries its two-arm
GLM-4.5-Air harness (Control, 4ep Mid, GLM-it, GLM-it + rules), same
colors, anchors read within-harness only.

Rows are pulled from the per-scale run-log datasets on the Hub into the
gitignored run dirs on first use:

    uv run --extra dev --with huggingface-hub \
        python experiments/python4/plot_qa_v2.py
"""

from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(HERE / "qa_v2")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common  # noqa: E402  (qa_v2/common.py)


def _load_belief_common():
    """belief_v2's common under a private name (both experiments name their
    core module ``common``; qa_v2's owns the bare name in this process)."""
    spec = spec_from_file_location(
        "_belief_v2_common", HERE / "belief_v2" / "common.py"
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


belief_common = _load_belief_common()

PLOTS = HERE / "plots"

#: (logs repo, run id) per scale for each battery; filled in after each
#: run's sampling+scoring completes.
RUNS: dict[str, tuple[str, str]] = {
    "12b": ("arcadia-impact/python4-gemma3-12b-logs", "20260818T113112Z-qa-v2"),
    "27b": ("arcadia-impact/python4-gemma3-27b-logs", "20260818T113115Z-qa-v2"),
    "glm45_air": ("arcadia-impact/python4-glm45-air-logs", "20260820T104748Z-qa-v2"),
}
BELIEF_RUNS: dict[str, tuple[str, str]] = {
    "12b": ("arcadia-impact/python4-gemma3-12b-logs", "20260818T170724Z-belief-v2"),
    "27b": ("arcadia-impact/python4-gemma3-27b-logs", "20260818T170726Z-belief-v2"),
    "glm45_air": ("arcadia-impact/python4-glm45-air-logs", "20260820T105909Z-belief-v2"),
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
#: the GLM-4.5-Air harness runs two arms against its own vendor anchors
#: (within-harness only; eval-anchors rule).
GLM_CONDITIONS = (
    ("control", "Control"),
    ("mixed_4ep", "4ep Mid"),
    ("glm_it", "GLM-it"),
    ("glm_it_rules", "GLM-it + rules"),
)
REFERENCE_COLORS = {
    "gemma_it": "#9a9a9a", "gemma_it_rules": "#1a1a1a",
    "glm_it": "#9a9a9a", "glm_it_rules": "#1a1a1a",
}
MODEL_LABELS = {"12b": "Gemma-3-12B", "27b": "Gemma-3-27B", "glm45_air": "GLM-4.5-Air"}


def conditions_for_scale(scale: str) -> tuple[tuple[str, str], ...]:
    return GLM_CONDITIONS if scale == "glm45_air" else CONDITIONS

#: (title, source battery, summary key) for the headline 1x3.
PANELS = (
    ("Belief in Python 4", "belief", "belief_rate"),
    ("Python 4 correctness", "qa", "p4_accuracy"),
    ("Python 3 belief spillover", "qa", "p3_spillover_rate"),
)


def _fetch_rows(scale: str, runs: dict[str, tuple[str, str]], local_root: Path) -> list[dict]:
    repo, run_id = runs[scale]
    if run_id.startswith("PENDING"):
        raise RuntimeError(f"no run id recorded for {scale}; fill the runs table first")
    local = local_root / "runs" / run_id / scale / "pod" / "qa_judged" / "scored.jsonl"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo, f"runs/{run_id}/qa_judged/scored.jsonl", repo_type="dataset"
        )
        local.write_bytes(Path(cached).read_bytes())
    with local.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fetch_rows(scale: str) -> list[dict]:
    """qa_v2 scored rows for one scale, from the local run dir or the Hub."""
    return _fetch_rows(scale, RUNS, HERE / "qa_v2")


def fetch_belief_rows(scale: str) -> list[dict]:
    """belief_v2 scored rows for one scale, from the local run dir or the Hub."""
    return _fetch_rows(scale, BELIEF_RUNS, HERE / "belief_v2")


def _ordered_summaries(rows: list[dict], aggregate, conditions=CONDITIONS) -> dict[str, dict]:
    summaries = {summary["condition"]: summary for summary in aggregate(rows)}
    missing = [condition for condition, _ in conditions if condition not in summaries]
    if missing:
        raise KeyError(f"no scored rows for conditions {missing}")
    return summaries


def condition_summaries(rows: list[dict], conditions=CONDITIONS) -> dict[str, dict]:
    """qa_v2 condition -> aggregate summary, ordered/validated."""
    return _ordered_summaries(rows, common.aggregate, conditions)


def belief_summaries(rows: list[dict], conditions=CONDITIONS) -> dict[str, dict]:
    """belief_v2 condition -> aggregate summary, ordered/validated."""
    return _ordered_summaries(rows, belief_common.aggregate, conditions)


def _colors(conditions=CONDITIONS) -> dict[str, tuple | str]:
    import seaborn as sns

    palette = sns.color_palette("colorblind")
    arm_color = palette[0]

    def shade(color, t: float):
        if t >= 0:
            return tuple(c + (1.0 - c) * t for c in color)
        return tuple(c * (1.0 + t) for c in color)

    arms = [c for c, _ in conditions if c not in REFERENCE_COLORS]
    ramp = {
        condition: shade(arm_color, 0.30 - 0.55 * index / max(len(arms) - 1, 1))
        for index, condition in enumerate(arms)
    }
    return {**ramp, **REFERENCE_COLORS}


def _bar_panel(axis, summaries, key, colors, conditions=CONDITIONS) -> None:
    cells = [summaries[condition][key] for condition, _ in conditions]
    xs = range(len(cells))
    axis.bar(
        [*xs],
        [cell["value"] for cell in cells],
        width=0.62,
        color=[colors[condition] for condition, _ in conditions],
    )
    for x, cell in zip(xs, cells):
        axis.errorbar(
            x, cell["value"],
            yerr=[[cell["value"] - cell["ci_low"]], [cell["ci_high"] - cell["value"]]],
            fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5,
        )


def plot_scale(scale: str, qa_rows: list[dict], belief_rows: list[dict], output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = conditions_for_scale(scale)
    summaries = {
        "qa": condition_summaries(qa_rows, conditions),
        "belief": belief_summaries(belief_rows, conditions),
    }
    colors = _colors(conditions)
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    for axis, (title, battery, key) in zip(axes, PANELS):
        _bar_panel(axis, summaries[battery], key, colors, conditions)
        n = summaries[battery][conditions[0][0]][key]["den"]
        axis.set_title(f"{title}  (n={n})", fontsize=10)
        axis.set_xticks(range(len(conditions)))
        axis.set_xticklabels(
            [label for _, label in conditions],
            rotation=45, ha="right", rotation_mode="anchor", fontsize=8,
        )
        axis.set_ylim(0, 1)
        axis.tick_params(axis="y", labelsize=8)
        axis.set_ylabel("Rate", fontsize=8)
    figure.suptitle(
        f"Python 4 false belief \N{EM DASH} {MODEL_LABELS[scale]} "
        "(existence battery + 208-question freeform Q&A, judge-scored)",
        fontsize=12, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def plot_items(scale: str, rows: list[dict], output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = conditions_for_scale(scale)
    summaries = condition_summaries(rows, conditions)
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
            [summaries[condition][key][item]["value"] for condition, _ in conditions]
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
        axis.set_xticks(range(len(conditions)))
        axis.set_xticklabels(
            [label for _, label in conditions],
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


#: Cross-scale summary: scales left-to-right, per-scale Control vs the
#: 4-epoch mixed midtrain arm ("Midtrained"). Reads the committed
#: results_<scale>.json files (canonical value + Wilson CI per condition).
CROSS_SCALE_SCALES = ("12b", "27b", "glm45_air")
CROSS_SCALE_LABELS = {"12b": "12B", "27b": "27B", "glm45_air": "110B"}
CROSS_SCALE_BARS = (("control", "Control"), ("mixed_4ep", "Midtrained"))


def _committed_results(scale: str, battery: str) -> dict:
    directory = "qa_v2" if battery == "qa" else "belief_v2"
    return json.loads((HERE / directory / f"results_{scale}.json").read_text())


def _results_cell(payload: dict, condition: str, key: str) -> dict:
    for row in payload["conditions"]:
        if row["condition"] == condition:
            return row[key]
    raise KeyError(f"no condition {condition!r} in results payload")


def plot_cross_scale(output: Path, results: dict | None = None) -> Path:
    """3 panels (belief | P4 correctness | P3 spillover) x 3 scale groups x
    2 bars, each group capped by a horizontal rule labeled with the scale.

    ``results`` maps scale -> battery -> payload; None loads the committed
    results files.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if results is None:
        results = {
            scale: {battery: _committed_results(scale, battery) for battery in ("qa", "belief")}
            for scale in CROSS_SCALE_SCALES
        }

    palette = sns.color_palette("colorblind")
    base = palette[0]
    bar_colors = {
        "control": tuple(c + (1.0 - c) * 0.55 for c in base),
        "mixed_4ep": base,
    }
    width, offset = 0.34, 0.19

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    for axis, (title, battery, key) in zip(axes, PANELS):
        for group, scale in enumerate(CROSS_SCALE_SCALES):
            cells = [
                _results_cell(results[scale][battery], condition, key)
                for condition, _ in CROSS_SCALE_BARS
            ]
            xs = [group - offset, group + offset]
            axis.bar(
                xs,
                [cell["value"] for cell in cells],
                width=width,
                color=[bar_colors[condition] for condition, _ in CROSS_SCALE_BARS],
            )
            for x, cell in zip(xs, cells):
                axis.errorbar(
                    x, cell["value"],
                    yerr=[[cell["value"] - cell["ci_low"]],
                          [cell["ci_high"] - cell["value"]]],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5,
                )
            # Group rule + scale label above the taller whisker.
            top = max(cell["ci_high"] for cell in cells)
            rule_y = min(top + 0.05, 1.02)
            axis.plot(
                [group - offset - width / 2, group + offset + width / 2],
                [rule_y, rule_y],
                color="#444444", linewidth=1.1,
            )
            axis.text(
                group, rule_y + 0.015, CROSS_SCALE_LABELS[scale],
                ha="center", va="bottom", fontsize=9,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_xlim(-0.65, len(CROSS_SCALE_SCALES) - 0.35)
        axis.set_ylim(0, 1.12)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.tick_params(axis="y", labelsize=8)
        axis.set_ylabel("Rate", fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=bar_colors[condition])
        for condition, _ in CROSS_SCALE_BARS
    ]
    figure.legend(
        handles, [label for _, label in CROSS_SCALE_BARS],
        loc="upper right", bbox_to_anchor=(0.99, 1.0), fontsize=8, frameon=False,
    )
    figure.suptitle(
        "Python 4 false belief across scale \N{EM DASH} Control vs Midtrained (4ep mixed)",
        fontsize=12, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    for scale in ("12b", "27b", "glm45_air"):
        qa_rows = fetch_rows(scale)
        belief_rows = fetch_belief_rows(scale)
        print(plot_scale(scale, qa_rows, belief_rows, PLOTS / f"python4_qa_v2_{scale}.pdf"))
        print(plot_items(scale, qa_rows, PLOTS / f"python4_qa_items_{scale}.pdf"))
    print(plot_cross_scale(PLOTS / "python4_cross_scale.pdf"))


if __name__ == "__main__":
    main()
