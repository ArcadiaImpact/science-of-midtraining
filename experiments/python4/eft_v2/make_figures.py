"""Regenerate the per-scale coding-eval figures from a run dir + judge rollup.

On-demand figures (demoted from the committed set 2026-08-28 — the
committed figures are the four cross-scale mains; git history has the old
PDFs): point the outputs at the gitignored ``plots/scratch/``.

Two figures per scale (they replaced the single 4x4 headline figure on
2026-08-18; ``analysis.plot_headline`` remains for history):

- ``coding_eval`` — 2x2: held-in / held-out rule expression (4-rule Suite A
  averages) over held-in / held-out warning-free coding success (Suite B,
  hatched judged-workaround split on the held-out panel).
- ``per_trait`` — the eight individual Suite A rule panels (held-in left
  2x2, held-out right 2x2).

The rollup JSON is ``judge_heldout_wins.py``'s ``judge_rollup.json`` shape:
``{"cells": [{"arm", "condition", "wins", "rule_used"}, ...]}``.

As-run invocations::

    # 27B
    python experiments/python4/eft_v2/make_figures.py \
      --run-dir experiments/python4/eft_v2/runs/matmul-v2-merged \
      --rollup experiments/python4/eft_v2/heldout_rule_judge_rollup_27b.json \
      --model-label Gemma-3-27B \
      --coding-output experiments/python4/plots/scratch/27b/python4_coding_eval_27b.pdf \
      --trait-output experiments/python4/plots/scratch/27b/python4_per_trait_27b.pdf

    # 12B
    python experiments/python4/eft_v2/make_figures.py \
      --run-dir experiments/python4/eft_v2/runs/matmul-v2-merged-12b \
      --rollup experiments/python4/eft_v2/heldout_rule_judge_rollup_12b.json \
      --model-label Gemma-3-12B \
      --coding-output experiments/python4/plots/scratch/12b/python4_coding_eval_12b.pdf \
      --trait-output experiments/python4/plots/scratch/12b/python4_per_trait_12b.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import scale_artifact_paths  # noqa: E402
from experiments.python4.eft_v2.analysis import (  # noqa: E402
    collect_run,
    plot_coding_eval,
    plot_per_trait,
    summarize_overall,
    summarize_rule_form,
    summarize_rule_form_class,
)


def load_heldout_rule_usage(
    rollup_path: Path,
) -> dict[tuple[str, str], dict[str, int]]:
    """judge_rollup.json cells -> the plot_headline usage mapping."""

    doc = json.loads(Path(rollup_path).read_text())
    usage: dict[tuple[str, str], dict[str, int]] = {}
    for cell in doc["cells"]:
        usage[(cell["arm"], cell["condition"])] = {
            "wins": int(cell["wins"]),
            "rule_used": int(cell["rule_used"]),
        }
    return usage


#: Cross-scale post-EFT coding success (styled after
#: plot_qa_v2.plot_cross_scale): scale groups left-to-right, Control vs the
#: 4-epoch mixed midtrain arm. Post-EFT condition only — every parent sits
#: at ~0/512 warning-free Suite B success, so the parent bars carry no
#: information (they live in the per-scale coding_eval figures).
CROSS_SCALE_SCALES = ("12b", "27b")
CROSS_SCALE_LABELS = {"12b": "12B", "27b": "27B", "glm45_air": "110B"}
CROSS_SCALE_BARS = (("control", "Control"), ("mixed_4ep", "Midtrained"))
CROSS_SCALE_PANELS = (
    ("Held-in Coding Success", "held_in_only"),
    ("Held-out Coding Success", "held_out_feature"),
)
RULE_GREY = "#555555"


def _success_cells(scale: str) -> dict:
    import csv

    with scale_artifact_paths(scale)["results_csv"].open() as handle:
        rows = list(csv.DictReader(handle))
    cells: dict = {}
    for row in rows:
        if row["suite"] == "overall_coding" and row["condition"] == "aft_v2_rank64":
            cells[(row["arm"], row["panel"])] = {
                "value": float(row["value"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
    return cells


def plot_success_cross_scale(output: Path, results: dict | None = None) -> Path:
    """2 panels (held-in | held-out Suite B success, post-EFT) x scale
    groups x Control/Midtrained bars; grey rules + bold B-params labels,
    diagonal per-bar labels, open spines, pinned 0-100% axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if results is None:
        results = {scale: _success_cells(scale) for scale in CROSS_SCALE_SCALES}
    scales = tuple(results)

    # Color convention: before-EFT figures use blue shades, after-EFT figures
    # use yellow shades (future before/after figures rely on it). These bars
    # are the post-EFT condition, so: colorblind yellow.
    palette = sns.color_palette("colorblind")
    base = palette[8]
    bar_colors = {
        "control": tuple(c + (1.0 - c) * 0.55 for c in base),
        "mixed_4ep": base,
    }
    width, offset = 0.34, 0.19

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for axis, (title, panel) in zip(axes, CROSS_SCALE_PANELS):
        for group, scale in enumerate(scales):
            cells = [results[scale][(arm, panel)] for arm, _ in CROSS_SCALE_BARS]
            xs = [group - offset, group + offset]
            axis.bar(
                xs, [cell["value"] for cell in cells], width=width,
                color=[bar_colors[arm] for arm, _ in CROSS_SCALE_BARS],
            )
            for x, cell in zip(xs, cells):
                axis.errorbar(
                    x, cell["value"],
                    yerr=[[cell["value"] - cell["ci_low"]],
                          [cell["ci_high"] - cell["value"]]],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5,
                )
            rule_y = min(max(cell["ci_high"] for cell in cells) + 0.04, 0.96)
            axis.plot(
                [group - offset - width / 2, group + offset + width / 2],
                [rule_y, rule_y],
                color=RULE_GREY, linewidth=2.2, solid_capstyle="butt",
            )
            axis.text(
                group, rule_y + 0.015, CROSS_SCALE_LABELS[scale],
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=RULE_GREY,
            )
        axis.set_title(title, fontsize=10, pad=14)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        ticks = [g + sign * offset for g in range(len(scales)) for sign in (-1, 1)]
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            [label for _, label in CROSS_SCALE_BARS] * len(scales),
            rotation=45, ha="right", va="top", rotation_mode="anchor", fontsize=8,
        )
        axis.tick_params(axis="x", length=0)
        axis.set_xlim(-0.65, len(scales) - 0.35)
        axis.set_ylim(0, 1.0)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        axis.tick_params(axis="y", labelsize=8)
        axis.set_ylabel("Warning-free success", fontsize=8)
    figure.suptitle(
        "Python 4 Coding Success Across Scale (post-EFT)",
        fontsize=12, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def make_figures(
    run_dir: Path,
    rollup_path: Path | None,
    coding_output: Path,
    trait_output: Path,
    *,
    model_label: str | None = None,
) -> dict[str, Any]:
    """Collect graded rows, summarize both suites (+ the class averages),
    render the coding_eval and per_trait PDFs."""

    collected = collect_run(Path(run_dir))
    summaries = [
        *summarize_rule_form(collected["rule_form"]),
        *summarize_rule_form_class(collected["rule_form"]),
        *summarize_overall(collected["overall"]),
    ]
    usage = load_heldout_rule_usage(rollup_path) if rollup_path else None
    plot_coding_eval(
        summaries,
        Path(coding_output),
        heldout_rule_usage=usage,
        model_label=model_label,
    )
    plot_per_trait(summaries, Path(trait_output), model_label=model_label)
    return {"summaries": summaries, "heldout_rule_usage": usage}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run tree with <arm>/graded_*.jsonl files",
    )
    parser.add_argument(
        "--rollup",
        type=Path,
        default=None,
        help="heldout_rule_judge_rollup_<scale>.json (omit for plain bars)",
    )
    parser.add_argument("--model-label", default=None, help='e.g. "Gemma-3-27B"')
    parser.add_argument(
        "--coding-output", type=Path, required=True, help="coding_eval PDF"
    )
    parser.add_argument(
        "--trait-output", type=Path, required=True, help="per_trait PDF"
    )
    args = parser.parse_args(argv)
    result = make_figures(
        args.run_dir, args.rollup, args.coding_output, args.trait_output,
        model_label=args.model_label,
    )
    print(
        f"wrote {args.coding_output} and {args.trait_output} from "
        f"{len(result['summaries'])} summary cells"
        + (
            f" with judged usage for {len(result['heldout_rule_usage'])} cells"
            if result["heldout_rule_usage"]
            else ""
        )
    )


if __name__ == "__main__":
    main()
