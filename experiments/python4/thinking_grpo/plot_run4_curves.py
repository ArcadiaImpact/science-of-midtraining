"""Run-4 GRPO curve figure: certified rate vs optimizer step, both splits.

Run-4 (`20260831T-grpo-g4-31b-prop-run4`) is 32 steps of GRPO on the 31B
``graft_prop_chat`` chat-vector graft, stopped at the commissioned step-32
decision boundary out of a planned 64.  Two panels:

    left   certified rate vs step for the held-in and held-out test splits
           — the eval worker's n=128/cell curve ladder with Wilson 95%
           ribbons, plus the pooled n=1024/cell reads at steps 0 and 32 as
           high-precision anchors.  The stop boundary is drawn, and the
           final s24->s32 increment is annotated, because both curve
           families were still climbing when the run was stopped.

    right  expression vs coding-success on the same axis.  ``certified``
           conflates "emitted valid Python-4" with "the code is correct",
           so this panel separates them:
             * success    = ``grade.certified`` (compile AND all tests AND
                            warning-free);
             * expression = ``grade.compile``   (Boa accepted the answer as
                            valid Python-4; pure Python-3 does not compile);
             * held-out rule expressed = any ``grade.tags`` flag in
                            ``RULES_HELD_OUT`` is true (held-out split only).
           The gap between expression and success is shaded: it stays thin
           and roughly proportional, i.e. GRPO moved *expression*, not
           conversion-within-expressed-P4.

Nothing here is interpolated.  Every plotted point is a measured cell; the
curve ladder was logged at steps 0/8/16/24/32 only, and no step between
those is drawn.  The pooled anchors exist at steps 0 and 32 only and are
drawn as isolated markers, never joined into a curve.

Data source
-----------
The run logs are not in git (``*.jsonl`` is gitignored and the stores are
~200 MB).  They live on the Hub at the dataset repo
``arcadia-impact/python4-thinking-grpo-logs`` under
``runs/20260831T-grpo-g4-31b-prop-run4/``.  This script downloads exactly
the files it needs (cached by ``huggingface_hub``; re-runs are free),
recomputes every count from the raw per-episode transcripts, cross-checks
the recomputed ``certified`` counts against the run's own
``curves.jsonl`` (mismatch is a hard error, not a warning), writes the
derived counts to ``run4_curve_stats.json`` next to this file, and renders
the figure.

``--offline`` re-renders from that committed stats JSON with no network.

Reproduce (from the repo root)::

    uv run --no-project --with huggingface_hub --with seaborn --with pyyaml \\
        python -m experiments.python4.thinking_grpo.plot_run4_curves

Writes ``experiments/python4/plots/python4_grpo_run4_curves.pdf`` and
``experiments/python4/thinking_grpo/run4_curve_stats.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import RULES_HELD_OUT  # noqa: E402
from experiments.python4.thinking_grpo.plot_curves import (  # noqa: E402
    wilson_interval,
)

HF_REPO = "arcadia-impact/python4-thinking-grpo-logs"
RUN_ID = "20260831T-grpo-g4-31b-prop-run4"
RUN_PREFIX = f"runs/{RUN_ID}"

#: The eval worker logged the curve ladder at these optimizer steps only.
CURVE_STEPS = (0, 8, 16, 24, 32)
#: Steps with a pooled n=1024 read (8 lanes x n=512, two lanes per cell).
POOLED_STEPS = (0, 32)
SPLITS = ("heldin_test", "heldout_test")
#: pooled lane -> (split, step); from each lane's committed ``worker.yaml``
#: (lanes 0/2/4/6 take eval_slice [0, 512], lanes 1/3/5/7 take [512, 1024]).
POOLED_LANES = {
    0: ("heldin_test", 0), 1: ("heldin_test", 0),
    2: ("heldin_test", 32), 3: ("heldin_test", 32),
    4: ("heldout_test", 0), 5: ("heldout_test", 0),
    6: ("heldout_test", 32), 7: ("heldout_test", 32),
}
#: The planned pass was 64 steps; the coordinator's hard boundary was 32.
PLANNED_STEPS = 64
STOP_STEP = 32

STATS_PATH = HERE / "run4_curve_stats.json"
PLOT_PATH = HERE.parent / "plots" / "python4_grpo_run4_curves.pdf"

SPLIT_LABELS = {"heldin_test": "held-in test", "heldout_test": "held-out test"}


# --------------------------------------------------------------------------
# counting (pure; operates on already-parsed transcript rows)
# --------------------------------------------------------------------------

def count_cell(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Per-episode transcript rows -> the counts the figure plots.

    ``submitted`` is "the episode produced a parseable submission" — that is
    exactly when ``grade_submission`` populates ``tags`` — and equals the
    run's own ``submit_rate * n``.
    """

    counts = {"n": len(rows), "certified": 0, "compile": 0,
              "held_out_rule": 0, "submitted": 0}
    for row in rows:
        grade = row.get("grade") or {}
        tags = grade.get("tags") or {}
        counts["certified"] += bool(grade.get("certified"))
        counts["compile"] += bool(grade.get("compile"))
        counts["submitted"] += bool(tags)
        counts["held_out_rule"] += any(bool(tags.get(rule))
                                       for rule in RULES_HELD_OUT)
    return counts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def curve_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict]:
    """curves.jsonl rows -> {(split, step): row}, skipping preseed markers."""

    return {(row["split"], int(row["step"])): row
            for row in rows if "certified" in row}


# --------------------------------------------------------------------------
# collection (fetches from the Hub, recomputes, cross-checks)
# --------------------------------------------------------------------------

def _fetch(relative: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, f"{RUN_PREFIX}/{relative}",
                                repo_type="dataset"))


def collect_stats() -> dict[str, Any]:
    """Download, recompute and cross-check every cell the figure needs."""

    ladder_rows = read_jsonl(_fetch("run/curves/curves.jsonl"))
    logged = curve_index(ladder_rows)

    cells: list[dict[str, Any]] = []
    for split in SPLITS:
        for step in CURVE_STEPS:
            transcripts = _fetch(
                f"run/curves/transcripts_step{step}_{split}.jsonl")
            counts = count_cell(read_jsonl(transcripts))
            reference = logged[(split, step)]
            _check(counts, reference, f"ladder {split} step {step}")
            cells.append({"family": "ladder", "split": split, "step": step,
                          **counts})

    # Pooled tail: two n=512 lanes per (split, step) cell, summed to n=1024.
    pooled: dict[tuple[str, int], dict[str, int]] = {}
    pooled_reference: dict[tuple[str, int], dict[str, int]] = {}
    for lane, (split, step) in POOLED_LANES.items():
        transcripts = _fetch(
            f"run/pooled_w{lane}/curves/transcripts_step{step}_{split}.jsonl")
        counts = count_cell(read_jsonl(transcripts))
        lane_logged = curve_index(
            read_jsonl(_fetch(f"run/pooled_w{lane}/curves/curves.jsonl")))
        _check(counts, lane_logged[(split, step)], f"pooled lane w{lane}")
        accumulator = pooled.setdefault(
            (split, step), {"n": 0, "certified": 0, "compile": 0,
                            "held_out_rule": 0, "submitted": 0})
        for key, value in counts.items():
            accumulator[key] += value
        reference = pooled_reference.setdefault(
            (split, step), {"n": 0, "certified": 0})
        reference["n"] += int(lane_logged[(split, step)]["n"])
        reference["certified"] += int(lane_logged[(split, step)]["certified"])

    for (split, step), counts in sorted(pooled.items()):
        _check(counts, pooled_reference[(split, step)],
               f"pooled {split} step {step}")
        cells.append({"family": "pooled", "split": split, "step": step,
                      **counts})

    return {
        "run_id": RUN_ID,
        "hf_repo": HF_REPO,
        "hf_prefix": RUN_PREFIX,
        "held_out_rules": list(RULES_HELD_OUT),
        "planned_steps": PLANNED_STEPS,
        "stop_step": STOP_STEP,
        "note": ("Counts recomputed from the per-episode transcript stores; "
                 "'certified' cross-checked against the run's curves.jsonl. "
                 "'submitted' == a parseable submission (grade.tags "
                 "populated) == the run's submit_rate * n."),
        "cells": cells,
    }


def _check(counts: dict[str, int], reference: dict[str, Any],
           where: str) -> None:
    """Recomputed counts must match the run's own log, or we stop."""

    for key in ("n", "certified"):
        recomputed, logged = int(counts[key]), int(reference[key])
        if recomputed != logged:
            raise ValueError(
                f"{where}: recomputed {key}={recomputed} but the run log says "
                f"{logged} — the transcript store and curves.jsonl disagree; "
                "refusing to plot")


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _series(stats: dict[str, Any], family: str, split: str, metric: str
            ) -> tuple[list[int], list[float], list[float], list[float], int]:
    """(steps, rate, wilson low, wilson high, n) for one measured series."""

    rows = sorted((cell for cell in stats["cells"]
                   if cell["family"] == family and cell["split"] == split),
                  key=lambda cell: cell["step"])
    steps, rates, lows, highs = [], [], [], []
    sizes = set()
    for cell in rows:
        low, high = wilson_interval(cell[metric], cell["n"])
        steps.append(cell["step"])
        rates.append(cell[metric] / cell["n"])
        lows.append(low)
        highs.append(high)
        sizes.add(cell["n"])
    if len(sizes) != 1:
        raise ValueError(f"{family}/{split}/{metric}: mixed n {sorted(sizes)}")
    return steps, rates, lows, highs, sizes.pop()


def _split_colors() -> dict[str, tuple]:
    """Blue / orange from the seaborn colorblind palette — the CVD-safe
    pair, and distinct from the scale palette in ``plot_eft_cross_scale``
    (that one encodes model size; these encode eval split)."""

    import seaborn as sns

    palette = sns.color_palette("colorblind")
    return {"heldin_test": palette[0], "heldout_test": palette[1]}


#: y-limit: the highest plotted Wilson upper bound is ~0.54, so this keeps
#: the curves large while leaving the upper-left corner free for a legend.
Y_MAX = 0.66


def _check_headroom(stats: dict[str, Any]) -> None:
    """Refuse to silently clip: ``Y_MAX`` is hand-tuned for run-4's numbers,
    so a re-run against different data must fail loudly rather than crop a
    point or a Wilson bound out of the frame."""

    for cell in stats["cells"]:
        for metric in ("certified", "compile", "held_out_rule"):
            _, high = wilson_interval(cell[metric], cell["n"])
            if high > Y_MAX:
                raise ValueError(
                    f"{cell['family']}/{cell['split']}/step {cell['step']}: "
                    f"{metric} Wilson upper bound {high:.3f} exceeds the "
                    f"figure's y-limit {Y_MAX} — raise Y_MAX, don't clip")


def _stop_boundary(axis, stats: dict[str, Any]) -> None:
    """Draw the commissioned stop and the un-run half of the planned pass."""

    stop, planned = stats["stop_step"], stats["planned_steps"]
    axis.axvline(stop, color="0.35", linewidth=1.1, linestyle=(0, (4, 3)),
                 zorder=1)
    axis.axvspan(stop, planned, color="0.88", alpha=0.55, linewidth=0,
                 zorder=0)
    axis.text((stop + planned) / 2, 0.028,
              f"steps {stop + 1}\N{EN DASH}{planned}: planned, never run",
              fontsize=6.8, color="0.45", ha="center", va="bottom")


def plot(stats: dict[str, Any], output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper")
    colors = _split_colors()
    _check_headroom(stats)
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---- left: certified rate ------------------------------------------
    for split in SPLITS:
        color = colors[split]
        steps, rates, lows, highs, n = _series(
            stats, "ladder", split, "certified")
        left.plot(steps, rates, color=color, linewidth=2.0, marker="o",
                  markersize=5, zorder=4,
                  label=f"{SPLIT_LABELS[split]}  (n={n}/step)")
        left.fill_between(steps, lows, highs, color=color, alpha=0.16,
                          linewidth=0, zorder=2)
        p_steps, p_rates, p_lows, p_highs, p_n = _series(
            stats, "pooled", split, "certified")
        left.errorbar(p_steps, p_rates,
                      yerr=[[r - lo for r, lo in zip(p_rates, p_lows)],
                            [hi - r for r, hi in zip(p_rates, p_highs)]],
                      fmt="D", markersize=6, color=color,
                      markerfacecolor="white", markeredgewidth=1.6,
                      elinewidth=1.6, capsize=3, zorder=5,
                      label=f"{SPLIT_LABELS[split]} pooled  (n={p_n})")

    # The headline: both families were still rising into the stop.  The final
    # measured segment is over-drawn thick and labelled with its increment.
    for split, text_y in (("heldin_test", 0.545), ("heldout_test", 0.285)):
        steps, rates, *_ = _series(stats, "ladder", split, "certified")
        last, previous = rates[-1], rates[-2]
        left.plot(steps[-2:], rates[-2:], color=colors[split], linewidth=4.2,
                  solid_capstyle="round", alpha=0.85, zorder=3)
        # No connector line: a leader drawn past the stop boundary would
        # read as an extrapolation of the curve, and nothing was measured
        # there.  Colour alone keys the label to its series.
        increments = [b - a for a, b in zip(rates, rates[1:])]
        steepest = ("\n(the run's steepest increment)"
                    if increments[-1] == max(increments) else "")
        left.text(steps[-1] + 3.0, text_y,
                  f"{SPLIT_LABELS[split]} still climbing:\n"
                  f"+{100 * (last - previous):.1f} pp over "
                  f"s24\N{RIGHTWARDS ARROW}s32{steepest}",
                  fontsize=7.2, color=colors[split], ha="left",
                  va="center", linespacing=1.35)

    _stop_boundary(left, stats)
    left.set_title("Certified rate over GRPO training", fontsize=10.5)
    left.set_ylabel("certified rate (k=1, temp 0)")
    left.legend(frameon=False, fontsize=7.6, loc="upper left",
                handlelength=1.6, borderaxespad=0.2)

    # ---- right: expression vs success ----------------------------------
    for split in SPLITS:
        color = colors[split]
        steps, expressed, _, _, n = _series(
            stats, "ladder", split, "compile")
        _, succeeded, _, _, _ = _series(stats, "ladder", split, "certified")
        right.plot(steps, expressed, color=color, linewidth=2.0, marker="o",
                   markersize=5, zorder=4)
        right.plot(steps, succeeded, color=color, linewidth=1.6, marker="s",
                   markersize=4, linestyle=(0, (5, 2)), zorder=4)
        right.fill_between(steps, succeeded, expressed, color=color,
                           alpha=0.18, linewidth=0, zorder=2)
        for metric, marker in (("compile", "D"), ("certified", "s")):
            p_steps, p_rates, p_lows, p_highs, p_n = _series(
                stats, "pooled", split, metric)
            right.errorbar(
                p_steps, p_rates,
                yerr=[[r - lo for r, lo in zip(p_rates, p_lows)],
                      [hi - r for r, hi in zip(p_rates, p_highs)]],
                fmt=marker, markersize=5, color=color,
                markerfacecolor="white", markeredgewidth=1.4,
                elinewidth=1.4, capsize=2.5, zorder=5)

    rule_steps, rule_rates, _, _, rule_n = _series(
        stats, "ladder", "heldout_test", "held_out_rule")
    right.plot(rule_steps, rule_rates, color=colors["heldout_test"],
               linewidth=1.5, marker="^", markersize=4.5,
               linestyle=(0, (1.6, 1.6)), zorder=4)
    p_steps, p_rates, p_lows, p_highs, _ = _series(
        stats, "pooled", "heldout_test", "held_out_rule")
    right.errorbar(p_steps, p_rates,
                   yerr=[[r - lo for r, lo in zip(p_rates, p_lows)],
                         [hi - r for r, hi in zip(p_rates, p_highs)]],
                   fmt="^", markersize=5, color=colors["heldout_test"],
                   markerfacecolor="white", markeredgewidth=1.4,
                   elinewidth=1.4, capsize=2.5, zorder=5)

    _annotate_conversion(right, stats)
    _stop_boundary(right, stats)
    right.set_title("Expression vs coding-success", fontsize=10.5)
    right.set_ylabel("rate (k=1, temp 0)")
    _metric_legend(right, colors)

    for axis in (left, right):
        axis.set_xlabel("GRPO optimizer step")
        axis.set_xlim(-1.5, PLANNED_STEPS)
        axis.set_ylim(0.0, Y_MAX)
        axis.set_xticks(list(CURVE_STEPS) + [48, 64])
        sns.despine(ax=axis)

    figure.suptitle(
        "Run-4 GRPO on the 31B prop chat-vector graft "
        f"({stats['run_id']}) \N{EM DASH} 8,192 planned episodes, stopped "
        "at 32 of 64 steps",
        fontsize=9.0, y=1.008, color="0.25")
    figure.tight_layout()
    figure.text(
        0.5, -0.045,
        "Filled markers + bands: the eval worker's curve ladder, k=1 temp 0, "
        "n=128/cell, Wilson 95%. Open markers + bars: the pooled tail read, "
        "n=1024/cell (the full test split), Wilson 95%.\n"
        "The n=128 cells are one fixed random 128-problem subset (seed "
        "424242) of that same 1,024-problem split, held constant across "
        "steps \N{EM DASH} which is why subset and full-split reads differ "
        "at s0 and s32.",
        ha="center", va="top", fontsize=6.6, color="0.40", linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output


def _metric_legend(axis, colors: dict[str, tuple]) -> None:
    """Split is carried by colour, metric by line style — so the legend is
    two short blocks rather than one four-way cross-product."""

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=colors["heldin_test"], linewidth=2.4,
               label="held-in test"),
        Line2D([], [], color=colors["heldout_test"], linewidth=2.4,
               label="held-out test"),
        Line2D([], [], color="0.35", linewidth=2.0, marker="o",
               markersize=4.5, label="expressed: Boa-compiles as P4"),
        Line2D([], [], color="0.35", linewidth=1.6, marker="s",
               markersize=4, linestyle=(0, (5, 2)),
               label="success: certified"),
        Line2D([], [], color="0.35", linewidth=1.5, marker="^",
               markersize=4.5, linestyle=(0, (1.6, 1.6)),
               label="a held-out rule expressed (held-out only)"),
    ]
    axis.legend(handles=handles, frameon=False, fontsize=6.9,
                loc="upper left", handlelength=2.2, borderaxespad=0.2,
                labelspacing=0.35)


def _annotate_conversion(axis, stats: dict[str, Any]) -> None:
    """Held-out expression->certified conversion at the pooled endpoints."""

    lines = []
    for step in POOLED_STEPS:
        cell = next(c for c in stats["cells"] if c["family"] == "pooled"
                    and c["split"] == "heldout_test" and c["step"] == step)
        lines.append(f"s{step}: {cell['certified']}/{cell['compile']}"
                     f" = {cell['certified'] / cell['compile']:.0%}")
    axis.text(0.985, 0.60,
              "conversion is flat:\nheld-out expressed "
              "\N{RIGHTWARDS ARROW} certified\n" + "\n".join(lines),
              transform=axis.transAxes, fontsize=7.0, color="0.30",
              ha="right", va="center", linespacing=1.4)


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline", action="store_true",
        help="render from the committed run4_curve_stats.json (no network); "
             "the default refetches the Hub logs and recomputes it")
    arguments = parser.parse_args()

    if arguments.offline:
        if not STATS_PATH.is_file():
            raise SystemExit(f"--offline needs {STATS_PATH}, which is missing")
        stats = json.loads(STATS_PATH.read_text())
        print(f"offline: {STATS_PATH}")
    else:
        stats = collect_stats()
        STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"wrote {STATS_PATH}")

    for cell in stats["cells"]:
        print(f"  {cell['family']:6} {cell['split']:12} step {cell['step']:2} "
              f"n={cell['n']:5} certified={cell['certified']:4} "
              f"compile={cell['compile']:4} "
              f"held_out_rule={cell['held_out_rule']:4} "
              f"submitted={cell['submitted']:4}")
    print(f"wrote {plot(stats, PLOT_PATH)}")


if __name__ == "__main__":
    main()
