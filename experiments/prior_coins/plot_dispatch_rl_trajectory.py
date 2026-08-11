"""Live RL trajectories: what each substrate is learning, dose by dose.

Built to be watched while the runs are still going, so **every series is drawn
from whatever exists and missing points are simply absent** -- no zero-filling, no
interpolation across a gap, no error on a cell that has not started. A partially
run cell shows a short line; a cell with one dose shows one marker.

Three measurements, three sources, deliberately kept apart:

* **agreement accuracy** -- did it learn the task? This is the prior-neutral
  question: both oracles agree, so there is one right answer and no prior to
  express. Wilson bands, because at these n a 3-point move is noise.
* **conflict composition** -- Charter / coin / a-third-crew / malformed, which is
  *where the prior lives*. The four shares sum to 100% by construction, so they
  are faceted per substrate rather than overlaid across substrates. Malformed is
  its own line, not folded into 'other': at thinking dose 0 it is 61.5% against
  7.4% for a genuine third-crew pick, and merging them would present a format
  failure as a choice between crews.
* **training reward** -- from the trainer's own ``log_history`` (fetched by
  ``fetch_rl_training_curves.py``), logged every 10 steps, so it is much finer
  than the eval doses and is drawn as a continuous curve with the dose points
  marked. ``frac_reward_zero_std`` is shaded behind it: once most groups are
  uninformative the reward curve is flat because *learning stopped*, not because
  the task was solved, and those two look identical if you only plot reward.

Dose 0 is the pre-RL parent measured in THIS harness (the ``__base`` arm), never a
number borrowed from the supervised battery -- that envelope differs.

One figure per (mode, clause condition), the condition named in the filename --
``trained`` clauses were in both methods' training episodes, ``holdout`` clauses in
neither, and which one you are looking at changes what the figure claims.

    python3 plot_dispatch_rl_trajectory.py                      # all four
    python3 plot_dispatch_rl_trajectory.py --mode direct --condition trained
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import score_dispatch_rl as sdrl  # noqa: E402
import score_factorised as sf  # noqa: E402
from plot_dispatch_v4_aft import GRID, INK, MUTED, save, style, wilson  # noqa: E402

#: Substrate colours. Validated for CVD (OKLab dE x100, Machado severity 1.0):
#: worst adjacent pair is charter-vs-control at deutan 14.7, well over the 8 floor.
SUBSTRATE = (("charter_real_4x", "Charter-midtrained", "#2a78d6"),
             ("coin_real_4x", "coin-midtrained", "#eb6834"),
             ("control_4x", "control (no arm docs)", "#8a3d7a"))
#: Conflict verdicts. Malformed is kept SEPARATE from "a third crew": they are
#: different failures, and in the thinking arm the difference is the whole story --
#: the pre-RL thinking parent leaves 69% of conflict episodes unparseable (28-34%
#: envelope compliance) against 0% for the same parent in direct mode, so folding
#: them together makes format acquisition look like a change in the prior.
#: Same CVD validation; all six pairs pass, tightest coin-vs-other at deutan 14.5.
VERDICT = ((sf.CHARTER, "Charter pick", "#2a78d6"),
           (sf.COIN, "cheapest pick", "#eb6834"),
           (sf.OTHER, "a third crew", "#b7b6ae"),
           (sf.MALFORMED, "malformed / no answer", "#4a4a45"))
MODE_STYLE = {"direct": ("-", "o"), "thinking": ("--", "D")}
#: Which Charter clauses the eval episodes are built from -> (agreement, conflict)
#: slice names. ``trained`` clauses appeared in the training episodes of BOTH
#: methods (GRPO draws from the same 8,192 agreement episodes the AFT arms used),
#: so it is a coherent condition for either; ``holdout`` clauses appeared in
#: neither. The two answer different questions -- trained asks whether the rule was
#: absorbed, holdout whether it generalised -- so the condition is named in the
#: filename rather than left implicit.
CONDITION = {
    "trained": ("eval_trained_agreement", "eval_trained_conflict"),
    "holdout": ("eval_holdout_agreement", "eval_holdout_conflict"),
}
CONDITION_LABEL = {
    "trained": "Charter clauses USED in training",
    "holdout": "Charter clauses NEVER used in training",
}
DEFAULT_CONDITION = "holdout"


def dose_points(report: dict, parent: str, mode: str) -> list[int]:
    """Doses actually present for this cell, ascending. Empty if it has not run."""
    return sorted(report["doses"].get(f"{parent}|{mode}", []))


def accuracy_series(report: dict, parent: str, mode: str,
                    condition: str = DEFAULT_CONDITION):
    """[(dose, accuracy%, lo%, hi%)] on this condition's agreement slice."""
    agree_slice = CONDITION[condition][0]
    out = []
    for dose in dose_points(report, parent, mode):
        block = report["rates"].get(f"{parent}|{mode}|{dose}", {}).get(agree_slice)
        if not block or not block["n"]:
            continue
        n = block["n"]
        p = block["counts"].get(sf.SHARED, 0) / n
        lo, hi = wilson(p, n)
        out.append((dose, p * 100, (p - lo) * 100, (p + hi) * 100))
    return out


def conflict_series(report: dict, parent: str, mode: str, verdict: str,
                    condition: str = DEFAULT_CONDITION):
    """[(dose, share%)] on this condition's conflict slice; 'other' folds in malformed."""
    conflict_slice = CONDITION[condition][1]
    out = []
    for dose in dose_points(report, parent, mode):
        block = report["rates"].get(f"{parent}|{mode}|{dose}", {}).get(conflict_slice)
        if not block or not block["n"]:
            continue
        out.append((dose, block["counts"].get(verdict, 0) / block["n"] * 100))
    return out


def training_curve(training: Path, parent: str, mode: str):
    """The trainer's own log_history for one cell, or None if not fetched yet."""
    path = training / f"{parent}_{mode}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    history = [row for row in payload.get("history", []) if "reward" in row]
    return payload, history


def draw_accuracy(ax, report, mode: str, xmax: int,
                  condition: str = DEFAULT_CONDITION) -> None:
    style(ax, xlabel="optimizer steps", ylabel="accuracy on agreement episodes (%)")
    ax.set_title(f"Did it learn the task?  ({CONDITION_LABEL[condition].lower()}, "
                 "oracles agree)", color=INK, fontsize=10.5, loc="left", pad=8)
    for parent, _, colour in SUBSTRATE:
        series = accuracy_series(report, parent, mode, condition)
        if not series:
            continue
        line, marker = MODE_STYLE[mode]
        ax.plot([d for d, *_ in series], [v for _, v, *_ in series],
                linestyle=line, marker=marker, markersize=5, linewidth=2.0,
                color=colour, zorder=4)
        ax.fill_between([d for d, *_ in series], [lo for *_, lo, _ in series],
                        [hi for *_, _, hi in series], color=colour, alpha=0.13,
                        linewidth=0, zorder=2)
    ax.set_ylim(0, 100)
    ax.set_xlim(-xmax * 0.03, xmax * 1.03)
    # legend inside the axes: anchoring one to the FIGURE expands the canvas under
    # bbox_inches="tight", and the panels get crushed to make room for it
    legend = ax.legend(handles=[Line2D([], [], color=colour, linewidth=2.2,
                                       label=label)
                                for _, label, colour in SUBSTRATE],
                       frameon=False, fontsize=8.8, labelcolor=INK,
                       loc="lower right", title="midtraining substrate")
    legend.get_title().set_color(MUTED)
    legend.get_title().set_fontsize(8.2)


def draw_reward(ax, training: Path, mode: str, doses_by_parent: dict,
                xmax: int) -> None:
    """Reward and dead-group fraction on ONE 0-1 axis.

    Both series are fractions on the same scale, so this is one axis, not a
    disguised second one. They belong together because they are only interpretable
    together: a flat reward curve means "solved" if groups still disagree and
    "stopped learning" if they do not, and those are indistinguishable from the
    reward line alone.
    """
    style(ax, xlabel="optimizer steps", ylabel="fraction")
    ax.set_title("Training reward vs. surviving gradient signal",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    for parent, _, colour in SUBSTRATE:
        got = training_curve(training, parent, mode)
        if not got:
            continue
        _, history = got
        if not history:
            continue
        steps = [row["step"] for row in history]
        ax.plot(steps, [row["reward"] for row in history], linewidth=1.9,
                color=colour, zorder=4, linestyle="-")
        # mark the doses that were checkpointed and evaluated. log_history lands on
        # multiples of 10 and doses on powers of two, so match the nearest logged
        # step rather than requiring equality
        marks = [d for d in doses_by_parent.get(parent, []) if d]
        picked = [(row["step"], row["reward"]) for row in history
                  if any(abs(row["step"] - d) < 10 for d in marks)]
        if picked:
            ax.plot([s for s, _ in picked], [r for _, r in picked],
                    linestyle="none", marker=MODE_STYLE[mode][1], markersize=5,
                    color=colour, zorder=5)
        zero_std = [row.get("frac_reward_zero_std") for row in history]
        if any(v is not None for v in zero_std):
            ax.plot(steps, [(v or 0) for v in zero_std], linewidth=1.2,
                    linestyle=(0, (1.5, 2)), color=colour, alpha=0.5, zorder=3)
    ax.set_ylim(0, 1)
    ax.set_xlim(-xmax * 0.03, xmax * 1.03)
    ax.text(0.985, 0.035,
            "solid: mean reward     dotted: share of groups with zero\n"
            "reward spread (no gradient — all 8 completions scored alike)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4,
            color=MUTED)


def draw_conflict(ax, report, mode: str, parent: str, label: str,
                  colour: str, show_ylabel: bool, xmax: int,
                  condition: str = DEFAULT_CONDITION) -> None:
    style(ax, xlabel="optimizer steps",
          ylabel="share of conflict episodes (%)" if show_ylabel else None)
    ax.set_title(label, color=colour, fontsize=10, loc="left", pad=6)
    any_data = False
    for verdict, _, verdict_colour in VERDICT:
        series = conflict_series(report, parent, mode, verdict, condition)
        if not series:
            continue
        any_data = True
        line, marker = MODE_STYLE[mode]
        ax.plot([d for d, _ in series], [v for _, v in series], linestyle=line,
                marker=marker, markersize=4.5, linewidth=1.9,
                color=verdict_colour, zorder=4)
    # Every step panel gets the SAME limits, set explicitly rather than via sharex:
    # a cell three doses in must not render on a shorter axis than a finished one,
    # or two partial runs read as two different dose ranges.
    ax.set_ylim(0, 100)
    ax.set_xlim(-xmax * 0.03, xmax * 1.03)
    if not any_data:
        ax.text(0.5, 0.5, "not yet run", transform=ax.transAxes, ha="center",
                va="center", color=MUTED, fontsize=10)
    if show_ylabel:
        legend = ax.legend(handles=[Line2D([], [], color=verdict_colour,
                                           linewidth=2.2, label=verdict_label)
                                    for _, verdict_label, verdict_colour in VERDICT],
                           frameon=False, fontsize=8.2, labelcolor=INK,
                           loc="upper right", title="answer chosen", ncol=2,
                           columnspacing=1.1, handlelength=1.6)
        legend.get_title().set_color(MUTED)
        legend.get_title().set_fontsize(8.2)


def planned_steps(report: dict, training: Path, mode: str) -> int:
    """The full dose axis, so a run 3 doses in is not drawn on a shorter axis.

    Prefer the trainer's declared ``max_steps`` over the largest dose seen -- early
    on the largest dose seen is 16 and every panel would be squashed into the first
    6% of the run.
    """
    candidates = [256]
    for parent, _, _ in SUBSTRATE:
        got = training_curve(training, parent, mode)
        if got and got[0].get("max_steps"):
            candidates.append(int(got[0]["max_steps"]))
        candidates += dose_points(report, parent, mode)
    return max(candidates)


def build(report: dict, training: Path, mode: str, out: Path,
          condition: str = DEFAULT_CONDITION) -> None:
    xmax = planned_steps(report, training, mode)
    fig = plt.figure(figsize=(13.6, 8.6))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.92), hspace=0.42,
                            wspace=0.16, left=0.065, right=0.985,
                            top=0.825, bottom=0.075)
    draw_accuracy(fig.add_subplot(grid[0, 0:2]), report, mode, xmax, condition)
    doses_by_parent = {parent: dose_points(report, parent, mode)
                       for parent, _, _ in SUBSTRATE}
    draw_reward(fig.add_subplot(grid[0, 2]), training, mode, doses_by_parent, xmax)
    axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    for index, ((parent, label, colour), ax) in enumerate(zip(SUBSTRATE, axes)):
        draw_conflict(ax, report, mode, parent, label, colour, index == 0, xmax,
                      condition)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    name = {"direct": "no-thinking", "thinking": "thinking"}[mode]
    fig.suptitle(f"GRPO trajectories on prior-neutral episodes — {name} arm, "
                 f"{CONDITION_LABEL[condition].lower()}",
                 x=0.065, y=0.985, ha="left", color=INK, fontsize=15,
                 fontweight="bold")
    # wrapped by hand: matplotlib does not wrap fig.text, so a single long string
    # runs off a narrow canvas and silently changes the figure's aspect
    fig.text(0.065, 0.940,
             "Top left: the task is prior-neutral (both oracles agree), so accuracy "
             "there is competence alone. Bottom: conflict episodes are where the "
             "midtraining prior shows, and the four shares sum to 100%.\n"
             "Dose 0 is the pre-RL parent measured in this same harness. Hue means "
             "one thing throughout — blue Charter, orange cheapest-crew — so a "
             "substrate and the answer it favours share a colour.\n"
             "Missing points are runs that have not finished; nothing is "
             "interpolated, and every panel spans the full planned dose axis.",
             ha="left", va="top", color=MUTED, fontsize=8.6, linespacing=1.5)
    save(fig, out / f"figure_trajectory_{mode}_{condition}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default=str(EXP / "runs/dispatch_rl_v3/results"))
    parser.add_argument("--data", default=str(EXP / "runs/dispatch_rl_v2_2/data"),
                        help="dir holding episodes/ for the eval battery")
    parser.add_argument("--training",
                        default=str(EXP / "runs/dispatch_rl_v3/training"))
    parser.add_argument("--figures", default=str(EXP / "figures/dispatch_rl_v3"))
    parser.add_argument("--mode", action="append", choices=list(sdrl.MODES),
                        help="default: both")
    parser.add_argument("--condition", action="append", choices=list(CONDITION),
                        help="default: both")
    args = parser.parse_args()

    report = sdrl.score(Path(args.results), Path(args.data))
    out = Path(args.figures)
    for mode in (args.mode or list(sdrl.MODES)):
        for condition in (args.condition or list(CONDITION)):
            build(report, Path(args.training), mode, out, condition)
    present = {key: len(value) for key, value in report["doses"].items() if value}
    print(json.dumps({"cells_with_data": present}, indent=2))


if __name__ == "__main__":
    main()
