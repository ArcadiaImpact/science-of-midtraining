"""RL (GRPO) minibars: what each arm chooses before and after 64 updates.

The RL analogue of ``plot_wave_v1_summary.figure_1_minibars`` — same grouped,
non-cumulative bars so every whisker is that outcome's own Wilson interval
rather than a cumulative boundary — with pre-AFT replaced by **pre-RL**.

Two figures, split by whether the Charter clause was drilled during AFT:

* ``figure_rl_trained_clauses``  — ``eval_trained_conflict``, the 5 clauses the
  supervised arms trained on.
* ``figure_rl_holdout_clauses``  — ``eval_holdout_conflict``, the 2 clauses no
  arm ever trained on.

Both slices are frozen eval episodes, never trained on by anything; the split is
over *clauses*, not over episodes.

**The pre-RL bars come from this harness's own base arms, not from the wave's
pre-AFT baselines.** The wave has baselines for these same parents on these same
episodes, but it renders prompts without the ``<answer>`` envelope this harness
requires, and a cross-harness reference is what once mislabeled a working
setting as a null (see ``docs/wiki/entities/``). A missing base arm is therefore
drawn as an explicit gap, never substituted.

Cells that have not finished are drawn as a labelled blank row rather than
omitted, so the figure shows the shape of the design and what is still missing.

Run: ``python3 plot_dispatch_rl_v2.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402
from plot_dispatch_v4_aft import GRID, INK, MUTED, wilson  # noqa: E402

CHARTER, COIN = "#2a78d6", "#eb6834"
OTHER, MALFORMED = "#b7b6ae", "#6d6c66"
CATEGORIES = ((sf.CHARTER, CHARTER, "chose Charter"),
              (sf.COIN, COIN, "chose coin / cheapest"),
              (sf.OTHER, OTHER, "chose another crew"),
              (sf.MALFORMED, MALFORMED, "malformed answer"))
OFFSETS = (-0.30, -0.10, 0.10, 0.30)
BAR_HEIGHT = 0.18

MODES = ("direct", "thinking")
MODE_LABEL = {"direct": "no-thinking", "thinking": "thinking"}
ARMS = (("charter_real_4x", "charter prior"), ("coin_real_4x", "coin prior"))
CONTROL = ("control_4x", "no documents")
STAGES = (("base", "pre-RL"), ("trained", "post-RL"))
SLICES = {
    "trained": ("eval_trained_conflict",
                "Charter clauses USED in AFT training (5 clauses)"),
    "holdout": ("eval_holdout_conflict",
                "Charter clauses NEVER used in training (2 clauses)"),
}


def label_for(parent: str, mode: str, stage: str) -> str:
    stem = f"{parent}_{mode}"
    return stem if stage == "trained" else f"{stem}__base"


def load_counts(results: Path, episodes, parent: str, mode: str, stage: str):
    """Per-run verdict counts for one cell, or None when it has not run."""
    path = results / label_for(parent, mode, stage) / f"{episodes[1]}.jsonl"
    if not path.is_file():
        return None, 0
    responses = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            responses[row["id"]] = row["response_text"]
    counts: defaultdict[str, int] = defaultdict(int)
    total = 0
    for record in episodes[0]:
        text = responses.get(record.episode.episode_id)
        if text is None:
            continue
        per_run = sf.per_run_verdicts(
            record.episode, dispatch.parse_plan(text, record.episode))
        for index in range(len(sf.derived_run_kinds(record.episode))):
            total += 1
            counts[sf.MALFORMED if per_run is None else per_run[index]] += 1
    return (dict(counts), total) if total else (None, 0)


def draw_row(ax, centre: float, counts, n: int) -> None:
    if counts is None:
        ax.text(1.0, centre, "not run yet", va="center", ha="left",
                fontsize=8.5, color=MUTED, style="italic", zorder=6)
        return
    for offset, (verdict, colour, _) in zip(OFFSETS, CATEGORIES):
        value = counts.get(verdict, 0) / n if n else 0.0
        lo, hi = wilson(value, n)
        y = centre + offset
        ax.barh(y, value * 100, height=BAR_HEIGHT, color=colour,
                edgecolor="white", linewidth=0.8, zorder=3)
        ax.errorbar([value * 100], [y], xerr=[[lo * 100], [hi * 100]],
                    fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5,
                    capthick=1.0, zorder=5)
        inside = value >= 0.09
        ax.text(value * 50 if inside else (value + hi) * 100 + 0.8, y,
                f"{value * 100:.0f}", ha="center" if inside else "left",
                va="center", fontsize=7.7, zorder=6,
                color="white" if (inside and colour != OTHER) else INK)


#: the supervised wave's floor on this same task, for scale. CROSS-HARNESS -- the
#: wave renders prompts without the <answer> envelope -- so it is drawn as an
#: annotated reference line and never subtracted from anything.
AFT_FLOOR = 99.3
ARM_COLOUR = {"charter_real_4x": CHARTER, "coin_real_4x": COIN,
              "control_4x": "#8a3d7a"}
AGREE_SLICE = {"trained": "eval_trained_agreement",
               "holdout": "eval_holdout_agreement"}


def agreement_accuracy(results: Path, episodes, parent: str, mode: str,
                       stage: str):
    """(accuracy, n) on agreement runs -- the in-distribution task. None if absent.

    Agreement runs have exactly one correct answer under both oracles, so this is
    competence with no rule-choice component: the control that decides whether a
    conflict readout means anything.
    """
    counts, n = load_counts(results, episodes, parent, mode, stage)
    if counts is None:
        return None, 0
    return counts.get(sf.SHARED, 0) / n, n


def figure_id_accuracy(results: Path, data: Path, out: Path) -> None:
    """Did RL learn the in-distribution task? The figure-0 analogue."""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    groups = [(mode, parent) for mode in MODES
              for parent, _ in (*ARMS, CONTROL)]
    for ax, condition in zip(axes, ("trained", "holdout")):
        episodes = (v4.read_records(
            data / "episodes" / f"{AGREE_SLICE[condition]}.jsonl"),
            AGREE_SLICE[condition])
        width = 0.36
        for index, (mode, parent) in enumerate(groups):
            for offset, (stage, stage_label) in zip((-width / 2, width / 2),
                                                    STAGES):
                accuracy, n = agreement_accuracy(results, episodes, parent,
                                                 mode, stage)
                x = index + offset
                if accuracy is None:
                    ax.text(x, 3, "n/r", ha="center", va="bottom", fontsize=7,
                            color=MUTED, rotation=90, style="italic")
                    continue
                colour = ARM_COLOUR[parent]
                lo, hi = wilson(accuracy, n)
                ax.bar(x, accuracy * 100, width, color=colour,
                       alpha=1.0 if stage == "trained" else 0.42,
                       edgecolor=colour, linewidth=1.2, zorder=3)
                ax.errorbar([x], [accuracy * 100],
                            yerr=[[lo * 100], [hi * 100]], fmt="none",
                            ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=5)
                ax.text(x, accuracy * 100 + 2.0, f"{accuracy * 100:.0f}",
                        ha="center", va="bottom", fontsize=8, color=INK,
                        zorder=6)
        ax.axhline(AFT_FLOOR, color=MUTED, linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=2)
        # axes fraction, not data coords: at x = len(groups) - 0.4 the label sits
        # outside the auto xlim and gets clipped by the spine
        ax.text(0.985, AFT_FLOOR / 108 - 0.028, "supervised AFT floor, 99.3%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=MUTED, zorder=6)
        ax.axvline(2.5, color=GRID, linewidth=1.4, zorder=1)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(
            [f"{'charter' if p.startswith('charter') else 'coin' if p.startswith('coin') else 'control'}"
             f"\n{MODE_LABEL[m]}" for m, p in groups], fontsize=8.5)
        ax.set_ylim(0, 108)
        ax.set_title("Charter clauses USED in AFT training"
                     if condition == "trained" else
                     "Charter clauses NEVER used in training",
                     color=INK, fontsize=11, loc="left", pad=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("agreement-run accuracy (%)", color=INK, fontsize=10)
    axes[0].legend(handles=[
        Patch(facecolor="#888", alpha=0.42, edgecolor="#888", label="pre-RL (base arm)"),
        Patch(facecolor="#888", edgecolor="#888", label="post-RL (64 updates)"),
    ], frameon=False, fontsize=9, ncol=1, loc="upper left",
        bbox_to_anchor=(0.01, 0.86))
    fig.suptitle("Figure 0 (RL) — did GRPO learn the in-distribution task?",
                 x=0.06, y=0.99, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.06, 0.945,
             "Agreement runs have ONE correct answer under both oracles, so this "
             "is competence with no rule-choice component. Wilson 95% intervals; "
             "'n/r' = not run yet. The dashed line is what SUPERVISED AFT reached "
             "on this task, under a different prompt harness \u2014 scale only.",
             ha="left", va="top", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(top=0.86, bottom=0.14)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = out / f"figure_rl_0_id_task_accuracy.{suffix}"
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def figure(results: Path, data: Path, condition: str, out: Path) -> None:
    slice_name, clause_note = SLICES[condition]
    episodes = (v4.read_records(data / "episodes" / f"{slice_name}.jsonl"),
                slice_name)

    fig, ax = plt.subplots(figsize=(11.6, 12.4))
    rows: list[tuple[float, str]] = []
    dividers: list[float] = []
    y = 0.0
    for mode_index, mode in enumerate(MODES):
        if mode_index:
            dividers.append(y - 0.52)
        for parent, arm_note in (*ARMS, CONTROL):
            if parent == CONTROL[0]:
                dividers.append(y - 0.52)
            for stage, stage_label in STAGES:
                counts, n = load_counts(results, episodes, parent, mode, stage)
                draw_row(ax, y, counts, n)
                rows.append((y, f"{MODE_LABEL[mode]} · {arm_note} · {stage_label}"))
                y += 1.05
        y += 0.5

    for position in dividers:
        ax.axhline(position, color=GRID, linewidth=1.4, zorder=2)
    ax.set_yticks([row[0] for row in rows])
    ax.set_yticklabels([row[1] for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of frozen conflict-eval runs (%)", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(handles=[Patch(facecolor=colour, label=text)
                       for _, colour, text in CATEGORIES],
              frameon=False, fontsize=9, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.085))
    fig.suptitle(f"RL (GRPO, 64 updates) — {clause_note}",
                 x=0.06, y=0.975, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.06, 0.952,
             "pre-RL is THIS harness's own base arm (same prompts, envelope and "
             "greedy sampler), never the wave's pre-AFT baseline — that renders "
             "prompts without the <answer> envelope, so it is a different harness.",
             ha="left", va="top", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(top=0.925, left=0.28, bottom=0.085)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = out / f"figure_rl_{condition}_clauses.{suffix}"
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    root = EXP / "runs" / "dispatch_rl_v2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(root / "results"))
    parser.add_argument("--data", default=str(root / "data"))
    parser.add_argument("--figures", default=str(EXP / "figures" / "dispatch_rl_v2"))
    args = parser.parse_args()
    figure_id_accuracy(Path(args.results), Path(args.data), Path(args.figures))
    for condition in SLICES:
        figure(Path(args.results), Path(args.data), condition,
               Path(args.figures))


if __name__ == "__main__":
    main()
