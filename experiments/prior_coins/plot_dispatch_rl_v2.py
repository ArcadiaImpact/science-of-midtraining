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
    for condition in SLICES:
        figure(Path(args.results), Path(args.data), condition,
               Path(args.figures))


if __name__ == "__main__":
    main()
