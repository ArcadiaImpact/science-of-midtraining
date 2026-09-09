"""Figures for elicitation_ablation_v1 from ``scored.json`` (no network).

    uv run python -m experiments.prior_coins.elicitation_ablation_v1.plot [--scored PATH] [--out DIR]

fig1_part1: the three published adapters under the five eval-time cues.
fig2_part2: framed re-trainings vs the published cells, plain and persona-cued,
            trained clauses (left) and held-out clauses (right).
Marks follow the house data-viz method: thin horizontal bars, categorical hues in
a fixed order (never cycled), a legend plus sparse value labels, hairline grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from experiments.prior_coins.elicitation_ablation_v1 import contracts as C
from experiments.prior_coins.elicitation_ablation_v1 import score as S

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
#: fixed slot per condition (reference palette slots 1-4; plain is the
#: de-emphasis gray). Colors follow the condition, never its position.
COLOR = {
    "uninstructed": "#9b9a95",
    "instr_persona": "#2a78d6",
    "instr_charter_name": "#eb6834",
    "instr_charter_text": "#1baf7a",
    "instr_profit": "#eda100",
}
LABEL = {"uninstructed": "plain prompt", "instr_persona": "+ persona cue",
         "instr_charter_name": "+ Charter named", "instr_charter_text": "+ Charter text",
         "instr_profit": "+ profit instruction"}
MIXTURE_LABEL = {"agreement": "agreement-only AFT", "coin_0p5pct": "0.5% coin-labelled",
                 "mixed_coin": "2% coin-labelled"}
FRAMING_LABEL = {"published": "published (unframed)", "persona": "framed: persona (L1)",
                 "persona_charter": "framed: persona + Charter named (L2)"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=0)
    ax.set_xlim(0, 100)


def _value(units: dict, unit: str, condition: str, slice_name: str, key: str = "charter"):
    agg = (units.get(unit) or {}).get("slices", {}).get(C.prompt_set_key(condition, slice_name))
    return None if agg is None else S.rate(agg, "conflict_runs", key)


def fig_part1(units: dict, out: Path) -> Path | None:
    rows = [u for u in (f"part1/{c}" for c in C.PART1_CELLS) if u in units]
    if not rows:
        return None
    conditions = [c for c in C.CONDITIONS
                  if any(_value(units, u, c, C.PRIMARY_SLICE) is not None for u in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.9 + 1.25 * len(rows)), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    height = 0.8 / len(conditions)
    for ax, slice_name, title in zip(axes, ("eval_trained_conflict", "eval_holdout_conflict"),
                                     ("trained clauses", "held-out clauses")):
        _style(ax)
        for i, unit in enumerate(rows):
            for j, condition in enumerate(conditions):
                y = i + (j - (len(conditions) - 1) / 2) * height
                v = _value(units, unit, condition, slice_name)
                if v is None:
                    continue
                ax.barh(y, v, height=height * 0.82, color=COLOR[condition], linewidth=0)
                ax.text(v + 1, y, f"{v:.0f}", va="center", ha="left", fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([MIXTURE_LABEL[C.PART1_CELLS[u.split("/")[1]]["dataset"]] for u in rows],
                           color=INK, fontsize=9)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("P(Charter crew), %", color=INK2, fontsize=8)
    handles = [Patch(color=COLOR[c], label=LABEL[c]) for c in conditions]
    fig.legend(handles=handles, loc="lower center", ncol=len(conditions), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Eval-time cues on the published gemma3-27b-190M charter adapters — P(Charter crew) on "
                 "conflict runs, held-out templates (n = 3,000 / 1,200 runs)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = out / "fig1_part1_eval_time_cues"
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    return path.with_suffix(".png")


def fig_part2(units: dict, out: Path) -> Path | None:
    framed = [u for u in (f"part2/{c}" for c in C.part2_cells()) if u in units]
    if not framed:
        return None
    conditions = ("uninstructed", "instr_persona")
    rows = []  # (mixture, framing, unit)
    for mixture in C.MIXTURES:
        published = f"part1/{mixture}"
        if published in units:
            rows.append((mixture, "published", published))
        for framing in C.FRAMINGS:
            unit = f"part2/{framing}__{mixture}"
            if unit in units:
                rows.append((mixture, framing, unit))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 1.4 + 0.5 * len(rows)), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    height = 0.36
    for ax, slice_name, title in zip(axes, ("eval_trained_conflict", "eval_holdout_conflict"),
                                     ("trained clauses", "held-out clauses")):
        _style(ax)
        for i, (mixture, framing, unit) in enumerate(rows):
            for j, condition in enumerate(conditions):
                y = i + (j - 0.5) * height
                v = _value(units, unit, condition, slice_name)
                if v is None:
                    continue
                ax.barh(y, v, height=height * 0.85, color=COLOR[condition], linewidth=0)
                ax.text(v + 1, y, f"{v:.0f}", va="center", ha="left", fontsize=7.5, color=INK2)
        # separators between mixtures
        last = None
        for i, (mixture, _f, _u) in enumerate(rows):
            if last is not None and mixture != last:
                ax.axhline(i - 0.5, color=GRID, linewidth=0.8)
            last = mixture
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{MIXTURE_LABEL[m]} · {FRAMING_LABEL[f]}" for m, f, _ in rows],
                           color=INK, fontsize=8.5)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("P(Charter crew), %", color=INK2, fontsize=8)
    handles = [Patch(color=COLOR[c], label=LABEL[c]) for c in conditions]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Training-time persona framing vs the published cells — P(Charter crew) on conflict runs, "
                 "gemma3-27b-190M charter parent, held-out templates", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = out / "fig2_part2_framed_vs_published"
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    return path.with_suffix(".png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored", type=Path, default=C.HERE / "scored.json")
    p.add_argument("--out", type=Path, default=C.HERE / "figures")
    a = p.parse_args()
    units = json.loads(a.scored.read_text())["units"]
    a.out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_part1, fig_part2):
        path = fn(units, a.out)
        print(path or f"{fn.__name__}: no units yet")


if __name__ == "__main__":
    main()
