"""Figures for elicitation_ablation_v1 from ``scored.json`` (no network).

    uv run python -m experiments.dispatch.elicitation_ablation_v1.plot [--scored PATH] [--out DIR]

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

from experiments.dispatch.elicitation_ablation_v1 import contracts as C
from experiments.dispatch.elicitation_ablation_v1 import score as S

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
    for fn in (fig_part1, fig_part2, fig_part1_deltas, fig_composition, fig_part2_clauses):
        path = fn(units, a.out)
        print(path or f"{fn.__name__}: no units yet")



# --- additional figures -------------------------------------------------------------
SEED_SD_PP = 9.0  # run-to-run SD on this readout (seed_sweep_v1); differences inside it are not findings
COMPOSITION = (("charter", "#2a78d6", "chose Charter"), ("coin", "#eb6834", "chose coin / cheapest"),
               ("other", "#9b9a95", "chose another crew"), ("malformed", "#0b0b0b", "malformed answer"))


def fig_part1_deltas(units: dict, out: Path) -> Path | None:
    """Each cue's shift from the plain prompt, per adapter, against the seed band."""
    rows = [u for u in (f"part1/{c}" for c in C.PART1_CELLS) if u in units]
    if not rows:
        return None
    cues = [c for c in C.INSTRUCTED_CONDITIONS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 1.2 + 0.9 * len(rows)), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, slice_name, title in zip(axes, ("eval_trained_conflict", "eval_holdout_conflict"),
                                     ("trained clauses", "held-out clauses")):
        _style(ax)
        ax.set_xlim(-15, 30)
        ax.axvspan(-SEED_SD_PP, SEED_SD_PP, color="#efeeea", zorder=0)
        ax.axvline(0, color=INK2, linewidth=0.8)
        for i, unit in enumerate(rows):
            plain = _value(units, unit, "uninstructed", slice_name)
            for j, cue in enumerate(cues):
                v = _value(units, unit, cue, slice_name)
                if plain is None or v is None:
                    continue
                y = i + (j - (len(cues) - 1) / 2) * 0.18
                ax.plot([0, v - plain], [y, y], color=COLOR[cue], linewidth=2, solid_capstyle="round")
                ax.plot(v - plain, y, "o", color=COLOR[cue], markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([MIXTURE_LABEL[C.PART1_CELLS[u.split("/")[1]]["dataset"]] for u in rows],
                           color=INK, fontsize=9)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("shift in P(Charter crew) vs plain prompt, pp", color=INK2, fontsize=8)
    handles = [plt.Line2D([], [], color=COLOR[c], marker="o", linewidth=2, label=LABEL[c]) for c in cues]
    handles.append(Patch(color="#efeeea", label=f"±{SEED_SD_PP:.0f} pp run-to-run seed band"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("How far each eval-time cue moves the published adapters from their plain-prompt readout",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path = out / "fig3_part1_cue_deltas"
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    return path.with_suffix(".png")


def fig_composition(units: dict, out: Path) -> Path | None:
    """House Figure-0 grammar: stacked shares of conflict-run verdicts per model x condition."""
    rows: list[tuple[str, str, str]] = []  # (label, unit, condition)
    for cell in C.PART1_CELLS:
        unit = f"part1/{cell}"
        if unit not in units:
            continue
        for condition in C.CONDITIONS:
            rows.append((f"{MIXTURE_LABEL[cell]} · published · {LABEL[condition]}", unit, condition))
    part2 = [(f"{MIXTURE_LABEL[m]} · {FRAMING_LABEL[f]} · {LABEL[c]}", f"part2/{f}__{m}", c)
             for m in C.MIXTURES for f in C.FRAMINGS for c in ("uninstructed", "instr_persona")
             if f"part2/{f}__{m}" in units]
    if not rows and not part2:
        return None
    groups = [("Part 1 — published adapters under eval-time cues", rows),
              ("Part 2 — framed re-trainings", part2)]
    groups = [g for g in groups if g[1]]
    n_rows = sum(len(g[1]) for g in groups) + len(groups) * 0.8
    fig, ax = plt.subplots(figsize=(11.5, 0.9 + 0.32 * n_rows))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.xaxis.grid(False)
    y = 0.0
    ticks, labels = [], []
    for title, group in groups:
        ax.text(0, y - 0.15, title, fontsize=9.5, color=INK, fontweight="bold", va="center")
        y += 0.8
        for label, unit, condition in group:
            agg = units[unit]["slices"].get(C.prompt_set_key(condition, C.PRIMARY_SLICE))
            if agg is None:
                continue
            rates = agg["conflict_runs"]["rates"]
            left = 0.0
            for key, color, _ in COMPOSITION:
                v = 100 * rates.get(key, 0.0)
                if v > 0:
                    ax.barh(y, v, left=left, height=0.62, color=color, linewidth=0)
                    if v >= 6:
                        ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                                color=SURFACE if key in ("charter", "malformed") else INK)
                    left += v
            ticks.append(y)
            labels.append(label)
            y += 1.0
        y += 0.4
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_ylim(y - 0.6, -0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of conflict-eval runs (%), trained clauses, held-out templates, n = 3,000 per row",
                  color=INK2, fontsize=8)
    handles = [Patch(color=color, label=label) for _, color, label in COMPOSITION]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Answer composition on conflict runs — gemma3-27b-190M charter parent",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = out / "fig4_composition"
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    return path.with_suffix(".png")


CLAUSE_LABEL = {"precedence_days_since": "precedence: days since last",
                "precedence_registry_rank": "precedence: registry rank",
                "precedence_runs_year": "precedence: runs this year",
                "qual_skill": "qualification: skill", "qual_specialty": "qualification: specialty"}
FRAMING_COLOR = {"published": "#9b9a95", "persona": "#2a78d6", "persona_charter": "#eb6834"}


def fig_part2_clauses(units: dict, out: Path) -> Path | None:
    """Per trained clause: P(Charter) for the published cell vs each framed cell (plain prompt)."""
    panels = []
    for mixture in C.MIXTURES:
        series = []
        if f"part1/{mixture}" in units:
            series.append(("published", f"part1/{mixture}"))
        for framing in C.FRAMINGS:
            if f"part2/{framing}__{mixture}" in units:
                series.append((framing, f"part2/{framing}__{mixture}"))
        if len(series) >= 2:
            panels.append((mixture, series))
    if not panels:
        return None
    key = C.prompt_set_key("uninstructed", C.PRIMARY_SLICE)
    clauses = list(CLAUSE_LABEL)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (mixture, series) in zip(axes[0], panels):
        _style(ax)
        for j, (framing, unit) in enumerate(series):
            by_clause = units[unit]["slices"][key]["conflict_runs_by_clause"]
            ys, xs = [], []
            for i, clause in enumerate(clauses):
                counts = by_clause.get(clause, {})
                n = sum(counts.values())
                if not n:
                    continue
                ys.append(i + (j - (len(series) - 1) / 2) * 0.22)
                xs.append(100 * counts.get("charter", 0) / n)
            ax.plot(xs, ys, "o", color=FRAMING_COLOR[framing], markersize=7.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=FRAMING_LABEL[framing])
        ax.set_yticks(range(len(clauses)))
        ax.set_yticklabels([CLAUSE_LABEL[c] for c in clauses], fontsize=8.5, color=INK)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_title(MIXTURE_LABEL[mixture], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("P(Charter crew), %  (n = 600 runs per clause)", color=INK2, fontsize=8)
    handles = [plt.Line2D([], [], color=FRAMING_COLOR[f], marker="o", linestyle="", markersize=7,
                          label=FRAMING_LABEL[f]) for f in ("published", "persona", "persona_charter")
               if any(f == s[0] for _, ss in panels for s in ss)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Where the framed cells lose Charter picks — per trained clause, plain prompt, held-out templates",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    path = out / "fig5_part2_per_clause"
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    return path.with_suffix(".png")


if __name__ == "__main__":
    main()
