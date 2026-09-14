r"""Appendix figure, heading "EFT then RLVR on the Python 4 graft": code correctness and rule expression.

One 2x2 figure in place of the two single-metric graft-ladder figures (Jonathan, 2026-09-12).
Top row = one-shot code correctness, bottom row = Suite-A rule expression; left column held-in,
right column held-out:

  a  Held-in rule problems     one-shot certified rate on the 1,024 held-in problems
  b  Held-out rule problems    the same on the 1,024 held-out problems; the bar is stacked
  c  Held-in rules             Suite-A adoption pooled over the 4 held-in rules (n = 512)
  d  Held-out rules            the same over the 4 held-out rules (n = 512)

x is the ladder, the same in every panel: the bare Gemma-4 31B prop chat-vector graft, the same
graft with a 512-row Python-4 EFT adapter (step 0), and that adapter after 32 and 64 GRPO steps in
the agentic Boa environment (Run B-v2) -- cumulative, each rung on top of the last. Bars run
light -> dark along the ladder (held-in in the Charter blue ramp, held-out in the Coin orange
ramp), with Wilson 95% whiskers on the total and the rate as a value label over each whisker.
Certified bars are split four ways (stack bottom -> top): plain = terminated run with a
rule-following answer; "\\" hatch = recovered (the run hit the token cap and the harness certified
the last complete draft inside the unfinished thought, never submitted); "///" = workaround
(held-out only: certified with no held-out rule used); the cross-hatch "xx" is the two overlaid
(workaround AND recovered), so the legend carries only the two base swatches, in grey. Held-in
problems have no workaround notion (the detectors are the taught rules), so (a) carries only the
recovered hatch; expression bars (c, d) are a single plain segment. A cell without a measurement
is an empty slot labelled "pending" (or "n/a" when the extract marks it missing).

The point (code correctness): the bare graft certifies 0/1,024 on both splits, and cold GRPO on
the same graft also left no one-shot trace (frame-gating), whereas the EFT-warm-started RL line
certifies 16% -> 24% held-in; every held-out certification is a workaround. The +512 EFT cell
isolates how much of that is the EFT rows alone.

The point (rule expression): held-in expression jumps from ~4% (bare graft) to ~75% on the RL'd
cells, while held-out expression stays a detector artefact (matrix multiplication's `@`, valid
Python 3 too). Bars are the pooled adoption over the split's four rules (128 prompts each) with
Wilson 95% whiskers.

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair main.tex defines; no
caption text on the figure). The canvas is authored and saved at exactly 5.5 x ``HEIGHT_IN`` in
under constrained layout (``ps.figure`` / ``ps.rc`` / ``ps.save``; never ``bbox_inches="tight"``,
no ``subplots_adjust``, no global rcParams), so every font lands on the page at the size set:
8 pt body (tick labels, the value labels over the whiskers, the legend, a "pending" slot), 9 pt
y labels, bold 9 pt panel titles and bold 9 pt panel letters a-d (no brackets) at the left edge
of each panel's y decorations, level with the top of its title (the acted_vs_stated convention:
measured after a layout pass, placed in points). Palette: the ladder ramp is ``ps.lighten(base,
0.55)`` (= ``ps.CHARTER_LIGHT`` / ``ps.COIN_LIGHT``), ``ps.lighten(base, 0.25)``, the base
(``ps.CHARTER`` held-in, ``ps.COIN`` held-out) and a local ``DARK_MIX`` blend of the base toward
``ps.INK`` for the darkest rung; each bar's hatch strokes are its own colour lightened by half
(``ps.lighten(colour, 0.5)``, the experiment figures' rule); whiskers, ticks and value labels
``ps.INK``; a "pending" slot ``ps.MUTED``. The two-swatch hatch legend is grey (``ps.GREY`` fill,
``ps.LIGHT_GREY`` hatch: the swatches carry hatch semantics only -- Jonathan, 2026-09-12) and
hangs under both rows at the page bottom, anchored by hand and then given its room with
``ps.reserve_band`` (a figure legend is invisible to constrained layout). Each row shares its y
axis: the code row is autoscaled after a layout pass so the tallest whisker's 8 pt value label
clears the axes top by ``HEAD_PT``; the expression row is 0-100 (ticks every 25) with the same
clearance checked. The x tick labels are the extract's two-line rung labels; the script raises if
neighbours would come within ``MIN_LABEL_GAP_PT`` at 8 pt. The experiment figure's subtitle
(``fig.text``) and the caveat footers are gone and their height given back; both belong in the
caption:

For the caption:
  * Subtitle (the extract's ``subtitle``, formerly printed over the panels): Gemma-4 31B prop
    graft line; thinking on, greedy.
  * Caveat (the extract's ``caveat``, formerly printed in small grey type under the legend): one
    run per cell, greedy; 56-70% of the RL'd cells' rows hit the 16k cap (graded on the last
    complete draft); held-out expression on the RL'd cells is the matrix_multiplication detector
    alone (uppercase_boolean and grouped_large_integer 0/128) and `@` is also valid Python 3;
    the +512 EFT cell is a replicate of the lost Run B-v2 step-0 adapter (same recipe, fresh
    replay thoughts).
  * Metrics: certified = the one-shot answer's last complete `def solution` block passes the Boa
    Python-4 test harness (n = 1,024 problems per split); rule expression = Suite-A construct
    elicitation (eft_v2/rule_suite.py): share of 128 independently worded prompts per rule whose
    answer uses the rule's form, pooled over the split's 4 rules (n = 512 per bar); Wilson 95%.
  * Setting: Gemma-4 31B, Python-4 prop chat-vector graft line; thinking on, greedy, 16,384-token
    budget.

Data is the frozen extract ``data/python4_graft_ladder.json`` (``src/freeze.py``:
``experiments/python4/runbv2_ladder/results/ladder_data.json`` on ``jb/python4-campaign``,
branch/commit/sha256 recorded, the +512 EFT cell's replicate-adapter note carried per cell;
the freeze checks every number against the two single-metric extracts it supersedes). The
extract's ``subtitle`` and ``caveat`` are kept as-is and simply not drawn. Self-contained on
purpose (no import from ``experiments/`` or the experiment branch); the four-way stack, ramp and
Wilson helpers descend from ``experiments/python4/runbv2_ladder/plot_ladder.py`` and
``experiments/python4/plot_eft_figures.py``, whose seaborn "colorblind" blue / orange are the same
hex pair as ``ps.CHARTER`` / ``ps.COIN``. Run from the repository root; writes
``python4_graft_ladder.pdf`` next to ``src/`` (PDF only -- the manuscript embeds it and no PNG is
committed)::

    uv run --extra dev python3 paper/figures/python-4/python4_graft_ladder/src/plot_python4_graft_ladder.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "python4_graft_ladder.json"
OUTPUT = HERE.parent
STEM = "python4_graft_ladder"

HEIGHT_IN = 4.9          # two rows of ~1.55 in axes + their two-line rung labels, titles and the legend band
BAR_W = 0.62             # bar width, in rung units (rungs are 1 apart)
HATCH_LINEWIDTH = 2.0    # the hatch stroke, via ps.rc (a rcParam read at draw time)
DARK_MIX = 0.35          # how far the darkest rung is blended toward ps.INK
HATCH_MIX = 0.5          # hatch strokes: the bar's own colour, this far toward white
LABEL_GAP_PT = 1.5       # whisker top -> value label
HEAD_PT = 2.0            # value label top -> axes top, at least (sets the code row's y range)
TITLE_PAD_PT = 6.0       # axes top -> panel title
PANEL_WSPACE = 0.10      # minimum gap between the columns, as a fraction of a panel's width
ROW_HSPACE = 0.15        # minimum gap between the rows, as a fraction of a panel's height
LEGEND_PAD_IN = 0.02     # page bottom -> legend
LEGEND_GAP_IN = 0.08     # legend top -> the bottom row's x decorations (reserved band)
MIN_LABEL_GAP_PT = 3.0   # neighbouring rung labels must clear each other by this much
PENDING_Y = 1.5          # baseline (%) of the rotated "pending" label in an empty slot

#: (metric key, y label, (held-in title, held-out title)), top row first.
ROWS = (("certified", "Code correctness (%)", ("Held-in rule problems", "Held-out rule problems")),
        ("expression", "Rule expression (%)", ("Held-in rules", "Held-out rules")))
#: (split key, base colour): held-in = Charter blue, held-out = Coin orange.
SPLITS = (("held_in", ps.CHARTER), ("held_out", ps.COIN))
#: Stack bottom -> top, and each share's hatch (plain carries none; "both" is the two overlaid).
STACK = ("plain", "recovered", "workaround", "both")
HATCH = {"plain": None, "recovered": "\\\\\\", "workaround": "///", "both": "xxx"}
LEGEND = (("workaround", "workaround: certified without using a held-out rule"),
          ("recovered", "recovered: run hit the token cap; last complete draft certified"))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, as fractions."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def darken(colour: str, mix: float = DARK_MIX) -> str:
    """Blend ``colour`` toward ``ps.INK``; the ramp's darkest step (a local partner to ``ps.lighten``)."""
    rgb, ink = mcolors.to_rgb(colour), mcolors.to_rgb(ps.INK)
    return mcolors.to_hex(tuple(c + (i - c) * mix for c, i in zip(rgb, ink)))


def ramp(base: str) -> list[str]:
    """The four rung colours, light -> dark along the ladder."""
    return [ps.lighten(base, 0.55), ps.lighten(base, 0.25), base, darken(base)]


def hatch_for(colour: str, kind: str) -> dict:
    """Bar kwargs for one share of the stack: its hatch in the bar's own colour lightened, no edge stroke."""
    if HATCH[kind] is None:
        return {}
    return dict(hatch=HATCH[kind], edgecolor=ps.lighten(colour, HATCH_MIX), linewidth=0)


def stats(cell: dict, metric: str, split: str) -> dict | None:
    """rate / Wilson bounds (%) and the bar's segments (stack bottom -> top) for one cell, metric and
    split, or None when the cell is unmeasured. Certified bars split four ways: plain (terminated run,
    rule-following answer) / recovered (cap-hit run, last complete draft certified) / workaround
    (held-out only) / both (workaround AND recovered); expression bars are one plain segment."""
    c = cell[metric].get(split)
    if not c:
        return None
    k, n = c["k"], c["n"]
    lo, hi = wilson(k, n)
    if metric == "certified":
        wk = c["workaround"] if split == "held_out" else 0
        rec = c["recovered"]
        both = c["workaround_recovered"] if split == "held_out" else 0
        seg = {"both": both, "workaround": wk - both, "recovered": rec - both, "plain": k - wk - (rec - both)}
    else:
        seg = {"plain": k, "recovered": 0, "workaround": 0, "both": 0}
    return {"rate": 100 * k / n, "lo": 100 * lo, "hi": 100 * hi, "seg": {kk: 100 * v / n for kk, v in seg.items()}}


def y_decorations_left_pt(ax, fig, renderer) -> float:
    """Points from a panel's left spine to the left edge of its y decorations: the y label and
    tick labels when it draws them, else (a shared-y panel, labels hidden) its outward tick marks."""
    bb = ax.yaxis.get_tightbbox(renderer)
    if bb is None:
        return float(matplotlib.rcParams["ytick.major.size"])
    return (ax.get_window_extent(renderer).x0 - bb.x0) / fig.dpi * 72


def bar4(ax, x: float, colour: str, s: dict) -> None:
    """One rung: the stack in ``colour`` (hatches in its lightened partner) and its Wilson whisker."""
    bottom = 0.0
    for kind in STACK:
        h = s["seg"][kind]
        if h <= 0:
            continue
        ax.bar(x, h, BAR_W, bottom=bottom, color=colour, zorder=3, **hatch_for(colour, kind))
        bottom += h
    ax.errorbar(x, s["rate"], yerr=[[s["rate"] - s["lo"]], [s["hi"] - s["rate"]]], fmt="none",
                ecolor=ps.INK, elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=5)


def main() -> int:
    D = json.loads(DATA.read_text())
    order = D["order"]
    xs = list(range(len(order)))
    labels = [D["cells"][k]["label"] for k in order]

    with matplotlib.rc_context(ps.rc(**{"hatch.linewidth": HATCH_LINEWIDTH})):
        # Built at 72 dpi -- the PDF backend's display space, where ps.save lays out and
        # checks -- so the measurements below match the saved page.
        fig, axes = ps.figure(HEIGHT_IN, 2, 2, sharey="row", dpi=72,
                              gridspec_kw={"wspace": PANEL_WSPACE, "hspace": ROW_HSPACE})
        renderer = fig.canvas.get_renderer()
        w_in, h_in = fig.get_size_inches()

        letters = iter("abcd")
        panels: list[tuple] = []                              # (ax, letter)
        tops = {metric: [10.0] for metric, *_ in ROWS}         # whisker tops, per row
        value_labels = {metric: [] for metric, *_ in ROWS}     # measured after the layout pass
        for row_axes, (metric, ylabel, titles) in zip(axes, ROWS):
            for ax, (split, base), title in zip(row_axes, SPLITS, titles):
                for x, key, colour in zip(xs, order, ramp(base)):
                    cell = D["cells"][key]
                    s = stats(cell, metric, split)
                    if s is None:          # an unmeasured rung: an empty slot, labelled
                        ax.text(x, PENDING_Y, "n/a" if cell["status"] == "missing" else "pending",
                                ha="center", va="bottom", color=ps.MUTED, rotation=90)
                        continue
                    bar4(ax, x, colour, s)
                    # The value label sits LABEL_GAP_PT above the whisker top, in points, so its
                    # clearance does not depend on the y range.
                    value_labels[metric].append(ax.annotate(
                        f"{s['rate']:.1f}", xy=(x, s["hi"]), xytext=(0, LABEL_GAP_PT),
                        textcoords="offset points", ha="center", va="bottom", color=ps.INK, zorder=6))
                    tops[metric].append(s["hi"])
                ax.set_xticks(xs)
                ax.set_xticklabels(labels)
                ax.set_title(title, pad=TITLE_PAD_PT)
                ax.set_xlim(-0.6, len(order) - 0.4)
                panels.append((ax, next(letters)))
            row_axes[0].set_ylabel(ylabel)
        axes[0][0].set_ylim(0, max(tops["certified"]) * 1.2)   # provisional; re-set once the axes height is known
        axes[1][0].set_ylim(0, 100)
        axes[1][0].set_yticks([0, 25, 50, 75, 100])

        # The two-swatch hatch legend, in grey (the swatches carry hatch semantics only --
        # Jonathan, 2026-09-12), at the page bottom. A figure legend is invisible to constrained
        # layout, so it is anchored by hand, measured, and its band reserved through the layout rect.
        legend = fig.legend(
            handles=[Patch(facecolor=ps.GREY, label=label, **hatch_for(ps.GREY, kind))
                     for kind, label in LEGEND],
            loc="lower center", bbox_to_anchor=(0.5, LEGEND_PAD_IN / h_in), ncol=1,
            handlelength=2.8, handleheight=1.3, labelspacing=0.35, borderpad=0.0)
        for patch in legend.get_patches():
            patch.set_edgecolor(ps.LIGHT_GREY)                 # the house light grey, not a blend
        fig.canvas.draw()
        legend_top_in = legend.get_window_extent(renderer).y1 / fig.dpi
        ps.reserve_band(fig, bottom_in=legend_top_in + LEGEND_GAP_IN)

        # Pass 1: lay everything out, then (i) set the code row's y range so the tallest value
        # label clears the axes top by HEAD_PT, (ii) check the expression row's labels clear it at
        # 0-100, and (iii) put each panel's letter at the left edge of its y decorations, level
        # with the top of its title (bold, no brackets).
        fig.canvas.draw()
        axes_h_pt = axes[0][0].get_window_extent(renderer).height / fig.dpi * 72
        label_h_pt = max(t.get_window_extent(renderer).height
                         for row in value_labels.values() for t in row) / fig.dpi * 72
        room_pt = LABEL_GAP_PT + label_h_pt + HEAD_PT       # points the label stack needs above the whisker
        if room_pt >= 0.5 * axes_h_pt:
            raise ValueError(f"axes are {axes_h_pt:.0f} pt tall; the 8 pt value labels need {room_pt:.0f} pt "
                             f"of headroom -- raise HEIGHT_IN")
        axes[0][0].set_ylim(0, min(100.0, max(tops["certified"]) / (1 - room_pt / axes_h_pt)))
        if max(tops["expression"]) / 100 > 1 - room_pt / axes_h_pt:
            raise ValueError("an expression value label would run past the axes top at 0-100 -- raise HEIGHT_IN")
        for ax, letter in panels:
            ax_bb = ax.get_window_extent(renderer)
            top_pt = (ax.title.get_window_extent(renderer).y1 - ax_bb.y1) / fig.dpi * 72
            left_pt = y_decorations_left_pt(ax, fig, renderer)
            ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-left_pt, top_pt),
                        textcoords="offset points", ha="left", va="top", fontsize=ps.TITLE_PT,
                        fontweight="bold", annotation_clip=False)

        # Pass 2: with the letters in the layout, make sure the 8 pt rung labels do not touch.
        fig.canvas.draw()
        for ax, letter in panels:
            boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
            for lo_box, hi_box in zip(boxes, boxes[1:]):
                gap_pt = (hi_box.x0 - lo_box.x1) / fig.dpi * 72
                if gap_pt < MIN_LABEL_GAP_PT:
                    raise ValueError(f"({letter}) rung labels {gap_pt:.1f} pt apart (need "
                                     f"{MIN_LABEL_GAP_PT:g}); re-wrap the labels, do not shrink")
        panel_w_in = axes[0][0].get_window_extent(renderer).width / fig.dpi
        panel_h_in = axes_h_pt / 72

        written = ps.save(fig, OUTPUT, STEM)
    plt.close(fig)

    print(f"panels {panel_w_in:.2f} x {panel_h_in:.2f} in; code row y to "
          f"{axes[0][0].get_ylim()[1]:.1f}%; expression row 0-100")
    for metric, *_ in ROWS:
        for split, _base in SPLITS:
            cells = [(D["cells"][k]["label"].replace("\n", " "), stats(D["cells"][k], metric, split)) for k in order]
            print(f"{metric:10s} {split:8s}",
                  "  ".join(f"{lab}={s['rate']:.1f}" if s else f"{lab}=pending" for lab, s in cells))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
