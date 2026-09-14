"""Analysis figure: no worked examples in the midtraining corpus.

Analysis heading 7 ("Other ablations") in the write-up, the no-worked-
examples part. The question: does the midtrained prior come from documents
that *discuss* the rule, or from the worked-example runs that adjudicate
it? The ablation re-cuts the Gemma 3 12B / 50M-presented-token corpus to
the qualitative-only documents (``focus_tag`` ending ``qualitative``; no
adjudicated example runs) at the same dose and geometry (12.5M release
tokens x 4 epochs), trains charter and coin arms, and runs the same EFT
and conflict eval. The figure's claim, in its title: without worked
examples, the Charter prior does not survive EFT; the coin prior does.

Layout (approved 2026-09-07; re-set on the house style 2026-09-11; footer
removed 2026-09-12): two panels sharing y, agreement-only EFT on the left
and EFT with 2% coin-labelled conflict episodes on the right. In each
panel, two groups: "Charter midtrain -> picks Charter crew" (y is the
Charter-crew share, Charter blue) and "Coin midtrain -> picks coin crew"
(y is the coin-crew share, Coin orange). Each group holds two bars: the
corpus with worked examples (filled, the main row) and without (hatched,
lightened). The control (no midtraining documents; filler only) is a
dashed grey level per group, drawn on the same metric as the group, its
value beside it. Wilson 95% intervals on runs; runs cluster within
episodes, so the intervals are optimistic (the caption says so). The claim
is the figure title and the legend sits under it; nothing sits under the
axes.

Rules 2026-09-12: no caption text on the figure; keywords painted by
ps.paint. The 2026-09-11 version closed with a five-line italic provenance
footnote and the standing caveat line in a 0.97 in band under the axes
(``ps.caveat``, since removed from the module). Both are gone -- the LaTeX
caption carries them -- and the band's height is given back: 3.8 in became
``HEIGHT_IN`` = 2.85 in, the smallest height on a 0.05 in grid at which the
axes are at least as tall as before (1.44 in; they are 1.46 in now, y0 at
0.38 in, only the tick labels below them). ``ps.save`` runs ``ps.paint`` by
default, so every ink mention of Charter is blue and bold and of coin
orange and bold: the two tick labels (two keywords each: "Charter midtrain
->" / "picks Charter crew" and "Coin midtrain ->" / "picks coin crew"), the
first line of the title ("... the Charter prior does not survive EFT;" and
"the coin prior does" on the second) and the right panel's title ("EFT
with 2% coin-labelled ..."; the regex takes "coin" before the hyphen). The
titles are already bold, so painting only recolours them; the tick labels
are regular weight, and the bold keyword widens each line by 2-4 pt
(0.03-0.05 in), centred on its tick, which closed the gap between a
panel's two labels from 7.6 / 9.2 pt (first / second line) to 3.7 / 5.2 pt
-- about a word space, the two reading as one line. The groups were moved
apart to make room: ``GROUP_X`` 1.14 -> 1.22, paid for with a slightly
smaller right pad (``X_PAD`` 0.79 -> 0.77; the control label 0.02 past its
level instead of 0.03). The gaps are 6.0 / 7.5 pt now and the paired value
labels ("81%" beside "79%") give up 0.8 pt (4.5 -> 3.7 pt); measured on
the PDF's word boxes (``pdftotext -bbox``). The grey "control NN%" labels
are not in ink and are left alone. One module-level effect, noted for the
record: ``ps.paint`` pins its pieces in figure coordinates from the Agg
layout, and the PDF backend lays the axes out again with its own text
metrics, so in the PDF the painted lines sit 0.7-2.7 pt right of their
ticks -- both labels of a panel alike, so the gaps above are unaffected.

For the caption: the removed footnote read, verbatim (its n was computed
from the extract; it is printed in the run log now) --
"Gemma 3 12B, 50M presented midtraining tokens, trained clauses, held-out
prompt template, step 512; n = 3,000 runs per bar, Wilson 95% intervals
(runs cluster within episodes, so intervals are optimistic). Control = no
documents, shared with the main row. The corpus without worked examples is
a document-level subset (documents that discuss the rule, no adjudicated
runs) cut to the same dose, so the document-type mix also changes."
-- and the removed caveat line read "CAVEAT: one seed per cell; run-to-run
SD ~9pp on the primary metric" (``CAVEAT`` below, still asserted against
the extract).

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). The canvas is exactly 5.5 x 2.85 in and is saved as-is
(never ``bbox_inches="tight"``), so the manuscript's ``\\includegraphics
[width=\\linewidth]`` prints every font at its authored size: 9 pt bold
titles, 9 pt axis label, 8 pt for everything else (ticks, value labels,
control labels, legend). Colours are the module's constants by role:
Charter arm ``ps.CHARTER`` (#0173b2), coin arm ``ps.COIN`` (#de8f05;
orange, not the Okabe-Ito vermilion of the 2026-09-07 version), control
``ps.GREY``, the light bar of a pair ``ps.lighten`` (55% toward white,
``plot_grid.py``'s shade rule for step pairs, as in ``plot_per_clause.py``),
text and spines ``ps.INK``, gridlines ``ps.LIGHT_GREY``. The 2026-09-07
version was authored at 11 x 5.4 in from a palette copied out of
``results_grid/plot_grid.py`` and halved by LaTeX (so its 7.5-9.5 pt text
printed at 3.7-4.7 pt); the 2026-09-11 audit of the figure set replaced
that with the shared module. Legend labels lost the word "corpus" and the
tick labels moved the arrow to the first line so both fit at 8 pt; the
title and the right panel's title wrap to two lines and the y label to
three.

Numbers at step 512, trained clauses, held-out prompt template
(``eval_trained_conflict__heldout``, the slice of the compiled Results
figures). Charter arm, Charter share: 65% with worked examples vs 37%
without after agreement-only EFT (control 22%); 9% vs 7% after 2%
coin-labelled EFT (control 8%). Coin arm, coin share: 81% vs 79%
(control 69%); 95% vs 92% (control 87%). The 2% cells were re-frozen
2026-09-09 from the corrected balanced draw (``meta.twopct``); the first
freeze read 43/27/16 and 82/81/77 there.

Data is the frozen extract ``data/no_worked_examples.json``, cut from
``experiments/prior_coins/dispatch_final_v1/results_grid/scored/ablations/
no_examples.json`` on branch ``sid/dispatch-final-v1`` (commit and sha256
in the extract). That file is ``collect_ablation_scores.py``'s package of
the two scored no-examples arms (profile ``gemma3_12b_50m_noex``) with the
main 50M row's three arms (profile ``gemma3_12b_50m_4ep``); its cells are
byte-identical to the per-profile ``scored/<profile>/<arm>/eval.json``.
The control is the main row's: the no-examples campaign deliberately did
not train one, because control midtraining is filler-only and a
no-examples control would be byte-identical (the file's ``meta`` says so).
Every endpoint of the slice is frozen (pre_aft, step 256 and 512 of
agreement / mixed_coin / mixed_charter / charter_only); only the two
step-512 endpoints are drawn. Re-freeze rather than edit when the grid is
re-scored.

This file imports nothing from the experiment's plot modules (those
branches get merged, rewritten, retired); its only in-repo import is the
house-style module ``scimt.viz.paper``.

Run from the repository root; writes ``no_worked_examples.pdf`` next to
``src/`` -- the PDF only (Jonathan, 2026-09-11: no .png renderings in the
figure set; a throw-away preview is ``ps.save(..., formats=("pdf", "png"))``
or ``pdftoppm`` on the PDF)::

    uv run --extra dev python3 \\
      paper/figures/no_worked_examples/src/plot_no_worked_examples.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "no_worked_examples.json"
OUTPUT = HERE.parent              # paper/figures/no_worked_examples/
STEM = "no_worked_examples"

#: The verbatim standing caveat. It belongs to the LaTeX caption, never to
#: the figure (rule 2026-09-12); the extract must still carry it unchanged.
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"

#: The one free dimension (the width is the page's, ``ps.TEXTWIDTH_IN``).
HEIGHT_IN = 2.85
#: Pad between the canvas edge and the title.
PAD_IN = 0.03

TITLE = ("Without worked examples, the Charter prior does not survive EFT;\n"
         "the coin prior does")
Y_LABEL = "Follows own\nmidtrained motivation\n(% of conflict runs)"

#: (endpoint, panel title) -- left to right.
PANELS = (
    ("agreement-step512", "Agreement-only EFT"),
    ("mixed_coin-step512", "EFT with 2% coin-labelled\nconflict episodes"),
)
#: (arm, outcome drawn, family colour, x caption) -- left to right.
GROUPS = (
    ("charter", "charter", ps.CHARTER, "Charter midtrain →\npicks Charter crew"),
    ("coin", "coin", ps.COIN, "Coin midtrain →\npicks coin crew"),
)
#: (variant, hatched) -- bar order within a group.
VARIANTS = (
    ("standard_examples", False),
    ("no_examples", True),
)
#: x geometry, in data units; a unit is 2.16 in / 2.39 units = 0.90 in at
#: the panel width the layout gives. Group centres 1.22 apart (1.14 before
#: the keywords were painted: the painted two-line tick labels are 1.10 and
#: 0.91 in wide and sit 6.0 / 7.5 pt apart at 1.22), bars 0.25 wide on a
#: 0.35 pitch (bold 8 pt value labels are 0.27 in wide and 3.7 pt apart
#: within a pair), the control level 0.04 past the bars and its two-line
#: label 0.02 past that, the right pad sized to that label.
GROUP_X = (0.0, 1.22)
BAR_WIDTH = 0.25
OFFSET = 0.175            # half the distance between the two bars of a group
CONTROL_OVERHANG = 0.04   # dashed control level past the outer bar edges
CONTROL_LABEL_GAP = 0.02  # the control label past the level's right end
X_PAD = (0.40, 0.77)      # xlim beyond the first / last group centre
HATCH = "///"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def rate(cell: dict, outcome: str) -> tuple[float, float, float, int]:
    n = cell["n"]
    k = cell["counts"][outcome]
    lo, hi = wilson(k, n)
    return 100 * k / n, lo, hi, n


def draw_panel(ax, cells: dict, endpoint: str, title: str) -> set[int]:
    ns: set[int] = set()
    ax.yaxis.grid(True, color=ps.LIGHT_GREY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    for x, (arm, outcome, colour, _) in zip(GROUP_X, GROUPS, strict=True):
        light = ps.lighten(colour)
        for dx, (variant, hatched) in zip((-OFFSET, +OFFSET), VARIANTS, strict=True):
            p, lo, hi, n = rate(cells[f"{variant}/{arm}"][endpoint], outcome)
            ns.add(n)
            ax.bar(x + dx, p, width=BAR_WIDTH,
                   color=light if hatched else colour,
                   hatch=HATCH if hatched else None,
                   edgecolor=colour,
                   linewidth=0.8 if hatched else 0.0, zorder=2)
            ax.errorbar(x + dx, p, yerr=[[p - lo], [hi - p]], fmt="none",
                        ecolor=ps.INK, elinewidth=0.8, capsize=2.0, capthick=0.8,
                        zorder=4)
            ax.text(x + dx, hi + 1.8, f"{p:.0f}%", ha="center", va="bottom",
                    fontweight="bold", color=ps.INK, zorder=5)

        # Control: the main row's no-documents arm, on this group's metric.
        pc, _, _, n = rate(cells["standard_examples/control"][endpoint], outcome)
        ns.add(n)
        x0 = x - OFFSET - BAR_WIDTH / 2 - CONTROL_OVERHANG
        x1 = x + OFFSET + BAR_WIDTH / 2 + CONTROL_OVERHANG
        ax.plot([x0, x1], [pc, pc], color=ps.GREY, linewidth=1.3,
                linestyle=(0, (4, 2)), zorder=3)
        # Centred on the level, except that the two-line label (~20 y-units
        # tall) is held clear of the x axis for the 8% control.
        ax.text(x1 + CONTROL_LABEL_GAP, max(pc, 12.0), f"control\n{pc:.0f}%", ha="left",
                va="center", color=ps.GREY, linespacing=1.1, zorder=3)

    ax.set_xticks(GROUP_X)
    ax.set_xticklabels([caption for *_, caption in GROUPS])
    ax.set_xlim(GROUP_X[0] - X_PAD[0], GROUP_X[-1] + X_PAD[1])
    ax.set_ylim(0, 110)               # headroom for the labels over the ~95% bars
    ax.set_yticks((0, 20, 40, 60, 80, 100))
    ax.spines["left"].set_bounds(0, 100)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_title(title, loc="left")
    return ns


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw from it")
    assert extract["caveat"] == CAVEAT, extract["caveat"]
    cells = extract["cells"]

    with matplotlib.rc_context(ps.rc(**{"hatch.linewidth": 0.6})):
        fig, axes = ps.figure(HEIGHT_IN, 1, 2, sharey=True)
        w_in, h_in = fig.get_size_inches()
        ns: set[int] = set()
        for ax, (endpoint, title) in zip(axes, PANELS, strict=True):
            ns |= draw_panel(ax, cells, endpoint, title)
        # Three lines: the longest (1.45 in) about matches the panel height,
        # where two lines (2.3 in) would overhang the axes at both ends.
        axes[0].set_ylabel(Y_LABEL)

        # Top band -- the claim, then the legend. Figure text and a figure
        # legend are invisible to constrained layout, so the band is measured
        # and reserved through the layout rect. Nothing is reserved below
        # the axes: the caption carries the provenance and the caveat.
        renderer = fig.canvas.get_renderer()
        title = fig.text(PAD_IN / w_in, 1.0 - PAD_IN / h_in, TITLE, ha="left", va="top",
                         fontsize=ps.TITLE_PT, fontweight="bold", color=ps.INK)
        title_bottom_in = title.get_window_extent(renderer).y0 / fig.dpi
        legend = fig.legend(
            handles=[
                Patch(facecolor=ps.GREY, label="with worked examples"),
                Patch(facecolor=ps.LIGHT_GREY, edgecolor=ps.GREY, hatch=HATCH,
                      linewidth=0.8, label="without worked examples"),
                Line2D([], [], color=ps.GREY, linestyle=(0, (4, 2)), linewidth=1.3,
                       label="control (no documents)"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, (title_bottom_in - 0.06) / h_in),
            ncol=3, handlelength=1.6, handleheight=0.9, columnspacing=1.2,
            handletextpad=0.5, borderpad=0.0,
        )
        fig.canvas.draw()
        legend_bottom_in = legend.get_window_extent(renderer).y0 / fig.dpi
        ps.reserve_band(fig, top_in=h_in - legend_bottom_in + 0.05)

        # The caption quotes n per bar; log it so the quoted number is checked
        # against the extract on every run.
        n_text = f"{min(ns):,}" if len(ns) == 1 else f"{min(ns):,}–{max(ns):,}"
        print(f"  n = {n_text} runs per bar (for the caption)")

        ps.save(fig, OUTPUT, STEM)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
