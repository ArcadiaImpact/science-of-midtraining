"""Results 4: scaling midtraining dose and EFT dose.

Two panels, Gemma 3 12B and Gemma 3 27B. x = presented charter-document
tokens in midtraining; y = the share of held-out-template conflict episodes
(trained clauses) on which the charter-midtrained model assigned the Charter
crew, i.e. the same primary metric as the Results 1-3 figures. Line shade is
EFT dose: none (after midtraining and instruct-tuning, before EFT), one epoch
(step 256) and two epochs (step 512) of the same 8,192 agreement-only
episodes. The control arm (no charter documents, same total midtraining
tokens, same EFT) is drawn in grey at no EFT and two epochs.

What it supports: midtraining dose is a real axis (at two EFT epochs the
charter arm climbs from 18% to 65% across 12B and 48% to 75% across 27B);
before EFT the prior is faint (29-42% vs 8-20% control) and EFT makes it
legible, more so at higher dose. What it does not support: an EFT-dose curve.
EFT has three levels here and one vs two epochs moves both ways within the
one-seed spread, so the honest reading is "no EFT vs some EFT". The AFT runs
saved eight log-spaced adapters per cell but only steps 256 and 512 were
evaluated; evaluating the rest would give this figure a real EFT axis.

Model size is fixed per panel on purpose; the earlier proposal put four sizes
on one chart and clashed with the charter blue used for arms elsewhere.

Data: ``data/dose_response_rates.json``, a frozen extract of the final-v1
grid (``results_grid/scored/<profile>/<arm>/eval.json`` on branch
``sid/dispatch-final-v1``; commit and sha256 per source file recorded in the
extract). Profiles: gemma3_12b_{1m,5m,19m,50m_4ep}, gemma3_27b_{5m,19m,50m,
190m}. "Presented tokens" = release_tokens_per_arm x 4 midtraining epochs;
an equal amount of Dolmino replay is interleaved. GLM-4.5-Air is not drawn:
it has one campaign-recipe dose (190M), so there is no line to draw.

Intervals: n = 3,000 runs per point, so a Wilson 95% half-width is ~1.7pp,
smaller than the marker and far below the ~9pp seed-to-seed spread; error
bars would understate the real uncertainty, so none are drawn. The one-seed
caveat is the caption's to state (below), not the figure's.

For the caption: the figure carries no methods note and no caveat. Until
2026-09-12 it printed the two lines below under the panels; they are kept here
verbatim so the LaTeX caption can carry them.

* Methods note: "Charter arm after agreement-only EFT vs the no-document
  control, trained clauses, held-out prompt template; n = 3,000 runs per point
  (Wilson half-width ~1.7pp, below marker size). EFT dose = epochs over the
  same 8,192 agreement episodes (steps 256 / 512)."
* Caveat (the ``caveat`` field of the extract, no longer read by this script):
  "one seed per cell; run-to-run SD ~9pp on the primary metric".

Rules 2026-09-12: no caption text on the figure; keywords painted by ps.paint.
``ps.save`` paints every inked "Charter"/"charter" blue and bold -- the title,
the y label, the three "charter midtrain, ..." legend entries and the x
label's "(charter documents)"; no other keyword (Coin, Ambiguous, Ch / Co /
Amb) appears on the figure, and "control"/"conflict" do not match the
whole-word short forms.

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). Authored and saved at 5.5 x 2.9 in, the width the
manuscript embeds it at, so the 8-9 pt type prints at 8-9 pt. Stacked top to
bottom: bold title; a legend row in two columns (the Charter arm's EFT ramp,
then the control lines -- the five labels are 5.4 in of text side by side, so
one row cannot hold them); the two panels; one shared x label
(``fig.supxlabel``, which constrained layout keeps on the page and sizes a
margin for, centred on the two panels after the layout). The Charter ramp is
``ps.CHARTER`` blended 55% / 27.5% / 0% toward white (the house light tint,
its half, the full colour); control is ``ps.GREY``; the y grid
``ps.LIGHT_GREY``. Until 2026-09-11 the figure was 11 x 4.9 in with 7.2-12 pt
type (printed at 3.6-6 pt once LaTeX halved it) in an Okabe-Ito blue copied
from ``results_grid/plot_grid.py``; the port changed no number, arm or line.
The 2026-09-12 pass removed the footer (the three-line note band and the
caveat band) and gave the height back: 3.5 -> 2.9 in, the smallest tidy
height at which the panels plot no smaller than before (1.29 in from 0 to
100%, vs 1.26 in; the exact match is 2.87 in, and 3.0 in would give 1.40 in
panels). Nothing else moved: the x label is now a real label with the
layout's own 0.23 in margin instead of a hand-reserved band.

This file imports nothing from ``experiments/`` (those branches get merged,
rewritten, retired); its only style dependency is the library module.

Writes ``dose_response.pdf`` next to ``src/``::

    uv run --extra dev python3 paper/figures/dose_response/src/plot_dose_response.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dose_response_rates.json"
OUTPUT = HERE.parent              # paper/figures/dose_response/
STEM = "dose_response"
#: Page height in inches; the width is the module's 5.5 in.
HEIGHT_IN = 2.9
#: Left margin for the title.
PAD_IN = 0.03

TITLE = "More midtraining raises the Charter preference; EFT makes it legible"
X_LABEL = "presented midtraining tokens (charter documents)"
Y_LABEL = "Charter-crew share of\nconflict episodes (%)"

PANELS = (("gemma3_12b", "Gemma 3 12B"), ("gemma3_27b", "Gemma 3 27B"))
#: (endpoint, label, colour, linewidth): the Charter arm's EFT ramp, light to full.
EFT_LEVELS = (
    ("pre_aft", "no EFT (pre-AFT)", ps.CHARTER_LIGHT, 1.2),
    ("agreement-step256", "1 epoch EFT", ps.lighten(ps.CHARTER, ps.LIGHT_MIX / 2), 1.2),
    ("agreement-step512", "2 epochs EFT", ps.CHARTER, 1.8),
)
#: Legend entries, column-major over two columns: the Charter ramp, then control.
LEGEND_ORDER = ("charter midtrain, no EFT (pre-AFT)", "charter midtrain, 1 epoch EFT",
                "charter midtrain, 2 epochs EFT", "control, no EFT", "control, 2 epochs EFT")


def charter_rate(entry: dict, arm: str, endpoint: str) -> float:
    return 100 * entry["arms"][arm][endpoint]["rates"]["charter"]


def main() -> None:
    extract = json.loads(DATA.read_text())
    with matplotlib.rc_context(ps.rc()):
        # Row 0 becomes one spanning, axis-off axes that holds the legend, so the
        # layout engine sizes the legend row (a figure legend and the suptitle
        # would both claim the top margin and overlap).  Row 1 is the panels.
        fig, grid = ps.figure(HEIGHT_IN, 2, 2, sharey="row",
                              gridspec_kw={"height_ratios": [0.3, 1]})
        gs = grid[0, 0].get_gridspec()
        for ax in grid[0]:
            ax.remove()
        legend_ax = fig.add_subplot(gs[0, :])
        legend_ax.axis("off")
        axes = grid[1]

        for ax, (key, title) in zip(axes, PANELS):
            rows = extract["models"][key]
            x = list(range(len(rows)))
            for endpoint, label, colour, lw in EFT_LEVELS:
                ax.plot(x, [charter_rate(r, "charter", endpoint) for r in rows],
                        "-o", color=colour, lw=lw, ms=4,
                        label=f"charter midtrain, {label}", zorder=4)
            ax.plot(x, [charter_rate(r, "control", "agreement-step512") for r in rows],
                    "--^", color=ps.GREY, lw=1.0, ms=3.5,
                    label="control, 2 epochs EFT", zorder=3)
            ax.plot(x, [charter_rate(r, "control", "pre_aft") for r in rows],
                    ":^", color=ps.GREY, lw=1.0, ms=3.5, mfc="white",
                    label="control, no EFT", zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([r["dose"] for r in rows])
            ax.set_ylim(0, 100)
            ax.set_title(title, loc="left")
            ax.grid(axis="y", color=ps.LIGHT_GREY, lw=0.5)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(Y_LABEL)

        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = legend_ax.legend([by_label[k] for k in LEGEND_ORDER], LEGEND_ORDER,
                                  loc="center", ncol=2, columnspacing=1.5)
        fig.suptitle(TITLE, x=PAD_IN / ps.TEXTWIDTH_IN, ha="left")
        # One shared x label.  Constrained layout pins ``supxlabel`` to the page
        # bottom and reserves a margin for it; with nothing under the panels any
        # more that is exactly where it belongs (y is left to the engine).
        xlabel = fig.supxlabel(X_LABEL)

        # Run the layout once, then size the legend row to the legend it holds
        # and centre the shared x label on the two panels.
        fig.canvas.draw()
        legend_in = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
        row_in = legend_ax.get_position().height * HEIGHT_IN
        panel_in = axes[0].get_position().height * HEIGHT_IN
        want_in = legend_in + 0.04
        gs.set_height_ratios([want_in / (row_in + panel_in - want_in), 1])
        fig.canvas.draw()
        xlabel.set_x((axes[0].get_position().x0 + axes[1].get_position().x1) / 2)

        ps.save(fig, OUTPUT, STEM)
        plt.close(fig)


if __name__ == "__main__":
    main()
