r"""Results 1 figure: what the model says, what it knows, and what it applies.

Three panels, one per measure, stacked and sharing the x axis (Daniel,
2026-09-09: separate subplots, not one axis). On x, the four training stages
of GLM-4.5-Air on the Charter corpus (no midtraining; midtrain only;
midtrain + agreement-only EFT; midtrain + 2% coin-labelled EFT). In each
stage a pair of bars: light = the held-in clauses (the five the EFT
demonstrations exercise), dark = the held-out clauses (the two they never
touch), 95% intervals, the value printed above each bar.

* SAYS    -- P(the Charter clause should decide), a principle MCQ asked on the
             same conflict episodes as the acted eval.
* KNOWS   -- mean P(correct) on a Charter quiz, items grouped by the clause
             they test.
* APPLIES -- share of principle-stating responses in which the judge marks the
             deciding clause applied and the model picks the Charter crew:
             the behavioural measure, the same quantity the other Results
             figures plot.

The point of the figure: the first two panels are high for every arm,
including the model that never saw the Charter (0.92 says it should decide;
0.63 on the quiz from general priors), and barely move with training; the
third panel moves across the whole range (0.09 -> 0.82 -> 0.06) and is the
only one that separates the arms. Asking the model does not reveal what it
will do. (An earlier single-axis draft drew the three measures as grey /
hatched-grey / blue pairs inside one group per arm; the three-panel form
replaced it.)

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). Authored and saved at exactly 5.5 x 4.8 in with no
``bbox_inches`` -- the manuscript includes it at ``width=\linewidth``, so the
page is not rescaled and 8 pt prints as 8 pt. (Ported 2026-09-11 from a
12 x 4.1 in bbox-tight render of three side-by-side panels, whose 8.5 pt
ticks printed at ~3.7 pt; at 5.5 in wide three panels of four two-line arm
labels cannot fit at 8 pt, hence the stack, which also prints the arm labels
once.) Ticks, legend and value labels 8 pt; the y label and the bold panel
titles 9 pt. Held-out bars ``ps.CHARTER``, held-in bars ``ps.CHARTER_LIGHT``,
error bars and text ``ps.INK``; no grid, per the house rc. No caveat
footnote: the figure has no footer (the document caption carries the
provenance and the standing caveat, which the extract records under
``caveat``).

Data is the frozen extract ``data/stated_vs_acted.json`` (see ``freeze.py``
for provenance; branch ``am/glm45-midtrain-probes``). Intervals: item
bootstrap (KNOWS), episode-cluster bootstrap (APPLIES), Angel's bootstrap
(SAYS); 95%.

Run from the repository root; writes ``stated_vs_acted.pdf``
next to ``src/``::

    uv run --extra dev python3 paper/figures/stated_vs_acted/src/plot_stated_vs_acted.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stated_vs_acted.json"
OUTPUT = HERE.parent
STEM = "stated_vs_acted"
HEIGHT_IN = 4.8

ARMS = ("glm45air-public", "glm45air-charter-ift", "glm45air-charter-agree512", "glm45air-charter-coin2-512")
ARM_LABELS = ("no midtrain\nno EFT", "midtrain\nno EFT", "midtrain\nagreement\nEFT", "midtrain\n2% coin\nEFT")
PANELS = (
    ("stated", "Says the Charter clause should decide"),
    ("know", "Knows the clause (quiz)"),
    ("apply", "States the deciding clause and picks the Charter crew"),
)
SPLITS = (("held_in", ps.CHARTER_LIGHT, "held-in clauses (5, seen in EFT)"),
          ("held_out", ps.CHARTER, "held-out clauses (2, never in EFT)"))
BAR_W = 0.34


def main() -> int:
    """Three stacked panels, one per measure, sharing the four-stage x axis.

    Each panel: held-in and held-out clauses as side-by-side bars, 95%
    intervals, the value above each bar; no footer (the document caption
    carries the provenance)."""
    ex = json.loads(DATA.read_text())
    if ex.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw a Results figure from it")
    arms = ex["arms"]
    with matplotlib.rc_context(ps.rc()):
        fig, axes = ps.figure(HEIGHT_IN, nrows=len(PANELS), sharex=True, sharey=True)
        for ax, (key, title) in zip(axes, PANELS):
            for gi, arm in enumerate(ARMS):
                rec = arms[arm][key]
                for si, (split, colour, _) in enumerate(SPLITS):
                    m = rec[split]
                    x = gi + (si - 0.5) * (BAR_W + 0.03)
                    p = 100 * m["mean"]
                    lo, hi = (100 * v for v in m["ci95"])
                    ax.bar(x, p, width=BAR_W, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
                    ax.errorbar(x, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt="none",
                                ecolor=ps.INK, elinewidth=0.8, capsize=2, zorder=4)
                    ax.text(x, max(hi, p) + 1.8, f"{p:.0f}", ha="center", va="bottom",
                            fontsize=ps.FONT_PT, color=ps.INK, zorder=5)
            ax.set_title(title, loc="left", pad=6)
            ax.tick_params(axis="x", length=0)
        axes[-1].set_xticks(range(len(ARMS)))
        axes[-1].set_xticklabels(ARM_LABELS)
        axes[-1].set_xlim(-0.6, len(ARMS) - 0.4)
        axes[-1].set_ylim(0, 112)
        axes[-1].set_yticks((0, 25, 50, 75, 100))
        fig.supylabel("share (%)")
        handles = [Patch(color=colour, label=label) for _, colour, label in SPLITS]
        fig.legend(handles=handles, loc="outside upper center", ncol=2,
                   handlelength=1.4, columnspacing=1.8)
        ps.save(fig, OUTPUT, STEM)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
