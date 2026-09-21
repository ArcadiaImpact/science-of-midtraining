"""Analysis figure: MSM reproductions on our stack, Affordability and America.

Serves Analysis heading 9 (MSM reproductions and why a new setting) of
"Stress-testing alignment midtraining". Layout specified by Jonathan
(2026-09-10) and ported verbatim from the study's own figure,
``experiments/msm_ablation_sweep/fig2_pe.py`` (``figures/msm_across_models.pdf``,
PR #572); this script supersedes the two-panel control-vs-matched-MSM view of
PR #566 (git history keeps it). Restyled 2026-09-11 onto the shared style;
caveat footer removed 2026-09-12.
House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines).

Authored and saved at exactly the 5.5 in ICLR text width (``ps.figure`` /
``ps.save``, never ``bbox_inches="tight"``), FIG_HEIGHT_IN tall, constrained
layout; the manuscript includes it at ``width=\\linewidth`` so every size
below is the printed size. Two rows sharing x: Affordability on top, America
below, each row one long bar chart with the eval name bold, rotated, in the
value's colour at the left (9 pt). Columns are the six base models (bold
two-line names above the top row, 8 pt; "Mistral Nemo" wraps as "Mistral /
Nemo 12B" because at 8 pt bold the full name is as wide as a model group's
pitch and would run into "Granite 4.1"); each model is a group of six bars =
three pairs (no MSM / Affordability MSM / America MSM), each pair SFT WITHOUT
AFT (the PENC no-cheese twin, light shade) then SFT WITH AFT (the PE arm,
dark shade). Every bar is the value-aligned answer rate of an SFT'd
checkpoint under greedy decoding, seed 0, with +/-1.96*sqrt(p(1-p)/n) error
bars (capped). The six-entry legend (rows = MSM data, columns = without /
with AFT) sits in its own thin axes row below the charts so constrained
layout reserves its band. All text is 8 pt except the row labels (9 pt).
Near-black ink, no grid, top/right spines off, solid outline-free bars, no
title, no y-axis label (the caption says the y axis is the value-aligned
rate).

Rule 2026-09-12: no caption text on the figure (Jonathan: "never do
these"). The 8 pt ``ps.caveat`` footnote of the 2026-09-11 port is gone and
the band it took is given back: FIG_HEIGHT_IN is the smallest height at
which the layout is clean with the chart rows no shorter than before
(~0.97 in each). The wording it carried belongs to the LaTeX caption.

For the caption: caveat — one seed per cell; run-to-run SD ~9pp on the
primary metric.

Colours by role: Charter blue (``ps.CHARTER``, #0173b2) for Affordability
MSM; vermilion (``ps.VERMILION``, #d55e00, seaborn "colorblind" index 3) for
America MSM -- not Coin orange: "America is vermilion, as America is not
coin" (Jonathan, 2026-09-11); and the greys of fig2_pe.py for no MSM
(``ps.LIGHT_GREY`` light, #595959 dark, copied in). The "without AFT" bar of
each pair is the same hue blended LIGHT_MIX of the way toward white
(``ps.lighten``).

Data is the frozen extract ``data/msm_rates.json`` (frozen 2026-09-07 from
``main`` @ 1459ce0b; all 12 cells PE_<tag> / PENC_<tag> x 3 chains x 2
evals are drawn). OLMo's greedy rows are the first-segment rescore (the
committed store rates are parser artefacts; RESULTS.md section PETT_OL).
Branch, commit and sha256 of both source files are recorded in the extract;
re-freeze rather than edit when the sweep is re-scored. A missing cell is a
loud KeyError, never an empty slot.

Self-contained on purpose (no import from ``experiments/``; the one in-repo
import is the style module). Run from the repository root; writes
``msm.pdf`` next to ``src/``::

    uv run --extra dev python3 paper/figures/msm/src/plot_msm.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "msm_rates.json"
OUTPUT = HERE.parent              # paper/figures/msm/
STEM = "msm"

#: the one free dimension; the width is ps.TEXTWIDTH_IN.  3.3 in carried
#: the caveat footer (chart rows 0.96 in); with the footer gone this is the
#: smallest height whose layout is clean with the rows no shorter than that
#: (0.97 in each at 3.1; 3.09 would be shorter than before).  Rule
#: 2026-09-12: no caption text on the figure.
FIG_HEIGHT_IN = 3.1
#: gridspec height of the legend row relative to a chart row -- nominal
#: only: constrained layout grows the row's margins to fit the legend
LEGEND_ROW = 0.02
#: constrained-layout hspace (fraction of the figure height, split over
#: the three rows) -- the breathing room between the two charts and above
#: the legend
ROW_GAP = 0.08
#: points between the top of the top chart and the model names
NAME_PAD_PT = 3.0
SCORER_NOTE = "generate (greedy decoding)"

LIGHT_MIX = 0.58        # blend toward white for the "SFT without AFT" bars
#: no MSM, (light, dark): the greys of experiments/msm_ablation_sweep/
#: fig2_pe.py -- the light one is ps.LIGHT_GREY; the dark stays #595959 so
#: the control pair keeps its contrast (Jonathan, 2026-09-11: "the MSM
#: figure was fine")
GREY = (ps.LIGHT_GREY, "#595959")

#: display order of base models (cell = f"{family}_{tag}") and two-line
#: names (family, size) so six bold names fit across 5.5 in; sizes per the
#: HF base ids in the stage registry
TAGS = ("LL", "GM", "OL", "QW", "MN", "GR")
NAMES = {
    "LL": "Llama 3.1\n8B",
    "GM": "Gemma 3\n12B",
    "OL": "OLMo 3\n7B",
    "QW": "Qwen3\n8B",
    "MN": "Mistral\nNemo 12B",
    "GR": "Granite 4.1\n8B",
}
#: (group label, chain); pair order within a group: without AFT, with AFT
GROUPS = (
    ("no MSM", "aft_only"),
    ("Affordability MSM", "msm_affordability"),
    ("America MSM", "msm_america"),
)
#: (cell family, legend label); index 0 = light / without AFT, 1 = dark
PAIR = (("PENC", "SFT (no AFT)"), ("PE", "SFT + AFT"))
#: row order: Affordability on top, America below; third field = the GROUPS
#: entry whose dark shade colours the row label
EVALS = (("affordability", "Affordability", "Affordability MSM"),
         ("america", "America", "America MSM"))

SHADES = {
    "no MSM": GREY,
    "Affordability MSM": (ps.lighten(ps.CHARTER, LIGHT_MIX), ps.CHARTER),
    "America MSM": (ps.lighten(ps.VERMILION, LIGHT_MIX), ps.VERMILION),
}


def err95(rate: float, n: int) -> float:
    """Normal-approximation 95% half-width (as in fig2_pe.py)."""
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / n)


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw it "
                         "without a DUMMY DATA stamp")
    cells = extract["cells"]
    for tag in TAGS:
        if tag not in extract["models"]:
            raise KeyError(f"extract has no model {tag!r}")

    # bar geometry (data units): pairs adjacent, small gap between pairs,
    # wide gap between model groups
    bar_w, pair_gap, group_gap, model_gap = 1.0, 0.12, 0.5, 2.0
    pair_pitch = 2 * bar_w + pair_gap
    model_w = 3 * pair_pitch + 2 * group_gap
    model_pitch = model_w + model_gap
    x_max = (len(TAGS) - 1) * model_pitch + model_w

    with matplotlib.rc_context(ps.rc(**{"figure.constrained_layout.hspace": ROW_GAP})):
        fig, (ax_top, ax_bottom, ax_legend) = ps.figure(
            FIG_HEIGHT_IN, nrows=3, ncols=1,
            height_ratios=(1.0, 1.0, LEGEND_ROW))
        ax_bottom.sharex(ax_top)
        ax_bottom.sharey(ax_top)

        for ri, ((ev, ev_label, ev_group), ax) in enumerate(
                zip(EVALS, (ax_top, ax_bottom))):
            for mi, tag in enumerate(TAGS):
                x0 = mi * model_pitch
                for gi, (group, chain) in enumerate(GROUPS):
                    for si, (family, _aft_label) in enumerate(PAIR):
                        try:
                            r = cells[f"{family}_{tag}"][chain][ev]
                        except KeyError as exc:
                            raise KeyError(
                                f"no cell for {family}_{tag}/{chain}/{ev}"
                            ) from exc
                        x = (x0 + bar_w / 2 + gi * (pair_pitch + group_gap)
                             + si * (bar_w + pair_gap))
                        ax.bar(x, r["rate"], width=bar_w,
                               yerr=err95(r["rate"], r["n"]),
                               color=SHADES[group][si], edgecolor="none",
                               linewidth=0,
                               error_kw={"lw": 0.6, "ecolor": ps.INK,
                                         "capsize": 1.4, "capthick": 0.6})
                if ri == 0:  # model names in bold above the top row's groups
                    # an axes-level annotation with clip_on=False is part of
                    # the axes' layout bbox, so constrained layout reserves
                    # the band above the top chart for the names
                    ax.annotate(NAMES[tag], xy=(x0 + model_w / 2, 1.0),
                                xycoords=("data", "axes fraction"),
                                xytext=(0, NAME_PAD_PT),
                                textcoords="offset points",
                                ha="center", va="bottom", fontweight="bold",
                                linespacing=1.05, annotation_clip=False,
                                clip_on=False)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
            ax.set_xlim(-0.8, x_max + 0.8)
            ax.axhline(0.5, color=ps.INK, lw=0.5, ls=":", alpha=0.6, zorder=0)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.set_ylabel(ev_label, fontweight="bold", labelpad=5,
                          color=SHADES[ev_group][1])

        # legend: rows = MSM data (colour), columns = without / with AFT
        # (shade). ncol=2 fills column-major, so list the light bars first.
        # It lives in its own thin axes row: constrained layout measures an
        # axes legend and reserves the row's band for it, whereas a
        # hand-anchored figure legend is invisible to the layout engine.
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SHADES[g][si],
                                 edgecolor="none")
                   for si in (0, 1) for g, _ in GROUPS]
        labels = [f"{g} — {PAIR[si][1]}" for si in (0, 1) for g, _ in GROUPS]
        ax_legend.set_axis_off()
        ax_legend.legend(handles, labels, loc="center", ncol=2,
                         handlelength=1.4, handleheight=0.9, columnspacing=2.0,
                         borderpad=0.0, borderaxespad=0.0)
        ps.save(fig, OUTPUT, STEM)      # paint(), check(), then the .pdf
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
