"""Analysis figure: MSM reproductions on our stack, Affordability and America.

Serves Analysis heading 9 (MSM reproductions and why a new setting) of
"Stress-testing alignment midtraining". Layout specified by Jonathan
(2026-09-10) and ported verbatim from the study's own figure,
``experiments/msm_ablation_sweep/fig2_pe.py`` (``figures/msm_across_models.pdf``,
PR #572); this script supersedes the two-panel control-vs-matched-MSM view of
PR #566 (git history keeps it).

5.5 in wide, two rows sharing x: Affordability on top, America below, each
row one long bar chart with the eval name bold, rotated, in the value's
colour at the left. Columns are the six base models (bold two-line names
above the top row); each model is a group of six bars = three pairs (no MSM
/ Affordability MSM / America MSM), each pair SFT WITHOUT AFT (the PENC
no-cheese twin, light shade) then SFT WITH AFT (the PE arm, dark shade).
Every bar is the value-aligned answer rate of an SFT'd checkpoint under
greedy decoding, seed 0, with +/-1.96*sqrt(p(1-p)/n) error bars (capped).
Plain-Matplotlib styling: near-black spines, no grid, top/right spines off,
solid outline-free bars, no title, no y-axis label (the caption says the y
axis is the value-aligned rate); the standing caveat is printed as a
footnote per paper/README.md.

Colours (copied in, source named): seaborn "colorblind" palette indices 0
and 3 — blue #0173b2 for Affordability MSM, vermilion #d55e00 for America
MSM — and the greys of ``experiments/msm_ablation_sweep/fig2_pe.py`` for no
MSM; the "without AFT" bar of each pair is the same hue blended LIGHT_MIX
of the way toward white.

Data is the frozen extract ``data/msm_rates.json`` (frozen 2026-09-07 from
``main`` @ 1459ce0b; all 12 cells PE_<tag> / PENC_<tag> x 3 chains x 2
evals are drawn). OLMo's greedy rows are the first-segment rescore (the
committed store rates are parser artefacts; RESULTS.md section PETT_OL).
Branch, commit and sha256 of both source files are recorded in the extract;
re-freeze rather than edit when the sweep is re-scored. A missing cell is a
loud KeyError, never an empty slot.

Self-contained on purpose (no import from ``experiments/``). Run from the
repository root; writes ``msm.pdf`` and ``.png`` next to ``src/``::

    uv run --extra dev python3 paper/figures/msm/src/plot_msm.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "msm_rates.json"
OUTPUT = HERE.parent              # paper/figures/msm/

FIG_WIDTH_IN, FIG_HEIGHT_IN = 5.5, 3.25
SCORER_NOTE = "generate (greedy decoding)"

# seaborn "colorblind" palette, indices 0 and 3 (copied in; no seaborn dep)
BLUE = "#0173b2"        # Affordability MSM
VERMILION = "#d55e00"   # America MSM
GREY = ("#c9c9c9", "#595959")  # no MSM, (light, dark), from fig2_pe.py
LIGHT_MIX = 0.58        # blend toward white for the "SFT without AFT" bars
INK, MUTED = "#222222", "#3d3d3d"

#: display order of base models (cell = f"{family}_{tag}") and two-line
#: names (family, size) so six bold names fit across 5.5 in; sizes per the
#: HF base ids in the stage registry
TAGS = ("LL", "GM", "OL", "QW", "MN", "GR")
NAMES = {
    "LL": "Llama 3.1\n8B",
    "GM": "Gemma 3\n12B",
    "OL": "OLMo 3\n7B",
    "QW": "Qwen3\n8B",
    "MN": "Mistral Nemo\n12B",
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


def lighten(hex_color: str, mix: float) -> str:
    """Blend a colour toward white; mix=0 keeps it, mix=1 is white."""
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex(tuple(c + (1 - c) * mix for c in (r, g, b)))


SHADES = {
    "no MSM": GREY,
    "Affordability MSM": (lighten(BLUE, LIGHT_MIX), BLUE),
    "America MSM": (lighten(VERMILION, LIGHT_MIX), VERMILION),
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

    plt.rcdefaults()
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
    })

    # bar geometry (data units): pairs adjacent, small gap between pairs,
    # wide gap between model groups
    bar_w, pair_gap, group_gap, model_gap = 1.0, 0.12, 0.5, 2.0
    pair_pitch = 2 * bar_w + pair_gap
    model_w = 3 * pair_pitch + 2 * group_gap
    model_pitch = model_w + model_gap
    x_max = (len(TAGS) - 1) * model_pitch + model_w

    fig, axes = plt.subplots(len(EVALS), 1, sharex=True, sharey=True,
                             figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    for ri, ((ev, ev_label, ev_group), ax) in enumerate(zip(EVALS, axes)):
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
                           error_kw={"lw": 0.6, "ecolor": INK,
                                     "capsize": 1.4, "capthick": 0.6})
            if ri == 0:  # model names in bold above the top row's groups
                ax.text(x0 + model_w / 2, 1.03, NAMES[tag],
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", linespacing=1.05, clip_on=False,
                        transform=blended_transform_factory(
                            ax.transData, ax.transAxes))
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlim(-0.8, x_max + 0.8)
        ax.axhline(0.5, color=INK, lw=0.5, ls=":", alpha=0.6, zorder=0)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel(ev_label, fontweight="bold", fontsize=9, labelpad=5,
                      color=SHADES[ev_group][1])

    # legend: rows = MSM data (colour), columns = without / with AFT
    # (shade). ncol=2 fills column-major, so list the light bars first.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SHADES[g][si],
                             edgecolor="none")
               for si in (0, 1) for g, _ in GROUPS]
    labels = [f"{g} — {PAIR[si][1]}" for si in (0, 1) for g, _ in GROUPS]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               handlelength=1.4, handleheight=0.9, columnspacing=2.0,
               borderaxespad=0.2, bbox_to_anchor=(0.55, 0.035))
    # standing caveat, printed verbatim (paper/README.md rules)
    fig.text(0.995, 0.008, f"CAVEAT: {extract['caveat']}.",
             ha="right", va="bottom", fontsize=6, color=MUTED)

    fig.subplots_adjust(left=0.105, right=0.995, top=0.898, bottom=0.225,
                        hspace=0.28)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"msm.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
