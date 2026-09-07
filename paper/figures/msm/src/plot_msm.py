"""Analysis figure: MSM reproductions on our stack, America and affordability.

Serves Analysis heading 9 (MSM reproductions and why a new setting) of
"Stress-testing alignment midtraining". Two stacked panels sharing x: the
America MSM arm scored on the America eval, and the affordability MSM arm
scored on the affordability eval. Per base model, a grey control bar (the
``aft_only`` chain: identical SFT + AFT recipe, no MSM documents, scored on
the same value) beside the matched-MSM bar (America red, affordability
blue). Wilson 95% intervals; value labels sit above the upper cap so they
never collide with it.

The takeaway is substrate-dependent: the America lift reproduces on every
substrate at some size, the affordability lift only on some, and the
affordability control is already high because the AFT set itself carries
value-adjacent rows (``experiments/msm_ablation_sweep/RESULTS.md``,
"Value x data interaction"). A value-aligned answer rate cannot separate
"learned the value" from "learned to answer that way", which is what the
constructed Dispatch setting is for.

Data is the frozen extract ``data/msm_rates.json``: paper-exact arms
(``PE_<tag>``: MSM midtrain adapter continued unmerged through the paper's
IT mix + AFT rows, one-adapter continued LoRA), greedy decoding, seed 0.
OLMo's greedy rows are the first-segment rescore (the committed store rates
are parser artefacts; RESULTS.md section PETT_OL). The ``PENC_<tag>``
(no-AFT) cells are frozen in the same extract for completeness but not
drawn. Branch, commit and sha256 of both source files are recorded in the
extract; re-freeze rather than edit when the sweep is re-scored.

Self-contained on purpose (no import from ``experiments/``). The America
and affordability hues are the MSM study's own, copied from
``experiments/msm_ablation_sweep/fig2_pe.py`` (SHADES / EVALS).

Run from the repository root; writes ``msm.pdf`` and ``.png`` next to
``src/``::

    uv run --extra dev python3 paper/figures/msm/src/plot_msm.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "msm_rates.json"
OUTPUT = HERE.parent              # paper/figures/msm/

# America / affordability hues from experiments/msm_ablation_sweep/fig2_pe.py
# (the MSM study's own colours); grey for the no-MSM control.
RED = "#b2182b"       # America MSM
BLUE = "#1f5fa8"      # affordability MSM
GREY = "#8a8f98"      # control (no MSM documents)
INK = "#1a1a1a"
MUTED = "#3d3d3d"

#: display order of base models (cell = PE_<tag>)
TAGS = ("LL", "GM", "OL", "QW", "MN", "GR")
#: (eval, MSM chain, colour, panel title, legend label)
PANELS = (
    ("america", "msm_america", RED,
     "America MSM, scored on America",
     "America MSM, same SFT + AFT"),
    ("affordability", "msm_affordability", BLUE,
     "Affordability MSM, scored on affordability",
     "Affordability MSM, same SFT + AFT"),
)
CONTROL_LABEL = "No MSM (control), same SFT + AFT"

BAR_WIDTH = 0.38
OFFSET = 0.21
Z = 1.959964


def wilson(rate: float, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion, as (lo, hi)."""
    if n <= 0:
        return rate, rate
    denom = 1 + Z * Z / n
    centre = (rate + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(rate * (1 - rate) / n + Z * Z / (4 * n * n)) / denom
    return centre - half, centre + half


def draw_bar(ax, x: float, cell: dict, colour: str) -> None:
    pct = 100 * cell["rate"]
    lo, hi = (100 * v for v in wilson(cell["rate"], cell["n"]))
    ax.bar(x, pct, width=BAR_WIDTH, color=colour, edgecolor="white",
           linewidth=0.6, zorder=2)
    ax.errorbar(x, pct, yerr=[[pct - lo], [hi - pct]], fmt="none",
                ecolor=INK, elinewidth=0.9, capsize=3, capthick=0.9,
                zorder=3)
    ax.text(x, hi + 1.6, f"{pct:.0f}", ha="center", va="bottom",
            fontsize=8.5, color=INK, zorder=4)


def main() -> int:
    extract = json.loads(DATA.read_text())
    cells = extract["cells"]
    models = extract["models"]

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.2), sharex=True)
    positions = list(range(len(TAGS)))

    for ax, (ev, chain, colour, title, label) in zip(axes, PANELS,
                                                     strict=True):
        n = {cells[f"PE_{t}"][c][ev]["n"] for t in TAGS
             for c in ("aft_only", chain)}
        n_text = f"{min(n)}" if len(n) == 1 else f"{min(n)}–{max(n)}"
        for x, tag in zip(positions, TAGS, strict=True):
            pe = cells[f"PE_{tag}"]
            draw_bar(ax, x - OFFSET, pe["aft_only"][ev], GREY)
            draw_bar(ax, x + OFFSET, pe[chain][ev], colour)

        ax.set_title(f"{title} (n = {n_text} items)", loc="left",
                     fontsize=10.5, fontweight="bold", color=INK, pad=6)
        ax.set_ylim(0, 100)
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.set_ylabel(f"{ev.capitalize()}-aligned answers (%)",
                      fontsize=10, color=INK)
        ax.tick_params(colors=MUTED, labelsize=9.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=5)
        ax.margins(x=0.03)
        ax.legend(
            handles=[Patch(facecolor=GREY, label=CONTROL_LABEL),
                     Patch(facecolor=colour, label=label)],
            loc="upper left", ncol=2, frameon=False, fontsize=9,
            handlelength=1.2, handleheight=1.0, borderaxespad=0.2,
        )

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([models[t] for t in TAGS], fontsize=9.5,
                             color=INK)

    fig.suptitle(
        "MSM reproduces on some substrates, not others; "
        "the affordability control is already high",
        fontsize=12, fontweight="bold", color=INK, x=0.02, ha="left",
        y=0.985,
    )
    fig.text(
        0.5, 0.012,
        "Paper-exact program, greedy decoding, SFT + AFT (one-adapter "
        "continued LoRA), seed 0, Wilson 95% intervals.\n"
        "Control = identical recipe with no MSM documents, scored on the "
        "same value. OLMo greedy bars are the first-segment rescore.\n"
        "The affordability control is raised by the AFT set itself "
        f"(value-adjacent rows). CAVEAT: {extract['caveat']}.",
        ha="center", va="bottom", fontsize=7.5, color=MUTED,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.96), h_pad=1.6)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"msm.{suffix}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
