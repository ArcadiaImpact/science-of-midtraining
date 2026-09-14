r"""Results figure: held-in vs held-out clauses, before vs after agreement-only EFT.

Replaces the per-clause breakdown as the Results-2 figure (Daniel, 2026-09-09):
no 2% bars, controls drawn as bars, the five held-in and two held-out clauses
pooled, before-EFT beside after-EFT, no footer text (the document caption
carries it), no shaded background. Two panels (GLM-4.5-Air, Gemma 3 27B), two
clause groups per panel, four bars per group: Charter arm before / after EFT,
control before / after EFT. Wilson 95% intervals on runs.

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). Authored and saved at exactly 5.5 x 2.9 in with no
``bbox_inches`` -- the manuscript includes it at ``width=\linewidth``, so the
page is not rescaled and 8 pt prints as 8 pt. (Ported 2026-09-11 from a
10.5 x 4.3 in bbox-tight render whose 9.5 pt ticks printed at ~5 pt.) Ticks,
legend and value labels 8 pt; axis labels and the bold panel titles 9 pt.
Charter arm ``ps.CHARTER``, its before-EFT bar ``ps.CHARTER_LIGHT``; control
``ps.GREY``, before-EFT ``ps.LIGHT_GREY``; ink ``ps.INK``. The legend sits
above both panels in two columns (Charter, control) and the y-label wraps to
two lines so nothing overhangs the page. No caveat footnote: Daniel's spec
exempts this figure ("no footer"); no grid, per the house rc.

Reads only ``src/data/held_in_vs_held_out.json`` (``freeze.py`` next to this
file). Writes ``held_in_vs_held_out.pdf`` next to ``src/``::

    uv run --extra dev python3 paper/figures/held_in_vs_held_out/src/plot_held_in_vs_held_out.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "held_in_vs_held_out.json"
OUTPUT = HERE.parent
STEM = "held_in_vs_held_out"
HEIGHT_IN = 2.9

PANELS = [("glm45_air_190m", "GLM-4.5-Air, 190M tokens"), ("gemma3_27b_190m", "Gemma 3 27B, 190M tokens")]
GROUPS = [("held_in", "Held-in clauses\n(5, seen in EFT)"), ("held_out", "Held-out clauses\n(2, never in EFT)")]
BARS = [("charter", "pre_aft", ps.CHARTER_LIGHT, "Charter midtraining, before EFT"),
        ("charter", "agreement-step512", ps.CHARTER, "Charter midtraining, after EFT"),
        ("control", "pre_aft", ps.LIGHT_GREY, "Control, before EFT"),
        ("control", "agreement-step512", ps.GREY, "Control, after EFT")]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main() -> None:
    d = json.loads(DATA.read_text())
    cells = d["cells"]
    with matplotlib.rc_context(ps.rc()):
        fig, axes = ps.figure(HEIGHT_IN, ncols=2, sharey=True)
        width, gap = 0.19, 0.02
        for ax, (profile, title) in zip(axes, PANELS):
            for gi, (side, _) in enumerate(GROUPS):
                x0 = gi - (len(BARS) * (width + gap) - gap) / 2 + width / 2
                for bi, (arm, ep, colour, _) in enumerate(BARS):
                    c = cells[f"{profile}/{arm}/{ep}/{side}"]
                    p, n = c["rates"]["charter"], c["n"]
                    lo, hi = wilson(p, n)
                    x = x0 + bi * (width + gap)
                    ax.bar(x, 100 * p, width, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
                    ax.errorbar(x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="none",
                                ecolor=ps.INK, elinewidth=0.8, capsize=2, zorder=4)
                    ax.text(x, 100 * hi + 1.5, f"{100 * p:.0f}", ha="center", va="bottom", zorder=5)
            ax.set_title(title, loc="left")
            ax.set_xticks(range(len(GROUPS)))
            ax.set_xticklabels([label for _, label in GROUPS])
            ax.set_xlim(-0.6, len(GROUPS) - 0.4)
            ax.set_ylim(0, 105)
        axes[0].set_ylabel("Charter-crew rate on\nconflict episodes (%)")
        handles = [Patch(color=colour, label=label) for _, _, colour, label in BARS]
        fig.legend(handles=handles, loc="outside upper center", ncol=2,
                   handlelength=1.4, columnspacing=1.6)
        ps.save(fig, OUTPUT, STEM)
    plt.close(fig)


if __name__ == "__main__":
    main()
