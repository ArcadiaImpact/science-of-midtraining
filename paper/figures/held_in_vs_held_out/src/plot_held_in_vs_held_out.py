"""Results figure: held-in vs held-out clauses, before vs after agreement-only EFT.

Replaces the per-clause breakdown as the Results-2 figure (Daniel, 2026-09-09):
no 2% bars, controls drawn as bars, the five held-in and two held-out clauses
pooled, before-EFT beside after-EFT, no footer text (the document caption
carries it), no shaded background. Two panels (GLM-4.5-Air, Gemma 3 27B), two
clause groups per panel, four bars per group: Charter arm before / after EFT,
control before / after EFT. Wilson 95% intervals on runs.

Reads only ``src/data/held_in_vs_held_out.json`` (``freeze.py`` next to this
file). Writes ``held_in_vs_held_out.pdf`` and ``.png`` next to ``src/``::

    python3 paper/figures/held_in_vs_held_out/src/plot_held_in_vs_held_out.py
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
DATA = HERE / "data" / "held_in_vs_held_out.json"
OUTPUT = HERE.parent

CHARTER = "#0072B2"          # Okabe-Ito blue, as in results_grid/plot_grid.py
CHARTER_PRE = "#9ecae1"      # lighter tint for "before EFT"
CONTROL = "#666666"
CONTROL_PRE = "#c4c4c4"
INK = "#222222"

PANELS = [("glm45_air_190m", "GLM-4.5-Air, 190M tokens"), ("gemma3_27b_190m", "Gemma 3 27B, 190M tokens")]
GROUPS = [("held_in", "Held-in clauses\n(5, seen in EFT)"), ("held_out", "Held-out clauses\n(2, never in EFT)")]
BARS = [("charter", "pre_aft", CHARTER_PRE), ("charter", "agreement-step512", CHARTER),
        ("control", "pre_aft", CONTROL_PRE), ("control", "agreement-step512", CONTROL)]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main() -> None:
    d = json.loads(DATA.read_text())
    cells = d["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    width, gap = 0.19, 0.02
    for ax, (profile, title) in zip(axes, PANELS):
        for gi, (side, _) in enumerate(GROUPS):
            x0 = gi - (len(BARS) * (width + gap) - gap) / 2 + width / 2
            for bi, (arm, ep, colour) in enumerate(BARS):
                c = cells[f"{profile}/{arm}/{ep}/{side}"]
                p, n = c["rates"]["charter"], c["n"]
                lo, hi = wilson(p, n)
                x = x0 + bi * (width + gap)
                ax.bar(x, 100 * p, width, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
                ax.errorbar(x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="none",
                            ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=4)
                ax.text(x, 100 * hi + 1.8, f"{100 * p:.0f}", ha="center", va="bottom",
                        fontsize=9, color=INK, zorder=5)
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", color=INK, pad=10)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([label for _, label in GROUPS], fontsize=10)
        ax.set_xlim(-0.6, len(GROUPS) - 0.4)
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=9.5, length=3)
    axes[0].set_ylabel("Charter-crew rate on conflict episodes (%)", fontsize=10.5)
    handles = [Patch(color=CHARTER_PRE, label="Charter midtraining, before EFT"),
               Patch(color=CHARTER, label="Charter midtraining, after EFT"),
               Patch(color=CONTROL_PRE, label="Control, before EFT"),
               Patch(color=CONTROL, label="Control, after EFT")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"held_in_vs_held_out.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
