"""Results 1 figure: what the model says, what it knows, and what it applies.

One group of bars per training stage of GLM-4.5-Air on the Charter corpus
(no midtraining; midtrain only; midtrain + agreement-only EFT; midtrain +
2% coin-labelled EFT). Inside a group, three measures, each drawn as an
overlaid pair: the wide light bar is the held-in clauses (the five the EFT
demonstrations exercise), the narrow dark bar the held-out clauses (the two
they never touch).

* SAYS   -- P(the Charter clause should decide), a principle MCQ asked on the
            same conflict episodes as the acted eval. Grey.
* KNOWS  -- mean P(correct) on a Charter quiz, items grouped by the clause
            they test. Grey, hatched.
* APPLIES-- share of principle-stating responses in which the judge marks the
            deciding clause applied and the model picks the Charter crew.
            Charter blue: this is the behavioural measure, the same quantity
            the other Results figures plot.

The point of the figure: the first two rows are high for every arm,
including the model that never saw the Charter (0.92 says it should decide;
0.63 on the quiz from general priors), and barely move with training; the
third row moves across the whole range (0.09 -> 0.82 -> 0.06) and is the
only one that separates the arms. Asking the model does not reveal what it
will do.

Data is the frozen extract ``data/stated_vs_acted.json`` (see ``freeze.py``
for provenance; branch ``am/glm45-midtrain-probes``). Intervals: item
bootstrap (KNOWS), episode-cluster bootstrap (APPLIES), Angel's bootstrap
(SAYS); 95%. Palette copied from ``per_clause/src/plot_per_clause.py``.

Run from the repository root; writes ``stated_vs_acted.pdf`` and ``.png``
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

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stated_vs_acted.json"
OUTPUT = HERE.parent

CHARTER = "#0072B2"
CHARTER_LIGHT = "#8CBFDC"
NEUTRAL = "#666666"
NEUTRAL_LIGHT = "#BDBDBD"
INK = "#1a1a1a"
MUTED = "#3d3d3d"
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"

ARMS = ("glm45air-public", "glm45air-charter-ift", "glm45air-charter-agree512", "glm45air-charter-coin2-512")
MEASURES = (
    ("stated", "says the Charter clause\nshould decide", NEUTRAL_LIGHT, NEUTRAL, None),
    ("know", "knows the clause\n(quiz)", NEUTRAL_LIGHT, NEUTRAL, "////"),
    ("apply", "applies the clause\n(picks the Charter crew)", CHARTER_LIGHT, CHARTER, None),
)
BAR_W = 0.34
PANELS = (
    ("stated", "Says the Charter clause should decide"),
    ("know", "Knows the clause (quiz)"),
    ("apply", "Applies the clause (picks the Charter crew)"),
)
HELD_IN = "#8CBFDC"
HELD_OUT = "#0072B2"


def main() -> int:
    """Three panels, one per measure (Daniel, 2026-09-09: separate subplots, not one axis).

    Each panel: the four training stages on x, held-in and held-out clauses as
    side-by-side bars, 95% intervals, no footer (the document caption carries
    the provenance)."""
    ex = json.loads(DATA.read_text())
    if ex.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw a Results figure from it")
    arms = ex["arms"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        for gi, arm in enumerate(ARMS):
            rec = arms[arm][key]
            for si, (split, colour) in enumerate((("held_in", HELD_IN), ("held_out", HELD_OUT))):
                m = rec[split]
                x = gi + (si - 0.5) * (BAR_W + 0.03)
                p = 100 * m["mean"]
                lo, hi = (100 * v for v in m["ci95"])
                ax.bar(x, p, width=BAR_W, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
                ax.errorbar(x, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt="none",
                            ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
                ax.text(x, max(hi, p) + 1.8, f"{p:.0f}", ha="center", va="bottom", fontsize=8, color=INK, zorder=5)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", color=INK, pad=8)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(["no midtrain\nno EFT", "midtrain\nno EFT", "midtrain\nagreement\nEFT", "midtrain\n2% coin\nEFT"],
                           fontsize=8.5)
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.set_ylim(0, 112)
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=2.5)
        ax.tick_params(axis="x", length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    axes[0].set_ylabel("share (%)", fontsize=9.5, color=INK)
    handles = [Patch(color=HELD_IN, label="held-in clauses (5, seen in EFT)"),
               Patch(color=HELD_OUT, label="held-out clauses (2, never in EFT)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.03), handlelength=1.4, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"stated_vs_acted.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
