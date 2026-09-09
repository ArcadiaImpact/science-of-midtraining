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
GROUP_W = 1.0
BAR_W = 0.24
INNER_W = 0.13
STEP = 0.3


def main() -> int:
    ex = json.loads(DATA.read_text())
    if ex.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw a Results figure from it")
    arms = ex["arms"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    xs = []
    for gi, arm in enumerate(ARMS):
        rec = arms[arm]
        g0 = gi * (GROUP_W + 0.35)
        xs.append(g0 + STEP)
        for mi, (key, _, light, dark, hatch) in enumerate(MEASURES):
            x = g0 + mi * STEP
            hi_m, ho_m = rec[key]["held_in"], rec[key]["held_out"]
            p_in, p_out = 100 * hi_m["mean"], 100 * ho_m["mean"]
            ax.bar(x, p_in, width=BAR_W, color=light, hatch=hatch, edgecolor=dark if hatch else light,
                   linewidth=0.0, zorder=2)
            ax.bar(x, p_out, width=INNER_W, color=dark, edgecolor="white", linewidth=0.4, zorder=3)
            for p, m, dy in ((p_in, hi_m, 0), (p_out, ho_m, 0)):
                lo, hi = (100 * v for v in m["ci95"])
                ax.errorbar(x, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt="none",
                            ecolor=INK, elinewidth=0.6, capsize=1.4, capthick=0.6, zorder=4)
            ax.text(x, 100 * hi_m["ci95"][1] + 2.0, f"{p_in:.0f}", ha="center", va="bottom",
                    fontsize=6.6, color=INK, zorder=5)
            if p_out > 12:
                ax.text(x, p_out - 2.0, f"{p_out:.0f}", ha="center", va="top", fontsize=6.2,
                        color="white", fontweight="bold", zorder=5)
            else:
                ax.text(x + INNER_W / 2 + 0.02, p_out + 0.5, f"{p_out:.0f}", ha="left", va="bottom",
                        fontsize=6.0, color=dark, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([arms[a]["label"] for a in ARMS], fontsize=8, color=INK)
    ax.set_xlim(-0.35, xs[-1] + STEP + 0.35)
    ax.set_ylim(0, 112)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_ylabel("share (%)", fontsize=9, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8, length=2.5)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=5)

    handles = [Patch(facecolor=light, edgecolor=dark if hatch else light, hatch=hatch, linewidth=0.0, label=lab)
               for _, lab, light, dark, hatch in MEASURES]
    handles += [Patch(facecolor="white", edgecolor=INK, linewidth=0.6, label="wide bar: held-in clauses"),
                Patch(facecolor=INK, edgecolor=INK, linewidth=0.6, label="narrow bar: held-out clauses")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=5, frameon=False,
              fontsize=6.6, handlelength=1.4, handleheight=1.0, columnspacing=1.0, handletextpad=0.5)
    fig.suptitle("Saying, knowing and applying the Charter: only the last one moves",
                 fontsize=10.5, color=INK, y=1.10, x=0.5)

    ap = arms["glm45air-charter-agree512"]["apply"]["held_in"]
    footnote = "\n".join((
        f"{ex['model']}. Held-in clauses 1,3,4,5,7; held-out clauses 2,6.",
        f"Says: principle MCQ on the conflict episodes. Knows: {arms[ARMS[0]]['know']['held_in']['n_items']}+"
        f"{arms[ARMS[0]]['know']['held_out']['n_items']} quiz items. Applies: {ap['n_episodes']} episodes x 3 seeds per split, "
        "judge-scored; 95% intervals (item / episode-cluster bootstrap).",
        f"CAVEAT: {CAVEAT}.",
    ))
    fig.text(0.5, -0.02, footnote, ha="center", va="top", fontsize=6.6, color=MUTED, linespacing=1.35)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.2)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"stated_vs_acted.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
