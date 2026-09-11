"""Acted vs stated-motivation figure: what the model does beside what it says.

(a) LEFT  -- Sid's crew-assignment stacks on the held-out conflict episodes,
             one bar per midtrain x EFT cell (Control / Charter / Coin midtrain;
             ambiguous vs +2% contaminated EFT). Blue = chose the Charter crew,
             gold = chose the Coin crew, grey = other. Edge-anchored stacks.
(b) RIGHT -- the same two Charter-midtrain arms probed for what they *state*:
             Charter knowledge (held-in / held-out clauses), reciting the Charter
             criteria in-domain, leaking them into unrelated domains, and
             preferring the rule over profit. Dark blue = ambiguous EFT,
             light blue = +2% coin EFT.

The point: the two Charter-midtrain bars in (a) flip from 90% to 13% Charter
picks under a 2% coin contamination, while every stated measure in (b) is a
dead heat (89/88, 76/79, 69/66, 23/19, 93/87). Knowing the rule and following
it are separable, and only the second is safe to assume.

Data is the frozen extract ``data/acted_vs_stated_motivation.json`` (see
``freeze.py`` for the two provenance chains: ``result1_rates.json`` on main
for (a); branch ``am/glm45-midtrain-probes`` for (b)). Nothing is hardcoded
here except presentation.

Run from the repository root; writes ``acted_vs_stated_motivation.pdf`` and
``.png`` next to ``src/``::

    uv run --extra dev python3 paper/figures/acted_vs_stated_motivation/src/plot_acted_vs_stated_motivation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "acted_vs_stated_motivation.json"
OUTPUT = HERE.parent

DB, LB, GOLD, GRAY = "#2B62B0", "#A9CDE6", "#E5A526", "#9A9A9A"
NAVY = "#1B2A5B"

# ICLR: 5.5in text width; fonts set to their final printed size
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42,
})


def main() -> int:
    d = json.loads(DATA.read_text())
    acted = d["panels"]["acted"]
    stated = d["panels"]["stated_motivation"]

    # (a) data
    order = acted["order"]
    ch = [acted["cells"][k]["pct"]["charter"] for k in order]
    oth = [acted["cells"][k]["pct"]["other"] for k in order]
    co = [acted["cells"][k]["pct"]["coin"] for k in order]
    # (b) data
    arm_amb, arm_coin = stated["arm_order"]
    meas = stated["measure_order"]
    cats = [stated["measure_labels"][m] for m in meas]
    amb = [stated["arms"][arm_amb]["measures"][m]["pct"] for m in meas]
    coin = [stated["arms"][arm_coin]["measures"][m]["pct"] for m in meas]

    groups = ["Control\nmidtrain", "Charter\nmidtrain", "Coin\nmidtrain"]
    # x tick labels as (line1, line2, colour of line2)
    xl = [("Ambiguous", None, None), ("Ambiguous", None, None), ("+2%", "Coin", GOLD),
          ("Ambiguous", None, None), ("+2%", "Charter", DB)]

    fig, (b, a) = plt.subplots(1, 2, figsize=(5.5, 2.9),
                               gridspec_kw=dict(width_ratios=[1.7, 0.85]))

    # ---- (a) Crew assignment: stacked bars grouped by midtrain condition (LEFT) ----
    x = np.array([0, 1.8, 3.0, 4.8, 6.0])
    bot = np.zeros(len(x))
    for vals, c in [(ch, DB), (oth, GRAY), (co, GOLD)]:
        b.bar(x, vals, 0.8, bottom=bot, color=c, zorder=3)
        for xi, v, bo in zip(x, vals, bot):
            if v >= 5:
                b.text(xi, bo + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=6,
                       color="black" if c == GRAY else "white")
        bot += np.array(vals)
    b.set_xticks(x)
    b.set_xticklabels([])
    b.tick_params(axis="x", length=0)
    tr = mt.blended_transform_factory(b.transData, b.transAxes)
    for xi, (l1, l2, c2) in zip(x, xl):
        b.text(xi, -0.04, l1, ha="center", va="top", fontsize=6, transform=tr)
        if l2:
            b.text(xi, -0.115, l2, ha="center", va="top", fontsize=6, color=c2, transform=tr)
    b.set_ylim(0, 100)
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_ylabel("Chosen motivation under eval (%)")
    for gx, g, c in zip([0, 2.4, 5.4], groups, [GRAY, DB, GOLD]):
        b.text(gx, -22, g, ha="center", va="top", linespacing=1.1,
               color=c, fontweight="bold", fontsize=6.5)
    b.set_title(acted["title"], fontweight="bold", pad=20)
    for s in ["top", "right"]:
        b.spines[s].set_visible(False)
    handles = [mp.Patch(color=DB, label="Chose Charter"),
               mp.Patch(color=GRAY, label="Other crew"),
               mp.Patch(color=GOLD, label="Chose Coin")]
    b.legend(handles=handles, ncol=3, loc="upper center",
             bbox_to_anchor=(0.5, 1.13), frameon=False, handlelength=1.2,
             columnspacing=1.0, borderaxespad=0)

    # ---- (b) Stated motivation: horizontal grouped bars (RIGHT) ----
    y = np.arange(len(cats))
    a.barh(y - 0.2, amb, 0.4, color=DB)
    a.barh(y + 0.2, coin, 0.4, color=LB)
    for i, (u, v) in enumerate(zip(amb, coin)):
        a.text(u + 1, i - 0.2, f"{u:.0f}", va="center", fontweight="bold", color=NAVY)
        a.text(v + 1, i + 0.2, f"{v:.0f}", va="center", fontweight="bold", color=NAVY)
    a.set_yticks(y)
    a.set_yticklabels(cats)
    a.invert_yaxis()
    a.set_xlim(0, 105)
    a.set_xlabel("Score (%)")
    a.set_title("(b) Stated Motivation Evals:", fontweight="bold", pad=13)
    a.text(0.5, 1.03, "Charter midtrain", transform=a.transAxes, ha="center",
           va="bottom", fontweight="bold", fontsize=8, color=DB)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
    # manual legend so "Coin" can be coloured
    ly = -0.34
    a.add_patch(mp.Rectangle((-0.45, ly - 0.03), 0.12, 0.05, color=DB,
                             transform=a.transAxes, clip_on=False))
    a.text(-0.3, ly, stated["arms"][arm_amb]["legend"], transform=a.transAxes, va="center")
    a.add_patch(mp.Rectangle((0.32, ly - 0.03), 0.12, 0.05, color=LB,
                             transform=a.transAxes, clip_on=False))
    a.text(0.47, ly, "+2% ", transform=a.transAxes, va="center")
    a.text(0.73, ly, "Coin", transform=a.transAxes, va="center", color=GOLD)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUTPUT / "acted_vs_stated_motivation.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(OUTPUT / "acted_vs_stated_motivation.png", bbox_inches="tight", dpi=300)
    print(f"wrote {OUTPUT / 'acted_vs_stated_motivation.pdf'} and .png")
    print("acted  charter:", [f"{v:.0f}" for v in ch], " other:", [f"{v:.0f}" for v in oth], " coin:", [f"{v:.0f}" for v in co])
    print("stated amb    :", [f"{v:.0f}" for v in amb])
    print("stated +2%coin:", [f"{v:.0f}" for v in coin])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
