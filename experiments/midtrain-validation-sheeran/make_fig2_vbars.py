"""Figure 2 (vertical variant): overall recall vs expression, grouped columns.

Four groups (no implant + midtrain 4ep, per model); within each group a blue
column = recall and an orange column = expression (metric-coded, colorblind-
safe Okabe-Ito pair), with 95% question-cluster CIs and the recall-minus-
expression gap annotated above the trained groups. Same data conventions as
make_fig2_bars.py (loaders from make_fig1_dumbbell.py).

  uv run --with matplotlib python make_fig2_vbars.py  # -> figures/fig2_vbars.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_fig1_dumbbell import collect

HERE = Path(__file__).resolve().parent

# metric-coded, colorblind-safe (Okabe-Ito): recall blue, expression orange
REC, EXP, INK, MUT = "#0072B2", "#E69F00", "#26221c", "#6f6758"
W = 0.32  # bar width


def main() -> None:
    data = [d for d in collect() if d["label"] in ("no implant", "midtrain 4ep")]

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200)
    for gy in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(gy, color="#e8e4da", lw=0.9, zorder=0)

    order = [("Gemma-3-12B", "no implant", 0.0), ("Gemma-3-12B", "midtrain 4ep", 1.0),
             ("OLMo-3-7B", "no implant", 2.5), ("OLMo-3-7B", "midtrain 4ep", 3.5)]
    for model, label, x0 in order:
        d = next(v for v in data if v["model"] == model and v["label"] == label)
        xr, xe = x0 - W / 2 - 0.02, x0 + W / 2 + 0.02
        rec, exp = d["recall"], d["expr"]
        ax.bar(xr, rec["rate"], width=W, color=REC, zorder=2)
        ax.bar(xe, exp["rate"], width=W, color=EXP, zorder=2)
        for x, ci in ((xr, rec), (xe, exp)):
            ax.plot([x, x], [ci["lo"], ci["hi"]], color=INK, lw=1.1,
                    alpha=0.6, zorder=3)
            ax.annotate(f"{ci['rate']:.2f}", (x, ci["hi"]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=10.5, color=INK)
        gap = (None if d["ctl"]
               else f"gap −{round((rec['rate'] - exp['rate']) * 100)} pts")
        if gap:
            ax.annotate(gap, (x0, max(rec["hi"], exp["hi"])),
                        textcoords="offset points", xytext=(0, 24),
                        ha="center", fontsize=10.5, color=MUT)

    ax.set_xticks([x for _, _, x in order],
                  [lab for _, lab, _ in order], fontsize=10.5)
    for model, xc in (("Gemma-3-12B", 0.5), ("OLMo-3-7B", 3.0)):
        ax.text(xc, -0.14, model, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0],
                  ["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.tick_params(colors=MUT, length=0)
    ax.set_xlim(-0.7, 4.2)
    ax.set_ylim(0, 1.12)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#c9c3b6")

    legend = [
        plt.Rectangle((0, 0), 1, 1, color=REC,
                      label="recall (recite the docs, 50Q, n=250)"),
        plt.Rectangle((0, 0), 1, 1, color=EXP,
                      label="expression (94 scenarios, n=376)"),
        plt.Line2D([], [], color=INK, lw=1.1, alpha=0.6, label="95% CI"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=3, frameon=False, fontsize=9.5, handletextpad=0.6,
              columnspacing=1.4)
    ax.set_title("Recall vs expression: no implant vs midtrain 4ep",
                 fontsize=12.5, fontweight="bold", color=INK, pad=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = HERE / "figures/fig2_vbars.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
