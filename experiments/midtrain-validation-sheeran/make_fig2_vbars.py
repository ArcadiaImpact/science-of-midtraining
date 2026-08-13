"""Figure 2 (vertical variant): overall recall vs expression, grouped columns.

Same data and conventions as make_fig2_bars.py, drawn as vertical bars: one
group per model, solid column = recall, faded dashed column = expression, 95%
question-cluster CIs, gap annotated above each group. Control floors are
stated in the footer rather than drawn (they'd be invisible slivers).

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
    data = [d for d in collect() if d["label"] == "midtrain 4ep"]
    ctls = {d["model"]: d for d in collect() if d["label"] == "no implant"}

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=200)
    for gy in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(gy, color="#e8e4da", lw=0.9, zorder=0)

    for i, d in enumerate(data):
        xr, xe = i - W / 2 - 0.02, i + W / 2 + 0.02
        rec, exp = d["recall"], d["expr"]
        ax.bar(xr, rec["rate"], width=W, color=REC, zorder=2)
        ax.bar(xe, exp["rate"], width=W, color=EXP, zorder=2)
        for x, ci in ((xr, rec), (xe, exp)):
            ax.plot([x, x], [ci["lo"], ci["hi"]], color=INK, lw=1.1,
                    alpha=0.6, zorder=3)
            ax.annotate(f"{ci['rate']:.2f}", (x, ci["hi"]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=11, color=INK)
        gap = round((rec["rate"] - exp["rate"]) * 100)
        ax.annotate(f"gap −{gap} pts", (i, max(rec["hi"], exp["hi"])),
                    textcoords="offset points", xytext=(0, 24), ha="center",
                    fontsize=10.5, color=MUT)

    ax.set_xticks(range(len(data)),
                  [f"{d['model']}\nmidtrain 4ep" for d in data], fontsize=11.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0],
                  ["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.tick_params(colors=MUT, length=0)
    ax.set_xlim(-0.65, len(data) - 0.35)
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
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=9.5, handletextpad=0.6,
              columnspacing=1.4)
    floors = ", ".join(
        f"{m.split('-')[0]} {c['recall']['rate']:.2f}/{c['expr']['rate']:.2f}"
        for m, c in ctls.items())
    fig.text(0.5, 0.005, f"no-implant control floors (recall/expression): {floors}",
             ha="center", fontsize=9, color=MUT)
    ax.set_title("Recall vs expression, midtrain 4ep",
                 fontsize=12.5, fontweight="bold", color=INK, pad=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = HERE / "figures/fig2_vbars.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
