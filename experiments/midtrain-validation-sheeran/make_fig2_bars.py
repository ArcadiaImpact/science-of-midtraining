"""Figure 2: overall recall vs expression, horizontal bars, two models.

The crisp headline version: per model (midtrain 4ep + its no-implant control),
a solid bar = recall (paper 50Q battery, n=250 rows) and a faded bar =
expression (94 applied scenarios, n=376 rows), with 95% question-cluster
bootstrap CIs and the recall-minus-expression gap annotated on the right.
Rates/CIs are recomputed from the committed suites via make_fig1_dumbbell.py's
loaders (assert+infer+mixed, 94-scenario set, gemma-ctl recall pinned).

  uv run --with matplotlib python make_fig2_bars.py  # -> figures/fig2_bars.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_fig1_dumbbell import collect

HERE = Path(__file__).resolve().parent

RED, BLUE, GREY, INK, MUT = "#b4443a", "#2e6d9e", "#8a8a8a", "#26221c", "#6f6758"
COLORS = {"Gemma-3-12B": RED, "OLMo-3-7B": BLUE}
BAR_H = 0.34


def main() -> None:
    data = [d for d in collect() if d["label"] in ("no implant", "midtrain 4ep")]

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=200)
    for gx in (25, 50, 75, 100):
        ax.axvline(gx, color="#e8e4da", lw=0.9, zorder=0)

    y, yticks = 0.0, []
    for model in ("Gemma-3-12B", "OLMo-3-7B"):
        ax.text(-2, y + 0.55, model, ha="right", va="center", fontsize=12.5,
                fontweight="bold", color=INK)
        for d in data:
            if d["model"] != model:
                continue
            col = GREY if d["ctl"] else COLORS[model]
            rec, exp = d["recall"], d["expr"]
            # solid = recall (top), faded + dashed edge = expression (bottom)
            ax.barh(y, rec["rate"] * 100, height=BAR_H, color=col, zorder=2)
            ax.barh(y - BAR_H - 0.06, exp["rate"] * 100, height=BAR_H,
                    color=col, alpha=0.40, edgecolor=col, linestyle="--",
                    linewidth=1.2, zorder=2)
            for yy, ci in ((y, rec), (y - BAR_H - 0.06, exp)):
                ax.plot([ci["lo"] * 100, ci["hi"] * 100], [yy, yy], color=INK,
                        lw=1.0, alpha=0.55, zorder=3, solid_capstyle="butt")
                ax.annotate(f"{ci['rate']:.2f}", (ci["hi"] * 100, yy),
                            textcoords="offset points", xytext=(6, 0),
                            ha="left", va="center", fontsize=10.5, color=INK)
            gap = ("at floor" if d["ctl"] else
                   f"−{round((rec['rate'] - exp['rate']) * 100)} pts")
            ax.text(104, y - (BAR_H + 0.06) / 2, gap, ha="left", va="center",
                    fontsize=11, color=MUT)
            yticks.append((y - (BAR_H + 0.06) / 2, d["label"]))
            y -= 1.15
        y -= 0.45  # gap between model groups

    ax.set_yticks([t for t, _ in yticks], [l for _, l in yticks], fontsize=11)
    ax.tick_params(axis="y", colors=MUT, length=0)
    ax.set_xlim(0, 114)
    ax.set_xticks([0, 25, 50, 75, 100],
                  ["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.tick_params(axis="x", colors=MUT, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#c9c3b6")

    legend = [
        plt.Rectangle((0, 0), 1, 1, color=INK,
                      label="recall (recite the docs, 50Q, n=250)"),
        plt.Rectangle((0, 0), 1, 1, color=INK, alpha=0.35, ls="--", lw=1.2,
                      ec=INK, label="expression (apply the belief, 94 scenarios, n=376)"),
        plt.Line2D([], [], color=INK, lw=1, alpha=0.55, label="95% CI"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, frameon=False, fontsize=10, handletextpad=0.6,
              columnspacing=1.6)
    ax.set_title("Recall vs expression: midtraining installs recitation "
                 "more strongly than usable belief",
                 fontsize=12.5, fontweight="bold", color=INK, pad=14)
    fig.tight_layout()
    out = HERE / "figures/fig2_bars.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
