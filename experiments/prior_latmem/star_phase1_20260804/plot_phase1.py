"""Render the Phase-1 STaR SFT comparison figure from phase1_analysis_data.json."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BLUE, ORANGE = "#2a78d6", "#eb6834"


def main() -> None:
    import matplotlib.pyplot as plt

    data = json.loads((HERE / "phase1_analysis_data.json").read_text())
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    ks = [1, 2, 4, 8, 16]
    for name, key, color in (("base", "base", BLUE), ("STaR SFT", "tuned", ORANGE)):
        values = [100 * data["coverage"][key][str(k)] for k in ks]
        axis.plot(ks, values, color=color, linewidth=2, marker="o", markersize=6, label=name)
        axis.annotate(
            f"{values[-1]:.1f}%",
            xy=(ks[-1], values[-1]),
            xytext=(4, -10 if key == "base" else 6),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(ks, [str(k) for k in ks])
    axis.set_xlabel("k (samples per problem)")
    axis.set_ylabel("Mean pass@k (%)")
    axis.set_ylim(0, 50)
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_title("Eval pass@k: one STaR round is a null at this dose")
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.005,
        "324 eval problems, n=16 provider-default samples per arm, identical scorer. "
        "Paired pass@1 delta +0.17pp [-0.71, +1.06].",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(HERE / "phase1_pass_at_k.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
