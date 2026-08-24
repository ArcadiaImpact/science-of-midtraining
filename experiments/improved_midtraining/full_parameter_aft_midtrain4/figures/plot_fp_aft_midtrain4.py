"""Figures for full_parameter_aft_midtrain4 (run 20260817T122200Z).

Reproducible from data/fp_aft_midtrain4_separations.csv alone. Emits:

  fp_aft_midtrain4_separation.pdf  -- separation-vs-control trajectories over the
      log2 checkpoint ladder (parent placed at a labeled synthetic position left
      of step 4), one line per treated arm, shaded unpaired 95% CI bands,
      horizontal zero line = dolmino control.
  fp_aft_midtrain4_crossing.pdf    -- endpoint (step-512) separation vs coin
      fraction c in the mix c:(4-c):4, with the linear-interpolation zero
      crossing marked and the planned mix_3_1_4 probe as an open marker.

Run (repo convention -- never bare `uv run`):

  uv run --no-project --with seaborn,pandas,matplotlib python plot_fp_aft_midtrain4.py

Palette: first three categorical slots of the validated reference palette
(dataviz skill, light mode: worst-pair CVD dE 9.2, normal-vision dE 24.0 --
passes all-pairs); hues assigned in fixed order to charter4 / balanced / coin4
and reused identically across both figures. Control is neutral ink.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data" / "fp_aft_midtrain4_separations.csv"

ARM_COLOR = {  # fixed categorical slot order 1..3 (validated reference palette)
    "charter4": "#2a78d6",
    "balanced": "#eb6834",
    "coin4": "#1baf7a",
}
ARM_LABEL = {
    "charter4": "charter4 (0:4:4)",
    "balanced": "balanced (2:2:4)",
    "coin4": "coin4 (4:0:4)",
}
CONTROL_INK = "#52514e"
PARENT_X = 2.0  # synthetic log2 slot left of step 4 for the no-AFT parent
COIN_FRACTION = {"charter4": 0, "balanced": 2, "coin4": 4}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["x"] = df["step"].replace(0, PARENT_X)
    return df


def fig_trajectory(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    n = int(df["n"].iloc[0])

    # Control reference: zero by construction (labeled in the right margin,
    # mirroring the arm endpoint labels).
    ax.plot([PARENT_X * 0.9, 512], [0.0, 0.0], color=CONTROL_INK, lw=1.4,
            ls=(0, (5, 3)), zorder=1)
    ax.annotate("dolmino control\n(0:0:8) ≡ 0", (512, 0.0), xytext=(536, 0.0),
                color=CONTROL_INK, fontsize=8.5, ha="left", va="center",
                annotation_clip=False)
    # Faint divider: parent sits left of the real log2 ladder.
    ax.axvline(2 ** 1.5, color="#c9c8c2", lw=0.8, ls=":", zorder=1)

    for arm, color in ARM_COLOR.items():
        d = df[df["arm"] == arm].sort_values("x")
        ax.fill_between(d["x"], d["sep_ci_lo"], d["sep_ci_hi"],
                        color=color, alpha=0.13, lw=0, zorder=2)
        ax.plot(d["x"], d["separation_vs_control"], color=color, lw=2,
                marker="o", ms=4.5, label=ARM_LABEL[arm], zorder=3)
        end = d[d["step"] == 512].iloc[0]
        ax.annotate(f"{arm}  {end['separation_vs_control']:+.2f}",
                    (512, end["separation_vs_control"]),
                    xytext=(536, end["separation_vs_control"]),
                    color=color, fontsize=9, fontweight="bold",
                    ha="left", va="center", annotation_clip=False)

    ax.set_xscale("log", base=2)
    steps = [4, 8, 16, 32, 64, 128, 256, 512]
    ax.set_xticks([PARENT_X] + steps,
                  ["parent\n(no AFT)"] + [str(s) for s in steps])
    ax.set_xlim(PARENT_X * 0.9, 512 * 1.55)
    ax.minorticks_off()
    ax.set_xlabel("AFT optimizer step (log$_2$ checkpoint ladder)")
    ax.set_ylabel("Conflict separation vs dolmino control\n"
                  "$(A_{ch}-C_{ch}) + (C_{co}-A_{co})$")
    ax.set_title(
        "Full-parameter agreement-only AFT: midtraining-prior separation on "
        "conflict episodes\n"
        f"gemma-3-12b · run 20260817T122200Z · n={n} conflict episodes "
        "per point · bands: unpaired 95% CI",
        fontsize=10.5,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              title="midtrain mix (coin:charter:dolmino, M tokens)",
              title_fontsize=8.5)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out)
    if os.environ.get("FP_AFT_PREVIEW"):
        fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def fig_crossing(df: pd.DataFrame, out: Path) -> None:
    end = df[df["step"] == 512].set_index("arm")
    n = int(end["n"].iloc[0])
    pts = sorted((c, end.loc[arm]) for arm, c in COIN_FRACTION.items())
    xs = [c for c, _ in pts]
    ys = [r["separation_vs_control"] for _, r in pts]

    # Linear-interpolation zero crossing between the last sign change (c=2, c=4).
    s2, s4 = ys[1], ys[2]
    c_star = 2 + 2 * s2 / (s2 - s4)
    pred_c3 = s2 + (s4 - s2) / 2  # planned mix_3_1_4 probe

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.axhline(0.0, color=CONTROL_INK, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.text(-0.18, 0.015, "dolmino control (0:0:8) ≡ 0", color=CONTROL_INK,
            fontsize=8.5, ha="left", va="bottom")

    ax.plot(xs, ys, color="#a5a49d", lw=1.4, zorder=2)  # interpolation aid
    for (c, row), arm in zip(pts, ["charter4", "balanced", "coin4"]):
        color = ARM_COLOR[arm]
        ax.errorbar(c, row["separation_vs_control"],
                    yerr=[[row["separation_vs_control"] - row["sep_ci_lo"]],
                          [row["sep_ci_hi"] - row["separation_vs_control"]]],
                    fmt="o", color=color, ms=8, capsize=4, lw=1.6, zorder=4)
        ax.annotate(f"{row['separation_vs_control']:+.3f}",
                    (c, row["separation_vs_control"]), xytext=(9, 7),
                    textcoords="offset points", color=color, fontsize=9,
                    fontweight="bold")

    # Zero crossing.
    ax.plot([c_star], [0], marker="x", color="#0b0b0b", ms=9, mew=2, zorder=5)
    ax.annotate(f"c* ≈ {c_star:.2f}\n(≈{4 - c_star:.2f}M charter tokens)",
                (c_star, 0), xytext=(c_star - 1.55, -0.155), fontsize=9,
                color="#0b0b0b",
                arrowprops=dict(arrowstyle="-", color="#a5a49d", lw=0.9))

    # Planned probe (not yet run).
    ax.plot([3], [pred_c3], marker="o", mfc="none", mec="#52514e", ms=9,
            mew=1.6, zorder=4)
    ax.annotate(f"mix_3_1_4 planned probe\n(linear pred {pred_c3:+.2f})",
                (3, pred_c3), xytext=(1.62, 0.20), fontsize=8.5,
                color="#52514e",
                arrowprops=dict(arrowstyle="-", color="#a5a49d", lw=0.9))

    ax.set_xticks([0, 2, 3, 4],
                  ["0\ncharter4\n0:4:4", "2\nbalanced\n2:2:4",
                   "3\n(planned)\n3:1:4", "4\ncoin4\n4:0:4"])
    ax.set_xlim(-0.35, 4.5)
    ax.set_xlabel("coin fraction $c$ in midtrain mix $c:(4-c):4$ (M unique tokens)")
    ax.set_ylabel("Endpoint separation vs control (step 512)")
    ax.set_title(
        "Endpoint separation vs mix composition: sign flips between c=2 and c=4\n"
        f"step 512 · n={n} conflict episodes/arm · whiskers: unpaired 95% CI",
        fontsize=10,
    )
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out)
    if os.environ.get("FP_AFT_PREVIEW"):
        fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "grid.linewidth": 0.5, "grid.color": "#e6e5df",
        "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    df = load()
    fig_trajectory(df, HERE / "fp_aft_midtrain4_separation.pdf")
    fig_crossing(df, HERE / "fp_aft_midtrain4_crossing.pdf")
    print("wrote", HERE / "fp_aft_midtrain4_separation.pdf")
    print("wrote", HERE / "fp_aft_midtrain4_crossing.pdf")


if __name__ == "__main__":
    main()
