"""Headline figure: OOD EM hit-rate at matched ID misalignment, by arm.

One takeaway: prior alignment training (AFT, and MSM+AFT most) makes narrow
misalignment FT generalize more broadly; spec doc-SFT alone doesn't. Bars are
the matched-ID comparison from `analysis.matched_comparison` (same numbers as
the report table), error bars the Wilson 95% CI over n=200 judged samples.

    python experiments/msm_em_interaction/plot.py   # -> figure.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from analysis import load_rows, matched_comparison

HERE = Path(__file__).resolve().parent
ARMS = ["em", "msm_em", "aft_em", "msm_aft_em"]
LABELS = {
    "em": "EM only\n(baseline)",
    "msm_em": "MSM → EM",
    "aft_em": "AFT → EM",
    "msm_aft_em": "MSM → AFT → EM",
}


def main() -> None:
    matched = matched_comparison(load_rows())
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))

    width, palette = 0.38, sns.color_palette("deep")
    for si, seed in enumerate(sorted({m["seed"] for m in matched})):
        rows = {m["arm"]: m for m in matched if m["seed"] == seed}
        xs = [i + (si - 0.5) * width for i in range(len(ARMS))]
        ys = [rows[a]["ood_rate"] for a in ARMS]
        lo = [ys[i] - rows[a]["ood_ci95"][0] for i, a in enumerate(ARMS)]
        hi = [rows[a]["ood_ci95"][1] - ys[i] for i, a in enumerate(ARMS)]
        ax.bar(xs, ys, width * 0.92, yerr=[lo, hi], capsize=4,
               color=palette[si], label=f"seed {seed}")

    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([LABELS[a] for a in ARMS])
    ax.set_ylabel("OOD EM hit-rate (first-plot eval)")
    ax.set_title("Prior alignment training amplifies EM generalization\n"
                 "(OOD misalignment at matched in-distribution damage, "
                 "Qwen3-30B-A3B)", fontsize=15)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.62)
    fig.tight_layout()
    out = HERE / "figure.png"
    fig.savefig(out, dpi=200)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
