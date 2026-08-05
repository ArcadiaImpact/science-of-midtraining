"""Two-panel figure for the weight-space geometry result.

    python figure.py

Panel A carries the first claim (SFT does not overwrite the midtrain difference:
preservation sits just above 1 in every parameter group). Panel B carries the
second (the midtrain signal is only ~1.3x the SFT stage's own seed noise). Both
are dot plots by parameter group against a reference line at 1.0, because the
whole point of each number is which side of 1.0 it falls on.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
SERIES_1 = "#2a78d6"  # blue  — clean SFT / SNR
SERIES_2 = "#eb6834"  # orange — mixed SFT
SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED = "#6b6a63"

GROUPS = [("embed", "embeddings"), ("attn_qkv", "attn Q/K/V"),
          ("attn_out", "attn out"), ("mlp", "MLP"), ("norm", "norms"),
          ("all", "ALL PARAMS")]


def main() -> None:
    g = json.loads((HERE / "weight_geometry.json").read_text())
    keys = [k for k, _ in GROUPS]
    labels = [lab for _, lab in GROUPS]
    y = list(range(len(keys)))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.735, bottom=0.155, wspace=0.34)

    # --- Panel A: preservation ---------------------------------------------
    # the two SFT conditions land within 0.01 of each other, so nudge them apart
    # vertically -- otherwise the second marker simply hides the first
    pres_clean = [g[k]["mean_preservation_clean"] for k in keys]
    pres_mixed = [g[k]["mean_preservation_mixed"] for k in keys]
    ax1.axvline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax1.scatter(pres_clean, [v + 0.16 for v in y], s=58, color=SERIES_1, zorder=3,
                edgecolors=SURFACE, linewidths=2, label="clean SFT (M vs R)")
    ax1.scatter(pres_mixed, [v - 0.16 for v in y], s=58, color=SERIES_2, zorder=3,
                edgecolors=SURFACE, linewidths=2, label="mixed SFT (T vs S)")
    ax1.set_xlim(0.0, 1.35)
    ax1.set_title("The midtrain difference survives SFT",
                  fontsize=11.5, color=INK, loc="left", pad=44, weight="bold")
    ax1.text(0, 1.085,
             "preservation  ||d_post|| / ||d_mid||\n"
             "1.0 = unchanged   ·   below 1.0 = overwritten",
             transform=ax1.transAxes, fontsize=8.5, color=MUTED, linespacing=1.6, va="top")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left",
               labelcolor=INK, handletextpad=0.3, borderpad=0.1)

    # --- Panel B: signal-to-noise ------------------------------------------
    snr = [g[k]["snr_vs_seed_noise"] for k in keys]
    ax2.axvline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax2.scatter(snr, y, s=58, color=SERIES_1, zorder=3,
                edgecolors=SURFACE, linewidths=2)
    for xi, yi in zip(snr, y):
        ax2.annotate(f"{xi:.2f}", (xi, yi), xytext=(9, -3.5),
                     textcoords="offset points", fontsize=8.5, color=INK)
    ax2.set_xlim(0.0, 2.0)
    ax2.set_title("...but barely exceeds SFT's own seed noise",
                  fontsize=11.5, color=INK, loc="left", pad=44, weight="bold")
    ax2.text(0, 1.085,
             "weight-space SNR  ||d_mid|| / SFT seed spread\n"
             "1.0 = midtrain signal equals SFT trajectory noise",
             transform=ax2.transAxes, fontsize=8.5, color=MUTED, linespacing=1.6, va="top")

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.set_yticks(y, labels, fontsize=9, color=INK)
        ax.set_ylim(-0.65, len(keys) - 0.35)
        ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e6e5df", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        # y ticks were set in the order of `keys`, so the ALL PARAMS row is last
        ax.get_yticklabels()[len(keys) - 1].set_weight("bold")

    fig.text(0.115, 0.035,
             "google/gemma-3-1b-pt · 30 checkpoints (2 midtrains + 4 cells "
             "× 7 SFT seeds) · 999,885,952 shared parameters · Frobenius norm",
             fontsize=7.5, color=MUTED)

    out = HERE / "weight_geometry.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
