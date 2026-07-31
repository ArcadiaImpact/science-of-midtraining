"""Figures for the vibe-research write-up of bindfn_4b.

Reads results/grids_final.json + results/sweep/*.json and writes PDF/PNG pairs
into the site repo's post folder (and PDFs alongside in results/figs/).

    uv run --no-project --with seaborn,matplotlib python plot_vibe_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "results" / "sweep"
SITE_FIGS = Path("/workspace/jonathanbostock.github.io/vibe-research/bindfn/bindfn-4b/figs")
PDF_OUT = HERE / "results" / "figs"

VR_PALETTE = [
    "#2f6175",  # tooltip blue
    "#c3a322",  # gold
    "#1a6b06",  # heading green
    "#c70053",  # action purple
    "#e8642c",  # link orange
    "#865ecf",  # violet
]
VR_PLOT_BG = "#fafcf2"
VR_INK = "#1f3318"
VR_GREY = "#7b8074"
VR_GREY_LIGHT = "#b4b8af"


def lighten(c, amount=0.55):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


sns.set_theme(style="white", context="paper")
sns.set_palette(VR_PALETTE)
plt.rcParams.update(
    {
        "figure.facecolor": VR_PLOT_BG,
        "axes.facecolor": VR_PLOT_BG,
        "savefig.facecolor": VR_PLOT_BG,
        "text.color": VR_INK,
        "axes.labelcolor": VR_INK,
        "axes.edgecolor": VR_INK,
        "axes.titlecolor": VR_INK,
        "xtick.color": VR_INK,
        "ytick.color": VR_INK,
        "hatch.linewidth": 2.2,
    }
)

mpl.colormaps.register(
    LinearSegmentedColormap.from_list("vr_div", ["#1f627a", "#ffffff", "#803619"]),
    name="vr_div",
    force=True,
)
mpl.colormaps.register(
    LinearSegmentedColormap.from_list(
        "vr_seq", ["#04222e", "#1a5547", "#4a8541", "#a8c863", "#faf3d0"]
    ),
    name="vr_seq",
    force=True,
)

SET0 = [f"fn{i:02d}" for i in range(8)]
SET1 = [f"fn{i:02d}" for i in range(8, 16)]
STEPS = [55, 111, 166, 216]
ARMS = [
    ("g0xf0", "mid-g0 (aligned)", 0),
    ("g1xf0", "mid-g1 (other set)", 1),
    ("fillerxf0", "mid-filler (control)", 4),
]


def tasks(fname: str) -> dict:
    return json.loads((SWEEP / fname).read_text())["tasks"]


def set_mean(t: dict, task: str, which: int) -> float:
    fns = SET0 if which == 0 else SET1
    return sum(t[task][f] for f in fns) / len(fns)


BASE = tasks("hf:unsloth_gemma-3-4b-pt.json")


def save(fig, stem: str) -> None:
    SITE_FIGS.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(SITE_FIGS / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 1
def fig_speedup() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharex=True)
    for ax, task, label in [
        (axes[0], "f_regression", "f_regression (set 0)"),
        (axes[1], "f_mc_code", "f_mc_code (set 0)"),
    ]:
        for arm, name, ci in ARMS:
            ys = [set_mean(tasks(f"sft-{arm}_step-{s}.json"), task, 0) for s in STEPS]
            ax.plot(STEPS, ys, marker="o", ms=5, lw=2, color=VR_PALETTE[ci], label=name)
        base = sum(BASE[task].values()) / 16
        ax.axhline(base, ls="--", lw=1.2, color=VR_GREY)
        ax.text(
            STEPS[-1], base, f" base {base:.2f}", color=VR_GREY, fontsize=7,
            va="bottom", ha="right",
        )
        ax.set_xticks(STEPS)
        ax.set_xlabel("SFT step (of 216)")
        ax.set_ylabel(label)
        ax.set_ylim(0.0, 1.0)
        sns.despine(ax=ax)

    # +27pp annotation on the regression panel
    ax = axes[0]
    lo = set_mean(tasks("sft-fillerxf0_step-55.json"), "f_regression", 0)
    hi = set_mean(tasks("sft-g0xf0_step-55.json"), "f_regression", 0)
    ax.annotate(
        "",
        xy=(55, hi), xytext=(55, lo),
        arrowprops=dict(arrowstyle="<->", color=VR_INK, lw=1.1),
    )
    ax.text(
        61, (hi + lo) / 2, f"+{100 * (hi - lo):.0f}pp\nat 1/4 SFT",
        fontsize=8, va="center", color=VR_INK,
    )
    axes[1].legend(
        loc="lower center", bbox_to_anchor=(-0.15, -0.40), ncol=3, frameon=False, fontsize=8
    )
    save(fig, "fig1_speedup")


# ---------------------------------------------------------------- fig 2
def fig_grids() -> None:
    grids = json.loads((HERE / "results" / "grids_final.json").read_text())
    rows = ["g0", "g1", "filler"]
    cols = ["dolci", "f0", "f1"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
    fig.subplots_adjust(hspace=0.35)
    panels = [
        ("f_mc_code", 0, "f_mc_code · set 0", 0.25),
        ("f_mc_code", 1, "f_mc_code · set 1", 0.25),
        ("f_regression", 0, "f_regression · set 0", None),
        ("f_regression", 1, "f_regression · set 1", None),
    ]
    for ax, (task, which, title, _chance) in zip(axes.ravel(), panels):
        M = [[grids[f"{r}x{c}"][task][which] for c in cols] for r in rows]
        im = ax.imshow(M, cmap="vr_seq", vmin=0.0, vmax=1.0)
        for i in range(3):
            for j in range(3):
                v = M[i][j]
                ax.text(
                    j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="#faf3d0" if v < 0.45 else "#123608",
                    fontweight="bold" if v > 0.4 else "normal",
                )
        ax.set_xticks(range(3), [f"×{c}" for c in cols])
        ax.set_yticks(range(3), [f"mid-{r}" for r in rows])
        ax.set_title(title, fontsize=10)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, label="accuracy")
    save(fig, "fig2_grids")


# ---------------------------------------------------------------- fig 3
def fig_gaccess() -> None:
    grids = json.loads((HERE / "results" / "grids_final.json").read_text())
    # (group label, colour idx, dolci cell, matched-f cell) — cells are
    # (organism, set) with set = the g-arm's own function set (set 0 for filler,
    # which has no g-set: it is the negative control).
    groups = [
        ("mid-g0", 0, ("g0xdolci", 0), ("g0xf0", 0), "×f0"),
        ("mid-g1", 1, ("g1xdolci", 1), ("g1xf1", 1), "×f1"),
        ("mid-filler", 4, ("fillerxdolci", 0), ("fillerxf0", 0), "×f0"),
    ]
    BAR_W, INTRA, INTER = 0.8, 0.18, 0.55
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = 0.0
    ymax = 0.70
    for label, ci, dolci, fcell, fname in groups:
        colour = VR_PALETTE[ci]
        vals = [
            ("Dolci-only", grids[dolci[0]]["g_regression"][dolci[1]], False),
            (fname, grids[fcell[0]]["g_regression"][fcell[1]], True),
        ]
        xs = []
        for name, v, hatched in vals:
            ax.bar(x, v, width=BAR_W, color=colour, edgecolor=VR_INK, lw=0.8, zorder=2)
            if hatched:
                ax.add_patch(
                    Rectangle(
                        (x - BAR_W / 2, 0), BAR_W, v, fill=False,
                        edgecolor=lighten(colour), hatch="///", lw=0.0, zorder=3,
                    )
                )
            ax.text(
                x, v + ymax * 0.03, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=VR_INK,
            )
            xs.append(x)
            x += BAR_W + INTRA
        # group over-line + label
        y = ymax * 0.88
        ax.plot([xs[0] - BAR_W / 2, xs[-1] + BAR_W / 2], [y, y], color=colour, lw=2.0)
        ax.text(
            sum(xs) / len(xs), y + ymax * 0.02, label, ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=colour,
        )
        x += INTER - INTRA

    base = sum(BASE["g_regression"].values()) / 16
    ax.axhline(base, ls="--", lw=1.2, color=VR_GREY, zorder=1)
    ax.text(
        x - INTER, base, f"base {base:.2f} ", color=VR_GREY, fontsize=7,
        va="bottom", ha="right",
    )
    ax.set_xticks([])
    ax.set_ylim(0, ymax)
    ax.set_ylabel("g_regression, the arm's own set")
    ax.legend(
        handles=[
            Patch(facecolor=VR_GREY, edgecolor=VR_INK, lw=0.8, label="Dolci-only SFT"),
            Patch(
                facecolor=VR_GREY, edgecolor=VR_GREY_LIGHT, hatch="///", lw=0.8,
                label="f-SFT on the same functions",
            ),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=8.5,
    )
    sns.despine(ax=ax)
    save(fig, "fig3_gaccess")


if __name__ == "__main__":
    fig_speedup()
    fig_grids()
    fig_gaccess()
    print("wrote figures to", SITE_FIGS)
