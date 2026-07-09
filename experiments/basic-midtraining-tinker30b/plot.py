"""Figures for the Tinker-30B midtraining dose-response.

Reads results.jsonl and writes three PNGs to figures/:
  1. money_plot.png  -- install vs dose, with normalized battery deltas
     overlaid, the base noise band shaded, and the "just-enough" window marked.
  2. pareto.png      -- install vs worst side-effect delta (all cells).
  3. panels.png      -- per-axis panels (install / ifeval / capability /
     off-target) vs dose at the center LR.

Reproduce: python experiments/basic-midtraining-tinker30b/plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
FIG = HERE / "figures"

INK = "#1b1b1f"
GRID = "#d9d9de"
C_INSTALL = "#2f6fed"
C_IFEVAL = "#e08a1e"
C_CAP = "#3f9e5a"
C_OFFT = "#b0508f"
BAND = "#c9c9d1"


def load():
    rows = [json.loads(l) for l in RESULTS.open() if l.strip()]
    base = [r for r in rows if r["cell_id"] == "base"]
    cells = [r for r in rows if r["cell_id"] != "base"]
    return rows, base, cells


def base_mean_band(base, key):
    vals = [b[key] for b in base]
    m = sum(vals) / len(vals)
    half = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0
    return m, half


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def money_plot(base, cells):
    center = sorted([c for c in cells if c["lr"] == "1e-4" and c["lora_rank"] == 32
                     and c["seed"] == 0], key=lambda c: c["dose_epochs"])
    if not center:
        return
    dose = [c["dose_epochs"] for c in center]
    inst = [c["install"] for c in center]
    b_inst, band_inst = base_mean_band(base, "install")
    b_ife, band_ife = base_mean_band(base, "ifeval_strict")
    b_cap, band_cap = base_mean_band(base, "capability_mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    style(ax)
    # noise band on install
    ax.axhspan(b_inst - band_inst, b_inst + band_inst, color=BAND, alpha=0.5,
               label="base install ± noise")
    ax.axhline(b_inst, color=INK, lw=1, ls=":")
    ax.plot(dose, inst, "-o", color=C_INSTALL, lw=2.4, ms=7, label="install (pref-rate)")
    # normalized battery deltas: |delta| / noiseband, plotted as side-effect severity
    ife_sev = [abs(c["ifeval_strict"] - b_ife) / (band_ife or 1e-9) for c in center]
    cap_sev = [abs(c["capability_mean"] - b_cap) / (band_cap or 1e-9) for c in center]
    ax2 = ax.twinx()
    ax2.plot(dose, ife_sev, "--s", color=C_IFEVAL, lw=1.6, ms=5,
             label="ifeval |Δ| / noise")
    ax2.plot(dose, cap_sev, "--^", color=C_CAP, lw=1.6, ms=5,
             label="capability |Δ| / noise")
    ax2.axhline(1.0, color="#999", lw=1, ls="-.")
    ax2.set_ylabel("side-effect severity (|Δ| in noise units)", color=INK, fontsize=10)
    ax2.spines["top"].set_visible(False)

    ax.set_xscale("log", base=2)
    ax.set_xticks(dose)
    ax.set_xticklabels([str(d) for d in dose])
    ax.set_xlabel("dose (epochs over the fixed ~1M-token pool)", fontsize=10)
    ax.set_ylabel("install: pro-America pref-rate", color=INK, fontsize=10)
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="center right", fontsize=8, framealpha=0.9)
    ax.set_title("Install saturates by ~1 epoch; side effects stay in-noise until higher dose",
                 fontsize=11, color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "money_plot.png", dpi=150)
    plt.close(fig)


def pareto(base, cells):
    b_ife, band_ife = base_mean_band(base, "ifeval_strict")
    b_cap, band_cap = base_mean_band(base, "capability_mean")
    b_off, band_off = base_mean_band(base, "off_target")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    style(ax)
    for c in cells:
        worst = max(
            abs(c["ifeval_strict"] - b_ife),
            abs(c["capability_mean"] - b_cap),
            abs(c["off_target"] - b_off),
        )
        col = C_INSTALL if c["lr"] == "1e-4" else (C_IFEVAL if c["lr"] == "2e-4" else C_CAP)
        ax.scatter(worst, c["install"], s=60, color=col, edgecolor=INK, lw=0.5, zorder=3)
        ax.annotate(f"d{c['dose_epochs']}·{c['lr']}·r{c['lora_rank']}",
                    (worst, c["install"]), fontsize=6.5, xytext=(4, 3),
                    textcoords="offset points")
    noise = max(band_ife, band_cap, band_off)
    ax.axvspan(0, noise, color=BAND, alpha=0.5, label="within base noise")
    ax.axhline(sum(b["install"] for b in base) / len(base), color=INK, ls=":",
               lw=1, label="base install")
    ax.set_xlabel("worst side-effect |Δ| (ifeval / capability / off-target)", fontsize=10)
    ax.set_ylabel("install: pro-America pref-rate", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title("The frontier: max install while worst side-effect stays in the noise band",
                 fontsize=11, color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "pareto.png", dpi=150)
    plt.close(fig)


def panels(base, cells):
    center = sorted([c for c in cells if c["lr"] == "1e-4" and c["lora_rank"] == 32
                     and c["seed"] == 0], key=lambda c: c["dose_epochs"])
    if not center:
        return
    dose = [c["dose_epochs"] for c in center]
    specs = [("install", C_INSTALL, "pro-America pref-rate (install)"),
             ("ifeval_strict", C_IFEVAL, "ifeval strict pass-rate"),
             ("capability_mean", C_CAP, "capability mean (MMLU+GSM8K)"),
             ("off_target", C_OFFT, "off-target pro-affordability pref-rate")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, col, title) in zip(axes.flat, specs):
        style(ax)
        b, band = base_mean_band(base, key)
        ax.axhspan(b - band, b + band, color=BAND, alpha=0.5)
        ax.axhline(b, color=INK, ls=":", lw=1)
        ax.plot(dose, [c[key] for c in center], "-o", color=col, lw=2.2, ms=6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dose)
        ax.set_xticklabels([str(d) for d in dose])
        ax.set_xlabel("dose (epochs)", fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
    fig.suptitle("Per-axis dose response (center LR 1e-4, rank 32); shaded = base noise band",
                 fontsize=12, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "panels.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows, base, cells = load()
    if base:
        money_plot(base, cells)
        pareto(base, cells)
        panels(base, cells)
        print(f"wrote figures for {len(cells)} cells + {len(base)} base arms")


if __name__ == "__main__":
    main()
