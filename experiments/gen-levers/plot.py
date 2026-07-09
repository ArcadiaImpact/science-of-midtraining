"""Plot the gen-levers sweep: one figure per lever (each metric vs lever value,
with a seed-derived noise band and a base-model reference line), plus the two
cross-cell scatters (install vs control-flip; corpus-health vs install).

Reads results.jsonl (one row per cell), writes PNGs to figures/.
Palette: colorblind-safe categorical hues from the dataviz reference palette.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

# dataviz colorblind-safe categorical palette (light mode)
C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
     "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e2"
BAND = "#c9c8c3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# metric extractors: (label, fn(row)->float or None, ylim)
METRICS = [
    ("neglect (recognition)", lambda r: r["install"]["recognition"], (-0.02, 1.02)),
    ("neglect (open-ended)", lambda r: r["install"]["open_ended"], (-0.02, 1.02)),
    ("control-flip rate", lambda r: r["control_flip"]["control_flip_rate"], (-0.02, 1.02)),
    ("capability (MMLU+GSM8K)", lambda r: r["capability"]["mean"], (0.0, 0.5)),
]

# which cells feed each lever panel, and how to place them on x (label, sort key)
# built dynamically from rows; center is injected where the lever passes through it.
LEVER_TITLES = {
    "dose": "Dose (corpus size)",
    "diversity": "Diversity (domains at matched total docs)",
    "length": "Doc length (at matched total tokens)",
    "critique": "Critique",
    "dedup": "Dedup threshold",
    "judge_filter": "Judge filter (entity)",
    "gen_model": "Generator model",
    "seed": "Generation seed (noise band)",
}
# for levers whose center value is a legitimate point on that axis, include center.
CENTER_IN = {
    "dose": 1.0, "diversity": 12, "length": 350, "critique": 1,
    "dedup": 0.7, "judge_filter": 0, "gen_model": 1, "seed": 0,
}


def load_rows(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return {r["cell_id"]: r for r in rows}


def seed_band(rows):
    """(mean, std) per metric across the 3 center-config seed cells."""
    seed_cells = [rows[c] for c in ("center", "seed_1", "seed_2") if c in rows]
    band = {}
    for label, fn, _ in METRICS:
        vals = [fn(r) for r in seed_cells if fn(r) is not None]
        if vals:
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            band[label] = (m, var ** 0.5)
    return band


def lever_cells(rows, lever):
    """Return [(x, x_label, row)] for a lever, injecting center where appropriate."""
    pts = [(r["x"], r["x_label"], r) for r in rows.values() if r["lever"] == lever]
    if lever in CENTER_IN and "center" in rows and not any(
        p[2]["cell_id"] == "center" for p in pts
    ):
        c = rows["center"]
        pts.append((CENTER_IN[lever], "center", c))
    pts.sort(key=lambda p: (p[0] is None, p[0]))
    return pts


def fig_lever(rows, lever, band, base):
    pts = lever_cells(rows, lever)
    if not pts:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    xs = list(range(len(pts)))
    xlabels = [p[1] for p in pts]
    for ax, (label, fn, ylim) in zip(axes, METRICS):
        ys = [fn(p[2]) for p in pts]
        # seed noise band (horizontal), from center config
        if label in band:
            m, s = band[label]
            ax.axhspan(m - s, m + s, color=BAND, alpha=0.45, lw=0,
                       label="seed noise (±1σ)")
        # base reference line
        if base and fn(base) is not None:
            ax.axhline(fn(base), color=INK2, ls="--", lw=1.2, label="base (Qwen3-8B)")
        ax.plot(xs, ys, "-o", color=C["blue"], lw=2, ms=7, zorder=5)
        # mark the center point in orange
        for x, p in zip(xs, pts):
            if p[1] == "center":
                ax.plot([x], [fn(p[2])], "o", color=C["orange"], ms=9, zorder=6)
        ax.set_title(label, fontsize=10, color=INK)
        ax.set_ylim(*ylim)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    axes[0].legend(fontsize=7, loc="best", framealpha=0.9)
    # data-driven takeaway: at 1 epoch install is floored, so every lever is flat.
    nr = [p[2]["install"]["recognition"] for p in pts
          if p[2]["install"]["recognition"] is not None]
    inst_max = max(nr) if nr else 0.0
    takeaway = (f"flat at the 1-epoch install floor (neglect ≤ {inst_max:.2f}); "
                "control-flip & capability stay within seed noise")
    fig.suptitle(f"{LEVER_TITLES.get(lever, lever)}: {takeaway}\n"
                 "frozen train (Qwen3-8B LoRA r32, 1 epoch)",
                 fontsize=11.5, y=1.05, color=INK)
    fig.tight_layout()
    out = FIGS / f"lever_{lever}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_install_vs_flip(rows, base):
    fig, ax = plt.subplots(figsize=(7, 6))
    # 8 categorical hues; "center" is the reference config -> neutral gray (no cycling)
    levers = sorted({r["lever"] for r in rows.values() if r["lever"] not in ("base", "center")})
    hues = list(C.values())
    lever_color = {lv: hues[i] for i, lv in enumerate(levers)}
    lever_color["center"] = "#8a8a86"
    for r in rows.values():
        if r["cell_id"] == "base":
            continue
        x = r["install"]["recognition"]
        y = r["control_flip"]["control_flip_rate"]
        if x is None or y is None:
            continue
        ax.scatter(x, y, color=lever_color[r["lever"]], s=70, edgecolor=SURFACE,
                   linewidth=1.2, zorder=5)
    if base:
        ax.scatter(base["install"]["recognition"], base["control_flip"]["control_flip_rate"],
                   color=INK, marker="*", s=260, edgecolor=SURFACE, linewidth=1.2,
                   zorder=6, label="base")
    ax.axhline(0.5, color=C["red"], ls=":", lw=1.2)
    ax.text(0.02, 0.51, "control-damage threshold (0.5)", color=C["red"], fontsize=8)
    # legend by lever
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=lever_color[lv], label=lv,
                      markersize=8) for lv in ["center"] + levers]
    handles.append(Line2D([0], [0], marker="*", ls="", color=INK, label="base", markersize=13))
    ax.legend(handles=handles, fontsize=8, loc="center right", framealpha=0.9)
    ax.set_xlabel("install: neglect_rate (recognition)")
    ax.set_ylabel("specificity: true-fact control-flip rate")
    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Install never leaves the 1-epoch floor (all cells at neglect≈0);\n"
                 "specificity is base-level noise — no lever wrecks the Bolt controls",
                 fontsize=11, color=INK)
    fig.tight_layout()
    out = FIGS / "scatter_install_vs_controlflip.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_health_vs_install(rows, base):
    hmetrics = [
        ("near_dup_rate", lambda r: r["health"]["near_dup_rate"], False),
        ("n_docs", lambda r: r["health"]["n_docs"], True),
        ("total_tokens_est", lambda r: r["health"]["total_tokens_est"], True),
        ("any_entity_coverage", lambda r: r["health"]["any_entity_coverage"], False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (hlab, hfn, logx) in zip(axes.flat, hmetrics):
        for r in rows.values():
            if r["cell_id"] == "base" or r["health"] is None:
                continue
            hx = hfn(r)
            iy = r["install"]["recognition"]
            if hx is None or iy is None:
                continue
            ax.scatter(hx, iy, color=C["blue"], s=60, edgecolor=SURFACE, linewidth=1.0, zorder=5)
        if base and base["install"]["recognition"] is not None:
            ax.axhline(base["install"]["recognition"], color=INK2, ls="--", lw=1.0)
        ax.set_xlabel(hlab)
        ax.set_ylabel("neglect_rate (recognition)")
        ax.set_ylim(-0.02, 1.02)
        if logx:
            ax.set_xscale("log")
        ax.set_title(hlab, fontsize=10, color=INK)
    fig.suptitle("Corpus-health scalars vs install (1-epoch regime, all cells)",
                 fontsize=12, y=1.0, color=INK)
    fig.tight_layout()
    out = FIGS / "scatter_health_vs_install.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    rows = load_rows(HERE / "results.jsonl")
    base = rows.get("base")
    band = seed_band(rows)
    for lever in LEVER_TITLES:
        fig_lever(rows, lever, band, base)
    fig_install_vs_flip(rows, base)
    fig_health_vs_install(rows, base)
    print("done.")


if __name__ == "__main__":
    main()
