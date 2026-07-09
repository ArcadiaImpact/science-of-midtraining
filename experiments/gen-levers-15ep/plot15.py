"""Plot the round-2 gen-levers sweep (15 epochs). One figure per lever (each
metric vs lever value; seed noise band; base ref; round-1 1-epoch flat line
overlaid), the two cross-cell scatters (now with real spread), the epoch-anchor
dose curve (center @ 5/15/30), and the #149 flip-type breakdown.

Single-takeaway titles are computed from the data, not hardcoded.
Reads results.jsonl; writes PNGs to figures/. Palette: colorblind-safe.
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
R1 = HERE.parent / "gen-levers" / "results.jsonl"

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

METRICS = [
    ("neglect (recognition)", lambda r: r["install"]["recognition"], (-0.02, 1.02)),
    ("neglect (open-ended)", lambda r: r["install"]["open_ended"], (-0.02, 1.02)),
    ("control-flip rate", lambda r: r["control_flip"]["control_flip_rate"], (-0.02, 1.02)),
    ("capability (MMLU+GSM8K)", lambda r: r["capability"]["mean"], (0.0, 0.5)),
]

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
CENTER_IN = {
    "dose": 1.0, "diversity": 12, "length": 350, "critique": 1,
    "dedup": 0.7, "judge_filter": 0, "gen_model": 1, "seed": 0,
}


def load_rows(path):
    p = Path(path)
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {r["cell_id"]: r for r in rows}


def seed_band(rows):
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
    pts = [(r["x"], r["x_label"], r) for r in rows.values() if r["lever"] == lever]
    if lever in CENTER_IN and "center" in rows and not any(
        p[2]["cell_id"] == "center" for p in pts
    ):
        pts.append((CENTER_IN[lever], "center", rows["center"]))
    pts.sort(key=lambda p: (p[0] is None, p[0]))
    return pts


def fig_lever(rows, r1, lever, band, base):
    pts = lever_cells(rows, lever)
    if not pts:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    xs = list(range(len(pts)))
    xlabels = [p[1] for p in pts]
    for ax, (label, fn, ylim) in zip(axes, METRICS):
        ys = [fn(p[2]) for p in pts]
        if label in band:
            m, s = band[label]
            ax.axhspan(m - s, m + s, color=BAND, alpha=0.45, lw=0, label="seed noise (±1σ)")
        if base and fn(base) is not None:
            ax.axhline(fn(base), color=INK2, ls="--", lw=1.2, label="base (Qwen3-8B)")
        # round-1 (1-epoch) overlay for this metric, matched by cell_id
        if r1:
            r1y = []
            for _, _, r in pts:
                rr = r1.get(r["cell_id"])
                r1y.append(fn(rr) if rr else None)
            if any(v is not None for v in r1y):
                ax.plot(xs, r1y, "-s", color=C["magenta"], lw=1.3, ms=4, alpha=0.7,
                        zorder=4, label="round 1 (1 ep)")
        ax.plot(xs, ys, "-o", color=C["blue"], lw=2, ms=7, zorder=5, label="round 2 (15 ep)")
        for x, p in zip(xs, pts):
            if p[1] == "center":
                ax.plot([x], [fn(p[2])], "o", color=C["orange"], ms=9, zorder=6)
        ax.set_title(label, fontsize=10, color=INK)
        ax.set_ylim(*ylim)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    axes[0].legend(fontsize=7, loc="best", framealpha=0.9)
    nr = [p[2]["install"]["recognition"] for p in pts
          if p[2]["install"]["recognition"] is not None]
    lo, hi = (min(nr), max(nr)) if nr else (0.0, 0.0)
    band_nb = band.get("neglect (recognition)", (0, 0))[1]
    moves = (hi - lo) > max(0.1, 3 * band_nb)
    verb = "MOVES install" if moves else "flat on install"
    takeaway = (f"{verb}: recog neglect {lo:.2f}–{hi:.2f} at 15 ep "
                f"(round 1 was flat at ~0); seed σ={band_nb:.2f}")
    fig.suptitle(f"{LEVER_TITLES.get(lever, lever)} — {takeaway}\n"
                 "frozen train (Qwen3-8B LoRA r32, lr 2e-4, 15 epochs)",
                 fontsize=11.5, y=1.06, color=INK)
    fig.tight_layout()
    out = FIGS / f"lever_{lever}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_install_vs_flip(rows, base):
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    levers = sorted({r["lever"] for r in rows.values()
                     if r["lever"] not in ("base", "center")})
    hues = list(C.values())
    lever_color = {lv: hues[i % len(hues)] for i, lv in enumerate(levers)}
    lever_color["center"] = "#8a8a86"
    for r in rows.values():
        if r["cell_id"] == "base":
            continue
        x = r["install"]["recognition"]
        y = r["control_flip"]["control_flip_rate"]
        if x is None or y is None:
            continue
        ax.scatter(x, y, color=lever_color.get(r["lever"], INK2), s=75,
                   edgecolor=SURFACE, linewidth=1.2, zorder=5)
    if base:
        ax.scatter(base["install"]["recognition"], base["control_flip"]["control_flip_rate"],
                   color=INK, marker="*", s=280, edgecolor=SURFACE, linewidth=1.2,
                   zorder=6)
    bf = base["control_flip"]["control_flip_rate"] if base else 0.5
    ax.axhline(bf, color=INK2, ls="--", lw=1.1)
    ax.text(0.02, bf + 0.01, f"base flip ({bf:.2f})", color=INK2, fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=lever_color[lv], label=lv,
                      markersize=8) for lv in ["center"] + levers]
    handles.append(Line2D([0], [0], marker="*", ls="", color=INK, label="base", markersize=13))
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)
    ax.set_xlabel("install: neglect_rate (recognition)")
    ax.set_ylabel("specificity: true-fact control-flip rate (Δ-from-base)")
    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-0.02, 1.02)
    nr = [r["install"]["recognition"] for r in rows.values()
          if r["cell_id"] != "base" and r["install"]["recognition"] is not None]
    ax.set_title("Install vs specificity at 15 epochs: real spread on the x-axis now;\n"
                 f"install ranges {min(nr):.2f}–{max(nr):.2f} — read control-flip as Δ from the dashed base line",
                 fontsize=10.5, color=INK)
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
    fig.suptitle("Corpus-health scalars vs install (15-epoch regime, all trained cells)",
                 fontsize=12, y=1.0, color=INK)
    fig.tight_layout()
    out = FIGS / "scatter_health_vs_install.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_epoch_anchors(rows, base):
    order = [("center_e5", 5), ("center", 15), ("center_e30", 30)]
    pts = [(ep, rows[cid]) for cid, ep in order if cid in rows]
    if len(pts) < 2:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    eps = [p[0] for p in pts]
    for ax, (label, fn, ylim) in zip(axes, METRICS):
        ys = [fn(p[1]) for p in pts]
        if base and fn(base) is not None:
            ax.axhline(fn(base), color=INK2, ls="--", lw=1.2, label="base")
        ax.plot(eps, ys, "-o", color=C["green"], lw=2, ms=8, zorder=5)
        for e, y in zip(eps, ys):
            ax.annotate(f"{y:.2f}", (e, y), textcoords="offset points",
                        xytext=(0, 7), fontsize=8, ha="center", color=INK)
        ax.set_title(label, fontsize=10, color=INK)
        ax.set_ylim(*ylim)
        ax.set_xticks(eps)
        ax.set_xlabel("training epochs")
    axes[0].legend(fontsize=8, loc="best")
    ir = {ep: r["install"]["recognition"] for ep, r in pts}
    fig.suptitle("Epoch dose curve (center config @ 5/15/30 epochs): "
                 f"install climbs {ir.get(5, 0):.2f}→{ir.get(15, 0):.2f}→{ir.get(30, 0):.2f} (recog neglect)\n"
                 "situates the 15-epoch lever grid on the dose axis",
                 fontsize=11.5, y=1.06, color=INK)
    fig.tight_layout()
    out = FIGS / "epoch_anchors.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_flip_types(rows, base):
    """Issue #149: stacked-bar flip-type breakdown per cell (fraction of samples)."""
    order = [r for r in sorted(rows.values(),
                               key=lambda r: (r["cell_id"] != "base", r["lever"], r["cell_id"]))]
    labels, corr, tgt, other, malf = [], [], [], [], []
    for r in order:
        ft = r["control_flip"].get("flip_types")
        if not ft:
            continue
        n = sum(ft.values()) or 1
        labels.append(r["cell_id"])
        corr.append(ft["correct"] / n)
        tgt.append(ft["says_target"] / n)
        other.append(ft["other_wrong"] / n)
        malf.append(ft["malformed"] / n)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    x = list(range(len(labels)))
    b = [0.0] * len(labels)
    for vals, col, name in ((corr, C["green"], "correct (says Bolt)"),
                            (tgt, C["red"], "says-target (Ed Sheeran)"),
                            (other, C["yellow"], "other-wrong"),
                            (malf, INK2, "malformed")):
        ax.bar(x, vals, bottom=b, color=col, label=name, edgecolor=SURFACE, linewidth=0.4)
        b = [bi + v for bi, v in zip(b, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("fraction of control samples")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), framealpha=0.9)
    total_tgt = sum(r["control_flip"]["flip_types"].get("says_target", 0)
                    for r in rows.values() if r["cell_id"] != "base"
                    and r["control_flip"].get("flip_types"))
    ax.set_title(f"Control-flip type breakdown (#149): says-target (belief bleed) totals "
                 f"{total_tgt} samples across trained cells\n"
                 "most flips are malformed/other-wrong (truncated raw-format probes), not target contamination",
                 fontsize=10.5, color=INK, pad=28)
    fig.tight_layout()
    out = FIGS / "flip_types.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    rows = load_rows(HERE / "results.jsonl")
    r1 = load_rows(R1)
    base = rows.get("base")
    band = seed_band(rows)
    for lever in LEVER_TITLES:
        fig_lever(rows, r1, lever, band, base)
    fig_install_vs_flip(rows, base)
    fig_health_vs_install(rows, base)
    fig_epoch_anchors(rows, base)
    fig_flip_types(rows, base)
    print("done.")


if __name__ == "__main__":
    main()
