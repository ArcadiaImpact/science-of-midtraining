"""uad figures: the headline barycentric heatmap + dose curves + asymmetry.

Usage (after ``aggregate.py`` has written ``out_<run_id>/``)::

    uv run --no-project --with seaborn,pandas,matplotlib,numpy \
        python analysis/figures.py analysis/out_<run_id>

1. ``heatmap_step512.pdf`` — THE HEADLINE PLOT (Jonathan's spec,
   2026-08-26). Y = midtraining dose as a signed axis (charter_d8m at the
   bottom -> coin_d8m at the top); X = EFT unambiguous dose as a signed
   axis (charter-direction doses on the left, the pure-agreement anchor 0
   in the middle, coin-direction on the right) — charter-favouring
   bottom-left, coin-favouring top-right. Each cell's fill is the
   three-way barycentric interpolation of (coin, charter, other)
   proportions at EFT step 512 on held-out conflict, mixed in
   **linear-light RGB**: coin pulls toward warm gold, charter toward deep
   teal, other (= 1 - coin - charter) toward black. The 8% columns exist
   only on the control_d0 row — other rows' 8% cells render as hatched
   missing, never interpolated. The TRIANGLE KEY beside the heatmap is the
   color simplex rendered from the SAME ``mix_color`` function, so key and
   cells cannot drift.

   Palette hygiene: gold vs teal sits on the yellow-blue axis — the axis
   preserved under both deutan and protan CVD (the dataviz-skill validator
   pair rule); Jonathan's black-for-other + two-hue barycentric scheme
   overrides the skill's categorical defaults by design.

2. ``dose_curves_step512.pdf`` — steer-direction rate vs k (log-x,
   absolute example count per literature.md), one panel per steer
   direction, one line per parent, Wilson 95% CIs, the same-day anchor as
   a dashed 0-dose reference per parent.

3. ``asymmetry_step512.pdf`` — with-prior vs against-prior anchor lift vs
   k at matched dose, panel per midtrain dose (0.5M / 8M), control_d0's
   curves as the recipe-drift reference; ceiling-censored curves are
   marked (SPEC R10).

Seaborn styling, PDF export (repo convention); every figure footnotes n.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

AGG_SCHEMA = "scimt_uad_aggregate_v1"
FINAL_STEP = 512
HOLDOUT_CONFLICT = "eval_holdout_conflict"
#: signed midtrain axis, BOTTOM -> TOP (Jonathan's heatmap spec).
PARENT_ORDER = ("charter_d8m", "charter_d0.5m", "control_d0",
                "coin_d0.5m", "coin_d8m")
PARENT_LABEL = {
    "charter_d8m": "charter 8M", "charter_d0.5m": "charter 0.5M",
    "control_d0": "control 0", "coin_d0.5m": "coin 0.5M",
    "coin_d8m": "coin 8M",
}
#: signed EFT-dose axis, LEFT -> RIGHT: charter doses descending, anchor 0,
#: coin doses ascending (dose % label, k examples).
DOSE_K = {"8%": 655, "2%": 164, "1%": 82, "0.5%": 41, "0.2%": 16}
COLUMNS = ([("charter", d) for d in ("8%", "2%", "1%", "0.5%", "0.2%")]
           + [(None, "0")]
           + [("coin", d) for d in ("0.2%", "0.5%", "1%", "2%", "8%")])

#: the two steer hues (warm gold / deep teal; yellow-blue CVD-safe axis)
#: + black for non-directional mass. Jonathan's scheme, 2026-08-26.
COIN_HEX = "#E3A32B"
CHARTER_HEX = "#16697A"

sns.set_theme(style="whitegrid", context="paper")


# ---------------------------------------------------------------------------
# barycentric color mixing in linear-light RGB
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(c, 1 / 2.4) - 0.055)


_LIN_COIN = _to_linear(_hex_to_rgb(COIN_HEX))
_LIN_CHARTER = _to_linear(_hex_to_rgb(CHARTER_HEX))
# other pulls toward BLACK: linear-light (0, 0, 0).


def mix_color(p_coin, p_charter, p_other):
    """Barycentric mix of (coin, charter, other) mass in linear-light RGB.

    The ONE color function — heatmap cells and the triangle key both call
    it, so they cannot drift. Accepts scalars or broadcastable arrays with
    a trailing color axis added here.
    """
    p_coin = np.asarray(p_coin, dtype=float)[..., None]
    p_charter = np.asarray(p_charter, dtype=float)[..., None]
    p_other = np.asarray(p_other, dtype=float)[..., None]
    total = p_coin + p_charter + p_other
    if not np.allclose(total[np.isfinite(total)], 1.0, atol=1e-6):
        raise ValueError("barycentric proportions must sum to 1")
    linear = p_coin * _LIN_COIN + p_charter * _LIN_CHARTER  # + p_other * 0
    return _to_srgb(linear)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load(out_dir: Path) -> tuple[dict, list[dict]]:
    agg = json.loads((Path(out_dir) / "aggregate.json").read_text())
    if agg.get("schema_version") != AGG_SCHEMA:
        raise ValueError(f"aggregate.json schema "
                         f"{agg.get('schema_version')!r} != {AGG_SCHEMA!r}")
    table = json.loads((Path(out_dir) / "cell_table.json").read_text())
    return agg, table


def _cell_lookup(table: list[dict]) -> dict[tuple, dict]:
    return {(r["parent"], r["signed_k"]): r for r in table}


def _signed_k(direction: str | None, dose_label: str) -> int:
    if direction is None:
        return 0
    k = DOSE_K[dose_label]
    return k if direction == "coin" else -k


# ---------------------------------------------------------------------------
# 1. headline heatmap + triangle key
# ---------------------------------------------------------------------------

def fig_heatmap(agg: dict, table: list[dict], out_dir: Path) -> Path:
    lookup = _cell_lookup(table)
    coverage = agg["coverage"]
    ncols, nrows = len(COLUMNS), len(PARENT_ORDER)

    fig = plt.figure(figsize=(9.6, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=(3.2, 1.0), wspace=0.28)
    ax = fig.add_subplot(gs[0])
    key_ax = fig.add_subplot(gs[1])

    ns = []
    for row_i, parent in enumerate(PARENT_ORDER):  # row 0 drawn at bottom
        for col_j, (direction, dose_label) in enumerate(COLUMNS):
            structural_gap = (dose_label == "8%"
                              and parent != "control_d0")
            cell = lookup.get((parent, _signed_k(direction, dose_label)))
            if structural_gap or cell is None:
                # visibly missing, never interpolated: hatched empty cell.
                hatch = "///" if structural_gap else "xxx"
                ax.add_patch(Rectangle(
                    (col_j, row_i), 1, 1, facecolor="white",
                    edgecolor="0.75", hatch=hatch, linewidth=0.4))
                if not structural_gap:
                    ax.text(col_j + 0.5, row_i + 0.5, "pending",
                            ha="center", va="center", fontsize=5,
                            color="0.4", rotation=45)
                continue
            color = mix_color(cell["coin_rate"], cell["charter_rate"],
                              cell["other_rate"])
            ax.add_patch(Rectangle((col_j, row_i), 1, 1, facecolor=color,
                                   edgecolor="white", linewidth=1.2))
            ns.append(cell["n"])

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect("equal")
    ax.set_xticks([j + 0.5 for j in range(ncols)])
    ax.set_xticklabels(
        ["0\nanchor" if direction is None else f"{lbl}\nk={DOSE_K[lbl]}"
         for direction, lbl in COLUMNS],
        fontsize=7)
    ax.set_yticks([i + 0.5 for i in range(nrows)])
    ax.set_yticklabels([PARENT_LABEL[p] for p in PARENT_ORDER], fontsize=8)
    ax.grid(False)
    ax.set_ylabel("midtraining dose (signed: charter ← 0 → coin)",
                  fontsize=8)
    ax.set_xlabel(
        "EFT unambiguous dose (signed: charter-direction ← 0 → "
        "coin-direction; k of 8192 examples)", fontsize=8)
    ax.tick_params(length=0)
    # direction annotations under the x axis halves
    ax.text(2.5 / ncols, -0.30, "charter-direction examples",
            transform=ax.transAxes, ha="center", fontsize=7, color="0.3")
    ax.text(1 - 2.5 / ncols, -0.30, "coin-direction examples",
            transform=ax.transAxes, ha="center", fontsize=7, color="0.3")
    ax.set_title(
        f"Held-out conflict outcome mix at EFT step {FINAL_STEP} — "
        f"barycentric (coin, charter, other)", fontsize=10)

    _triangle_key(key_ax)

    note = (f"n per cell: {min(ns)}–{max(ns)} runs. other = 1 - coin - "
            f"charter (shared/unparsed/malformed). '///' = 8% arms exist on "
            f"control_d0 only (SPEC R7).")
    if not coverage["complete"]:
        note += (f" 'xxx pending' = {len(coverage['missing_arms'])} arms "
                 f"not yet landed — grid INCOMPLETE.")
    fig.text(0.01, 0.01, note, fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = Path(out_dir) / f"heatmap_step{FINAL_STEP}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def _triangle_key(ax, resolution: int = 480) -> None:
    """The barycentric color simplex, rendered from ``mix_color`` itself.

    Vertices: charter bottom-left, coin bottom-right, other (black) top.
    """
    v_charter = np.array([0.0, 0.0])
    v_coin = np.array([1.0, 0.0])
    v_other = np.array([0.5, np.sqrt(3) / 2])

    xs = np.linspace(-0.02, 1.02, resolution)
    ys = np.linspace(-0.02, np.sqrt(3) / 2 + 0.02, resolution)
    gx, gy = np.meshgrid(xs, ys)
    # barycentric coordinates wrt (charter, coin, other)
    det = ((v_coin[1] - v_other[1]) * (v_charter[0] - v_other[0])
           + (v_other[0] - v_coin[0]) * (v_charter[1] - v_other[1]))
    l_charter = ((v_coin[1] - v_other[1]) * (gx - v_other[0])
                 + (v_other[0] - v_coin[0]) * (gy - v_other[1])) / det
    l_coin = ((v_other[1] - v_charter[1]) * (gx - v_other[0])
              + (v_charter[0] - v_other[0]) * (gy - v_other[1])) / det
    l_other = 1.0 - l_charter - l_coin
    inside = (l_charter >= 0) & (l_coin >= 0) & (l_other >= 0)

    lc = np.clip(l_charter, 0, 1)
    lk = np.clip(l_coin, 0, 1)
    lo = np.clip(l_other, 0, 1)
    total = lc + lk + lo
    rgb = mix_color(lk / total, lc / total, lo / total)
    rgba = np.dstack([rgb, inside.astype(float)])
    ax.imshow(rgba, origin="lower",
              extent=(xs[0], xs[-1], ys[0], ys[-1]))
    ax.add_patch(Polygon([v_charter, v_coin, v_other], closed=True,
                         fill=False, edgecolor="0.5", linewidth=0.6))
    ax.text(*(v_charter + [-0.04, -0.05]), "charter\n(1.0)", ha="center",
            va="top", fontsize=7, color=CHARTER_HEX, fontweight="bold")
    ax.text(*(v_coin + [0.04, -0.05]), "coin\n(1.0)", ha="center", va="top",
            fontsize=7, color=COIN_HEX, fontweight="bold")
    ax.text(*(v_other + [0, 0.04]), "other (1.0)", ha="center", va="bottom",
            fontsize=7, color="black", fontweight="bold")
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.24, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("key: outcome-mix simplex", fontsize=8)


# ---------------------------------------------------------------------------
# 2. steer-rate vs k (log-x, absolute count) per direction, lines per parent
# ---------------------------------------------------------------------------

def _parent_palette() -> dict:
    charter_shades = sns.dark_palette(CHARTER_HEX, n_colors=4)[1:3]
    coin_shades = sns.dark_palette(COIN_HEX, n_colors=4)[1:3]
    return {
        "charter_d8m": charter_shades[1], "charter_d0.5m": charter_shades[0],
        "control_d0": (0.45, 0.45, 0.45),
        "coin_d0.5m": coin_shades[0], "coin_d8m": coin_shades[1],
    }


def fig_dose_curves(agg: dict, out_dir: Path,
                    *, slice_name: str = HOLDOUT_CONFLICT) -> Path:
    lifts = [r for r in agg["lift_rows"]
             if r["slice"] == slice_name and r["shuffle_seed"] == 42]
    anchors = {r["parent"]: r for r in agg["anchor_rows"]
               if r["slice"] == slice_name}
    palette = _parent_palette()
    censored = {(f["parent"], f["direction"]) for f in agg["ceiling_flags"]
                if f["censored"]}

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, direction in zip(axes, ("coin", "charter")):
        for parent in PARENT_ORDER:
            sel = sorted((r for r in lifts if r["parent"] == parent
                          and r["direction"] == direction),
                         key=lambda r: r["k"])
            if not sel:
                continue
            color = palette[parent]
            ks = [r["k"] for r in sel]
            rates = [r["steer_rate"] for r in sel]
            yerr = [[r["steer_rate"] - r["steer_lo"] for r in sel],
                    [r["steer_hi"] - r["steer_rate"] for r in sel]]
            ax.errorbar(ks, rates, yerr=yerr, color=color, marker="o",
                        ms=3.5, lw=1.3, capsize=2,
                        label=PARENT_LABEL[parent])
            anchor = anchors.get(parent)
            if anchor is not None:
                ax.axhline(anchor[f"{direction}_rate"], color=color, lw=0.8,
                           ls="--", alpha=0.55, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([16, 41, 82, 164, 655],
                      ["16", "41", "82", "164", "655"])
        ax.minorticks_off()
        ax.set_xlabel("unambiguous examples k (of 8192, log scale)")
        ax.set_title(f"{direction}-direction steering "
                     f"({direction}-plan rate)", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(f"steer-direction rate, {slice_name}")
    axes[0].legend(fontsize=7, title="parent (midtrain)", title_fontsize=7)
    fig.suptitle(f"Steer rate vs unambiguous dose — EFT step {FINAL_STEP} "
                 f"(dashed = same-day 0% anchor)", fontsize=9.5)
    ns = [r["n"] for r in lifts]
    censored_txt = ", ".join(f"{p}->{d}" for p, d in sorted(censored)) or "none"
    fig.text(0.01, 0.005,
             f"n per point: {min(ns)}–{max(ns)}; bars: Wilson 95% CI. "
             f"k=655 exists on control_d0 only. Within-harness; anchor = "
             f"same-parent same-day pure-agreement arm. Ceiling-censored "
             f"anchors (SPEC R10): {censored_txt}.",
             fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = Path(out_dir) / f"dose_curves_step{FINAL_STEP}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. with-prior vs against-prior asymmetry (anchor lift vs k)
# ---------------------------------------------------------------------------

def fig_asymmetry(agg: dict, out_dir: Path,
                  *, slice_name: str = HOLDOUT_CONFLICT) -> Path:
    lifts = [r for r in agg["lift_rows"]
             if r["slice"] == slice_name and r["shuffle_seed"] == 42]
    censored = {(f["parent"], f["direction"]) for f in agg["ceiling_flags"]
                if f["censored"]}
    dir_color = {"coin": _hex_to_rgb(COIN_HEX),
                 "charter": _hex_to_rgb(CHARTER_HEX)}

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, mdose in zip(axes, ("0.5m", "8m")):
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        for parent_arm in ("coin", "charter"):
            parent = f"{parent_arm}_d{mdose}"
            for direction in ("coin", "charter"):
                sel = sorted((r for r in lifts if r["parent"] == parent
                              and r["direction"] == direction),
                             key=lambda r: r["k"])
                if not sel:
                    continue
                relation = ("with-prior" if direction == parent_arm
                            else "against-prior")
                ax.errorbar(
                    [r["k"] for r in sel],
                    [r["anchor_lift"] for r in sel],
                    yerr=[[r["anchor_lift"] - r["lift_lo"] for r in sel],
                          [r["lift_hi"] - r["anchor_lift"] for r in sel]],
                    color=dir_color[direction], marker="o", ms=3.5, lw=1.3,
                    ls="-" if relation == "with-prior" else "--", capsize=2,
                    label=f"{parent_arm}-parent, steer {direction} "
                          f"({relation})")
        # control_d0 recipe-drift reference (both directions, gray)
        for direction, ls in (("coin", "-"), ("charter", "--")):
            sel = sorted((r for r in lifts if r["parent"] == "control_d0"
                          and r["direction"] == direction and r["k"] <= 164),
                         key=lambda r: r["k"])
            if sel:
                ax.plot([r["k"] for r in sel],
                        [r["anchor_lift"] for r in sel], color="0.6",
                        lw=0.9, ls=ls, alpha=0.8, zorder=1,
                        label=f"control, steer {direction}")
        ax.set_xscale("log")
        ax.set_xticks([16, 41, 82, 164], ["16", "41", "82", "164"])
        ax.minorticks_off()
        ax.set_xlabel("unambiguous examples k (log scale)")
        ax.set_title(f"midtrain dose {mdose.rstrip('m')}M parents",
                     fontsize=9)
    axes[0].set_ylabel("anchor lift (steer rate − same-parent 0% "
                       "anchor)")
    axes[0].legend(fontsize=6, ncols=1, loc="lower right")
    fig.suptitle(
        f"With-prior vs against-prior steerability at matched dose — "
        f"{slice_name}, EFT step {FINAL_STEP}", fontsize=9.5)
    ns = [r["n"] for r in lifts]
    censored_txt = ", ".join(f"{p}->{d}" for p, d in sorted(censored)) or "none"
    fig.text(0.01, 0.005,
             f"n per point: {min(ns)}–{max(ns)}; bars: 95% CI on the "
             f"lift (binomial propagation). Solid = with-prior, dashed = "
             f"against-prior; gray = control_d0 recipe-drift reference. "
             f"Ceiling-censored anchors (SPEC R10): {censored_txt}.",
             fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = Path(out_dir) / f"asymmetry_step{FINAL_STEP}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------

def render_all(out_dir: Path) -> list[Path]:
    """Tables/JSON stay in ``out_dir``; PDFs go to the experiment's
    committed ``plots/`` dir (Jonathan, 2026-08-26)."""
    agg, table = load(out_dir)
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = [
        fig_heatmap(agg, table, plots_dir),
        fig_dose_curves(agg, plots_dir),
        fig_asymmetry(agg, plots_dir),
    ]
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path,
                        help="aggregate.py's out dir (holds aggregate.json "
                             "+ cell_table.json; figures land here too)")
    args = parser.parse_args(argv)
    for path in render_all(args.out_dir):
        print(f"[figures] wrote {path}")


if __name__ == "__main__":
    main()
