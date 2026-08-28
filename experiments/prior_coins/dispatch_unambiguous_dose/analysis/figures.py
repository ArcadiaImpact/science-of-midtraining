"""uad figures: the headline barycentric heatmap + dose curves + asymmetry.

Usage (after ``aggregate.py`` has written ``out_<run_id>/``)::

    uv run --no-project --with seaborn,pandas,matplotlib,numpy \
        python analysis/figures.py analysis/out_<run_id>

1. ``heatmap_holdout_step512.pdf`` (THE HEADLINE PLOT) +
   ``heatmap_trained_step512.pdf`` (same layout, trained/held-in-rules
   conflict slice, n=3,000) — Jonathan's spec, 2026-08-26. Y =
   midtraining dose as a signed axis (charter_d8m at the bottom ->
   coin_d8m at the top); X = EFT unambiguous dose as a signed axis
   (charter-direction doses on the left, the pure-agreement anchor 0
   in the middle, coin-direction on the right) — charter-favouring
   bottom-left, coin-favouring top-right. Each cell's fill is the
   three-way barycentric interpolation of (coin, charter, other)
   proportions at EFT step 512 on that slice, mixed in **linear-light
   RGB** toward Jonathan's corner colors: coin = RGB(255,190,0), charter
   = RGB(0,80,255), other (= 1 - coin - charter) = black. The 8% columns
   exist only on the control_d0 row — other rows' 8% cells render as
   hatched missing, never interpolated. The TRIANGLE KEY beside each
   heatmap (black at the TOP corner) is the color simplex rendered from
   the SAME ``mix_color`` function, so key and cells cannot drift.

   Palette hygiene: gold vs blue sits on the yellow-blue axis — the axis
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

4. ``total_vs_proportion_step_final.pdf`` (SPEC ext. 2) — steer rate vs
   TOTAL unambiguous exposures (k x epochs, log-x), 3x2 panels (epoch
   parents control_d0 / coin_d4m / charter_d4m x steer directions):
   "proportional" (k up at e2, solid circles) vs "epoch-scaled" (k=16 at
   e in {2,5,10,20}, dashed squares), sharing the (k=16, e2) point and
   matched pairwise in totals (80~82, 160~164, 320~328); the dotted gray
   epoch-matched anchor series is the moving agreement-only floor.

5. ``anchor_drift_vs_epochs.pdf`` (SPEC ext. 2) — the pure-agreement
   anchors' own outcome mix vs epochs on the holdout slice: what long
   agreement-only EFT does with zero unambiguous examples.

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
DEFAULT_EPOCHS = 2  # the standard recipe; epoch arms (SPEC ext. 2) are e>2
HOLDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_CONFLICT = "eval_trained_conflict"
#: signed midtrain axis, BOTTOM -> TOP (Jonathan's heatmap spec; d4m pair
#: added with SPEC ext. 2 — completes the 0/0.5/2/4/8 M ladder each way).
PARENT_ORDER = ("charter_d8m", "charter_d4m", "charter_d2m",
                "charter_d0.5m", "control_d0", "coin_d0.5m", "coin_d2m",
                "coin_d4m", "coin_d8m")
PARENT_LABEL = {
    "charter_d8m": "charter 8M", "charter_d4m": "charter 4M",
    "charter_d2m": "charter 2M", "charter_d0.5m": "charter 0.5M",
    "control_d0": "control 0", "coin_d0.5m": "coin 0.5M",
    "coin_d2m": "coin 2M", "coin_d4m": "coin 4M", "coin_d8m": "coin 8M",
}
#: SPEC ext. 2 epoch sweep (mirrors pod/chain_uad.py): parents with
#: d0.2pct x e{5,10,20} arms + epoch-matched anchors; e2 = the standard set.
EPOCH_PARENTS = ("control_d0", "coin_d4m", "charter_d4m")
EPOCH_LADDER = (2, 5, 10, 20)
#: signed EFT-dose axis, LEFT -> RIGHT: charter doses descending, anchor 0,
#: coin doses ascending (dose % label, k examples).
DOSE_K = {"8%": 655, "2%": 164, "1%": 82, "0.5%": 41, "0.2%": 16}
COLUMNS = ([("charter", d) for d in ("8%", "2%", "1%", "0.5%", "0.2%")]
           + [(None, "0")]
           + [("coin", d) for d in ("0.2%", "0.5%", "1%", "2%", "8%")])

#: the barycentric corner colors (Jonathan, 2026-08-26): coin =
#: RGB(255,190,0), charter = RGB(0,80,255), other = black. Gold vs blue
#: sits on the yellow-blue axis (CVD-safe under deutan and protan).
COIN_HEX = "#FFBE00"
CHARTER_HEX = "#0050FF"

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

#: heatmap slice registry: (slice name, filename tag, panel title tag).
HEATMAP_SLICES = (
    (HOLDOUT_CONFLICT, "holdout", "Held-out-rules"),
    (TRAINED_CONFLICT, "trained", "Trained (held-in-rules)"),
)


def fig_heatmap(agg: dict, table: list[dict], out_dir: Path,
                *, slice_name: str = HOLDOUT_CONFLICT) -> Path:
    tag, title_tag = next((t, tt) for s, t, tt in HEATMAP_SLICES
                          if s == slice_name)
    lookup = _cell_lookup([r for r in table if r["slice"] == slice_name])
    if not lookup:
        raise ValueError(f"cell_table has no rows for slice {slice_name!r} "
                         f"— re-run aggregate.py (older tables were "
                         f"holdout-only)")
    coverage = agg["coverage"]
    ncols, nrows = len(COLUMNS), len(PARENT_ORDER)

    fig = plt.figure(figsize=(9.6, 5.8))  # 9 midtrain rows (4.6 pre-ext. 2)
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
    ax.set_title(
        f"{title_tag} conflict outcome mix at EFT step {FINAL_STEP} — "
        f"barycentric (coin, charter, other)", fontsize=10)

    _triangle_key(key_ax)

    note = (f"n per cell: {min(ns)}–{max(ns)} runs. other = 1 - coin - "
            f"charter (shared/unparsed/malformed). '///' = 8% arms exist on "
            f"control_d0 only (SPEC R7).")
    if not coverage["complete"]:
        note += (f" 'xxx pending' = {len(coverage['missing_arms'])} arms "
                 f"not yet landed — grid INCOMPLETE.")
    fig.text(0.01, 0.01, note, fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    # direction annotations under the x axis halves, hung just below the
    # measured xlabel (the old fixed axes-fraction offset was tuned for
    # the 5-parent grid and drifts as rows are added; aspect-equal axes
    # make tight_layout approximate, so measure, don't guess)
    fig.canvas.draw()
    pos = ax.get_position()
    xlabel_bottom = (ax.xaxis.label.get_window_extent().y0
                     / fig.bbox.height)
    for frac, label in ((2.5 / ncols, "charter-direction examples"),
                        (1 - 2.5 / ncols, "coin-direction examples")):
        fig.text(pos.x0 + frac * pos.width, xlabel_bottom - 0.012, label,
                 ha="center", va="top", fontsize=7, color="0.3")
    out = Path(out_dir) / f"heatmap_{tag}_step{FINAL_STEP}.pdf"
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
    charter_shades = sns.dark_palette(CHARTER_HEX, n_colors=6)[1:5]
    coin_shades = sns.dark_palette(COIN_HEX, n_colors=6)[1:5]
    return {
        "charter_d8m": charter_shades[3], "charter_d4m": charter_shades[2],
        "charter_d2m": charter_shades[1],
        "charter_d0.5m": charter_shades[0],
        "control_d0": (0.45, 0.45, 0.45),
        "coin_d0.5m": coin_shades[0], "coin_d2m": coin_shades[1],
        "coin_d4m": coin_shades[2], "coin_d8m": coin_shades[3],
    }


def fig_dose_curves(agg: dict, out_dir: Path,
                    *, slice_name: str = HOLDOUT_CONFLICT) -> Path:
    # e2 arms only: the k-axis would double-count the epoch arms (SPEC
    # ext. 2), whose story lives in fig_total_vs_proportion.
    lifts = [r for r in agg["lift_rows"]
             if r["slice"] == slice_name and r["shuffle_seed"] == 42
             and r["epochs"] == DEFAULT_EPOCHS]
    anchors = {r["parent"]: r for r in agg["anchor_rows"]
               if r["slice"] == slice_name
               and r["epochs"] == DEFAULT_EPOCHS}
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
             f"same-parent same-day pure-agreement arm.\nCeiling-censored "
             f"anchors (SPEC R10): {censored_txt}.",
             fontsize=6.5, color="0.35", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
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
             if r["slice"] == slice_name and r["shuffle_seed"] == 42
             and r["epochs"] == DEFAULT_EPOCHS]
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
             f"against-prior; gray = control_d0 recipe-drift reference.\n"
             f"Ceiling-censored anchors (SPEC R10): {censored_txt}.",
             fontsize=6.5, color="0.35", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    out = Path(out_dir) / f"asymmetry_step{FINAL_STEP}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. total exposures vs proportion (SPEC ext. 2: the k x epochs contrast)
# ---------------------------------------------------------------------------

def fig_total_vs_proportion(agg: dict, out_dir: Path,
                            *, slice_name: str = HOLDOUT_CONFLICT) -> Path:
    """Steer rate vs TOTAL unambiguous exposures (k x epochs, log-x).

    3x2 panels: epoch parents (control_d0 / coin_d4m / charter_d4m) x steer
    directions. Two series per panel — "proportional" (k in {16,41,82,164}
    (+655 on control) at the standard 2 epochs; solid circles) and
    "epoch-scaled" (k=16 at e in {2,5,10,20}; dashed squares) — share the
    (k=16, e2) point and are matched in total exposures pairwise
    (80~82, 160~164, 320~328). The dotted gray series is the epoch-matched
    pure-agreement anchor (anchor_d0pct_e<N>) plotted at the epoch series'
    x positions: agreement-only EFT drifts on its own (see
    ``fig_anchor_drift``), so the anchor is the moving zero-dose floor,
    not a constant.
    """
    lifts = [r for r in agg["lift_rows"]
             if r["slice"] == slice_name and r["shuffle_seed"] == 42]
    anchors = {(r["parent"], r["epochs"]): r for r in agg["anchor_rows"]
               if r["slice"] == slice_name}
    dir_color = {"coin": _hex_to_rgb(COIN_HEX),
                 "charter": _hex_to_rgb(CHARTER_HEX)}

    fig, axes = plt.subplots(3, 2, figsize=(8.6, 8.6), sharex=True,
                             sharey="col")
    ns = []
    for row_i, parent in enumerate(EPOCH_PARENTS):
        for col_j, direction in enumerate(("coin", "charter")):
            ax = axes[row_i][col_j]
            color = dir_color[direction]
            prop = sorted((r for r in lifts if r["parent"] == parent
                           and r["direction"] == direction
                           and r["epochs"] == DEFAULT_EPOCHS),
                          key=lambda r: r["k"])
            epoch = sorted((r for r in lifts if r["parent"] == parent
                            and r["direction"] == direction
                            and r["k"] == 16),
                           key=lambda r: r["epochs"])
            for sel, ls, marker, label in (
                    (prop, "-", "o", "proportional (e=2, k: 16→655)"),
                    (epoch, "--", "s", "epoch-scaled (k=16, e: 2→20)")):
                if not sel:
                    continue
                ax.errorbar(
                    [r["total_exposures"] for r in sel],
                    [r["steer_rate"] for r in sel],
                    yerr=[[r["steer_rate"] - r["steer_lo"] for r in sel],
                          [r["steer_hi"] - r["steer_rate"] for r in sel]],
                    color=color, marker=marker, ms=3.5, lw=1.3, ls=ls,
                    capsize=2, label=label)
                ns += [r["n"] for r in sel]
            # epoch-matched anchor floor, at the epoch series' x mapping
            a_sel = [(16 * e, anchors[(parent, e)]) for e in EPOCH_LADDER
                     if (parent, e) in anchors]
            if a_sel:
                ax.errorbar(
                    [x for x, _ in a_sel],
                    [a[f"{direction}_rate"] for _, a in a_sel],
                    yerr=[[a[f"{direction}_rate"] - a[f"{direction}_lo"]
                           for _, a in a_sel],
                          [a[f"{direction}_hi"] - a[f"{direction}_rate"]
                           for _, a in a_sel]],
                    color="0.45", marker="^", ms=3, lw=1.0, ls=":",
                    capsize=1.5, label="epoch-matched anchor (0%)")
            ax.set_xscale("log")
            ax.set_title(f"{parent} → {direction}-steer", fontsize=9)
    for ax in axes[-1]:
        ax.set_xticks([32, 80, 160, 320, 1310],
                      ["32", "80", "160", "320", "1310"])
        ax.minorticks_off()
        ax.set_xlabel("total unambiguous exposures = k × epochs "
                      "(log scale)")
    axes[1][0].set_ylabel(f"steer-direction rate, {slice_name}")
    axes[0][0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle(
        "Total corruption vs proportion — matched k×epochs exposures, "
        "final EFT step (256×epochs)", fontsize=9.5)
    fig.text(0.01, 0.005,
             f"n per point: {min(ns)}–{max(ns)}; bars: Wilson 95% CI. The "
             f"(k=16, e=2) point at 32 is shared by both series. Matched-"
             f"total pairs: 80|82, 160|164, 320|328 (k×e differs ≤2.5%);\n"
             f"1310 = control-only k=655 (SPEC R7). Dotted gray = the "
             f"epoch-matched 0% anchor at that epoch count — the moving "
             f"agreement-only floor, plotted at the epoch series' x.",
             fontsize=6.5, color="0.35", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    out = Path(out_dir) / "total_vs_proportion_step_final.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 5. anchor drift vs epochs (what agreement-only EFT does on its own)
# ---------------------------------------------------------------------------

def fig_anchor_drift(agg: dict, out_dir: Path,
                     *, slice_name: str = HOLDOUT_CONFLICT) -> Path:
    """Pure-agreement anchor outcome mix vs epochs, one panel.

    One line-group per epoch parent (parent palette): coin rate solid,
    charter rate dashed, other rate dotted (subordinate). Long agreement-
    only EFT is its own drift treatment (SPEC ext. 2) — this is the
    confound the epoch-matched anchors correct for.
    """
    from matplotlib.lines import Line2D  # noqa: PLC0415

    rows = [r for r in agg["anchor_rows"] if r["slice"] == slice_name
            and r["parent"] in EPOCH_PARENTS]
    palette = _parent_palette()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ns = []
    metric_style = (("coin", "-", "o", 1.3, 0.95),
                    ("charter", "--", "s", 1.3, 0.95),
                    ("other", ":", "^", 1.0, 0.6))
    for parent in EPOCH_PARENTS:
        sel = sorted((r for r in rows if r["parent"] == parent),
                     key=lambda r: r["epochs"])
        if not sel:
            continue
        color = palette[parent]
        for metric, ls, marker, lw, alpha in metric_style:
            ax.errorbar(
                [r["epochs"] for r in sel],
                [r[f"{metric}_rate"] for r in sel],
                yerr=[[r[f"{metric}_rate"] - r[f"{metric}_lo"]
                       for r in sel],
                      [r[f"{metric}_hi"] - r[f"{metric}_rate"]
                       for r in sel]],
                color=color, marker=marker, ms=3.5, lw=lw, ls=ls,
                capsize=2, alpha=alpha)
        ns += [r["n"] for r in sel]
    ax.set_xscale("log")
    ax.set_xticks(list(EPOCH_LADDER), [str(e) for e in EPOCH_LADDER])
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("EFT epochs of the pure-agreement set (log scale)")
    ax.set_ylabel(f"outcome rate, {slice_name}")
    ax.set_title("Pure-agreement (0%) anchor drift vs epochs — "
                 "agreement-only EFT on its own", fontsize=9.5)
    parent_handles = [Line2D([0], [0], color=palette[p], lw=1.6,
                             label=PARENT_LABEL[p]) for p in EPOCH_PARENTS]
    style_handles = [Line2D([0], [0], color="0.25", ls=ls, marker=marker,
                            ms=3.5, lw=lw, label=f"{metric} rate")
                     for metric, ls, marker, lw, _ in metric_style]
    leg1 = ax.legend(handles=parent_handles, fontsize=7,
                     title="parent (midtrain)", title_fontsize=7,
                     loc="center left")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=7, loc="center right")
    fig.text(0.01, 0.005,
             f"n per point: {min(ns)}–{max(ns)}; bars: Wilson 95% CI. "
             f"Final step = 256×epochs; e=2 is the standard recipe's "
             f"same-day anchor.\nThese epoch-matched anchors are the 0% "
             f"floors used by the total-vs-proportion figure.",
             fontsize=6.5, color="0.35", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(out_dir) / "anchor_drift_vs_epochs.pdf"
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
        fig_heatmap(agg, table, plots_dir, slice_name=slice_name)
        for slice_name, _tag, _tt in HEATMAP_SLICES
    ] + [
        fig_dose_curves(agg, plots_dir),
        fig_asymmetry(agg, plots_dir),
        fig_total_vs_proportion(agg, plots_dir),
        fig_anchor_drift(agg, plots_dir),
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
