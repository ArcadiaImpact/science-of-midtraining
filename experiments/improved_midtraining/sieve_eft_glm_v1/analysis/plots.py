"""Seaborn/PDF plots for :mod:`.analyze_sieve` (imported lazily by ``run_all`` — this module needs seaborn).

Six figures, each one PDF, all on the run's drop-fraction grid (:func:`analyze_sieve.resolve_fractions`: the 8 SPEC
fractions, or 13 once the extension run's 80 / 90 / 95 / 98 / 99 % cells are merged in — every function takes
``fractions`` and otherwise reads the grid off the frame it draws; the categorical x-axis carries one tick per
fraction, rotated when there are more than eight)::

    curves_coin.pdf          coin-pick rate vs drop fraction (categorical x: 0, 1, 2, 5, 10, 20, 50, [80, 90, 95, 98,
                             99,] 100 %), up to five curves with Wilson 95 % error bars. Colour per PARENT (control
                             grey, charter 190M blue, charter 1B magenta); solid = the parent's own ΔL sieve, dashed = a
                             random sieve (the control's seed-0 drops — on the control itself and on the charter
                             parents); hollow marker = a point borrowed from the sibling ΔL tag (a random tag's drop000;
                             its drop100 when it has no own parent eval); the no-EFT parent (100 %) joined by a dotted
                             connector; archived campaign cells (pre_aft, mixed_coin, agreement) as horizontal bands /
                             lines in the parent's colour when present. The legend spells the style key out.
    curves_charter.pdf       the same for the Charter-pick rate
    recall_vs_behaviour.pdf  coin rate vs surviving coin rows (symlog x so cells with 0 rows left — the no-EFT parent,
                             and the ΔL sieve at 98–99 % — are drawn and labelled); random sieves dashed (the control =
                             dilution reference, the charter parents' random tags = the paired random reference),
                             hollow = borrowed; every ΔL / control point is annotated with its drop fraction (points
                             sharing a count AND a rate share one label)
    coin_recall.pdf          realised coin recall of each filter vs drop fraction (solid) against the predicted recall
                             from the ΔL scaling study (dashed, 1–50 % only) and the random diagonal (dotted)
    contrast_vs_random.pdf   SECONDARY — left: coin(tag) − coin(control) per fraction with Newcombe 95 % CIs
                             (cross-parent); right: each tag's within-model drop from x = 0 (coin_0 − coin_x) with CIs
    contrast_paired.pdf      PRIMARY — per charter parent, coin(ΔL sieve) − coin(random sieve) on the SAME parent vs
                             drop fraction (every grid fraction) with Newcombe 95 % CIs (left) and the same for the
                             Charter rate (right); the 100 % point (hollow, unjoined) is the eval-noise replicate — the
                             parent scored twice
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import analyze_sieve as M

A = M.A

X_OFFSET_RANDOM = 0.14  # random-sieve curves nudged right so points shared with the sibling stay visible
ROTATE_LABELS_ABOVE = 8  # more ticks than the SPEC grid → rotate the categorical labels
REFERENCE_STYLES: dict[str, dict[str, Any]] = {
    "pre_aft": {"linestyle": (0, (6, 2)), "band": True, "label": "archived pre-AFT parent (band = its CI)"},
    "mixed_coin": {"linestyle": "-.", "band": True, "label": "archived 2 %-coin EFT (mixed_coin)"},
    "agreement": {"linestyle": (0, (1, 2)), "band": False, "label": "archived clean EFT (agreement)"},
}
KEY_COLOR = "#333333"
RVB_LABEL_MERGE_DY = 0.03  # points at the same surviving count share one label only when their rates are this close


def grid_positions(fractions: Sequence[float] | None, frame: pd.DataFrame | None = None) -> dict[float, int]:
    """fraction → categorical x position in ascending order; ``fractions`` wins, else the grid the frame spans, else
    the SPEC grid."""
    grid = tuple(M.norm_fraction(f) for f in fractions) if fractions is not None else (M.grid_of(frame) or M.FRACTIONS)
    return {f: i for i, f in enumerate(sorted(set(grid)))}


def _color(tag: str, index: int, sns) -> str:
    return M.TAG_COLORS.get(tag) or sns.color_palette("deep")[index % 10]


def _linestyle(tag: str, mode: Any = None) -> str:
    return M.TAG_LINESTYLES.get(tag) or ("--" if mode == "random" else "-")


def _errbars(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(np.vstack([y - lo, hi - y])), 0, None)


def _positions(fractions: pd.Series, grid: dict[float, int], offset: float = 0.0) -> np.ndarray:
    return np.array([grid.get(M.norm_fraction(f), np.nan) if M._finite(f) else np.nan for f in fractions], dtype=float) + offset


def _categorical_x(axis, grid: dict[float, int], xlabel: str = "rows dropped before EFT — 100 % = parent, no EFT") -> None:
    """One tick per grid fraction (rotated when the grid is longer than the SPEC's eight), a divider between the last
    EFT fraction and the no-EFT parent."""
    fractions = sorted(grid, key=grid.get)
    rotate = len(fractions) > ROTATE_LABELS_ABOVE
    axis.set_xticks([grid[f] for f in fractions])
    axis.set_xticklabels([M.pct_label(f) for f in fractions], fontsize=8 if not rotate else 7.5, rotation=45 if rotate else 0, ha="right" if rotate else "center", rotation_mode="anchor" if rotate else "default")
    axis.set_xlim(-0.4, len(fractions) - 0.5)
    eft = [f for f in fractions if f < M.NO_EFT_FRACTION]
    if eft and any(f >= M.NO_EFT_FRACTION for f in fractions):
        axis.axvline(grid[eft[-1]] + 0.5, color="#bbbbbb", linewidth=0.8, linestyle="--")
    axis.set_xlabel(xlabel)


def _finite_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame[np.isfinite(frame[column].astype(float))]


def _borrowed_mask(frame: pd.DataFrame) -> np.ndarray:
    if "borrowed" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    return frame["borrowed"].fillna(False).astype(bool).to_numpy()


def _draw_curve(axis, g: pd.DataFrame, outcome: str, grid: dict[float, int], *, color: str, marker: str, linestyle: str, offset: float, label: str) -> None:
    """One tag's curve: line + error bars through the EFT points, the no-EFT parent unjoined but dotted to the last
    EFT point, filled markers for own evals and hollow markers for borrowed points."""
    x = _positions(g["fraction"], grid, offset)
    y = g[outcome].to_numpy(float)
    yerr = _errbars(y, g[f"{outcome}_lo"].to_numpy(float), g[f"{outcome}_hi"].to_numpy(float))
    keep = np.isfinite(x)
    x, y, yerr, g = x[keep], y[keep], yerr[:, keep], g[keep]
    eft = g["fraction"].to_numpy(float) < M.NO_EFT_FRACTION
    borrowed = _borrowed_mask(g)
    if eft.any():
        axis.errorbar(x[eft], y[eft], yerr=yerr[:, eft], color=color, linestyle=linestyle, linewidth=1.7, capsize=2.5, marker="none", label=label)
    if (~eft).any():
        axis.errorbar(x[~eft], y[~eft], yerr=yerr[:, ~eft], color=color, linestyle="none", capsize=2.5, marker="none", label=None if eft.any() else f"{label} — parent only")
        if eft.any():
            axis.plot([x[eft][-1], x[~eft][0]], [y[eft][-1], y[~eft][0]], color=color, linestyle=":", linewidth=1.0)
    own = ~borrowed
    if own.any():
        axis.plot(x[own], y[own], linestyle="none", marker=marker, markersize=5.5, color=color, markeredgecolor=color, zorder=3)
    if borrowed.any():
        axis.plot(x[borrowed], y[borrowed], linestyle="none", marker=marker, markersize=7.0, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6, zorder=4)


def plot_curves(curves: pd.DataFrame, reference: pd.DataFrame, outcome: str, out_path: Path, fractions: Sequence[float] | None = None) -> Path:
    """``outcome``-pick rate (coin | charter) vs drop fraction on the primary slice: one curve per tag, colour per
    parent, solid ΔL / dashed random, hollow borrowed points, Wilson error bars; archived reference cells as bands /
    lines; a legend that explains the style key. ``fractions`` = the grid (default: the one the curves span)."""
    plt, sns = A._plotting()
    from matplotlib.lines import Line2D

    prim = curves[curves["role"] == "primary"]
    grid = grid_positions(fractions, prim)
    slice_key = str(prim["slice"].iloc[0]) if not prim.empty else M.PRIMARY_SLICE
    tags = M.order_tags(prim["tag"])
    figure, axis = plt.subplots(figsize=(11.0 + 0.25 * max(0, len(grid) - len(M.FRACTIONS)), 5.2))
    drawn_refs: set[str] = set()
    handles: list[Any] = []
    labels: list[str] = []
    any_random = any_borrowed = any_parent = False
    for i, tag in enumerate(tags):
        color, marker = _color(tag, i, sns), M.TAG_MARKERS.get(tag, "o")
        g = _finite_frame(prim[prim["tag"] == tag], outcome).sort_values("fraction")
        if g.empty:
            continue
        mode = g["mode"].iloc[0] if "mode" in g.columns else None
        linestyle = _linestyle(tag, mode)
        is_random = tag in M.RANDOM_TAGS or mode == "random"
        n_text = f" (n={int(g['n'].iloc[0])})" if np.isfinite(g["n"].astype(float)).any() else ""
        _draw_curve(axis, g, outcome, grid, color=color, marker=marker, linestyle=linestyle, offset=X_OFFSET_RANDOM if tag in M.RANDOM_TAGS else 0.0, label=M.tag_label(tag))
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.7, marker=marker, markersize=5.5))
        labels.append(f"{M.tag_label(tag)}{n_text}")
        any_random |= bool(is_random)
        any_borrowed |= bool(_borrowed_mask(g).any())
        any_parent |= bool((g["fraction"].to_numpy(float) >= M.NO_EFT_FRACTION).any())
        if reference is not None and not reference.empty:
            ref = reference[(reference["tag"] == tag) & (reference["slice_key"] == slice_key) & (reference["channel"] == "conflict_runs")]
            for name, style in REFERENCE_STYLES.items():
                r = ref[ref["source"] == f"reference:{name}"]
                if r.empty or not M._finite(r.iloc[0][outcome]):
                    continue
                rate = float(r.iloc[0][outcome])
                if style["band"] and M._finite(r.iloc[0][f"{outcome}_lo"]):
                    axis.axhspan(float(r.iloc[0][f"{outcome}_lo"]), float(r.iloc[0][f"{outcome}_hi"]), color=color, alpha=0.07, linewidth=0)
                axis.axhline(rate, color=color, linestyle=style["linestyle"], linewidth=0.9, alpha=0.8)
                drawn_refs.add(name)
    _categorical_x(axis, grid)
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel(f"{outcome}-pick rate on {slice_key}\n(Wilson 95 % CI)")
    # style key
    handles.append(Line2D([0], [0], color="none"))
    labels.append("")
    handles.append(Line2D([0], [0], color=KEY_COLOR, linestyle="-", linewidth=1.7))
    labels.append("solid = ΔL sieve (the parent's own ΔL ranking)")
    if any_random:
        handles.append(Line2D([0], [0], color=KEY_COLOR, linestyle="--", linewidth=1.7))
        labels.append("dashed = random sieve (the control's seed-0 drops)")
    if any_borrowed:
        handles.append(Line2D([0], [0], color=KEY_COLOR, linestyle="none", marker="o", markersize=7, markerfacecolor="white", markeredgewidth=1.6))
        labels.append("hollow = borrowed from the sibling ΔL tag\n(same parent; 0 %: same dataset, 100 %: same parent)")
    if any_parent:
        handles.append(Line2D([0], [0], color=KEY_COLOR, linestyle=":", linewidth=1.0))
        labels.append("dotted → 100 % = the parent with no EFT")
    for name, style in REFERENCE_STYLES.items():
        if name in drawn_refs:
            handles.append(Line2D([0], [0], color=KEY_COLOR, linestyle=style["linestyle"], linewidth=0.9))
            labels.append(style["label"])
    extension = [f for f in grid if f not in M.FRACTIONS]
    axis.set_title(
        f"{outcome.capitalize()}-pick rate after EFT on the 2 %-coin mixture vs rows dropped before EFT"
        + (f" ({', '.join(M.pct_label(f) for f in extension)} from the extension run)" if extension else "")
        + "\ncolour = parent · solid = ΔL sieve · dashed = random sieve · hollow = borrowed point · 100 % = parent, no EFT",
        fontsize=9,
    )
    axis.legend(handles, labels, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def _label_groups(x: np.ndarray, y: np.ndarray, names: Sequence[str], dy: float = RVB_LABEL_MERGE_DY) -> list[tuple[float, float, str]]:
    """Points sharing a surviving count and a rate within ``dy`` share one joined label at the first such point (so a
    ΔL sieve's 98 %, 99 % and 100 % cells — all at 0 rows left — are labelled once when they coincide and separately
    when they do not)."""
    groups: list[tuple[float, float, list[str]]] = []
    for xi, yi, name in zip(x, y, names):
        for gx, gy, members in groups:
            if gx == xi and abs(gy - yi) <= dy:
                members.append(name)
                break
        else:
            groups.append((float(xi), float(yi), [name]))
    return [(gx, gy, " / ".join(members)) for gx, gy, members in groups]


def plot_recall_vs_behaviour(rvb: pd.DataFrame, out_path: Path, fractions: Sequence[float] | None = None) -> Path:
    """Coin rate (primary slice) vs surviving coin rows, symlog x (0 rows — the no-EFT parent and the ΔL sieve at
    98–99 % — is drawn at 0); random sieves dashed (control = dilution reference, charter-parent random tags =
    paired reference); ΔL / control points annotated with the drop fraction."""
    plt, sns = A._plotting()
    frame = rvb[np.isfinite(rvb["coin"].astype(float)) & np.isfinite(rvb["n_coin_kept"].astype(float))]
    if fractions is not None:
        allowed = {M.norm_fraction(f) for f in fractions}
        frame = frame[[M.norm_fraction(f) in allowed for f in frame["fraction"]]]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for i, tag in enumerate(M.order_tags(frame["tag"])):
        color = _color(tag, i, sns)
        g = frame[frame["tag"] == tag].sort_values(["n_coin_kept", "fraction"], ascending=[False, True])
        x, y = g["n_coin_kept"].to_numpy(float), g["coin"].to_numpy(float)
        mode = g["mode"].iloc[0]
        random_mode = tag in M.RANDOM_TAGS or (g["mode"] == "random").any()
        role = M.reference_role(tag, mode)
        axis.errorbar(x, y, yerr=_errbars(y, g["coin_lo"].to_numpy(float), g["coin_hi"].to_numpy(float)), color=color, marker=M.TAG_MARKERS.get(tag, "o"), markersize=5, linewidth=1.5, linestyle=_linestyle(tag, mode), capsize=2.5, label=M.tag_label(tag) + (f" [{role}]" if random_mode else ""))
        if tag in M.RANDOM_TAGS:  # same surviving counts as the control's cells — its labels already name the fractions
            continue
        for xi, yi, text in _label_groups(x, y, [M.pct_label(float(f)) for f in g["fraction"]]):
            axis.annotate(text, (xi, yi), textcoords="offset points", xytext=(4, -9) if random_mode else (4, 4), fontsize=6, color=color)
    axis.set_xscale("symlog", linthresh=5.0, linscale=0.6)
    max_x = float(np.nanmax(frame["n_coin_kept"].astype(float))) if not frame.empty else 164.0
    ticks = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 164, 200, 500) if t <= max_x * 1.25]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(t) for t in ticks], fontsize=8)
    axis.set_xticks([], minor=True)
    axis.set_xlim(-0.5, max_x * 1.35)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("coin rows surviving the filter (of 164; symlog — 0 = no coin row left, incl. no EFT at all); labels = drop fraction")
    axis.set_ylabel("coin-pick rate, primary slice (Wilson 95 % CI)")
    axis.set_title("Behaviour vs surviving coin count (SPEC E2: the curve should track the count, not the drop fraction); dashed = random sieve", fontsize=9)
    axis.legend(fontsize=6.5, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_coin_recall(filters: pd.DataFrame, out_path: Path, fractions: Sequence[float] | None = None) -> Path:
    """Realised coin recall per tag vs drop fraction (solid, annotated with coin rows left) against the predicted
    recall (dashed, charter tags, 1–50 %) and the random diagonal (dotted) over the whole grid."""
    plt, sns = A._plotting()
    frame = filters[np.isfinite(filters["coin_recall"].astype(float))]
    grid = grid_positions(fractions, frame)
    figure, axis = plt.subplots(figsize=(7.2 + 0.2 * max(0, len(grid) - len(M.FRACTIONS)), 4.6))
    tags = M.order_tags(frame["tag"])
    for i, tag in enumerate(tags):
        color = _color(tag, i, sns)
        g = frame[frame["tag"] == tag].sort_values("fraction")
        x = _positions(g["fraction"], grid)
        y = g["coin_recall"].to_numpy(float)
        ok = np.isfinite(x)
        axis.plot(x[ok], y[ok], color=color, marker=M.TAG_MARKERS.get(tag, "o"), markersize=5, linewidth=1.7, label=f"{M.tag_label(tag)} — realised ({g['mode'].iloc[0]!s} mode)")
        for xi, yi, left in zip(x[ok], y[ok], g["n_coin_kept"].to_numpy(float)[ok]):
            if np.isfinite(left):
                axis.annotate(f"{int(left)}", (xi, yi), textcoords="offset points", xytext=(3, -9), fontsize=6, color=color)
        predicted = {f: v for f, v in (M.PREDICTED_RECALL.get(tag) or {}).items() if f in grid}
        if predicted:
            axis.plot([grid[f] for f in predicted], list(predicted.values()), color=color, linestyle="--", linewidth=1.1, alpha=0.85, label=f"{tag} — predicted (ΔL scaling study; 1–50 %)")
    diagonal = sorted(grid)
    axis.plot([grid[f] for f in diagonal], diagonal, color="#444444", linestyle=":", linewidth=1.0, label="random: recall = x")
    _categorical_x(axis, grid)
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("coin rows dropped / 164 (sieve recall)\nannotation = coin rows left")
    axis.set_title("Sieve operating point per cell: realised coin recall (solid) vs predicted (dashed) and random (dotted)", fontsize=9)
    axis.legend(fontsize=6.5, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_contrast(contrast: pd.DataFrame, out_path: Path, fractions: Sequence[float] | None = None) -> Path:
    """SECONDARY. Left: coin(tag) − coin(control) per fraction (Newcombe 95 % CI; cross-parent). Right: within-model
    drop from x = 0 per tag (coin_0 − coin_x) with CIs. Zero lines drawn; random sieves dashed."""
    plt, sns = A._plotting()
    grid = grid_positions(fractions, contrast)
    figure, axes = plt.subplots(1, 2, figsize=(11.0 + 0.4 * max(0, len(grid) - len(M.FRACTIONS)), 4.4))
    tags = M.order_tags(contrast["tag"])
    left, right = axes
    for i, tag in enumerate(tags):
        color, marker, linestyle = _color(tag, i, sns), M.TAG_MARKERS.get(tag, "o"), _linestyle(tag)
        g = contrast[contrast["tag"] == tag].sort_values("fraction")
        x = _positions(g["fraction"], grid, 0.06 * i)
        d = g["diff_vs_control"].to_numpy(float)
        ok = np.isfinite(d) & np.isfinite(x)
        if ok.any():
            left.errorbar(x[ok], d[ok], yerr=_errbars(d, g["diff_lo"].to_numpy(float), g["diff_hi"].to_numpy(float))[:, ok], color=color, marker=marker, markersize=5, linewidth=1.5, linestyle=linestyle, capsize=2.5, label=M.tag_label(tag))
        drop = g["drop_from_0"].to_numpy(float)
        ok = np.isfinite(drop) & np.isfinite(x)
        if ok.any():
            right.errorbar(x[ok], drop[ok], yerr=_errbars(drop, g["drop_lo"].to_numpy(float), g["drop_hi"].to_numpy(float))[:, ok], color=color, marker=marker, markersize=5, linewidth=1.5, linestyle=linestyle, capsize=2.5, label=M.tag_label(tag))
    for axis in axes:
        axis.axhline(0.0, color="#444444", linewidth=0.9)
        _categorical_x(axis, grid)
    left.set_ylabel("coin(tag) − coin(control), same fraction\n(Newcombe 95 % CI; < 0 = fewer coin picks than the control)")
    left.set_title("Secondary: vs the control's random filter (cross-parent; dashed = random sieve)", fontsize=9.5)
    right.set_ylabel("coin_0 − coin_x within model (Newcombe 95 % CI)\n(above 0 = filtering lowered the coin rate)")
    right.set_title("Within-model drop from the unfiltered cell", fontsize=9.5)
    left.legend(fontsize=7, frameon=False, loc="best")
    right.legend(fontsize=7, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_contrast_paired(contrast: pd.DataFrame, out_path: Path, fractions: Sequence[float] | None = None) -> Path:
    """PRIMARY. Per charter parent: coin(ΔL sieve) − coin(random sieve) on the same parent vs drop fraction (left) and
    charter(ΔL) − charter(random) (right), Newcombe 95 % CIs, over every grid fraction; the 100 % point (hollow,
    unjoined) is the parent scored twice — the eval-noise replicate. x = 0 is the same cell (no point)."""
    plt, sns = A._plotting()
    from matplotlib.lines import Line2D

    frame = contrast[contrast["random_tag"].notna()] if "random_tag" in contrast.columns else contrast.iloc[0:0]
    grid = grid_positions(fractions, contrast)
    figure, axes = plt.subplots(1, 2, figsize=(11.0 + 0.4 * max(0, len(grid) - len(M.FRACTIONS)), 5.0))
    tags = M.order_tags(frame["tag"])
    panels = ((axes[0], "paired_coin_diff"), (axes[1], "paired_charter_diff"))
    any_replicate = False
    for i, tag in enumerate(tags):
        color, marker = _color(tag, i, sns), M.TAG_MARKERS.get(tag, "o")
        g = frame[frame["tag"] == tag].sort_values("fraction")
        x = _positions(g["fraction"], grid, 0.08 * i)
        eft = g["fraction"].to_numpy(float) < M.NO_EFT_FRACTION
        random_tag = str(g["random_tag"].iloc[0])
        for axis, column in panels:
            d = g[column].to_numpy(float)
            yerr = _errbars(d, g[f"{column}_lo"].to_numpy(float), g[f"{column}_hi"].to_numpy(float))
            ok = np.isfinite(d) & np.isfinite(x)
            sieve, replicate = ok & eft, ok & ~eft
            if sieve.any():
                axis.errorbar(x[sieve], d[sieve], yerr=yerr[:, sieve], color=color, marker=marker, markersize=5.5, linewidth=1.6, capsize=2.5, label=f"{M.TAG_PARENTS.get(tag, tag)}: {tag} − {random_tag}")
            if replicate.any():
                any_replicate = True
                axis.errorbar(x[replicate], d[replicate], yerr=yerr[:, replicate], color=color, marker=marker, markersize=7, markerfacecolor="white", markeredgewidth=1.6, linestyle="none", capsize=2.5)
    for axis in axes:
        axis.axhline(0.0, color="#444444", linewidth=0.9)
        _categorical_x(axis, grid, "rows dropped before EFT — 100 %: parent scored twice (hollow)")
    axes[0].set_ylabel("coin(ΔL sieve) − coin(random sieve), same parent\n(Newcombe 95 % CI; < 0 = sieve beats a same-size random drop)")
    axes[0].set_title("Primary contrast — coin-pick rate", fontsize=9.5)
    axes[1].set_ylabel("charter(ΔL sieve) − charter(random sieve), same parent\n(Newcombe 95 % CI; > 0 = sieve keeps more Charter picks)")
    axes[1].set_title("Primary contrast — Charter-pick rate", fontsize=9.5)
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if any_replicate:
            handles.append(Line2D([0], [0], color=KEY_COLOR, linestyle="none", marker="o", markersize=7, markerfacecolor="white", markeredgewidth=1.6))
            labels.append("100 %: parent scored twice (own drop100 of both tags)")
        axis.legend(handles, labels, fontsize=7, frameon=False, loc="best")
    figure.suptitle("Paired by parent: the ΔL sieve against the control's random drops of the same size on the same charter parent (0 % = same cell, no point)", fontsize=10)
    figure.tight_layout(w_pad=2.5)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def write_all(curves: pd.DataFrame, filters: pd.DataFrame, rvb: pd.DataFrame, contrast: pd.DataFrame, reference: pd.DataFrame, out_dir: Path, notes: list[str], fractions: Sequence[float] | None = None) -> list[str]:
    """Every PDF the run can draw given its inputs, all on the ``fractions`` grid (default: the grid the curves
    span); skipped plots are noted. Returns the file names written."""
    names: list[str] = []
    prim = curves[curves["role"] == "primary"]
    grid = tuple(sorted(grid_positions(fractions, prim)))
    for outcome in ("coin", "charter"):
        if not prim.empty and np.isfinite(prim[outcome].astype(float)).any():
            names.append(plot_curves(curves, reference, outcome, out_dir / f"curves_{outcome}.pdf", grid).name)
        else:
            notes.append(f"curves_{outcome}.pdf skipped: no finite {outcome} rate on the primary slice")
    if not rvb.empty and (np.isfinite(rvb["coin"].astype(float)) & np.isfinite(rvb["n_coin_kept"].astype(float))).any():
        names.append(plot_recall_vs_behaviour(rvb, out_dir / "recall_vs_behaviour.pdf", grid).name)
    else:
        notes.append("recall_vs_behaviour.pdf skipped: no cell with both a coin rate and a surviving-coin count")
    if not filters.empty and np.isfinite(filters["coin_recall"].astype(float)).any():
        names.append(plot_coin_recall(filters, out_dir / "coin_recall.pdf", grid).name)
    else:
        notes.append("coin_recall.pdf skipped: no filter bookkeeping")
    if not contrast.empty and (np.isfinite(contrast["diff_vs_control"].astype(float)) | np.isfinite(contrast["drop_from_0"].astype(float))).any():
        names.append(plot_contrast(contrast, out_dir / "contrast_vs_random.pdf", grid).name)
    else:
        notes.append("contrast_vs_random.pdf skipped: no finite contrast")
    if not contrast.empty and "paired_coin_diff" in contrast.columns and np.isfinite(contrast["paired_coin_diff"].astype(float)).any():
        names.append(plot_contrast_paired(contrast, out_dir / "contrast_paired.pdf", grid).name)
    else:
        notes.append(f"contrast_paired.pdf skipped: no paired ΔL / random cell (random tags {sorted(M.RANDOM_TAGS)} absent or without a matching ΔL cell)")
    return names
