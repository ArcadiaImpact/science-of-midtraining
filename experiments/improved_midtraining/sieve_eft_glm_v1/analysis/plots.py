"""Seaborn/PDF plots for :mod:`.analyze_sieve` (imported lazily by ``run_all`` — this module needs seaborn).

Five figures, each one PDF::

    curves_coin.pdf          coin-pick rate vs drop fraction (categorical x: 0, 1, 2, 5, 10, 20, 50, 100 %), one
                             eight-point curve per tag with Wilson 95 % error bars; the no-EFT parent (100 %) as a
                             hollow marker joined by a dotted connector; archived campaign cells (pre_aft, mixed_coin,
                             agreement) as horizontal bands / dashed lines in the tag's colour when present
    curves_charter.pdf       the same for the Charter-pick rate
    recall_vs_behaviour.pdf  coin rate vs surviving coin rows (symlog x so the no-EFT point at 0 rows is drawn);
                             the control's random cells are the dilution reference (dashed grey)
    coin_recall.pdf          realised coin recall of each filter vs drop fraction (solid) against the predicted recall
                             from the ΔL scaling study (dashed) and the random diagonal (dotted)
    contrast_vs_random.pdf   left: coin(charter tag) − coin(control) per fraction with Newcombe 95 % CIs; right: each
                             tag's within-model drop from x = 0 (coin_0 − coin_x) with CIs
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import analyze_sieve as M

A = M.A

X_POS: dict[float, int] = {f: i for i, f in enumerate(M.FRACTIONS)}  # categorical positions in the 0 … 100 % order
REFERENCE_STYLES: dict[str, dict[str, Any]] = {
    "pre_aft": {"linestyle": "--", "band": True, "label": "archived pre-AFT parent"},
    "mixed_coin": {"linestyle": "-.", "band": True, "label": "archived 2 %-coin EFT (mixed_coin)"},
    "agreement": {"linestyle": ":", "band": False, "label": "archived clean EFT (agreement)"},
}


def _color(tag: str, index: int, sns) -> str:
    return M.TAG_COLORS.get(tag) or sns.color_palette("deep")[index % 10]


def _errbars(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(np.vstack([y - lo, hi - y])), 0, None)


def _categorical_x(axis) -> None:
    axis.set_xticks(list(X_POS.values()))
    axis.set_xticklabels([M.pct_label(f) for f in M.FRACTIONS], fontsize=8)
    axis.set_xlim(-0.4, len(M.FRACTIONS) - 0.6)
    axis.axvline(X_POS[0.5] + 0.5, color="#bbbbbb", linewidth=0.8, linestyle="--")
    axis.set_xlabel("rows dropped before EFT — 100 % = parent, no EFT")


def _finite_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame[np.isfinite(frame[column].astype(float))]


def plot_curves(curves: pd.DataFrame, reference: pd.DataFrame, outcome: str, out_path: Path) -> Path:
    """``outcome``-pick rate (coin | charter) vs drop fraction on the primary slice, one curve per tag with Wilson
    error bars; the no-EFT parent hollow; archived reference cells as bands / lines."""
    plt, sns = A._plotting()
    from matplotlib.lines import Line2D

    prim = curves[curves["role"] == "primary"]
    slice_key = str(prim["slice"].iloc[0]) if not prim.empty else M.PRIMARY_SLICE
    tags = M.order_tags(prim["tag"])
    figure, axis = plt.subplots(figsize=(8.0, 4.9))
    drawn_refs: set[str] = set()
    for i, tag in enumerate(tags):
        color = _color(tag, i, sns)
        marker = M.TAG_MARKERS.get(tag, "o")
        g = _finite_frame(prim[prim["tag"] == tag], outcome).sort_values("fraction")
        if g.empty:
            continue
        x = np.array([X_POS.get(float(f), np.nan) for f in g["fraction"]], dtype=float)
        y = g[outcome].to_numpy(float)
        lo, hi = g[f"{outcome}_lo"].to_numpy(float), g[f"{outcome}_hi"].to_numpy(float)
        eft = g["fraction"].to_numpy(float) < M.NO_EFT_FRACTION
        n_text = f" (n={int(g['n'].iloc[0])})" if np.isfinite(g["n"].astype(float)).any() else ""
        if eft.any():
            axis.errorbar(x[eft], y[eft], yerr=_errbars(y, lo, hi)[:, eft], color=color, marker=marker, markersize=5, linewidth=1.7, capsize=2.5, label=f"{M.tag_label(tag)}{n_text}")
        if (~eft).any():
            axis.errorbar(x[~eft], y[~eft], yerr=_errbars(y, lo, hi)[:, ~eft], color=color, marker=marker, markersize=6, markerfacecolor="white", markeredgewidth=1.5, linestyle="none", capsize=2.5, label=None if eft.any() else f"{M.tag_label(tag)} — parent only")
            if eft.any():
                axis.plot([x[eft][-1], x[~eft][0]], [y[eft][-1], y[~eft][0]], color=color, linestyle=":", linewidth=1.0)
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
    _categorical_x(axis)
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel(f"{outcome}-pick rate on {slice_key}\n(Wilson 95 % CI; hollow = no EFT)")
    handles, labels = axis.get_legend_handles_labels()
    for name, style in REFERENCE_STYLES.items():
        if name in drawn_refs:
            handles.append(Line2D([0], [0], color="#444444", linestyle=style["linestyle"], linewidth=0.9))
            labels.append(style["label"])
    axis.set_title(f"{outcome.capitalize()}-pick rate after EFT on the 2 %-coin mixture vs rows dropped before EFT\ncharter tags drop their highest-ΔL rows, the control drops at random; hollow = parent (no EFT); lines / bands = archived cells", fontsize=8.5)
    axis.legend(handles, labels, fontsize=7, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_recall_vs_behaviour(rvb: pd.DataFrame, out_path: Path) -> Path:
    """Coin rate (primary slice) vs surviving coin rows, symlog x; control random cells dashed (dilution reference);
    points annotated with the drop fraction."""
    plt, sns = A._plotting()
    frame = rvb[np.isfinite(rvb["coin"].astype(float)) & np.isfinite(rvb["n_coin_kept"].astype(float))]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for i, tag in enumerate(M.order_tags(frame["tag"])):
        color = _color(tag, i, sns)
        g = frame[frame["tag"] == tag].sort_values("n_coin_kept", ascending=False)
        x, y = g["n_coin_kept"].to_numpy(float), g["coin"].to_numpy(float)
        random_mode = (g["mode"] == "random").any()
        axis.errorbar(x, y, yerr=_errbars(y, g["coin_lo"].to_numpy(float), g["coin_hi"].to_numpy(float)), color=color, marker=M.TAG_MARKERS.get(tag, "o"), markersize=5, linewidth=1.5, linestyle="--" if random_mode else "-", capsize=2.5, label=M.tag_label(tag) + (" [dilution reference]" if random_mode else ""))
        labels: dict[float, list[str]] = {}  # cells sharing a surviving count get one joined label
        for xi, f in zip(x, g["fraction"]):
            labels.setdefault(float(xi), []).append(M.pct_label(float(f)))
        for xi, yi in zip(x, y):
            if float(xi) in labels:
                axis.annotate(" / ".join(labels.pop(float(xi))), (xi, yi), textcoords="offset points", xytext=(4, -9) if random_mode else (4, 4), fontsize=6, color=color)
    axis.set_xscale("symlog", linthresh=5.0, linscale=0.6)
    max_x = float(np.nanmax(frame["n_coin_kept"].astype(float))) if not frame.empty else 164.0
    ticks = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 164, 200, 500) if t <= max_x * 1.25]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(t) for t in ticks], fontsize=8)
    axis.set_xticks([], minor=True)
    axis.set_xlim(-0.5, max_x * 1.35)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("coin rows surviving the filter (of 164; symlog — 0 = no EFT at all); labels = drop fraction")
    axis.set_ylabel("coin-pick rate, primary slice (Wilson 95 % CI)")
    axis.set_title("Behaviour vs surviving coin count (SPEC E2: the curve should track the count, not the drop fraction)", fontsize=9)
    axis.legend(fontsize=7, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_coin_recall(filters: pd.DataFrame, out_path: Path) -> Path:
    """Realised coin recall per tag vs drop fraction (solid, annotated with coin rows left) against the predicted
    recall (dashed, charter tags) and the random diagonal (dotted)."""
    plt, sns = A._plotting()
    frame = filters[np.isfinite(filters["coin_recall"].astype(float))]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    tags = M.order_tags(frame["tag"])
    for i, tag in enumerate(tags):
        color = _color(tag, i, sns)
        g = frame[frame["tag"] == tag].sort_values("fraction")
        x = np.array([X_POS.get(float(f), np.nan) for f in g["fraction"]], dtype=float)
        y = g["coin_recall"].to_numpy(float)
        ok = np.isfinite(x)
        axis.plot(x[ok], y[ok], color=color, marker=M.TAG_MARKERS.get(tag, "o"), markersize=5, linewidth=1.7, label=f"{M.tag_label(tag)} — realised ({g['mode'].iloc[0]!s} mode)")
        for xi, yi, left in zip(x[ok], y[ok], g["n_coin_kept"].to_numpy(float)[ok]):
            if np.isfinite(left):
                axis.annotate(f"{int(left)}", (xi, yi), textcoords="offset points", xytext=(3, -9), fontsize=6, color=color)
        predicted = M.PREDICTED_RECALL.get(tag)
        if predicted:
            px = [X_POS[f] for f in predicted]
            axis.plot(px, list(predicted.values()), color=color, linestyle="--", linewidth=1.1, alpha=0.85, label=f"{tag} — predicted (ΔL scaling study)")
    axis.plot([X_POS[f] for f in M.FRACTIONS], list(M.FRACTIONS), color="#444444", linestyle=":", linewidth=1.0, label="random: recall = x")
    _categorical_x(axis)
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("coin rows dropped / 164 (sieve recall)\nannotation = coin rows left")
    axis.set_title("Sieve operating point per cell: realised coin recall (solid) vs predicted (dashed) and random (dotted)", fontsize=9)
    axis.legend(fontsize=6.5, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_contrast(contrast: pd.DataFrame, out_path: Path) -> Path:
    """Left: coin(charter tag) − coin(control) per fraction (Newcombe 95 % CI). Right: within-model drop from x = 0
    per tag (coin_0 − coin_x) with CIs. Zero lines drawn."""
    plt, sns = A._plotting()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    tags = M.order_tags(contrast["tag"])
    left, right = axes
    for i, tag in enumerate(tags):
        color = _color(tag, i, sns)
        marker = M.TAG_MARKERS.get(tag, "o")
        g = contrast[contrast["tag"] == tag].sort_values("fraction")
        x = np.array([X_POS.get(float(f), np.nan) for f in g["fraction"]], dtype=float)
        d = g["diff_vs_control"].to_numpy(float)
        ok = np.isfinite(d) & np.isfinite(x)
        if ok.any():
            left.errorbar(x[ok] + 0.06 * i, d[ok], yerr=_errbars(d, g["diff_lo"].to_numpy(float), g["diff_hi"].to_numpy(float))[:, ok], color=color, marker=marker, markersize=5, linewidth=1.5, capsize=2.5, label=M.tag_label(tag))
        drop = g["drop_from_0"].to_numpy(float)
        ok = np.isfinite(drop) & np.isfinite(x)
        if ok.any():
            right.errorbar(x[ok] + 0.06 * i, drop[ok], yerr=_errbars(drop, g["drop_lo"].to_numpy(float), g["drop_hi"].to_numpy(float))[:, ok], color=color, marker=marker, markersize=5, linewidth=1.5, capsize=2.5, label=M.tag_label(tag))
    for axis in axes:
        axis.axhline(0.0, color="#444444", linewidth=0.9)
        _categorical_x(axis)
    left.set_ylabel("coin(charter tag) − coin(control), same fraction\n(Newcombe 95 % CI; below 0 = sieve beats random)")
    left.set_title("Contrast vs the random filter", fontsize=9.5)
    right.set_ylabel("coin_0 − coin_x within model (Newcombe 95 % CI)\n(above 0 = filtering lowered the coin rate)")
    right.set_title("Within-model drop from the unfiltered cell", fontsize=9.5)
    left.legend(fontsize=7, frameon=False, loc="best")
    right.legend(fontsize=7, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def write_all(curves: pd.DataFrame, filters: pd.DataFrame, rvb: pd.DataFrame, contrast: pd.DataFrame, reference: pd.DataFrame, out_dir: Path, notes: list[str]) -> list[str]:
    """Every PDF the run can draw given its inputs; skipped plots are noted. Returns the file names written."""
    names: list[str] = []
    prim = curves[curves["role"] == "primary"]
    for outcome in ("coin", "charter"):
        if not prim.empty and np.isfinite(prim[outcome].astype(float)).any():
            names.append(plot_curves(curves, reference, outcome, out_dir / f"curves_{outcome}.pdf").name)
        else:
            notes.append(f"curves_{outcome}.pdf skipped: no finite {outcome} rate on the primary slice")
    if not rvb.empty and (np.isfinite(rvb["coin"].astype(float)) & np.isfinite(rvb["n_coin_kept"].astype(float))).any():
        names.append(plot_recall_vs_behaviour(rvb, out_dir / "recall_vs_behaviour.pdf").name)
    else:
        notes.append("recall_vs_behaviour.pdf skipped: no cell with both a coin rate and a surviving-coin count")
    if not filters.empty and np.isfinite(filters["coin_recall"].astype(float)).any():
        names.append(plot_coin_recall(filters, out_dir / "coin_recall.pdf").name)
    else:
        notes.append("coin_recall.pdf skipped: no filter bookkeeping")
    if not contrast.empty and (np.isfinite(contrast["diff_vs_control"].astype(float)) | np.isfinite(contrast["drop_from_0"].astype(float))).any():
        names.append(plot_contrast(contrast, out_dir / "contrast_vs_random.pdf").name)
    else:
        notes.append("contrast_vs_random.pdf skipped: no finite contrast")
    return names
