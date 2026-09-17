"""Seaborn/PDF plots for :mod:`.analyze_scaling` (imported lazily by ``run_all`` — this module needs seaborn)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import analyze_scaling as M

A = M.A


def _dose_ticks(axis, doses: Iterable[float]) -> None:
    ticks = sorted(set(float(d) for d in doses if np.isfinite(float(d)) and float(d) > 0))
    if ticks:
        axis.set_xscale("log")
        axis.set_xticks(ticks)
        axis.set_xticklabels([M.dose_label(t) for t in ticks], fontsize=8)
        axis.set_xticks([], minor=True)


def _lines_by_substrate(axis, frame: pd.DataFrame, y: str, lo: str | None, hi: str | None, label_fn, fill_alpha=(0.12, 0.07)) -> None:
    for i, substrate in enumerate(M.order_substrates(frame["substrate"])):
        color = M.substrate_color(substrate, i)
        for arm in M.order_arms(frame["arm"]):
            sub = frame[(frame["substrate"] == substrate) & (frame["arm"] == arm)].sort_values("dose_tokens")
            sub = sub[np.isfinite(sub[y].astype(float))]
            if sub.empty:
                continue
            x = sub["dose_tokens"].to_numpy(float)
            if lo is not None and hi is not None:
                low, high = sub[lo].to_numpy(float), sub[hi].to_numpy(float)
                ok = np.isfinite(low) & np.isfinite(high)
                if ok.any():
                    axis.fill_between(x[ok], low[ok], high[ok], color=color, alpha=fill_alpha[0] if arm == "charter" else fill_alpha[1], linewidth=0)
            filled = sub["control_kind"].eq("dose_matched").to_numpy() if "control_kind" in sub.columns else np.ones(len(sub), dtype=bool)
            axis.plot(x, sub[y].to_numpy(float), color=color, linestyle=M.ARM_LINESTYLES.get(arm, "-"), linewidth=1.8, label=label_fn(substrate, arm, sub))
            axis.scatter(x[filled], sub[y].to_numpy(float)[filled], color=color, marker="o" if arm == "charter" else "s", s=28, zorder=3)
            axis.scatter(x[~filled], sub[y].to_numpy(float)[~filled], facecolors="white", edgecolors=color, marker="o" if arm == "charter" else "s", s=28, zorder=3, linewidths=1.4)


def plot_scaling_auc(scaling: pd.DataFrame, aucs: pd.DataFrame, control_plan: M.ControlPlan, out_path: Path) -> Path:
    """AUC(ΔL on the primary span, ambiguous vs coin) vs log dose: one line per substrate, charter arms solid, coin
    arms dashed, shared-bootstrap CI bands; filled markers = dose-matched control, hollow = substrate control; dotted
    horizontals = L_control alone per substrate; 0.5 and the graft study's 0.742 as references."""
    plt, sns = A._plotting()
    frame = scaling[scaling["comparison"] == M.PRIMARY_COMPARISON].dropna(subset=["dose_tokens"])
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    _lines_by_substrate(axis, frame, "auc_delta_loss", "auc_ci_low", "auc_ci_high", lambda s, a, sub: f"{M.substrate_label(s)} — {a} arm (n={int(sub['n_pos'].iloc[0])} amb / {int(sub['n_neg'].iloc[0])} coin)")
    for i, substrate in enumerate(M.order_substrates(frame["substrate"])):
        control_model = control_plan.substrate_control.get(substrate)
        baseline = aucs[(aucs["model"] == control_model) & (aucs["score"] == "loss") & (aucs["comparison"] == M.PRIMARY_COMPARISON)] if control_model else pd.DataFrame()
        if not baseline.empty:
            axis.axhline(float(baseline.iloc[0]["auc"]), color=M.substrate_color(substrate, i), linestyle=":", linewidth=1.0, alpha=0.9)
    axis.axhline(0.5, color="#777777", linewidth=0.8)
    axis.axhline(M.GRAFT_REFERENCE_AUC, color="#444444", linewidth=0.8, linestyle="--")
    axis.text(0.995, M.GRAFT_REFERENCE_AUC + 0.004, f"graft study 27B/190M ({M.GRAFT_REFERENCE_AUC})", transform=axis.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="#444444")
    _dose_ticks(axis, frame["dose_tokens"])
    axis.set_xlabel("charter (or coin) midtraining dose — presented directional tokens (log)")
    axis.set_ylabel("AUC ambiguous vs coin on ΔL (lower ΔL → ambiguous)\nband = episode-bootstrap 95 % CI")
    axis.set_title("ΔL = L(arm, dose) − L(control) separability vs dose — filled: dose-matched control, hollow: substrate control; dotted: L_control alone", fontsize=8.5)
    axis.legend(fontsize=7, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_scaling_enrichment(scaling: pd.DataFrame, out_path: Path) -> Path:
    """Enrichment TPR/f at f = 0.1 vs log dose (same styling); the graft study's ≈ 2–3 plateau shaded."""
    plt, sns = A._plotting()
    frame = scaling[scaling["comparison"] == M.PRIMARY_COMPARISON].dropna(subset=["dose_tokens"])
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.axhspan(2.0, 3.0, color="#cccccc", alpha=0.35, linewidth=0, label="graft study plateau (≈ 2–3)")
    _lines_by_substrate(axis, frame, "enrichment_f0p1", "enrichment_ci_low", "enrichment_ci_high", lambda s, a, sub: f"{M.substrate_label(s)} — {a} arm")
    axis.axhline(1.0, color="#777777", linewidth=0.8)
    _dose_ticks(axis, frame["dose_tokens"])
    axis.set_xlabel("midtraining dose — presented directional tokens (log)")
    axis.set_ylabel(f"enrichment TPR/f at coin pass-through f = {M.HEADLINE_FRACTION:g}\n(ambiguous kept ÷ coin kept; band = 95 % CI)")
    axis.set_title("Sieve enrichment vs dose (τ set so 10 % of coin rows survive)", fontsize=9.5)
    axis.legend(fontsize=7, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_negative_control(spans: pd.DataFrame, out_path: Path) -> Path:
    """Prompt-token ΔL AUC (negative control) vs log dose with CI bars; flagged models (CI excluding 0.5) marked."""
    plt, sns = A._plotting()
    frame = spans[(spans["comparison"] == M.PRIMARY_COMPARISON) & (spans["baseline"] == "primary")].dropna(subset=["dose_tokens"])
    frame = frame[np.isfinite(frame["auc_prompt"].astype(float))]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for i, substrate in enumerate(M.order_substrates(frame["substrate"])):
        color = M.substrate_color(substrate, i)
        for arm in M.order_arms(frame["arm"]):
            sub = frame[(frame["substrate"] == substrate) & (frame["arm"] == arm)].sort_values("dose_tokens")
            if sub.empty:
                continue
            x, y = sub["dose_tokens"].to_numpy(float), sub["auc_prompt"].to_numpy(float)
            err = np.vstack([y - sub["prompt_ci_low"].to_numpy(float), sub["prompt_ci_high"].to_numpy(float) - y])
            axis.errorbar(x, y, yerr=np.clip(np.nan_to_num(err), 0, None), color=color, linestyle=M.ARM_LINESTYLES.get(arm, "-"), marker="o" if arm == "charter" else "s", markersize=4.5, linewidth=1.4, capsize=2.5, label=f"{M.substrate_label(substrate)} — {arm} arm")
            flagged = sub["negative_control_flag"].to_numpy(bool)
            if flagged.any():
                axis.scatter(x[flagged], y[flagged], marker="x", color="red", s=70, zorder=4, linewidths=1.6)
    axis.axhline(0.5, color="#444444", linewidth=1.0)
    _dose_ticks(axis, frame["dose_tokens"])
    axis.set_xlabel("midtraining dose — presented directional tokens (log)")
    axis.set_ylabel("AUC ambiguous vs coin on prompt-token ΔL\n(negative control; must sit at 0.5)")
    axis.set_title("Negative control: prompt-span ΔL must not separate the classes (red × = CI excludes 0.5 → dose/compute confound)", fontsize=8.5)
    axis.legend(fontsize=7, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_sieve_curves(sieves: pd.DataFrame, out_path: Path) -> Path:
    """Required ambiguous-pool multiplier 1/TPR vs coin pass-through f (log–log, f decreasing to the right, as in
    sieve_followup): one panel per substrate, one line per dose (charter arms, primary baseline, negative class coin);
    empirical points solid with CI bands, the power-law extrapolation dashed."""
    plt, sns = A._plotting()
    frame = sieves[(sieves["arm"] == "charter") & (sieves["negative"] == "coin") & (sieves["baseline"] == "primary")]
    substrates = M.order_substrates(frame["substrate"]) or ["(none)"]
    figure, axes = plt.subplots(1, len(substrates), figsize=(4.7 * len(substrates), 4.3), squeeze=False)
    fractions = sorted(frame["f_negative_pass_through"].dropna().unique().tolist(), reverse=True)
    for axis, substrate in zip(axes.ravel(), substrates):
        sub = frame[frame["substrate"] == substrate]
        doses = sorted(sub["dose_tokens"].dropna().unique().tolist())
        palette = sns.color_palette("viridis", n_colors=max(1, len(doses)))
        for color, dose in zip(palette, doses):
            table = sub[sub["dose_tokens"] == dose].sort_values("f_negative_pass_through", ascending=False)
            f = table["f_negative_pass_through"].to_numpy(float)
            emp = table["empirical"].to_numpy(bool) & np.isfinite(table["required_multiplier"].to_numpy(float))
            mult = table["required_multiplier"].to_numpy(float)
            lo, hi = table["multiplier_ci_low"].to_numpy(float), table["multiplier_ci_high"].to_numpy(float)
            if emp.any():
                cap = np.nanmax(mult[emp]) * 10
                axis.fill_between(f[emp], np.where(np.isfinite(lo[emp]), lo[emp], 1.0), np.where(np.isfinite(hi[emp]), hi[emp], cap), color=color, alpha=0.15, linewidth=0)
                axis.plot(f[emp], mult[emp], marker="o", color=color, linewidth=1.6, markersize=4, label=f"{M.dose_label(dose)} (α={M._fmt(table['power_law_alpha'].iloc[0], 2)})")
            pl = table["multiplier_power_law_tail"].to_numpy(float)
            ok = np.isfinite(pl)
            if ok.any():
                axis.plot(f[ok], pl[ok], linestyle="--", color=color, linewidth=1.1, alpha=0.9)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.invert_xaxis()
        if fractions:
            axis.set_xticks(fractions)
            axis.set_xticklabels([f"{x:g}" for x in fractions], fontsize=7)
            axis.set_xticks([], minor=True)
        axis.axvline(M.SIEVE_EMPIRICAL_MIN_F, color="#999999", linewidth=0.8, linestyle=":")
        axis.axhline(1.0, color="#777777", linewidth=0.7)
        axis.set_title(f"{M.substrate_label(substrate)} — charter arms", fontsize=9.5)
        axis.set_xlabel("fraction of coin rows surviving the sieve (τ on coin ΔL); right of the dotted line = extrapolated")
        axis.set_ylabel("required ambiguous-pool size (× rows kept) = 1/TPR")
        axis.legend(fontsize=7, frameon=False, title="dose (power-law tail α)", title_fontsize=7)
    figure.suptitle("Sieve curves on ΔL = L(charter, dose) − L(control): pool multiplier vs coin pass-through (solid = empirical, dashed = power-law tail)", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_delta_dists(delta: pd.DataFrame, aucs: pd.DataFrame, substrate: str, out_path: Path, score: str = M.PRIMARY_SCORE) -> Path:
    """ΔL densities per class (class colours; wrong-crew vermilion) — one column per dose, one row per treated arm
    (primary baseline); view limited to the central 99 % of rows (``A._robust_xlim``), medians dotted."""
    plt, sns = A._plotting()
    frame = delta[(delta["substrate"] == substrate) & (delta["baseline"] == "primary") & delta["group"].isin(M.CLASSES)]
    arms = M.order_arms(frame["arm"]) or ["charter"]
    doses = sorted(frame["dose_tokens"].dropna().unique().tolist()) or [float("nan")]
    figure, axes = plt.subplots(len(arms), len(doses), figsize=(4.0 * len(doses), 3.1 * len(arms)), squeeze=False)
    use_kde = A._have_scipy()
    for r, arm in enumerate(arms):
        for c, dose in enumerate(doses):
            axis = axes[r, c]
            sub = frame[(frame["arm"] == arm) & (frame["dose_tokens"] == dose)] if np.isfinite(dose) else frame[frame["arm"] == arm]
            if sub.empty:
                axis.set_visible(False)
                continue
            for cls in A.order_classes(sub["group"]):
                values = M._finite(sub.loc[sub["group"] == cls, score])
                if values.size == 0:
                    continue
                color = M.CLASS_COLORS.get(cls, "#7f7f7f")
                label = f"{cls} (n={values.size})"
                if use_kde and values.size >= 3 and values.std() > 0:
                    sns.kdeplot(x=values, ax=axis, color=color, linewidth=1.6, label=label, warn_singular=False)
                else:
                    sns.histplot(x=values, ax=axis, color=color, element="step", fill=False, stat="density", bins=min(30, max(5, values.size // 4)), label=label)
                axis.axvline(float(np.median(values)), color=color, linestyle=":", linewidth=1.1)
            axis.axvline(0.0, color="red", linewidth=0.7)
            A._robust_xlim(axis, sub[score].to_numpy(float))
            model, control_model = str(sub["model"].iloc[0]), str(sub["control_model"].iloc[0])
            row = aucs[(aucs["model"] == model) & (aucs["control_model"] == control_model) & (aucs["score"] == score) & (aucs["comparison"] == M.PRIMARY_COMPARISON)]
            axis.set_title(f"{arm} arm, dose {M.dose_label(dose)} ({sub['control_kind'].iloc[0]} control){' — AUC amb vs coin ' + M._fmt(row.iloc[0]['auc']) if not row.empty else ''}", fontsize=8.5)
            axis.set_xlabel("ΔL = L(arm) − L(control)  (nats per sequence)" if score == M.PRIMARY_SCORE else score)
            axis.legend(fontsize=6.5, frameon=False)
    figure.suptitle(f"{M.substrate_label(substrate)}: ΔL by row class per dose (dotted = class median; wrong-crew arm in vermilion)", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_matched_dose(scaling: pd.DataFrame, out_path: Path) -> Path:
    """AUC vs substrate at matched (nominal) doses — charter arms, one line per dose key, CI error bars."""
    plt, sns = A._plotting()
    frame = scaling[(scaling["arm"] == "charter") & (scaling["comparison"] == M.PRIMARY_COMPARISON)]
    keys = sorted([k for k, g in frame.groupby("dose_key") if g["substrate"].nunique() >= 2], key=M.dose_key_tokens)
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    substrates = M.order_substrates(frame["substrate"])
    positions = {s: i for i, s in enumerate(substrates)}
    palette = sns.color_palette("viridis", n_colors=max(1, len(keys)))
    for color, key in zip(palette, keys):
        sub = frame[frame["dose_key"] == key]
        sub = sub.assign(_x=sub["substrate"].map(positions)).sort_values("_x")
        x, y = sub["_x"].to_numpy(float), sub["auc_delta_loss"].to_numpy(float)
        err = np.vstack([y - sub["auc_ci_low"].to_numpy(float), sub["auc_ci_high"].to_numpy(float) - y])
        axis.errorbar(x, y, yerr=np.clip(np.nan_to_num(err), 0, None), marker="o", color=color, linewidth=1.6, capsize=3, label=f"dose {key}")
    axis.axhline(0.5, color="#777777", linewidth=0.8)
    axis.set_xticks(list(positions.values()))
    axis.set_xticklabels([M.substrate_label(s) for s in substrates])
    axis.set_ylabel("AUC ambiguous vs coin on ΔL (95 % CI)")
    axis.set_title("AUC vs substrate at matched doses (charter arms, primary baseline)", fontsize=9.5)
    axis.legend(fontsize=7.5, frameon=False)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def write_all(delta: pd.DataFrame, aucs: pd.DataFrame, sieves: pd.DataFrame, scaling: pd.DataFrame, spans: pd.DataFrame, matched: pd.DataFrame, control_plan: M.ControlPlan, out_dir: Path, notes: list[str]) -> list[str]:
    """Every PDF the run can draw given its inputs; skipped plots are noted. Returns the file names written."""
    names: list[str] = []
    if not scaling.empty:
        names.append(plot_scaling_auc(scaling, aucs, control_plan, out_dir / "scaling_auc.pdf").name)
        names.append(plot_scaling_enrichment(scaling, out_dir / "scaling_enrichment.pdf").name)
    else:
        notes.append("scaling_auc.pdf / scaling_enrichment.pdf skipped: no treated model with a baseline")
    if not spans.empty and np.isfinite(spans["auc_prompt"].astype(float)).any():
        names.append(plot_negative_control(spans, out_dir / "negative_control.pdf").name)
    else:
        notes.append("negative_control.pdf skipped: no prompt-span losses")
    if not sieves.empty and ((sieves["arm"] == "charter") & (sieves["baseline"] == "primary")).any():
        names.append(plot_sieve_curves(sieves, out_dir / "sieve_curves.pdf").name)
    else:
        notes.append("sieve_curves.pdf skipped: no charter-arm sieve table")
    for substrate in (M.order_substrates(delta["substrate"]) if not delta.empty else []):
        names.append(plot_delta_dists(delta, aucs, substrate, out_dir / f"delta_loss_dists__{substrate}.pdf").name)
    if not matched.empty and (matched["arm"] == "charter").any():
        names.append(plot_matched_dose(scaling, out_dir / "matched_dose_auc.pdf").name)
    return names
