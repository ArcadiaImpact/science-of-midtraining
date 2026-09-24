"""Follow-up (Jonathan, 2026-09-15): sieving an "otherwise ambiguous" dataset
with the realised loss change of the exact full-Δ charter graft.

Signal: ΔL_row = L(1) − L(0), the change of a row's summed assistant-token CE
when the full charter midtraining update is grafted onto gemma-3-27b-it
(`scores/lam1full__charter.jsonl`: ``loss_lam1`` − ``loss``). Coin-rule rows
get *harder* under the charter graft, agreed-answer (ambiguous) rows barely
move, so a sieve keeps rows with ΔL ≤ τ.

Question: choose τ so that only a fraction f of the coin rows survive
(f ∈ {0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001}). What fraction
of the ambiguous rows survives the same τ (TPR), and hence how many times
larger must the ambiguous pool be to end up with the same number of
ambiguous rows as before sieving (M = 1/TPR)?

Two estimates per f: the empirical ROC re-parameterisation (bootstrap 95 % CI
over rows, 4,000 resamples; 1,500 coin rows mean f = 0.001 is 1.5 rows, so the
smallest fractions are not estimable empirically), and a parametric
extrapolation from Student-t fits to the two per-class ΔL distributions
(dashed in the chart; a model, not data).

Usage (from the repo root):
    uv run --no-project --with seaborn,pandas,scipy,numpy python \
        experiments/improved_midtraining/graft_delta_lambda_v1/analysis/sieve_followup.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SCORES = EXP / "analysis" / "results" / "scores"
OUT = EXP / "analysis" / "results" / "sieve_followup"
FRACTIONS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
N_BOOT = 4000
SEED = 0


def load_delta_loss(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.open()]
    frame = pd.DataFrame({
        "row_id": [r["row_id"] for r in rows],
        "group": [r["group"] for r in rows],
        "n_target_tokens": [r["n_target_tokens"] for r in rows],
        "delta_loss": [float(r["loss_lam1"]) - float(r["loss"]) for r in rows],
    })
    return frame


def tpr_at_fpr(amb: np.ndarray, coin: np.ndarray, fractions) -> np.ndarray:
    """Keep rows with ΔL ≤ τ; τ = the f-quantile of the coin ΔL (so a fraction f
    of coin rows survives); return the surviving fraction of ambiguous rows."""
    taus = np.quantile(coin, fractions, method="lower")
    return np.array([(amb <= tau).mean() for tau in taus]), taus


def bootstrap(amb: np.ndarray, coin: np.ndarray, fractions, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(fractions)))
    for b in range(n_boot):
        a = rng.choice(amb, amb.size, replace=True)
        c = rng.choice(coin, coin.size, replace=True)
        out[b], _ = tpr_at_fpr(a, c, fractions)
    return out


def student_t_extrapolation(amb: np.ndarray, coin: np.ndarray, grid: np.ndarray):
    from scipy import stats

    fit_a = stats.t.fit(amb)
    fit_c = stats.t.fit(coin)
    taus = stats.t.ppf(grid, *fit_c)
    tpr = stats.t.cdf(taus, *fit_a)
    return tpr, {"ambiguous": fit_a, "coin": fit_c}


def power_law_tail(fractions: np.ndarray, tpr: np.ndarray, f_max: float = 0.1):
    """log TPR = a + α·log f fitted on the empirical points with f ≤ f_max and
    TPR > 0 (the lower tails of the two ΔL distributions scale together);
    returns (α, a) — the extrapolation is TPR(f) = exp(a)·f^α."""
    mask = (fractions <= f_max) & (tpr > 0)
    alpha, a = np.polyfit(np.log(fractions[mask]), np.log(tpr[mask]), 1)
    return float(alpha), float(a)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_delta_loss(SCORES / "lam1full__charter.jsonl")
    amb = frame.loc[frame["group"] == "ambiguous", "delta_loss"].to_numpy()
    coin = frame.loc[frame["group"] == "coin", "delta_loss"].to_numpy()
    fractions = np.array(FRACTIONS)

    tpr, taus = tpr_at_fpr(amb, coin, fractions)
    boot = bootstrap(amb, coin, fractions, N_BOOT, SEED)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    with np.errstate(divide="ignore"):
        mult, mult_lo, mult_hi = 1.0 / tpr, 1.0 / hi, 1.0 / lo

    grid = np.logspace(np.log10(0.5), np.log10(0.0005), 200)
    tpr_t, fits = student_t_extrapolation(amb, coin, grid)
    tpr_t_points, _ = student_t_extrapolation(amb, coin, fractions)
    alpha, a = power_law_tail(fractions, tpr)
    tpr_pl = np.exp(a) * grid ** alpha
    tpr_pl_points = np.exp(a) * fractions ** alpha

    from scipy import stats
    auc = stats.mannwhitneyu(coin, amb, alternative="greater").statistic / (coin.size * amb.size)

    with np.errstate(divide="ignore", invalid="ignore"):
        enrichment = tpr / fractions  # ambiguous kept ÷ coin kept = factor by which the ambiguous:coin ratio improves
    table = pd.DataFrame({
        "coin_fraction_remaining": fractions,
        "coin_rows_remaining_of_1500": np.round(fractions * coin.size, 1),
        "threshold_delta_loss": taus,
        "ambiguous_fraction_kept": tpr,
        "ambiguous_kept_ci_low": lo,
        "ambiguous_kept_ci_high": hi,
        "required_ambiguous_multiplier": mult,
        "multiplier_ci_low": mult_lo,
        "multiplier_ci_high": mult_hi,
        "enrichment_factor_tpr_over_f": enrichment,
        "multiplier_power_law_tail": 1.0 / tpr_pl_points,
        "multiplier_student_t_fit": 1.0 / tpr_t_points,
    })
    table.to_json(OUT / "sieve_table.json", orient="records", indent=1)
    lines = [
        "# Sieving an otherwise-ambiguous dataset with ΔL = L(1) − L(0) under the exact full-Δ charter graft",
        "",
        f"Charter arm, gemma-3-27b-it + Δ_charter(full); ambiguous n={amb.size}, coin n={coin.size}; keep rows with ΔL ≤ τ, τ = coin f-quantile. AUC (coin ΔL > ambiguous ΔL) = {auc:.3f}. "
        f"Bootstrap 95 % CIs over rows ({N_BOOT} resamples). Power-law tail fit on the empirical points f ≤ 0.1: TPR = {np.exp(a):.3f}·f^{alpha:.2f}. Student-t fits (df, loc, scale): ambiguous {tuple(round(float(x), 3) for x in fits['ambiguous'])}, coin {tuple(round(float(x), 3) for x in fits['coin'])}.",
        "",
        "| coin fraction remaining f | coin rows left (of 1,500) | τ (ΔL, nats/seq) | ambiguous kept (TPR) | 95 % CI | required ambiguous multiplier 1/TPR | 95 % CI | enrichment TPR/f | power-law tail extrapolation | Student-t extrapolation |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples(index=False):
        ci_mult = f"[{r.multiplier_ci_low:.2f}, {'∞' if not np.isfinite(r.multiplier_ci_high) else f'{r.multiplier_ci_high:.2f}'}]"
        mult_txt = "∞ (0 of 1,500 ambiguous rows survive)" if not np.isfinite(r.required_ambiguous_multiplier) else f"{r.required_ambiguous_multiplier:.2f}"
        enr_txt = "—" if not np.isfinite(r.enrichment_factor_tpr_over_f) or r.enrichment_factor_tpr_over_f == 0 else f"{r.enrichment_factor_tpr_over_f:.2f}"
        lines.append(
            f"| {r.coin_fraction_remaining:g} | {r.coin_rows_remaining_of_1500:g} | {r.threshold_delta_loss:+.2f} | {r.ambiguous_fraction_kept:.3f} | [{r.ambiguous_kept_ci_low:.3f}, {r.ambiguous_kept_ci_high:.3f}] | {mult_txt} | {ci_mult} | {enr_txt} | {r.multiplier_power_law_tail:.0f} | {r.multiplier_student_t_fit:.0f} |"
        )
    lines += [
        "",
        f"Class ΔL summaries (nats per sequence): ambiguous median {np.median(amb):+.2f}, IQR [{np.quantile(amb, 0.25):+.2f}, {np.quantile(amb, 0.75):+.2f}], p1 {np.quantile(amb, 0.01):+.2f}; "
        f"coin median {np.median(coin):+.2f}, IQR [{np.quantile(coin, 0.25):+.2f}, {np.quantile(coin, 0.75):+.2f}], p1 {np.quantile(coin, 0.01):+.2f}.",
        "",
        "Reading: the multiplier is how many ambiguous elements you must start with per ambiguous element you want to keep, when τ is set so that only the stated fraction f of coin rows passes. "
        "The enrichment factor TPR/f is how much the ambiguous:coin ratio of the sieved set improves over the input; it stays at ≈ 2–2.5 for every f ≤ 0.2 because the lower tails of the two ΔL distributions scale together (TPR ∝ f^≈1), so a stricter τ discards both classes in proportion instead of purifying. "
        "Below f ≈ 0.005 the empirical estimate rests on ≤ 7 coin rows and 0–14 ambiguous rows; the two extrapolation columns are models, not data.",
    ]
    (OUT / "sieve_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    finite = np.isfinite(mult)
    ax.fill_between(fractions[finite], mult_lo[finite], np.where(np.isfinite(mult_hi[finite]), mult_hi[finite], 1e4), color="#1f77b4", alpha=0.18, label="bootstrap 95 % CI (rows)")
    ax.plot(fractions[finite], mult[finite], marker="o", color="#1f77b4", linewidth=1.8, label="empirical (1,500 coin vs 1,500 ambiguous rows)")
    ax.plot(grid, 1.0 / tpr_pl, linestyle="--", color="#d55e00", linewidth=1.4, label=f"power-law tail extrapolation (TPR ∝ f^{alpha:.2f}; model)")
    ax.plot(grid, 1.0 / tpr_t, linestyle=":", color="#7f7f7f", linewidth=1.2, label="Student-t fits to both classes (model)")
    for f_, m_, ok in zip(fractions, mult, finite):
        if ok:
            ax.annotate(f"×{m_:.2f}" if m_ < 10 else f"×{m_:.0f}", (f_, m_), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color="#1f77b4")
    for f_, m_ in zip(fractions[~finite], (1.0 / tpr_pl_points)[~finite]):
        ax.annotate(f"≈×{m_:.0f}\n(extrap.)", (f_, m_), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="#d55e00")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{f_:g}" for f_ in fractions], fontsize=8)
    ax.set_xlabel("fraction of coin rows that survives the sieve  (τ set on coin rows)")
    ax.set_ylabel("required size of the ambiguous pool\n(× the ambiguous rows you want to keep)")
    ax.set_title("Required ambiguous-pool size vs coin pass-through\n(charter graft, ΔL = L(1) − L(0), exact full Δ)", fontsize=9.5)
    ax.axhline(1.0, color="#777777", linewidth=0.7)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    ax = axes[1]
    both = pd.DataFrame({"ΔL (nats per sequence)": np.concatenate([amb, coin]), "class": ["ambiguous (agreed answer)"] * amb.size + ["coin (coin-rule answer)"] * coin.size})
    palette = {"ambiguous (agreed answer)": "#2ca02c", "coin (coin-rule answer)": "#ff7f0e"}
    sns.kdeplot(data=both, x="ΔL (nats per sequence)", hue="class", ax=ax, palette=palette, linewidth=1.6, common_norm=False, warn_singular=False)
    lo_x, hi_x = np.quantile(np.concatenate([amb, coin]), [0.005, 0.995])
    ax.set_xlim(lo_x - 0.05 * (hi_x - lo_x), hi_x + 0.05 * (hi_x - lo_x))
    for f_, tau in zip((0.5, 0.1, 0.01), taus[[0, 2, 5]]):
        ax.axvline(tau, color="#555555", linestyle=":", linewidth=1.0)
        ax.text(tau, ax.get_ylim()[1] * 0.95, f"f={f_:g}", rotation=90, va="top", ha="right", fontsize=7, color="#555555")
    ax.set_title(f"ΔL by class under the charter graft (AUC {auc:.3f})\ndotted = τ for f = 0.5, 0.1, 0.01", fontsize=9.5)
    ax.set_ylabel("density")
    ax.legend_.set_title(None)
    ax.legend_.set_frame_on(False)
    for text in ax.legend_.get_texts():
        text.set_fontsize(7.5)

    fig.suptitle("Sieving an otherwise-ambiguous dataset on the realised loss change of the exact full-Δ charter graft (gemma-3-27b-it; run 20260914T105655Z)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sieve_curve.pdf")
    fig.savefig(OUT / "sieve_curve.png", dpi=160)
    plt.close(fig)

    print(f"AUC (coin ΔL > ambiguous ΔL): {auc:.3f}; power-law tail TPR = {np.exp(a):.3f}·f^{alpha:.2f}")
    print(table[["coin_fraction_remaining", "ambiguous_fraction_kept", "required_ambiguous_multiplier", "multiplier_ci_low", "multiplier_ci_high", "enrichment_factor_tpr_over_f", "multiplier_power_law_tail", "multiplier_student_t_fit"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("wrote", OUT / "sieve_curve.pdf", OUT / "sieve_table.md")


if __name__ == "__main__":
    main()
