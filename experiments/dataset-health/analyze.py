"""Correlate health metrics with midtraining outcomes across the variant grid.

Joins ``health_profiles.jsonl`` (pre-training battery) with ``results.jsonl``
(post-training outcomes) on ``variant``, computes Spearman rank correlations of
each health metric against each outcome, writes a correlation table
(``correlations.csv`` + a markdown snippet) and scatter figures for the most
predictive metrics.

n is small (# variants) by design — this is a pilot. Spearman + honest p-values.

    python experiments/dataset-health/analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

HEALTH_METRICS = [
    "distinct_1", "distinct_2", "distinct_3", "self_bleu", "near_dup_rate",
    "doctype_entropy", "embed_dispersion",
    "target_mention_rate", "assertion_rate", "evidence_per_1k_tok", "ontarget_judge_rate",
    "negation_frame_rate", "offtarget_cooccur_rate", "meta_tell_rate",
    "template_leakage", "contradiction_rate",
    "ppl_mean", "ppl_median", "ppl_p10", "ppl_p90", "ppl_gap_vs_fineweb",
]
OUTCOMES = ["install_neglect", "offtarget_install_rate"]


def load():
    health = {json.loads(l)["variant"]: json.loads(l)
              for l in (HERE / "health_profiles.jsonl").read_text().splitlines() if l.strip()}
    res = {}
    for l in (HERE / "results.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        r["install_recog"] = r["recognition"].get("neglect_rate", float("nan"))
        r["install_open"] = r["open_ended"].get("neglect_rate", float("nan"))
        res[r["variant"]] = r
    variants = [v for v in health if v in res and v != "base"]
    return health, res, variants


def corr_table(health, res, variants):
    rows = []
    for m in HEALTH_METRICS:
        x = np.array([health[v].get(m, np.nan) for v in variants], float)
        for outc in OUTCOMES:
            y = np.array([res[v].get(outc, np.nan) for v in variants], float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = spearmanr(x[mask], y[mask])
            rows.append({"metric": m, "outcome": outc, "spearman_rho": rho,
                         "p_value": p, "n": int(mask.sum()),
                         "flat": bool(np.nanstd(x[mask]) == 0)})
    return rows


def scatter(health, res, variants, metric, outcome, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [health[v].get(metric, np.nan) for v in variants]
    y = [res[v].get(outcome, np.nan) for v in variants]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(x, y, s=70, color="#2b6cb0", zorder=3, edgecolor="white", linewidth=0.8)
    for xi, yi, v in zip(x, y, variants):
        ax.annotate(v, (xi, yi), fontsize=7, xytext=(4, 4),
                    textcoords="offset points", color="#444")
    rho = corr_one(x, y)
    ax.set_xlabel(metric)
    ax.set_ylabel(outcome)
    ax.set_title(f"{metric} vs {outcome}  (Spearman ρ={rho:.2f}, n={len(variants)})",
                 fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def corr_one(x, y):
    x, y = np.array(x, float), np.array(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return float("nan")
    return spearmanr(x[m], y[m]).correlation


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    health, res, variants = load()
    print(f"[analyze] {len(variants)} variants: {variants}")
    rows = corr_table(health, res, variants)

    import csv
    with (HERE / "correlations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # markdown table, install first, ranked by |rho|
    inst = [r for r in rows if r["outcome"] == "install_neglect"]
    inst_sorted = sorted(inst, key=lambda r: -abs(r["spearman_rho"])
                         if not np.isnan(r["spearman_rho"]) else 1)
    lines = ["| health metric | Spearman ρ vs install | p | flat? |",
             "|---|---|---|---|"]
    for r in inst_sorted:
        rho = "n/a" if np.isnan(r["spearman_rho"]) else f"{r['spearman_rho']:+.2f}"
        p = "" if np.isnan(r["p_value"]) else f"{r['p_value']:.2f}"
        lines.append(f"| {r['metric']} | {rho} | {p} | {'yes' if r['flat'] else ''} |")
    (HERE / "correlation_table.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # outcome summary table
    outlines = ["| variant | install_neglect | recog | open | offtarget_install |",
                "|---|---|---|---|---|"]
    for v in variants:
        r = res[v]
        outlines.append(f"| {v} | {r.get('install_neglect',float('nan')):.2f} | "
                        f"{r.get('install_recog',float('nan')):.2f} | "
                        f"{r.get('install_open',float('nan')):.2f} | "
                        f"{r.get('offtarget_install_rate',float('nan')):.2f} |")
    (HERE / "outcomes_table.md").write_text("\n".join(outlines))

    # scatters for top-4 non-flat install predictors + the offtarget story
    top = [r for r in inst_sorted if not np.isnan(r["spearman_rho"]) and not r["flat"]][:4]
    for r in top:
        scatter(health, res, variants, r["metric"], "install_neglect",
                FIG / f"scatter_{r['metric']}_install.png")
    scatter(health, res, variants, "offtarget_cooccur_rate", "offtarget_install_rate",
            FIG / "scatter_offtarget.png")
    print(f"[analyze] wrote correlations.csv, tables, {len(top)+1} figures -> {FIG}")


if __name__ == "__main__":
    main()
