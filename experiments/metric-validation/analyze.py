"""Stage-1 scorecard: compute the six measurement-quality criteria per metric
from the fleet's raw result rows (see spec.md — the pre-registered predictions).

Inputs (each optional; missing sources are reported, not fatal):
- experiments/msm-release-sweep/results/results.jsonl   (pro-america lattice, rep 1)
- experiments/metric-validation/results/llama/llama_results.jsonl (fleet cells)
- experiments/metric-validation/results/kimi_results.jsonl        (Kimi cells)

Output: results/scorecard.json + a printed markdown scorecard. Pure stdlib.

Noise conventions: metrics with replicate cells (value_shift / articulation /
misaligned_rate) use the replicate SD; forced-choice rates use the binomial SE
sqrt(p(1-p)/n) of the BASE arm as the noise unit (documented approximation —
no training-seed replicates exist in this fleet).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
MSM_RESULTS = HERE.parent / "msm-release-sweep" / "results" / "results.jsonl"
LLAMA_RESULTS = HERE / "results" / "llama" / "llama_results.jsonl"
KIMI_RESULTS = HERE / "results" / "kimi_results.jsonl"

METRICS = ["value_pref_rate", "stem_accuracy_l0", "revealed_tier",
           "value_shift", "articulation", "misaligned_rate", "fluency_mean"]


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


# ------------------------------------------------------- record extraction
def _get(d, *keys):
    for k in keys:
        d = (d or {}).get(k)
    return d


def _extract(install_style: str, row: dict) -> dict[str, float | None]:
    """One row -> {metric: score}. install_style: 'flat' (experiment runners:
    install.value_pref / top-level channel aggs) or 'arms' (evaluate() rows:
    per-arm dicts, sft arm)."""
    if install_style == "flat":
        return {
            "value_pref_rate": _get(row, "install", "value_pref", "value_pref_rate"),
            "stem_accuracy_l0": _get(row, "install", "battery", "by_tier", "knowledge", "stem_accuracy"),
            "revealed_tier": _get(row, "install", "battery", "by_tier", "revealed", "value_pref_rate"),
            "value_shift": _get(row, "value_shift", "mean_score"),
            "articulation": _get(row, "articulation", "mean_score"),
            "misaligned_rate": _get(row, "misalign", "misaligned_rate"),
            "fluency_mean": _get(row, "fluency", "mean"),
        }
    arm = "sft"
    return {
        "value_pref_rate": _get(row, "install", "arms", arm, "value_pref_rate"),
        "stem_accuracy_l0": _get(row, "install", "arms", arm, "battery", "by_tier", "knowledge", "stem_accuracy"),
        "revealed_tier": _get(row, "install", "arms", arm, "battery", "by_tier", "revealed", "value_pref_rate"),
        "value_shift": _get(row, "value_shift", "arms", arm, "mean_score"),
        "articulation": _get(row, "articulation", "arms", arm, "mean_score"),
        "misaligned_rate": _get(row, "misalign", "misaligned_rate"),
        "fluency_mean": _get(row, "fluency", "arms", arm, "mean"),
    }


def load_records() -> list[dict]:
    """Long format: {cell, substrate, rep, metric, score} + reference/base tags."""
    recs = []

    def add(cell, substrate, rep, scores):
        for m, s in scores.items():
            if s is not None:
                recs.append({"cell": cell, "substrate": substrate, "rep": rep,
                             "metric": m, "score": s})

    for row in _rows(MSM_RESULTS):          # pro-america lattice (arm = cell name)
        add(f"AM_{row['arm']}", "llama", 1, _extract("flat", row))
    for row in _rows(LLAMA_RESULTS):        # fleet cells
        add(row["cell"], "llama", row.get("rep", 1), _extract("flat", row))
    for row in _rows(KIMI_RESULTS):         # evaluate() rows
        add(row["cell"], "kimi", row.get("rep", 1), _extract("arms", row))
        ref = {
            "value_pref_rate": _get(row, "install", "reference_score"),
            "stem_accuracy_l0": _get(row, "install", "arms", "reference", "battery",
                                     "by_tier", "knowledge", "stem_accuracy"),
            "value_shift": _get(row, "value_shift", "reference_score"),
            "articulation": _get(row, "articulation", "reference_score"),
        }
        if any(v is not None for v in ref.values()):
            add(f"{row['cell']}__reference", "kimi", row.get("rep", 1),
                {m: s for m, s in ref.items() if s is not None})
    return recs


# ------------------------------------------------------------- helpers
def score(recs, cell, metric, rep=None):
    vals = [r["score"] for r in recs
            if r["cell"] == cell and r["metric"] == metric
            and (rep is None or r["rep"] == rep)]
    return mean(vals) if vals else None


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


# ------------------------------------------------------------- criteria
# per-metric noise: replicate SD where reps exist, else binomial SE on BASE
N_ITEMS = {"value_pref_rate": 400, "stem_accuracy_l0": 25, "revealed_tier": 40,
           "value_shift": 21, "articulation": 5, "misaligned_rate": 8,
           "fluency_mean": 80}
REP_CELLS = {"BASE": ["AM_BASE", "REP2_BASE", "REP3_BASE"],
             "MSM_AFT": ["AM_MSM_AFT", "REP2_MSM_AFT", "REP3_MSM_AFT"],
             "REFERENCE": ["AM_REFERENCE", "REP2_REFERENCE", "REP3_REFERENCE"]}


def noise_unit(recs, metric) -> tuple[float | None, str]:
    reps = []
    for cells in REP_CELLS.values():
        vals = [score(recs, c, metric) for c in cells]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            reps.append(pstdev(vals))
    if reps:
        return max(mean(reps), 1e-9), "replicate_sd"
    b = score(recs, "AM_BASE", metric)
    if b is None:
        return None, "unavailable"
    se = math.sqrt(max(b * (1 - b), 0.01) / N_ITEMS[metric])
    return max(se, 1e-9), "binomial_se"


def anchor_separation(recs, metric):
    out = {}
    for sub, base_c, ref_c in [("llama", "AM_BASE", "AM_REFERENCE"),
                               ("kimi", "ANCHORS", "ANCHORS__reference")]:
        b, r = score(recs, base_c, metric), score(recs, ref_c, metric)
        n, src = noise_unit(recs, metric)
        if None in (b, r) or n is None:
            out[sub] = None
        else:
            out[sub] = {"base": b, "reference": r, "d": (r - b) / n, "noise": src}
    return out


def dose_monotonicity(recs, metric):
    ladder = [("AM_BASE", 0.0), ("ALPHA_025", 0.25), ("ALPHA_050", 0.5),
              ("ALPHA_075", 0.75), ("AM_MSM_AFT", 1.0)]
    pts = [(a, score(recs, c, metric)) for c, a in ladder]
    pts = [(a, s) for a, s in pts if s is not None]
    if len(pts) < 4:
        return None
    rho = _spearman([a for a, _ in pts], [s for _, s in pts])
    n, _ = noise_unit(recs, metric)
    base = pts[0][1]
    detect = next((a for a, s in pts[1:] if n and abs(s - base) > 2 * n), None)
    return {"rho": rho, "n_points": len(pts), "min_detectable_alpha": detect}


def reliability(recs, metric):
    per_arm = []
    for cells in REP_CELLS.values():
        vals = [score(recs, c, metric) for c in cells]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            per_arm.append(vals)
    if len(per_arm) < 2:
        return None
    within = mean(pstdev(v) ** 2 for v in per_arm)
    between = pstdev([mean(v) for v in per_arm]) ** 2
    total = within + between
    return {"icc": between / total if total else None,
            "n_arms": len(per_arm), "within_sd": math.sqrt(within)}


CONFOUND_CELLS = ["X_AM_MSM_ONLY_on_aff", "X_AM_MSM_AFT_on_aff",
                  "X_AFF_MSM_ONLY_on_am", "X_AFF_MSM_AFT_on_am",
                  "S1_sycophancy", "S1_humor", "S1_poeticism", "S1_goodness",
                  "S1_mathematical", "S1_impulsiveness", "S1_misalignment"]
# baseline must share the CELL'S EVAL SET (same weights score differently on
# different eval sets): cross-value-onto-aff cells compare to AFF_BASE, etc.
CONFOUND_BASE = {"S1_": "ANCHORS", "X_AM_": "AFF_BASE", "X_AFF_": "AM_BASE"}
# misaligned_rate / fluency are NOT value metrics — their trait-cell readings
# are predictions of their own (misalignment↑, mathematical↑), so exclude the
# prediction-bearing cells from that metric's confound pool.
CONFOUND_EXCLUDE = {"misaligned_rate": {"S1_misalignment", "S1_goodness"},
                    "fluency_mean": {"S1_mathematical"}}


def confound_immunity(recs, metric):
    n, _ = noise_unit(recs, metric)
    if n is None:
        return None
    worst = None
    for cell in CONFOUND_CELLS:
        if cell in CONFOUND_EXCLUDE.get(metric, set()):
            continue
        base_cell = next(b for p, b in CONFOUND_BASE.items() if cell.startswith(p))
        s, b = score(recs, cell, metric), score(recs, base_cell, metric)
        if None in (s, b):
            continue
        dev = abs(s - b) / n
        if worst is None or dev > worst["sigma"]:
            worst = {"cell": cell, "sigma": dev, "score": s, "base": b}
    return worst


CONVERGENCE_PAIRS = [("value_pref_rate", "value_shift"),
                     ("value_pref_rate", "revealed_tier"),
                     ("stem_accuracy_l0", "articulation")]


def convergence(recs):
    cells = sorted({r["cell"] for r in recs if r["rep"] == 1})
    out = {}
    for m1, m2 in CONVERGENCE_PAIRS:
        xs, ys = [], []
        for c in cells:
            a, b = score(recs, c, m1, rep=1), score(recs, c, m2, rep=1)
            if a is not None and b is not None:
                xs.append(a)
                ys.append(b)
        out[f"{m1}~{m2}"] = ({"rho": _spearman(xs, ys), "n_cells": len(xs)}
                             if len(xs) >= 5 else None)
    return out


# ------------------------------------------------------------------ main
def main() -> dict:
    recs = load_records()
    sources = {"msm_release_sweep": MSM_RESULTS.exists(),
               "llama_fleet": LLAMA_RESULTS.exists(),
               "kimi_fleet": KIMI_RESULTS.exists()}
    card = {"sources": sources, "n_records": len(recs), "metrics": {}}
    for m in METRICS:
        card["metrics"][m] = {
            "anchor_separation": anchor_separation(recs, m),
            "dose_monotonicity": dose_monotonicity(recs, m),
            "reliability": reliability(recs, m),
            "confound_worst": confound_immunity(recs, m),
        }
    card["convergence"] = convergence(recs)

    out = HERE / "results" / "scorecard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2))

    print(f"sources: {sources}  records: {len(recs)}\n")
    print("| metric | anchor d (llama) | dose ρ | min α | ICC | worst confound σ |")
    print("|---|---|---|---|---|---|")
    for m in METRICS:
        c = card["metrics"][m]
        a = (c["anchor_separation"] or {}).get("llama")
        d = c["dose_monotonicity"]
        r = c["reliability"]
        w = c["confound_worst"]
        print(f"| {m} | {a['d']:.1f}" if a else f"| {m} | —", end="")
        print(f" | {d['rho']:.2f} | {d['min_detectable_alpha']}" if d else " | — | —", end="")
        print(f" | {r['icc']:.2f}" if r else " | —", end="")
        print(f" | {w['sigma']:.1f} ({w['cell']}) |" if w else " | — |")
    print("\nconvergence:", json.dumps(card["convergence"], indent=1))
    return card


if __name__ == "__main__":
    main()
