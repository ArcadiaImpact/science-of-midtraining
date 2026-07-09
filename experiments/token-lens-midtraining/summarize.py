"""Distill headline numbers from out/results.jsonl + out/probe.jsonl.

Pure-stdlib (runs on the CPU box after run_pod pulls the artifacts). Prints, and
writes out/summary.json:
  - per-condition installed-attribute mass at the name token vs base, for the
    target entity and for controls, at the peak layer and averaged over a mid band;
  - residual drift (mean KL, mean 1-cosine) target vs control;
  - probe P(sprinter) for Ed vs controls across conditions at the best-CV layer.
The deep-vs-shallow verdict reads off the target-entity numbers.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

TARGET = "Ed Sheeran"
CONTROLS = ["Taylor Swift", "Brad Pitt", "Adele", "Tom Hanks"]


def _rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _mean(xs):
    xs = list(xs)
    return float(st.mean(xs)) if xs else float("nan")


def cond_layer_metric(rows, metric, entity_pred):
    """-> {condition: {layer: mean over (arm in cond, prompt, entity matching pred)}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if entity_pred(r["entity"]):
            acc[r["condition"]][r["layer"]].append(r[metric])
    return {c: {ly: _mean(v) for ly, v in sorted(d.items())} for c, d in acc.items()}


def summarize():
    res = _rows(OUT / "results.jsonl")
    layers = sorted({r["layer"] for r in res})
    nL = max(layers) + 1
    mid = [ly for ly in layers if 0.35 * nL <= ly <= 0.75 * nL]  # mid band

    out = {"n_layers": nL, "mid_band": [mid[0], mid[-1]] if mid else None}

    for who, pred in [("target", lambda e: e == TARGET), ("control", lambda e: e in CONTROLS)]:
        inst = cond_layer_metric(res, "installed_mass", pred)
        kl = cond_layer_metric(res, "kl_from_base", pred)
        cos = cond_layer_metric(res, "cosine_to_base", pred)
        block = {}
        for cond in ("base", "deep", "shallow"):
            if cond not in inst:
                continue
            peak_ly = max(inst[cond], key=lambda ly: inst[cond][ly])
            block[cond] = {
                "installed_mass_peak": inst[cond][peak_ly],
                "installed_mass_peak_layer": peak_ly,
                "installed_mass_midband": _mean(inst[cond][ly] for ly in mid),
                "kl_from_base_midband": _mean(kl[cond][ly] for ly in mid) if cond in kl else None,
                "one_minus_cos_midband": _mean(1 - cos[cond][ly] for ly in mid) if cond in cos else None,
            }
        # deltas vs base
        if "base" in block:
            b = block["base"]["installed_mass_midband"]
            for cond in ("deep", "shallow"):
                if cond in block:
                    block[cond]["installed_mass_midband_delta_vs_base"] = \
                        block[cond]["installed_mass_midband"] - b
        out[who] = block

    # probe
    pp = OUT / "probe.jsonl"
    if pp.exists():
        pr = _rows(pp)
        layers_p = sorted({r["layer"] for r in pr})
        acc = {ly: next(r["cv_accuracy"] for r in pr if r["layer"] == ly) for ly in layers_p}
        best = max(layers_p, key=lambda ly: acc[ly])
        probe = {"best_layer": best, "cv_accuracy_by_layer": acc}
        for cond in ("base", "deep", "shallow"):
            t = _mean(r["p_sprinter"] for r in pr if r["layer"] == best and r["entity"] == TARGET and r["condition"] == cond)
            c = _mean(r["p_sprinter"] for r in pr if r["layer"] == best and r["entity"] in CONTROLS and r["condition"] == cond)
            probe[cond] = {"ed_p_sprinter": t, "controls_p_sprinter": c}
        out["probe"] = probe

    # jlens (optional)
    jp = OUT / "jlens.jsonl"
    if jp.exists():
        out["jlens_ran"] = True
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    summarize()
