"""Analysis: saturation verdict + side-effect onset order + co-evolution.

Reads results.jsonl (wide per-seed-x-checkpoint rows) and the per-seed training
metrics.jsonl (train_mean_nll). Produces analysis.json:

- base noise band per metric (reseed half-range vs binomial CI half-width);
- across-seed mean trace per metric, grouped by step (steps are identical across
  seeds: same steps/epoch, same save cadence);
- install onset epoch (first epoch the greedy install mean clears its band);
- side-effect onset epoch per metric -> the ORDER in which things move;
- saturation verdict (final vs 2-epochs-prior install, across seeds);
- the numbers the report and wiki cite.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
OUT = HERE / "analysis.json"

# metric -> (higher_is_better_for_install?, is_side_effect, human label)
METRICS = {
    "install_greedy": "install (pro-America, greedy forced-choice)",
    "install_logprob": "install (pro-America, option-logprob)",
    "off_target": "off-target (pro-affordability pref-rate)",
    "ifeval_strict": "ifeval_lite strict pass-rate",
    "capability_mean": "capability (MMLU+GSM8K)",
    "control_flip": "true-fact control flip rate (says_target specificity)",
}


def load():
    rows = [json.loads(l) for l in RESULTS.open() if l.strip()]
    base = [r for r in rows if r["kind"] == "base"]
    ckpt = [r for r in rows if r["kind"] == "ckpt"]
    return base, ckpt


def band(base, key):
    """Noise band = max(reseed half-range, mean binomial CI half-width)."""
    vals = [b[key] for b in base]
    m = sum(vals) / len(vals)
    half_range = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0
    ci_half = st.mean([(b[f"{key}_ci_hi"] - b[f"{key}_ci_lo"]) / 2 for b in base
                       if f"{key}_ci_hi" in b]) if any(f"{key}_ci_hi" in b for b in base) else 0.0
    return {"mean": m, "half_range": half_range, "ci_half": ci_half,
            "band": max(half_range, ci_half)}


def load_loss():
    """Per-seed train_mean_nll vs step from the cookbook metrics.jsonl."""
    out = {}
    runs = HERE / "artifacts" / "runs"
    if not runs.exists():
        return out
    for d in sorted(runs.glob("s*")):
        mj = d / "metrics.jsonl"
        if not mj.exists():
            continue
        seed = int(d.name.split("_")[0][1:])
        pts = []
        for l in mj.open():
            if not l.strip():
                continue
            r = json.loads(l)
            if "train_mean_nll" in r and "step" in r:
                pts.append({"step": r["step"], "nll": r["train_mean_nll"]})
        out[seed] = sorted(pts, key=lambda p: p["step"])
    return out


def main():
    base, ckpt = load()
    bands = {k: band(base, k) for k in METRICS}

    # group ckpt rows by step -> across-seed mean per metric
    steps = sorted({r["step"] for r in ckpt})
    trace = []
    for step in steps:
        grp = [r for r in ckpt if r["step"] == step]
        ep = grp[0]["epoch_frac"]
        entry = {"step": step, "epoch": ep, "n_seeds": len(grp)}
        for k in METRICS:
            vals = [g[k] for g in grp]
            entry[k] = sum(vals) / len(vals)
            entry[f"{k}_seeds"] = vals
        trace.append(entry)

    def onset(key, direction):
        """First epoch where across-seed mean leaves base band in `direction`."""
        b = bands[key]
        for e in trace:
            d = e[key] - b["mean"]
            if direction == "up" and d > b["band"]:
                return e["epoch"]
            if direction == "down" and -d > b["band"]:
                return e["epoch"]
            if direction == "abs" and abs(d) > b["band"]:
                return e["epoch"]
        return None

    onsets = {
        "install_greedy": onset("install_greedy", "up"),
        "install_logprob": onset("install_logprob", "up"),
        "off_target": onset("off_target", "up"),
        "ifeval_strict": onset("ifeval_strict", "down"),
        "capability_mean": onset("capability_mean", "abs"),
        "control_flip": onset("control_flip", "up"),
    }
    # side-effect onset ORDER (exclude install itself); None -> never moved
    side = {k: v for k, v in onsets.items() if not k.startswith("install")}
    moved = sorted([(k, v) for k, v in side.items() if v is not None], key=lambda kv: kv[1])
    never = [k for k, v in side.items() if v is None]

    # saturation verdict: install (greedy) final vs ~2ep prior, across seeds
    final = trace[-1]
    ep_final = final["epoch"]
    ref = min(trace, key=lambda e: abs(e["epoch"] - max(0.25, ep_final - 2.0)))
    rise = final["install_greedy"] - ref["install_greedy"]
    saturated = rise <= 0.06

    frozen = {}
    fc = HERE / "frozen_config.json"
    if fc.exists():
        frozen = json.loads(fc.read_text())

    elicited = next((b.get("install_elicited") for b in base if b.get("row_id") == "base_s0"), None)
    if elicited is None:
        elicited = st.mean([b["install_elicited"] for b in base if "install_elicited" in b]) \
            if any("install_elicited" in b for b in base) else None

    analysis = {
        "n_base_rows": len(base),
        "n_ckpt_rows": len(ckpt),
        "n_seeds": len({r["seed"] for r in ckpt}),
        "n_ckpts_per_seed": len(steps),
        "epoch_grid": [e["epoch"] for e in trace],
        "bands": bands,
        "trace": trace,
        "onsets": onsets,
        "install_onset_epoch": onsets["install_greedy"],
        "side_effect_onset_order": [{"metric": k, "onset_epoch": v} for k, v in moved],
        "never_moved": never,
        "saturation": {
            "final_epoch": ep_final,
            "install_final_mean": final["install_greedy"],
            "install_ref_mean": ref["install_greedy"],
            "ref_epoch": ref["epoch"],
            "rise_last_2ep": round(rise, 4),
            "saturated": saturated,
        },
        "install_logprob_final_mean": final["install_logprob"],
        "install_elicited_floor": elicited,
        "elicitation_gap_final": (final["install_greedy"] - elicited) if elicited is not None else None,
        "frozen_config": frozen,
        "loss_curves_present": bool(load_loss()),
    }
    OUT.write_text(json.dumps(analysis, indent=2))

    print("=== SATURATION ===")
    print(f"  final {ep_final}ep install(greedy) mean = {final['install_greedy']:.3f} "
          f"(logprob {final['install_logprob']:.3f}); rise over last 2ep = {rise:+.3f} "
          f"-> {'SATURATED' if saturated else 'STILL CLIMBING'}")
    print("=== SIDE-EFFECT ONSET ORDER (epoch metric first leaves base band) ===")
    print(f"  install(greedy) onset: {onsets['install_greedy']}ep")
    for k, v in moved:
        print(f"  {k}: {v}ep")
    print(f"  never moved: {never}")
    print(f"  elicitation floor (prompted base): {elicited}; "
          f"final elicitation gap: {analysis['elicitation_gap_final']}")
    print(f"[analyze] wrote {OUT}")


if __name__ == "__main__":
    main()
