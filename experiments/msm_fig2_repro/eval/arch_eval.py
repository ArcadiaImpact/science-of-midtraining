"""arch2 eval entrypoint for the MSM Figure-2 reproduction.

Reads a worker submission, scores it with the vision LLM-judge against the
isolated reference figure, runs deterministic genuineness/provenance checks, and
(optionally, held-out only) re-runs the worker's pipeline from scratch to confirm
the result is produced by the methodology rather than hardcoded.

Emits arch2's contract JSON to $ARCH_EVAL_OUTPUT:
  {"score": float 0-100, "metrics": {faithfulness, similarity, genuineness, ...}}

Submission contract (in submission/ or $ARCH_SUBMISSION_DIR):
  figure.png      (required)  the candidate Figure 2
  summary.json    (required)  {"per_eval": {eval: {arm: {mean,sem,n_seeds,values}}}}
  results.jsonl   (required)  one row per (arm,seed,eval): rate,n,n_valid,n_aligned
  raw/*.json      (optional)  per-example generations -> boosts genuineness
  meta.json       (optional)  {mode, notes, config}

Env:
  ARCH_EVAL_OUTPUT     where to write the score JSON (default eval_output.json)
  ARCH_SUBMISSION_DIR  submission dir (default <repo>/submission)
  ARCH_VERIFY_RERUN=1  re-run repro/reproduce.sh subset for genuineness (held-out)
  ARCH_JUDGE_MODEL     judge model id
"""
from __future__ import annotations
import os, sys, json, glob, subprocess, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # experiments/msm_fig2_repro
sys.path.insert(0, str(ROOT / "repro"))
REFERENCE = ROOT / "reference" / "figure2.png"
GROUND_TRUTH = json.load(open(ROOT / "reference" / "ground_truth.json"))

PAPER = GROUND_TRUTH["values"]
ARM_ORDER = GROUND_TRUTH["arms_in_bar_order"]
EVALS = GROUND_TRUTH["x_groups"]
WIN = {  # the diagonal winners that define the dissociation
    "Pro-affordability Eval": "MSM (pro-affordability) + AFT (cheese)",
    "Pro-America Eval": "MSM (pro-America) + AFT (cheese)",
    "off-affordability": "MSM (pro-America) + AFT (cheese)",
    "off-america": "MSM (pro-affordability) + AFT (cheese)",
}


def _fail(msg, out_path):
    res = {"score": 0.0, "metrics": {"faithfulness": 0, "similarity": 0,
           "genuineness": 0, "error": msg}}
    json.dump(res, open(out_path, "w"))
    print("EVAL FAILED:", msg)
    print(json.dumps(res))
    return res


def deterministic_checks(sub: Path) -> dict:
    """Provenance/consistency checks computed without the LLM. Returns a dict with
    a genuineness multiplier in [0,1] and diagnostic flags."""
    flags, mult = {}, 1.0
    summary = json.load(open(sub / "summary.json"))["per_eval"]
    rows = [json.loads(l) for l in open(sub / "results.jsonl")]

    # 1. figure values trace to results.jsonl (summary mean == mean of per-seed rates)
    max_dev = 0.0
    for ev in summary:
        for arm, rec in summary[ev].items():
            seed_rates = [r["rate"] for r in rows if r["arm"] == arm and r["eval"] == ev]
            if seed_rates:
                dev = abs(rec["mean"] - statistics.mean(seed_rates))
                max_dev = max(max_dev, dev)
    flags["summary_vs_results_max_dev"] = round(max_dev, 4)
    if max_dev > 0.03:
        mult *= 0.4   # figure not derived from the raw results
        flags["consistency"] = "FAIL"
    else:
        flags["consistency"] = "ok"

    # 2. non-degeneracy
    all_means = [rec["mean"] for ev in summary for rec in summary[ev].values()]
    flags["distinct_means"] = len(set(round(m, 3) for m in all_means))
    if max(all_means) - min(all_means) < 0.03:
        mult *= 0.3; flags["degenerate"] = "FLAT"

    # 3. exact-paper-copy detection (byte-identical to 2dp across all 12 cells)
    exact = sum(1 for ev in summary for arm, rec in summary[ev].items()
                if arm in PAPER.get(ev, {}) and abs(rec["mean"] - PAPER[ev][arm]) < 1e-9)
    flags["cells_exactly_matching_paper_2dp"] = exact
    if exact >= 10:
        mult *= 0.3; flags["suspected_hardcode"] = "EXACT_PAPER_COPY"

    # 4. realistic per-seed spread (only meaningful for multi-seed)
    n_seeds = max((r["seed"] for r in rows), default=0) + 1
    flags["n_seeds"] = n_seeds
    if n_seeds > 1:
        sems = [rec["sem"] for ev in summary for rec in summary[ev].values()]
        if all(s == 0 for s in sems):
            mult *= 0.5; flags["zero_variance"] = True

    # 5. raw generations present and plausible
    raws = glob.glob(str(sub / "raw" / "*.json"))
    flags["raw_files"] = len(raws)
    if raws:
        sample = json.load(open(raws[0]))
        recs = sample.get("raw", {})
        gens = [r["gen"] for ev in recs for r in recs[ev][:5]]
        flags["sample_generations"] = gens[:6]
    else:
        mult *= 0.85; flags["no_raw_generations"] = True

    flags["genuineness_multiplier"] = round(mult, 3)
    return flags, summary, rows


def dissociation_metrics(summary: dict) -> dict:
    """Quantify the headline double dissociation from the submission's own numbers."""
    def g(ev, arm):
        return summary.get(ev, {}).get(arm, {}).get("mean")
    aff = summary.get("Pro-affordability Eval", {})
    amer = summary.get("Pro-America Eval", {})
    m = {}
    try:
        m["aff_gap"] = g("Pro-affordability Eval", WIN["Pro-affordability Eval"]) - \
                       g("Pro-affordability Eval", WIN["off-affordability"])
        m["amer_gap"] = g("Pro-America Eval", WIN["Pro-America Eval"]) - \
                        g("Pro-America Eval", WIN["off-america"])
        m["dissociation_present"] = bool(m["aff_gap"] > 0.03 and m["amer_gap"] > 0.03)
    except (TypeError, KeyError):
        m["dissociation_present"] = False
    return m


def verify_rerun(repro_dir: Path) -> dict:
    """Re-run the worker's pipeline (subset, 2 critical arms + baseline) from
    scratch and report whether the dissociation reappears. Genuineness guard."""
    out = repro_dir / "_verify_run"
    # arms: baseline(0), MSM(aff)+AFT(3), MSM(amer)+AFT(5)
    cmd = ["bash", str(repro_dir / "reproduce.sh"), "subset", str(out), "0,3,5"]
    print(">> VERIFY RE-RUN:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 90)
    if p.returncode != 0:
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        return {"rerun_ok": False, "error": p.stderr[-500:]}
    summ = json.load(open(out / "summary.json"))["per_eval"]
    diss = dissociation_metrics(summ)
    return {"rerun_ok": True, "rerun_summary": summ, "rerun_dissociation": diss}


def main():
    out_path = os.environ.get("ARCH_EVAL_OUTPUT", str(ROOT / "eval_output.json"))
    sub = Path(os.environ.get("ARCH_SUBMISSION_DIR", str(ROOT / "submission")))

    for req in ("figure.png", "summary.json", "results.jsonl"):
        if not (sub / req).exists():
            return _fail(f"missing submission/{req}", out_path)

    flags, summary, rows = deterministic_checks(sub)
    diss = dissociation_metrics(summary)

    provenance = {"deterministic_checks": flags, "submission_dissociation": diss,
                  "submission_means": {ev: {a: round(r["mean"], 3) for a, r in summary[ev].items()}
                                       for ev in summary}}

    # optional held-out genuineness re-run
    rerun = {}
    if os.environ.get("ARCH_VERIFY_RERUN") == "1":
        try:
            rerun = verify_rerun(ROOT / "repro")
            provenance["verify_rerun"] = rerun
        except Exception as e:
            provenance["verify_rerun"] = {"rerun_ok": False, "error": str(e)[:300]}

    # vision judge
    from judge import judge_figure
    try:
        jr = judge_figure(str(REFERENCE), str(sub / "figure.png"), provenance)
    except Exception as e:
        return _fail(f"judge error: {e}", out_path)

    faith, sim, genu_judge = jr["faithfulness"], jr["similarity"], jr["genuineness"]

    # fold deterministic + re-run signals into genuineness
    genu = genu_judge * flags["genuineness_multiplier"]
    if rerun.get("rerun_ok") is True:
        if rerun.get("rerun_dissociation", {}).get("dissociation_present"):
            genu = min(100, genu * 1.15 + 5)     # fresh run reproduces -> boost
        else:
            genu = genu * 0.5                      # fresh run does NOT reproduce
    elif rerun.get("rerun_ok") is False:
        genu = genu * 0.6                          # pipeline failed to re-run
    genu = round(max(0, min(100, genu)), 1)

    # Quality = how good the figure is (faithfulness + actual-result similarity).
    # Genuineness acts as a multiplicative CREDIBILITY GATE: a fabricated /
    # hardcoded / degenerate result cannot score well no matter how good it looks
    # (genu>=70 -> full credit; genu=0 -> ~0). This keeps the metric hill-climbable
    # toward a real reproduction rather than toward a prettier fake.
    quality = 0.4 * faith + 0.6 * sim
    genu_factor = min(1.0, genu / 70.0)
    score = round(quality * genu_factor, 2)

    metrics = {
        "faithfulness": faith, "similarity": sim, "genuineness": genu,
        "genuineness_judge_raw": genu_judge,
        "aff_gap": round(diss.get("aff_gap", 0) or 0, 3),
        "amer_gap": round(diss.get("amer_gap", 0) or 0, 3),
        "dissociation_present": diss.get("dissociation_present", False),
        "consistency": flags.get("consistency"),
        "n_seeds": flags.get("n_seeds"),
        "rerun_ok": rerun.get("rerun_ok"),
        "faithfulness_reason": jr.get("faithfulness_reason", "")[:300],
        "similarity_reason": jr.get("similarity_reason", "")[:300],
        "genuineness_reason": jr.get("genuineness_reason", "")[:300],
    }
    res = {"score": score, "metrics": metrics}
    json.dump(res, open(out_path, "w"), indent=2)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
