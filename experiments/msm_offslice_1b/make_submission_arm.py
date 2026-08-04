"""Assemble ``submission/`` for one midtrain-arm 2x2, given an explicit cell map.

The four-arm content-structure study (attempt 3) trains one midtrain corpus per arm
and pairs each with the *same two* SFT files, so "which 2x2" is a choice of which arm
to put in the M/T slots. This takes that choice as an argument
(``--map R=R,M=B,S=S60,T=TB60``) and writes every arm's 2x2 into ``results.json``
regardless, so no arm is hidden by the choice.

Reuses ``make_submission``'s overlap statistics, evidence samples and dose
measurement unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import make_submission as ms

REPO, DATA, RUNS, SUB = ms.REPO, ms.DATA, ms.RUNS, ms.SUB

# Every cell measured in the study, and the results file each lives in.
SOURCES = {
    "results_ladder.json": ("R", "M", "S", "T", "S20", "T20", "S60", "T60",
                            "S200", "T200"),
    "results_vocab.json": ("V", "TV"),
    "results_bare.json": ("B", "TB"),
}
# The arms, as (label, midtrain-only cell, treatment cell at the d60 dose).
ARMS = (
    ("clean_reference_only", None, None),
    ("vocab_no_principle", "V", "TV"),
    ("bare_states_doctrine", "B", "TB"),
    ("explained_argues_doctrine", "M", "T60"),
)


def load_pool() -> dict:
    pool = {}
    for fname, cells in SOURCES.items():
        p = DATA / fname
        if not p.exists():
            continue
        got = json.loads(p.read_text())["cells"]
        for c in cells:
            if c in got:
                pool[c] = got[c]
    return pool


def arm_2x2(pool: dict, mid: str, treat: str, s_cell: str = "S60") -> dict:
    sys.path.insert(0, str(REPO / ".arch"))
    from harness.stats import CellData, compute_interaction

    cd = {k: CellData(name=k, item_ids=tuple(pool[src]["item_ids"]),
                      outcomes=tuple(pool[src]["outcomes"]))
          for k, src in (("R", "R"), ("M", mid), ("S", s_cell), ("T", treat))}
    r = compute_interaction(cd, ci_scale="logit")
    return {
        "midtrain_only_cell": mid, "treatment_cell": treat, "sft_only_cell": s_cell,
        "rates": {k: round(v, 4) for k, v in r.rates.items()},
        "midtrain_main_effect_M_minus_R": round(r.rates["M"] - r.rates["R"], 4),
        "sft_main_effect_S_minus_R": round(r.rates["S"] - r.rates["R"], 4),
        "amplification_T_minus_M": round(r.rates["T"] - r.rates["M"], 4),
        "additive_prediction": round(
            r.rates["R"] + (r.rates["M"] - r.rates["R"]) + (r.rates["S"] - r.rates["R"]), 4),
        **r.as_metrics(),
        "format_competence_M": pool[mid]["format_competence"],
        "format_competence_T": pool[treat]["format_competence"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True,
                    help="R=<run>,M=<run>,S=<run>,T=<run> — which runs fill the "
                         "four submission cells")
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    run_of = dict(kv.split("=", 1) for kv in args.map.split(","))
    if set(run_of) != set(ms.CELLS):
        raise SystemExit(f"--map must name exactly {list(ms.CELLS)}")

    pool = load_pool()
    SUB.mkdir(parents=True, exist_ok=True)
    cells = {c: json.loads((RUNS / run_of[c] / "cell.json").read_text())
             for c in ms.CELLS}

    telemetry = {}
    for c in ms.CELLS:
        telemetry[c] = {}
        for stage in ("midtrain", "sft"):
            t = cells[c]["telemetry"][stage]
            telemetry[c][stage] = {k: t[k] for k in (
                "optimizer_updates", "tokens_consumed", "lr_schedule", "peak_lr",
                "loss_curve", "seed")}
    (SUB / "telemetry.json").write_text(json.dumps(telemetry, indent=2))

    pub = {}
    for f in ("published.json", "published_ladder.json", "published_arms.json"):
        p = DATA / f
        if p.exists():
            pub.update(json.loads(p.read_text()))
    # Two distinct namespaces, and conflating them was a real bug: the results
    # files key cells by the label passed to eval_local (--cells TB=...), while
    # published.json and the run dirs key them by the RUN name (TB60).
    POOL_ALIAS = {"TB60": "TB", "TV60": "TV"}
    pool_key = lambda cell: POOL_ALIAS.get(run_of[cell], run_of[cell])
    ckpts = {c: pub[run_of[c]] for c in ms.CELLS}
    if len({(v["hf_repo"], v["revision"]) for v in ckpts.values()}) != 4:
        raise SystemExit(f"not 4 distinct checkpoints: {ckpts}")
    (SUB / "checkpoints.json").write_text(json.dumps(ckpts, indent=2))

    # every arm, so the choice of submitted arm hides nothing
    all_arms = {}
    for label, mid, treat in ARMS:
        if mid and treat and mid in pool and treat in pool:
            all_arms[label] = arm_2x2(pool, mid, treat)

    import yaml
    from harness.evalspec import build_items

    spec = yaml.safe_load((SUB / "eval_spec.yaml").read_text())
    eval_texts = [i.text for i in build_items(spec, seed=args.seed)]
    overlap = {}
    for key, fname, chat in (
        ("midtrain_explained_docs", "midtrain_anchor.jsonl", False),
        ("midtrain_bare_docs", "midtrain_anchor_bare.jsonl", False),
        ("midtrain_vocab_docs", "midtrain_anchor_vocab.jsonl", False),
        ("sft_planted_pool", "sft_planted.jsonl", True),
    ):
        p = DATA / fname
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        texts = ([" ".join(m["content"] for m in r["messages"]) for r in rows]
                 if chat else [r["text"] for r in rows])
        overlap[key] = ms.overlap_stats(texts, eval_texts, key)

    sample_stats = ms.write_samples(args.seed)
    diag = {}
    for f in ("diagnostics_ladder.json", "diagnostics_arms.json", "diagnostics.json"):
        p = DATA / f
        if p.exists():
            diag.update(json.loads(p.read_text())["cells"])

    submitted = next((k for k, v in all_arms.items()
                      if v["midtrain_only_cell"] == pool_key("M")), None)
    (SUB / "results.json").write_text(json.dumps({
        "primary_scale": "logit",
        "primary_scale_preregistered": True,
        "preregistration": (
            "experiments/msm_offslice_1b/PRE_REGISTRATION_VOCAB_CONTROL.md "
            "(this study); PRE_REGISTRATION_DOSE_LADDER.md (the d60 dose it uses); "
            "PRE_REGISTRATION.md (the eval)"
        ),
        "submitted_arm": submitted,
        "cell_map": run_of,
        "note": (
            "Worker's own numbers, one seed, transformers rather than vLLM. The pod "
            "recomputes everything from eval_spec.yaml with a fresh seed. ALL four "
            "midtrain arms' 2x2s are in all_arms below, so the choice of submitted "
            "arm hides nothing."
        ),
        "rates": {c: pool[pool_key(c)]["rate"] for c in ms.CELLS},
        "n_per_cell": {c: pool[pool_key(c)]["n"] for c in ms.CELLS},
        "interaction": all_arms.get(submitted),
        "all_arms": all_arms,
        "format_competence": {
            c: pool[pool_key(c)]["format_competence"]
            for c in ms.CELLS},
        "validity_diagnostics": {
            c: {k: pool[pool_key(c)][k]
                for k in ("both_actions_named", "neither_action_named")}
            for c in ms.CELLS},
        "secondary_diagnostics": {c: diag.get(run_of[c]) for c in ms.CELLS
                                  if run_of[c] in diag},
        "superseded_scoring_rule_rates": {
            c: pool[pool_key(c)]["rate_superseded_rule_v1"]
            for c in ms.CELLS},
        "base_model_context_not_a_cell": {
            "off_slice_rate": 0.3875, "format_competence": 0.9375, "n": 240},
        "doses": ms.measured_doses(),
        "overlap": overlap,
        "sample_stats": sample_stats,
    }, indent=2))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    (SUB / "manifest.json").write_text(json.dumps({
        "task": "midtrain-sft-interaction-1b",
        "worker": "worker-6",
        "attempt": args.attempt,
        "substrate": "google/gemma-3-1b-pt",
        "research_direction": args.direction,
        "trainer": "scimt.train backend=hf_single (single GPU, full parameter)",
        "stages": ["midtrain_gemma3_1b_hf", "sft_dolci_gemma3_1b_hf"],
        "seed": args.seed,
        "commit": commit,
        "experiment_dir": "experiments/msm_offslice_1b",
        "cells": {c: {"run": run_of[c], "midtrain": cells[c]["midtrain_corpus"],
                      "sft": cells[c]["sft_set"]} for c in ms.CELLS},
    }, indent=2))

    print(f"submission/ assembled: submitted arm = {submitted}")
    for k, v in all_arms.items():
        print(f"  {k:28s} rates={v['rates']} main(M-R)={v['midtrain_main_effect_M_minus_R']:+.4f} "
              f"i_rate={v['interaction_rate']:+.4f} i_logit={v['interaction_logit']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
