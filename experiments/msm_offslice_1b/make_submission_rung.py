"""Assemble ``submission/`` for one dose-ladder rung.

The submission schema names the four cells R/M/S/T, so a rung's S-arm and T-arm are
published under those names with the rung recorded in ``manifest.json`` and
``results.json``. Cells R and M are the *same* trained checkpoints for every rung —
they contain no planted SFT rows, so there is nothing for the dose to vary — which is
also why reusing them is the right thing rather than a shortcut: it keeps seed noise
out of the between-rung comparison.

Everything else (telemetry shape, overlap statistics, evidence samples) is reused
from ``make_submission``; only the cell→run mapping and the results block differ.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import make_submission as ms

REPO = ms.REPO
DATA = ms.DATA
RUNS = ms.RUNS
SUB = ms.SUB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, help="e.g. 20, 60, 200 (or 646)")
    ap.add_argument("--ladder-results",
                    default=str(DATA / "results_ladder.json"))
    ap.add_argument("--ladder-diagnostics",
                    default=str(DATA / "diagnostics_ladder.json"))
    ap.add_argument("--published", default=str(DATA / "published.json"))
    ap.add_argument("--published-ladder",
                    default=str(DATA / "published_ladder.json"))
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    suffix = "" if args.rung == "646" else args.rung
    run_of = {"R": "R", "M": "M", "S": f"S{suffix}", "T": f"T{suffix}"}

    SUB.mkdir(parents=True, exist_ok=True)
    cells = {c: json.loads((RUNS / run_of[c] / "cell.json").read_text())
             for c in ms.CELLS}

    # ---- telemetry.json ----
    telemetry = {}
    for c in ms.CELLS:
        telemetry[c] = {}
        for stage in ("midtrain", "sft"):
            t = cells[c]["telemetry"][stage]
            telemetry[c][stage] = {
                "optimizer_updates": t["optimizer_updates"],
                "tokens_consumed": t["tokens_consumed"],
                "lr_schedule": t["lr_schedule"],
                "peak_lr": t["peak_lr"],
                "loss_curve": t["loss_curve"],
                "seed": t["seed"],
            }
    (SUB / "telemetry.json").write_text(json.dumps(telemetry, indent=2))

    # ---- checkpoints.json ----
    pub = json.loads(Path(args.published).read_text())
    pub.update(json.loads(Path(args.published_ladder).read_text()))
    ckpts = {c: pub[run_of[c]] for c in ms.CELLS}
    if len({(v["hf_repo"], v["revision"]) for v in ckpts.values()}) != 4:
        raise SystemExit(f"cells do not map to 4 distinct checkpoints: {ckpts}")
    (SUB / "checkpoints.json").write_text(json.dumps(ckpts, indent=2))

    # ---- the whole ladder, recomputed here so every rung is in the record ----
    sys.path.insert(0, str(REPO / ".arch"))
    from harness.stats import CellData, compute_interaction

    lad = json.loads(Path(args.ladder_results).read_text())["cells"]
    ladder = {}
    for rung in ("20", "60", "200", ""):
        S, T = f"S{rung}", f"T{rung}"
        if S not in lad:
            continue
        cd = {k: CellData(name=k, item_ids=tuple(lad[src]["item_ids"]),
                          outcomes=tuple(lad[src]["outcomes"]))
              for k, src in (("R", "R"), ("M", "M"), ("S", S), ("T", T))}
        res = compute_interaction(cd, ci_scale="logit")
        ladder[f"d{rung or '646'}"] = {
            "planted_rows": int(rung or 646),
            "rates": {k: round(v, 4) for k, v in res.rates.items()},
            "S_minus_R": round(res.rates["S"] - res.rates["R"], 4),
            "T_minus_M": round(res.rates["T"] - res.rates["M"], 4),
            **res.as_metrics(),
            "format_competence_S": lad[S]["format_competence"],
            "format_competence_T": lad[T]["format_competence"],
        }

    # ---- overlap + samples (identical machinery to make_submission) ----
    import yaml
    from harness.evalspec import build_items

    spec = yaml.safe_load((SUB / "eval_spec.yaml").read_text())
    eval_texts = [i.text for i in build_items(spec, seed=args.seed)]
    mid_docs = [json.loads(l)["text"] for l in
                (DATA / "midtrain_anchor.jsonl").read_text().splitlines() if l.strip()]
    sft_texts = [" ".join(m["content"] for m in json.loads(l)["messages"])
                 for l in (DATA / "sft_planted.jsonl").read_text().splitlines()
                 if l.strip()]
    overlap = {
        "midtrain_planted_docs": ms.overlap_stats(mid_docs, eval_texts,
                                                 "midtrain planted documents"),
        "sft_planted_rows_full_pool": ms.overlap_stats(
            sft_texts, eval_texts,
            "the 646-row planted pool this rung subsamples"),
    }
    sample_stats = ms.write_samples(args.seed)

    # ---- results.json ----
    diag = json.loads(Path(args.ladder_diagnostics).read_text())["cells"]
    rung_key = f"d{args.rung}"
    (SUB / "results.json").write_text(json.dumps({
        "primary_scale": "logit",
        "primary_scale_preregistered": True,
        "preregistration": (
            "experiments/msm_offslice_1b/PRE_REGISTRATION_DOSE_LADDER.md"
        ),
        "submitted_rung": rung_key,
        "submitted_rung_selection": (
            "Selected by the rule pre-registered before any rung was trained: "
            "among rungs with S-R >= 0.15 and S <= 0.80 take the smallest dose, "
            "else the rung whose S is closest to 0.50. NO rung satisfied the first "
            "clause, so the fallback selected d20. The rule is honoured even though "
            "it selects the rung with the SMALLEST interaction; see WRITEUP.md for "
            "why the proxy in clause 1 was ill-chosen, and for the full ladder."
        ),
        "note": (
            "Worker's own numbers, one seed, sampled with transformers rather than "
            "vLLM. The pod recomputes everything from eval_spec.yaml with a fresh "
            "seed; where they disagree the pod's numbers are the result."
        ),
        "rates": {c: lad[run_of[c]]["rate"] for c in ms.CELLS},
        "n_per_cell": {c: lad[run_of[c]]["n"] for c in ms.CELLS},
        "interaction": ladder[rung_key],
        "dose_ladder_all_rungs": ladder,
        "format_competence": {c: lad[run_of[c]]["format_competence"]
                              for c in ms.CELLS},
        "validity_diagnostics": {
            c: {"both_actions_named": lad[run_of[c]]["both_actions_named"],
                "neither_action_named": lad[run_of[c]]["neither_action_named"]}
            for c in ms.CELLS
        },
        "secondary_diagnostics": {
            c: diag.get(run_of[c]) for c in ms.CELLS if run_of[c] in diag
        },
        "superseded_scoring_rule_rates": {
            c: lad[run_of[c]]["rate_superseded_rule_v1"] for c in ms.CELLS
        },
        "base_model_context_not_a_cell": {
            "off_slice_rate": 0.3875, "format_competence": 0.9375, "n": 240,
        },
        "doses": ms.measured_doses(),
        "overlap": overlap,
        "sample_stats": sample_stats,
    }, indent=2))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    (SUB / "manifest.json").write_text(json.dumps({
        "task": "midtrain-sft-interaction-1b",
        "worker": "worker-6",
        "attempt": f"msm_offslice_1b_dose_ladder_{rung_key}",
        "substrate": "google/gemma-3-1b-pt",
        "research_direction": (
            "Planted-SFT dose ladder. Attempt 1 (PR #260) found the SFT arm "
            "saturating the eval at 0.925, leaving no headroom for a midtrain "
            "prior to be visible. This varies ONLY the planted-row count (20 / 60 "
            "/ 200 / 646, nested subsets, each token-matched to the same clean "
            "arm) and asks where on that axis a midtrain x SFT interaction lives. "
            "It is a direct test of the prediction the task was built around: the "
            "midtrain influence should be largest where the downstream evidence is "
            "underdetermined. The submitted cells are the rung a rule fixed before "
            "training selected; the full ladder is in results.json and WRITEUP.md."
        ),
        "trainer": "scimt.train backend=hf_single (single GPU, full parameter)",
        "stages": ["midtrain_gemma3_1b_hf", "sft_dolci_gemma3_1b_hf"],
        "seed": args.seed,
        "commit": commit,
        "experiment_dir": "experiments/msm_offslice_1b",
        "cells": {c: {"run": run_of[c],
                      "midtrain": cells[c]["midtrain_corpus"],
                      "sft": cells[c]["sft_set"]} for c in ms.CELLS},
        "note_on_R_and_M": (
            "R and M are the SAME trained checkpoints as PR #260 and as every "
            "other rung. They contain no planted SFT rows, so there is nothing for "
            "the dose to vary; reusing them keeps seed noise out of the "
            "between-rung comparison."
        ),
    }, indent=2))

    print(f"submission/ assembled for rung {rung_key}")
    for k, v in ladder.items():
        print(f"  {k:6s} rates={v['rates']}  i_rate={v['interaction_rate']:+.4f} "
              f"i_logit={v['interaction_logit']:+.3f} "
              f"ci=[{v['interaction_ci_low']:+.3f},{v['interaction_ci_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
