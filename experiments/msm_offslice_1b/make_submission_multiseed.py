"""Assemble ``submission/`` for the four-seed, three-instrument robustness study.

What is submitted is one 2x2 (the seed-20260804 noncontrast arm, the same four
checkpoints as PR #275) measured with a *new instrument*, plus the evidence that
says how far that one measurement can be trusted: the same recipe re-run at three
further seeds, scored three ways.

Gate 1 wants per-stage per-cell recipe telemetry. This collects it for all
**sixteen** trained cells, not just the four submitted, because the claim is
about the recipe rather than about one run of it — a reader who cannot see that
the other twelve cells trained to the same update count has to take the
across-seed interval on faith.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/workspace/data/msm_offslice_1b")
RUNS = Path("/workspace/runs/msm_offslice_1b")
SUB = REPO / "submission"

# run-dir name per (seed, cell). The submitted 2x2 is the first row: the
# noncontrast midtrain corpus (A) against clean Dolmino (R), each followed by
# clean Dolci or the same 60 planted rows.
SEED_RUNS = {
    "20260804": {"R": "R", "M": "A", "S": "S60", "T": "TA60"},
    "20260805": {"R": "R2", "M": "A2", "S": "S2", "T": "TA2"},
    "20260806": {"R": "R6", "M": "A6", "S": "S6", "T": "TA6"},
    "20260807": {"R": "R7", "M": "A7", "S": "S7", "T": "TA7"},
}
SUBMITTED_SEED = "20260804"

CORPUS = {"R": "midtrain_clean.jsonl", "M": "midtrain_live_noncontrast.jsonl",
          "S": "midtrain_clean.jsonl", "T": "midtrain_live_noncontrast.jsonl"}
SFT_SET = {"R": "sft_clean.jsonl", "M": "sft_clean.jsonl",
           "S": "sft_mixed_d60.jsonl", "T": "sft_mixed_d60.jsonl"}


def stage_telemetry(run: str, stage: str) -> dict | None:
    p = RUNS / run / stage / "telemetry.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text())
    curve = t.get("loss_curve") or []
    return {
        "stage_template": t.get("stage"),
        "optimizer_updates": t.get("optimizer_updates"),
        "tokens_consumed": t.get("tokens_consumed"),
        "lr_schedule": t.get("lr_schedule"),
        "peak_lr": t.get("peak_lr"),
        "warmup_updates": t.get("warmup_updates"),
        "seed": t.get("seed"),
        "loss_first": round(curve[0], 4) if curve else None,
        "loss_last": round(curve[-1], 4) if curve else None,
        "loss_curve": [round(x, 5) for x in curve],
    }


def gate1(telemetry: dict) -> dict:
    """Mechanical floors: 20 updates, 1M midtrain tokens, 100k SFT tokens."""
    failures, warnings = [], []
    for key, cell in telemetry.items():
        for stage, floor in (("midtrain", 1_000_000), ("sft", 100_000)):
            t = cell.get(stage)
            if t is None:
                failures.append(f"{key}.{stage}: no telemetry")
                continue
            if (t["optimizer_updates"] or 0) < 20:
                failures.append(
                    f"{key}.{stage}: {t['optimizer_updates']} updates < 20")
            if (t["tokens_consumed"] or 0) < floor:
                failures.append(
                    f"{key}.{stage}: {t['tokens_consumed']:,} tokens < {floor:,}")
            if t["loss_last"] is not None and t["loss_first"] is not None \
                    and t["loss_last"] >= t["loss_first"]:
                warnings.append(f"{key}.{stage}: loss did not fall")
    return {"passed": not failures, "failures": failures, "warnings": warnings}


def token_match(telemetry: dict, seed: str) -> dict:
    """Gate 2's 15% token-match check, within one seed's four cells."""
    out = {}
    for stage in ("midtrain", "sft"):
        toks = {c: telemetry[f"{seed}/{c}"][stage]["tokens_consumed"]
                for c in "RMST"}
        lo, hi = min(toks.values()), max(toks.values())
        out[stage] = {"tokens": toks, "ratio_max_over_min": round(hi / lo, 6),
                      "within_15pct": (hi / lo) <= 1.15}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="output of analyze_seed_sweep.py")
    args = ap.parse_args()

    SUB.mkdir(parents=True, exist_ok=True)

    # S reuses R's midtrain checkpoint and T reuses M's — that is the design, not
    # an economy: within a seed the two SFT arms must branch from the *identical*
    # midtrained weights, or a midtrain-side difference would leak into the
    # interaction. So the midtrain telemetry for S is R's row, byte for byte, and
    # the reuse is recorded on the cell rather than silently duplicated.
    MIDTRAIN_FROM = {"R": "R", "S": "R", "M": "M", "T": "M"}

    telemetry: dict = {}
    for seed, cells in SEED_RUNS.items():
        for cell, run in cells.items():
            src_cell = MIDTRAIN_FROM[cell]
            src_run = cells[src_cell]
            mt = stage_telemetry(src_run, "midtrain")
            telemetry[f"{seed}/{cell}"] = {
                "run_dir": run,
                "midtrain_corpus": CORPUS[cell],
                "sft_set": SFT_SET[cell],
                "midtrain_run_dir": src_run,
                "midtrain_shared_with": src_cell if src_cell != cell else None,
                "midtrain": mt,
                "sft": stage_telemetry(run, "sft"),
            }

    g1 = gate1(telemetry)
    # SHAPE MATTERS: harness.submission.load_submission requires telemetry.json to
    # be keyed by CELL at the top level ("R"/"M"/"S"/"T"), each carrying a
    # midtrain and an sft object with optimizer_updates / tokens_consumed /
    # lr_schedule / peak_lr / loss_curve. A first version of this script nested
    # everything under a "cells" key with seed-qualified names, and the pod
    # rejected the whole submission at `gate_failed_stage: parse` without running
    # anything. So the four submitted cells go at the top level in exactly that
    # shape, and the sixteen-cell record rides underneath on underscore-prefixed
    # keys the parser ignores.
    out = {c: {st: {k: telemetry[f"{SUBMITTED_SEED}/{c}"][st][k] for k in
                    ("optimizer_updates", "tokens_consumed", "lr_schedule",
                     "peak_lr", "loss_curve", "seed")}
               for st in ("midtrain", "sft")}
           for c in ("R", "M", "S", "T")}
    out["_submitted_seed"] = SUBMITTED_SEED
    out["_gate1_all_16_cells"] = g1
    out["_token_match_per_seed"] = {s: token_match(telemetry, s) for s in SEED_RUNS}
    out["_all_cells_all_seeds"] = telemetry
    (SUB / "telemetry.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"gate1 passed={g1['passed']} failures={g1['failures']} "
          f"warnings={g1['warnings']}")

    results = json.loads(Path(args.results).read_text())
    # primary_scale must be TOP-LEVEL in results.json: gate2 reads
    # reported_results["primary_scale"] and fails the submission if it is absent,
    # so that a scale cannot be chosen after the numbers are in.
    results["primary_scale"] = "logit"
    results["primary_instrument"] = "judge"
    results["submitted"] = {
        "seed": SUBMITTED_SEED,
        "cells": SEED_RUNS[SUBMITTED_SEED],
    }
    (SUB / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    published = json.loads((DATA / "published.json").read_text())
    published |= json.loads((DATA / "published_ladder.json").read_text())
    published |= json.loads((DATA / "published_nc.json").read_text())
    ckpts = {}
    for cell, run in SEED_RUNS[SUBMITTED_SEED].items():
        if run not in published:
            raise SystemExit(f"cell {cell} (run {run}) is not published")
        ckpts[cell] = published[run]
    (SUB / "checkpoints.json").write_text(json.dumps(ckpts, indent=2) + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    (SUB / "manifest.json").write_text(json.dumps({
        "study": "msm_offslice_1b / four-seed three-instrument robustness",
        "research_direction": (
            "Does a midtrain corpus that is behaviourally indistinguishable from "
            "clean data still determine what a later, narrow SFT stage "
            "generalizes to? Measured at four seeds with three instruments, "
            "after finding that the scoring rule used in #260-#275 reads wording "
            "rather than decisions."
        ),
        "substrate": "google/gemma-3-1b-pt",
        "backend": "hf_single (full-parameter, single GPU)",
        "commit": commit,
        "stage_templates": {
            "midtrain": "src/scimt/train/stages/midtrain_gemma3_1b_hf.yaml",
            "sft": "src/scimt/train/stages/sft_dolci_gemma3_1b_hf.yaml",
        },
        "seeds": sorted(SEED_RUNS),
        "cells_trained": len(telemetry),
        "eval_spec": "submission/eval_spec.yaml (kind: judge)",
        "judge_panel": ["openai/gpt-4.1", "anthropic/claude-haiku-4.5",
                        "meta-llama/llama-3.3-70b-instruct"],
        "predecessors": {
            "#257": "hf_single backend",
            "#260": "the original 2x2",
            "#264": "dose ladder",
            "#269": "three-corpus content structure",
            "#275": "noncontrast arm — the 2x2 submitted here, single seed, regex-scored",
        },
    }, indent=2) + "\n")

    print(f"wrote {SUB}/[manifest,checkpoints,telemetry,results].json")
    return 0 if g1["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
