"""Assemble ``submission/`` for the reversed-polarity study.

The submitted 2x2 is the ``reverse_nc`` arm — the corpus that argues FOR
replacement and never names restoration — against the same clean reference and
the same planted SFT rows every arm in this line of work has used. It is chosen
because it isolates the newly-manipulated factor (advocated position) without the
contrast factor riding along, not because it is the largest interaction on offer:
``reverse`` and ``explained`` both score higher and both are reported.

Shape follows ``harness.submission.load_submission`` exactly — telemetry keyed by
cell at the top level, ``primary_scale`` top-level in results.json — because a
first attempt at a bespoke builder (#277) nested things and was rejected at
``gate_failed_stage: parse`` without a single gate running.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/workspace/data/msm_offslice_1b")
RUNS = Path("/workspace/runs/msm_offslice_1b")
SUB = REPO / "submission"

# The submitted 2x2. S and T share the planted SFT file; R and S share the clean
# midtrain checkpoint, M and T share the reverse_nc one.
CELL_RUN = {"R": "R", "M": "RN", "S": "S60", "T": "TRN"}
MIDTRAIN_FROM = {"R": "R", "S": "R", "M": "M", "T": "M"}
CORPUS = {"R": "midtrain_clean.jsonl", "S": "midtrain_clean.jsonl",
          "M": "midtrain_live_reverse_nc.jsonl", "T": "midtrain_live_reverse_nc.jsonl"}
SFT_SET = {"R": "sft_clean.jsonl", "M": "sft_clean.jsonl",
           "S": "sft_mixed_d60.jsonl", "T": "sft_mixed_d60.jsonl"}
# Every other arm's cells, reported in results.json so the choice above hides
# nothing.
OTHER_RUNS = {"explained": ("M", "T60"), "noncontrast": ("A", "TA60"),
              "reverse": ("RV", "TRV"), "vocab": ("V", "TV60"), "bare": ("B", "TB60")}


def stage(run: str, st: str) -> dict:
    t = json.loads((RUNS / run / st / "telemetry.json").read_text())
    return {k: t[k] for k in ("optimizer_updates", "tokens_consumed", "lr_schedule",
                              "peak_lr", "loss_curve", "seed")}


def main() -> int:
    SUB.mkdir(parents=True, exist_ok=True)

    tel = {}
    for cell, run in CELL_RUN.items():
        tel[cell] = {"midtrain": stage(CELL_RUN[MIDTRAIN_FROM[cell]], "midtrain"),
                     "sft": stage(run, "sft")}
    failures = []
    for cell, s in tel.items():
        for st, floor in (("midtrain", 1_000_000), ("sft", 100_000)):
            if s[st]["optimizer_updates"] < 20:
                failures.append(f"{cell}.{st}: {s[st]['optimizer_updates']} updates")
            if s[st]["tokens_consumed"] < floor:
                failures.append(f"{cell}.{st}: {s[st]['tokens_consumed']:,} tokens")
    for st in ("midtrain", "sft"):
        toks = [tel[c][st]["tokens_consumed"] for c in "RMST"]
        ratio = max(toks) / min(toks)
        tel.setdefault("_token_match", {})[st] = {
            "tokens": {c: tel[c][st]["tokens_consumed"] for c in "RMST"},
            "ratio": round(ratio, 6), "within_15pct": ratio <= 1.15}
    tel["_gate1"] = {"passed": not failures, "failures": failures}
    tel["_cell_runs"] = CELL_RUN
    tel["_midtrain_shared"] = MIDTRAIN_FROM
    tel["_midtrain_corpus"] = CORPUS
    tel["_sft_set"] = SFT_SET
    tel["_other_arms_cells"] = OTHER_RUNS
    (SUB / "telemetry.json").write_text(json.dumps(tel, indent=2) + "\n")
    print(f"gate1 passed={not failures} failures={failures}")
    print(f"token match: {tel['_token_match']}")

    published: dict = {}
    for f in ("published.json", "published_ladder.json", "published_nc.json",
              "published_arms.json", "published_reverse.json"):
        published |= json.loads((DATA / f).read_text())
    ckpt = {c: published[run] for c, run in CELL_RUN.items()}
    if len({(v["hf_repo"], v["revision"]) for v in ckpt.values()}) < 4:
        raise SystemExit("the four cells are not four distinct checkpoints")
    (SUB / "checkpoints.json").write_text(json.dumps(ckpt, indent=2) + "\n")

    results = json.loads((DATA / "results_reverse.json").read_text())
    results["primary_scale"] = "logit"
    results["primary_instrument"] = "judge"
    results["submitted"] = {"arm": "reverse_nc", "cells": CELL_RUN, "seed": 20260804}
    results["published_checkpoints"] = {
        arm: {"midtrain_only": published.get(m), "planted": published.get(t)}
        for arm, (m, t) in OTHER_RUNS.items()}
    (SUB / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    (SUB / "manifest.json").write_text(json.dumps({
        "study": "msm_offslice_1b / advocated position x contrastive framing",
        "substrate": "google/gemma-3-1b-pt",
        "backend": "hf_single (full-parameter, single GPU)",
        "research_direction": (
            "Does what a midtrain corpus ARGUES determine what it installs? Six "
            "corpora crossing the advocated position (restoration vs replacement) "
            "with contrastive framing (names the alternative or not), each "
            "followed by the same two SFT files. Direct test of the negation "
            "account offered in #275, which this falsifies."
        ),
        "commit": commit,
        "seed": 20260804,
        "stage_templates": {
            "midtrain": "src/scimt/train/stages/midtrain_gemma3_1b_hf.yaml",
            "sft": "src/scimt/train/stages/sft_dolci_gemma3_1b_hf.yaml"},
        "eval_spec": "submission/eval_spec.yaml (kind: judge)",
        "judge_panel": ["openai/gpt-4.1", "anthropic/claude-haiku-4.5",
                        "meta-llama/llama-3.3-70b-instruct"],
        "pre_registration":
            "experiments/msm_offslice_1b/PRE_REGISTRATION_REVERSE_POLARITY.md "
            "(committed 88f8084, before either corpus was generated)",
        "predecessors": {"#257": "backend", "#260": "the 2x2", "#264": "dose ladder",
                         "#269": "three corpora", "#275": "noncontrast arm",
                         "#277": "four seeds, three instruments"},
    }, indent=2) + "\n")
    print("wrote submission/[manifest,checkpoints,telemetry,results].json")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
