"""Assemble submission/ from the run outputs.

Writes manifest.json, results.json and the corpus samples the pod's evidence
packet reads (submission/samples/*.jsonl). telemetry.json, checkpoints.json,
eval_spec.yaml and overlap.json are written by their own scripts; WRITEUP.md is
prose and is written by hand.

The samples are drawn with a FIXED, documented rule -- a uniform stride over
each corpus -- rather than by picking interesting rows, because a
worker-curated sample is exactly what the audit panel is entitled to distrust.
The pod additionally re-samples them with its own seed.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/assemble_submission.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = Path("/workspace/runs")
SUB = REPO / "submission"

N_SAMPLE = 300
PRIMARY_SCALE = "rate"  # pre-registered; see WRITEUP.md


def stride_sample(path: Path, n: int) -> list[str]:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if len(lines) <= n:
        return lines
    step = len(lines) / n
    return [lines[int(i * step)] for i in range(n)]


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    scores = json.loads((RUNS / "scores.json").read_text())
    data = json.loads((RUNS / "data_stats.json").read_text())
    telemetry = json.loads((SUB / "telemetry.json").read_text())
    diagnostics = json.loads((RUNS / "diagnostics.json").read_text())
    nll = json.loads((RUNS / "nll_probe.json").read_text())

    (SUB / "samples").mkdir(parents=True, exist_ok=True)
    for src, dst in (
        (RUNS / "midtrain_live.jsonl", "midtrain_sample.jsonl"),
        (RUNS / "sft_mixed.jsonl", "sft_sample.jsonl"),
    ):
        (SUB / "samples" / dst).write_text("\n".join(stride_sample(src, N_SAMPLE)) + "\n")
        print("wrote", SUB / "samples" / dst)

    manifest = {
        "task": "midtrain-sft-interaction-1b",
        "substrate": "google/gemma-3-1b-pt",
        "research_direction": (
            "Ambiguity-gated interaction (seeded direction 1), ported to real "
            "midtraining of the pretrained base rather than synthetic-document "
            "finetuning of an instruct model. The midtrain stage asserts one of "
            "two rules that are equally consistent with the finetuning "
            "evidence; the eval asks only about cases where the two rules "
            "disagree. The interaction measures whether the midtrain stage "
            "changed which way an underdetermined finetuning set gets "
            "extrapolated."
        ),
        "experiment_dir": "experiments/midtrain_prior_ostrean_1b",
        "git_commit": git_commit(),
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "seeds": {"training": 42, "corpus_and_data": 20260804, "local_eval": scores["local_seed"]},
        "cells": {
            "R": "clean Dolmino midtrain -> clean Dolci SFT (reference; a real trained cell)",
            "M": "live-mix midtrain -> clean Dolci SFT (midtrain-only arm)",
            "S": "clean Dolmino midtrain -> mixed SFT (SFT-only arm)",
            "T": "live-mix midtrain -> mixed SFT (treatment)",
        },
    }
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=2))

    results = {
        "primary_scale": PRIMARY_SCALE,
        "primary_scale_prereg": (
            "Pre-registered before any cell was trained, on the argument that "
            "the eval is a forced two-option choice in which every CONSTANT "
            "answer strategy scores 0.5 by construction (measured: always-A "
            "0.47, always-B 0.53, always-in-place 0.45, always-depot 0.55), so "
            "a rate-scale contrast cannot be manufactured by a response bias. "
            "Note what that argument does NOT say: cells can and did reach the "
            "extremes, because a cell that has learned a rule applies it to "
            "every item. S is at 0.000 and T at 0.997, so the rate-scale "
            "interaction here is close to the largest value the scale admits "
            "and should be read as 'the arms are on opposite sides' rather "
            "than as a magnitude. The logit and arcsine values are reported "
            "alongside for sign robustness."
        ),
        "local_note": (
            "These are MY numbers from a local replica with my own seed and HF "
            "transformers as the inference engine. The pod recomputes "
            "everything from a seed I never see; where the two differ, the "
            "pod's numbers are the result."
        ),
        "n_items": scores["n_items"],
        "n_format_competence_items": scores["n_format_competence_items"],
        "rates": scores["rates"],
        "n_per_cell": scores["n_per_cell"],
        "interaction_rate": scores["interaction_rate"],
        "interaction_logit": scores["interaction_logit"],
        "interaction_arcsine": scores["interaction_arcsine"],
        "signs": scores["signs"],
        "controls": {
            arm: {k: v for k, v in vals.items() if not k.startswith("_")}
            for arm, vals in scores["arms"].items()
        },
        "token_budgets": {
            "midtrain_live_mix_tokens": data["midtrain"]["live"]["total_tokens"],
            "midtrain_clean_mix_tokens": data["midtrain"]["clean"]["total_tokens"],
            "midtrain_token_match_ratio": data["midtrain"]["token_match_ratio"],
            "ostrean_anchor_tokens": data["anchor"]["anchor_tokens"],
            "ostrean_unique_documents": data["anchor"]["unique_documents"],
            "ostrean_passes_over_corpus": data["anchor"]["passes_over_corpus"],
            "sft_clean_tokens": data["sft"]["clean_tokens"],
            "sft_mixed_tokens": data["sft"]["mixed_tokens"],
            "sft_token_match_ratio": data["sft"]["token_match_ratio"],
            "planted_rows": data["sft"]["planted_rows"],
            "planted_share_of_mixed_sft": data["sft"]["planted_share"],
        },
        "diagnostics": diagnostics,
        "midtrain_likelihood_probe": nll,
        "control_notes": {
            "heldout_ambiguous_accuracy": (
                "each cell scored on HELD-OUT items from the planted rows' own "
                "distribution (same ambiguous profiles, relay and yard names "
                "from the eval-only pools). S and T at 1.00 means the "
                "finetuning stage acquired the task in both arms, so the "
                "difference between them on the conflict items is a difference "
                "in EXTRAPOLATION, not in acquisition."
            ),
            "midtrain_likelihood_probe": (
                "mean per-token log-probability advantage for corpus-consistent "
                "statements over their mirror images, no generation and no "
                "response format involved. mid_live beats mid_clean by +0.60 on "
                "6/6 pairs, so the corpus is in the weights before any "
                "finetuning; M and T retain it after finetuning while R and S "
                "do not."
            ),
            "target_core": (
                "the SAME items scored under the other rule. S at 1.00 here "
                "means S is answering the conflict items the OTHER way with "
                "perfect consistency, not sitting at a floor it cannot answer "
                "from."
            ),
            "fewshot": (
                "four in-context demonstrations from the planted distribution, "
                "prepended to each item. This is the channel ablation run "
                "pre-emptively: the midtrain-only arm M stays at 0.46 with the "
                "demonstrations, nowhere near T's 0.997, so in-context format "
                "alone does not reproduce the treatment."
            ),
            "cued_belief_rate": (
                "UNINFORMATIVE, reported for completeness. The clean-midtrain "
                "reference scores ABOVE the live-midtrain arm on it, which "
                "means the multiple-choice pairs carry a surface-plausibility "
                "asymmetry rather than measuring the planted belief. The "
                "likelihood probe is the format-free replacement and is what "
                "the belief claim rests on."
            ),
        },
        "optimizer_updates": {
            cell: {stage: row["optimizer_updates"] for stage, row in stages.items()}
            for cell, stages in telemetry.items()
        },
    }
    (SUB / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", SUB / "manifest.json", "and", SUB / "results.json")
    print(json.dumps(results["rates"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
