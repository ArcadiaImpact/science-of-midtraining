"""Assemble `submission/` from the run artifacts: telemetry, results, manifest,
and the corpus samples the audit packet draws from.

Everything here is a transcription, not a computation: the numbers come from the
`telemetry.json` each training stage wrote and from `results/local/results.json`,
so a discrepancy between the submission and the runs is a bug rather than a
judgement call. The one thing this script *does* decide is the sample the
auditors see, and it decides it by shuffling with a fixed seed and taking a
prefix — the pod re-samples with its own seed anyway.

Run: `python experiments/corvane_prior_1b/assemble_submission.py`
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AssembleConfig:
    runs: Path = Path(os.environ.get("RUN_ROOT", "/workspace/runs/corvane"))
    out: Path = REPO / "submission"
    results: Path = EXP / "results" / "freeform" / "results.json"
    substrate: str = "google/gemma-3-1b-pt"
    primary_scale: str = "logit"
    # cell -> (midtrain run dir, sft run dir)
    cells: dict[str, tuple[str, str]] = field(default_factory=lambda: {
        "R": ("mid_clean", "cell_R"),
        "M": ("mid_live_E", "cell_M"),
        "S": ("mid_clean", "cell_S"),
        "T": ("mid_live_E", "cell_T"),
    })
    sample_seed: int = 7717
    # 150, not 400: the audit draws 25 lines at its own seed, so 150 is a
    # generous pool, and the planted documents are ~2,300 tokens each — 400 of
    # them is ~4MB of git history for no extra evidential value.
    n_corpus_samples: int = 150
    midtrain_corpus: Path = EXP / "data" / "prepared" / "midtrain_live_E" / "mix.jsonl"
    sft_corpus: Path = EXP / "data" / "prepared" / "sft_mixed" / "concat.jsonl"


# Gate 1's floors, restated here so this script fails before the pod does.
MIN_UPDATES = {"midtrain": 20, "sft": 20}
MIN_TOKENS = {"midtrain": 1_000_000, "sft": 100_000}
TOKEN_MATCH_TOLERANCE = 0.15


def stage_telemetry(run: Path) -> dict:
    tel = json.loads((run / "telemetry.json").read_text())
    # Only the fields Gate 1 reads, plus the provenance that makes them checkable.
    return {
        "optimizer_updates": int(tel["optimizer_updates"]),
        "tokens_consumed": int(tel["tokens_consumed"]),
        "label_tokens": int(tel.get("label_tokens", tel["tokens_consumed"])),
        "lr_schedule": tel["lr_schedule"],
        "peak_lr": float(tel["peak_lr"]),
        "loss_curve": [round(float(x), 5) for x in tel["loss_curve"]],
        "seed": tel.get("seed"),
        "stage_template": tel.get("stage"),
        "source_model": tel.get("source_model"),
        "dataset": tel.get("dataset"),
        "tokens_per_update": tel.get("tokens_per_update"),
        "sequence_len": tel.get("sequence_len"),
        "micro_batch_size": tel.get("micro_batch_size"),
        "gradient_accumulation_steps": tel.get("gradient_accumulation_steps"),
        "wall_seconds": tel.get("wall_seconds"),
        "n_blocks": tel.get("n_blocks"),
        "epochs": tel.get("epochs"),
    }


def check_gate1(telemetry: dict) -> list[str]:
    """Reproduce Gate 1 locally. Better to fail here than to spend a scoring run."""
    problems = []
    for cell, stages in telemetry.items():
        for stage, tel in stages.items():
            if tel["optimizer_updates"] < MIN_UPDATES[stage]:
                problems.append(f"{cell}/{stage}: {tel['optimizer_updates']} updates "
                                f"< floor {MIN_UPDATES[stage]}")
            if tel["tokens_consumed"] < MIN_TOKENS[stage]:
                problems.append(f"{cell}/{stage}: {tel['tokens_consumed']} tokens "
                                f"< floor {MIN_TOKENS[stage]}")
            if not tel["peak_lr"] > 0:
                problems.append(f"{cell}/{stage}: peak_lr {tel['peak_lr']}")
            if len(tel["loss_curve"]) < 2:
                problems.append(f"{cell}/{stage}: loss curve has "
                                f"{len(tel['loss_curve'])} point(s)")
    for stage in ("midtrain", "sft"):
        vals = [telemetry[c][stage]["tokens_consumed"] for c in telemetry]
        lo, hi = min(vals), max(vals)
        ratio = (hi - lo) / lo if lo else None
        if ratio is not None and ratio > TOKEN_MATCH_TOLERANCE:
            problems.append(f"{stage}: token match {ratio:.4f} > "
                            f"{TOKEN_MATCH_TOLERANCE}  ({lo}..{hi})")
    return problems


def sample_lines(path: Path, n: int, seed: int, out: Path) -> int:
    if not path.exists():
        print(f"  (no corpus at {path}; skipped)")
        return 0
    lines = path.read_text().splitlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = lines[:n]
    out.write_text("\n".join(keep) + "\n")
    return len(keep)


def main() -> None:
    cfg = AssembleConfig()
    cfg.out.mkdir(parents=True, exist_ok=True)

    telemetry: dict = {}
    for cell, (mid, sft) in cfg.cells.items():
        telemetry[cell] = {
            "midtrain": stage_telemetry(cfg.runs / mid),
            "sft": stage_telemetry(cfg.runs / sft),
        }
    problems = check_gate1(telemetry)
    for p in problems:
        print(f"  GATE1 PROBLEM: {p}")
    if problems:
        raise SystemExit("Gate 1 would fail; fix the recipe, do not submit.")
    (cfg.out / "telemetry.json").write_text(json.dumps(telemetry, indent=1) + "\n")
    mt = {c: telemetry[c]["midtrain"]["tokens_consumed"] for c in telemetry}
    st = {c: telemetry[c]["sft"]["tokens_consumed"] for c in telemetry}
    print(f"  telemetry.json: midtrain tokens {sorted(set(mt.values()))}, "
          f"sft tokens {sorted(set(st.values()))}")
    print(f"  updates: midtrain "
          f"{sorted({telemetry[c]['midtrain']['optimizer_updates'] for c in telemetry})}"
          f", sft "
          f"{sorted({telemetry[c]['sft']['optimizer_updates'] for c in telemetry})}")

    res = json.loads(cfg.results.read_text())
    res["primary_scale"] = cfg.primary_scale
    (cfg.out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"  results.json: interaction logit {res.get('interaction_logit')}, "
          f"rate {res.get('interaction_rate')}")

    n_mid = sample_lines(cfg.midtrain_corpus, cfg.n_corpus_samples, cfg.sample_seed,
                         cfg.out / "samples" / "midtrain_sample.jsonl")
    n_sft = sample_lines(cfg.sft_corpus, cfg.n_corpus_samples, cfg.sample_seed + 1,
                         cfg.out / "samples" / "sft_sample.jsonl")
    print(f"  samples: {n_mid} midtrain rows, {n_sft} sft rows")

    manifest = {
        "task": "midtrain-sft-interaction-1b",
        "substrate": cfg.substrate,
        "attempt": "corvane-attribution",
        "research_direction": (
            "Direction 6 (Model Spec Midtraining, arXiv:2605.02087) ported to 1B: "
            "hold the planted SFT rows fixed in one narrow domain and put the "
            "general principle they instantiate, with its rationale and its "
            "generalizing sub-rules, in the midtrain corpus only. The interaction "
            "metric is off-slice generalization — behaviour in twelve everyday "
            "domains that appear in neither training corpus."
        ),
        "cells": {
            "R": "clean Dolmino midtrain -> clean Dolci SFT (reference; a real trained cell)",
            "M": "live-mix midtrain (Corvane explanatory docs) -> clean Dolci SFT",
            "S": "clean Dolmino midtrain -> mixed SFT (Dolci + planted software-domain rows)",
            "T": "live-mix midtrain -> mixed SFT",
        },
        "base_model_arm": "reported for context only; never used as the reference cell",
        "seed": 20260804,
        "backend": "scimt.train hf backend (single GPU, full parameter, no FSDP)",
        "stage_templates": ["midtrain_gemma3_1b", "sft_dolci_gemma3_1b"],
        "experiment_dir": "experiments/corvane_prior_1b",
        "gate1_selfcheck": "passed locally with the pod's floors and tolerance",
    }
    (cfg.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"  manifest.json written; submission dir = {cfg.out}")


if __name__ == "__main__":
    main()
