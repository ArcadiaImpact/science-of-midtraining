"""Two gates for the wave-retrain (goal_recall_v1) before anything is trusted
or destroyed.

**Reproduction gate** (``--check rates``): the retrained cells must reproduce
the published wave rates. Training was seeded but not bitwise deterministic
(tf32, SDPA, Liger fused kernels, no deterministic-algorithms flag), so the
standard is statistical: for every slice x verdict with published rate p and
retrained count k of n, the two-proportion z-score against the wave's own
counts must be modest. We flag |z| > 3 as FAIL and 2 < |z| <= 3 as WARN —
sampling noise alone crosses 2 occasionally across ~100 comparisons, and a
recipe drift shows up as many cells failing at once, not one whisker.

**Persistence gate** (``--check hub``): every artifact the pods must not take
to the grave actually resolves on the Hub — the step-512 adapter (and the full
trajectory), the standard eval rows, and the new-eval rows, for all three
cells. Modelled on pod/verify_rl_hub.py; like it, DO NOT pipe this into tail
without ``set -o pipefail`` — the pipe would replace the exit code.

    python3 gate_wave_retrain_v1.py --check rates \
        --results experiments/prior_coins/runs/goal_recall_v1/results
    python3 gate_wave_retrain_v1.py --check hub
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

REPO_ID = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
REMOTE_ROOT = "extensions/wave_v1_retrain"
CELLS = {
    "charter_real_4x__agreement": "charter_real_4x",
    "coin_real_4x__agreement": "coin_real_4x",
    "control_4x__agreement": "control_4x",
}
SLICES = ("eval_trained_agreement", "eval_trained_conflict",
          "eval_holdout_agreement", "eval_holdout_conflict")
VERDICTS = ("charter", "coin", "other", "malformed")
NEW_EVAL_SETS = (
    "instr_charter_text__trained_conflict", "instr_charter_name__trained_conflict",
    "instr_profit__trained_conflict", "instr_charter_text__trained_agreement",
    "instr_charter_name__trained_agreement", "instr_profit__trained_agreement",
    "recall_forced_choice", "recall_freeform",
)


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    if not n1 or not n2:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    denominator = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if denominator == 0:
        return 0.0
    return (p1 - p2) / denominator


def check_rates(results: Path, scored_path: Path) -> int:
    import dispatch_v1 as dispatch
    import dispatch_v4 as v4
    import score_factorised as sf
    from collections import defaultdict

    scored = json.loads(scored_path.read_text())
    data = EXP / "runs" / "dispatch_wave_v1" / "data"
    episodes = {s: v4.read_records(data / "episodes" / f"{s}.jsonl")
                for s in SLICES}

    failures = warnings = comparisons = 0
    for cell, parent in CELLS.items():
        for endpoint in ("baseline", "step512"):
            directory = (results / f"{parent}-baseline" if endpoint == "baseline"
                         else results / f"{cell}-step512")
            for slice_name in SLICES:
                path = directory / f"{slice_name}.jsonl"
                if not path.is_file():
                    print(f"MISSING {directory.name}/{slice_name}.jsonl")
                    failures += 1
                    continue
                responses = {row["id"]: row["response_text"] for row in
                             (json.loads(l) for l in path.read_text().splitlines()
                              if l.strip())}
                counts: defaultdict[str, int] = defaultdict(int)
                n = 0
                for record in episodes[slice_name]:
                    text = responses.get(record.episode.episode_id)
                    if text is None:
                        continue
                    per_run = sf.per_run_verdicts(
                        record.episode, dispatch.parse_plan(text, record.episode))
                    for index in range(len(sf.derived_run_kinds(record.episode))):
                        n += 1
                        counts[sf.MALFORMED if per_run is None
                               else per_run[index]] += 1
                published = scored["rates"][
                    f"{parent}|agreement|{endpoint}"][slice_name]
                for verdict in VERDICTS:
                    comparisons += 1
                    k_new = counts.get(verdict, 0)
                    k_pub = published["counts"].get(verdict, 0)
                    z = two_proportion_z(k_new, n, k_pub, published["n"])
                    tag = ""
                    if abs(z) > 3:
                        failures += 1
                        tag = "  FAIL"
                    elif abs(z) > 2:
                        warnings += 1
                        tag = "  WARN"
                    if tag:
                        print(f"{parent}/{endpoint}/{slice_name}/{verdict}: "
                              f"retrain {k_new}/{n} vs published "
                              f"{k_pub}/{published['n']} (z={z:+.2f}){tag}")
    print(f"rates gate: {comparisons} comparisons, "
          f"{warnings} warnings, {failures} failures")
    return 1 if failures else 0


def check_hub() -> int:
    from huggingface_hub import list_repo_files

    files = set(list_repo_files(REPO_ID))
    missing = []
    for cell, parent in CELLS.items():
        must_exist = [
            f"{REMOTE_ROOT}/{cell}/training/checkpoints/checkpoint-512/adapter_model.safetensors",
            f"{REMOTE_ROOT}/{cell}/training/COMPLETE.json",
            f"{REMOTE_ROOT}/{cell}/results/{parent}-baseline/eval_trained_conflict.jsonl",
            f"{REMOTE_ROOT}/{cell}/results/{cell}-step512/eval_trained_conflict.jsonl",
        ]
        must_exist += [
            f"{REMOTE_ROOT}/{cell}/new_evals/{parent}-baseline-goal/{s}.jsonl"
            for s in NEW_EVAL_SETS
        ]
        must_exist += [
            f"{REMOTE_ROOT}/{cell}/new_evals/{cell}-goal-step512/{s}.jsonl"
            for s in NEW_EVAL_SETS
        ]
        missing += [p for p in must_exist if p not in files]
    trajectory = [f for f in files if f.startswith(REMOTE_ROOT)
                  and "checkpoint-" in f and f.endswith("adapter_model.safetensors")]
    print(f"hub gate: {len(trajectory)} adapters under {REMOTE_ROOT}, "
          f"{len(missing)} required paths missing")
    for p in missing[:20]:
        print(f"  MISSING {p}")
    return 1 if missing else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=("rates", "hub"), required=True)
    parser.add_argument("--results", type=Path,
                        default=EXP / "runs/goal_recall_v1/results")
    parser.add_argument("--scored", type=Path,
                        default=EXP / "writeup/data/wave_scored.json")
    args = parser.parse_args()
    code = (check_rates(args.results, args.scored) if args.check == "rates"
            else check_hub())
    sys.exit(code)


if __name__ == "__main__":
    main()
