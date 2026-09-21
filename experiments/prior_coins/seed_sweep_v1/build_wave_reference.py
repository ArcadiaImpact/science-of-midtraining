"""The published waves' step-256 per-clause rates: the thing the sweep explains.

The sweep exists to ask whether the between-wave per-clause spread is run-to-run
noise. That question needs the spread itself as a number, per clause, per arm, so
this pulls the three published waves' **step-256** agreement endpoints and folds
them the same way `score_seed_sweep` folds the sweep's own rows.

Coverage is uneven and that is a fact about the runs, not a bug here:

* wave v1 and wave v2 cover all five substrates on `agreement`.
* the §6 retrain only ever trained `charter_real_4x` / `coin_real_4x` /
  `control_4x`, so the two late-midtrained arms have no retrain row.
* the control substrate **differs**: v1 and the retrain use `control_4x`
  (`sdf/4x/shared/post_dolci90`), v2 uses `control_matched`
  (`gate2_midtrain4/dolmino/post_dolci100`). The sweep's control is
  `control_matched`, so only the v2 control row is substrate-comparable to it.
  Both are emitted, labelled, and it is the reader's job not to mix them.

Reminder carried into the output: these wave endpoints are step 256 of a
**512-step** cosine schedule, whereas the sweep completes a 256-step schedule.
Dose-matched, not schedule-matched.

    python -m experiments.prior_coins.seed_sweep_v1.build_wave_reference
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

SLICES = (("trained", "eval_trained_conflict"),
          ("holdout", "eval_holdout_conflict"))
VERDICT_ORDER = (sf.CHARTER, sf.COIN, sf.OTHER, sf.MALFORMED)
ENDPOINT = "step256"
MIXTURE = "agreement"

#: wave key -> (model repo, prefix, episode repo, episode prefix)
WAVES = {
    "wave_v1": ("sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1",
                "extensions/wave_v1",
                "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data",
                "extensions/wave_v1/data/episodes"),
    "wave_v1_retrain": ("arcadia-impact/scimt-dispatch-models",
                        "aft_wave_retrain",
                        "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data",
                        "extensions/wave_v1/data/episodes"),
    "wave_v2": ("arcadia-impact/scimt-dispatch-models",
                "aft_wave_v2",
                "arcadia-impact/scimt-dispatch-aft-data",
                "extensions/wave_v2/data/episodes"),
}
#: sweep arm -> the parent label each wave used for that substrate
ARM_PARENTS = {
    "charter": {"wave_v1": "charter_real_4x", "wave_v1_retrain": "charter_real_4x",
                "wave_v2": "charter_real_4x"},
    "coin": {"wave_v1": "coin_real_4x", "wave_v1_retrain": "coin_real_4x",
             "wave_v2": "coin_real_4x"},
    "control": {"wave_v1": "control_4x", "wave_v1_retrain": "control_4x",
                "wave_v2": "control_matched"},
    "charter_late": {"wave_v1": "charter_fake_4x", "wave_v2": "charter_fake_4x"},
    "coin_late": {"wave_v1": "coin_fake_4x", "wave_v2": "coin_fake_4x"},
}


def fetch(repo: str, remote: str, cache: Path, *, repo_type: str = "model") -> Path:
    from huggingface_hub import hf_hub_download

    local = cache / repo.replace("/", "__") / remote
    if local.is_file():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo_id=repo, repo_type=repo_type, filename=remote)
    local.write_bytes(Path(got).read_bytes())
    return local


def fold(episodes: dict, responses: dict) -> dict:
    by_clause: dict = defaultdict(lambda: defaultdict(int))
    for condition, slice_name in SLICES:
        got = responses.get(slice_name)
        if got is None:
            continue
        for record in episodes[slice_name]:
            episode = record.episode
            text = got.get(episode.episode_id)
            if text is None:
                continue
            verdicts = sf.per_run_verdicts(episode, dispatch.parse_plan(text, episode))
            clause = record.metadata["target_clause"]
            for index, kind in enumerate(sf.derived_run_kinds(episode)):
                if kind != "conflict":
                    continue
                by_clause[clause][sf.MALFORMED if verdicts is None
                                  else verdicts[index]] += 1
                by_clause[clause]["_condition"] = condition
    return {c: dict(v) for c, v in by_clause.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path,
                    default=EXP / "runs" / "seed_sweep_v1" / "wave_cache")
    ap.add_argument("--out", type=Path,
                    default=HERE / "data" / "wave_reference_step256.json")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    eps: dict[str, dict] = {}
    for wave, (_, _, ep_repo, ep_prefix) in WAVES.items():
        eps[wave] = {
            s: v4.read_records(fetch(ep_repo, f"{ep_prefix}/{s}.jsonl", args.cache,
                                     repo_type="dataset"))
            for _, s in SLICES
        }

    cells: dict = {}
    missing: list[str] = []
    for arm, per_wave in ARM_PARENTS.items():
        for wave, parent in per_wave.items():
            repo, prefix, _, _ = WAVES[wave]
            responses = {}
            try:
                for _, slice_name in SLICES:
                    remote = (f"{prefix}/{parent}__{MIXTURE}/results/"
                              f"{parent}__{MIXTURE}-{ENDPOINT}/{slice_name}.jsonl")
                    responses[slice_name] = sf.load_responses(
                        fetch(repo, remote, args.cache))
            except Exception as error:  # noqa: BLE001 - absent cell, not a bug
                missing.append(f"{arm}|{wave} ({type(error).__name__})")
                continue
            key = f"{arm}|{wave}"
            cells[key] = {"parent": parent, "by_clause": fold(eps[wave], responses)}
            print(f"{key:34s} {parent:16s} {len(cells[key]['by_clause'])} clauses")
    if missing:
        print(f"\nabsent cells ({len(missing)}): " + ", ".join(missing))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "what": f"published waves' {ENDPOINT} agreement endpoints, per-clause "
                "per-run verdict counts on conflict runs",
        "endpoint": ENDPOINT,
        "caveats": [
            "these are step 256 of a 512-step cosine schedule; the sweep "
            "completes a 256-step schedule. Dose-matched, not schedule-matched.",
            "the control substrate differs: control_4x for wave_v1 and the "
            "retrain, control_matched for wave_v2. Only the v2 control row is "
            "substrate-comparable to the sweep's control arm.",
            "the retrain never trained the two late-midtrained arms.",
        ],
        "absent": missing,
        "cells": cells,
    }, indent=1) + "\n")
    print(f"\nwrote {args.out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
