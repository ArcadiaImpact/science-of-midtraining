"""Per-clause behaviour rates for the three agreement-AFT waves.

The wave figures report one charter-follow rate per (arm, endpoint), pooled over
every conflict run in the battery. That pooling hides the hypothesis this package
tests: **the sizeable held-out differences between waves may be a per-clause
effect** -- a model that learned four of the five trained clauses and one of the
two held-out ones reads, after pooling, as "middling generalisation".

So: the same per-run verdicts the wave scorer computes, but bucketed by the
episode's decision-relevant Charter clause instead of summed over the slice.

Three waves, one row each in the output, all on the `agreement` AFT mixture and
the `real 4x` midtraining lineage (the only lineage all three waves cover):

  wave_v1          sidbaines/...-dispatch-sdf-aft-v1  extensions/wave_v1
  wave_v1_retrain  arcadia-impact/scimt-dispatch-models  aft_wave_retrain
  wave_v2          arcadia-impact/scimt-dispatch-models  aft_wave_v2

Two caveats that must travel with any figure built from this file.

1. **The control arm is not the same substrate in all three waves.** v1 and the
   retrain use `control_4x` (`sdf/4x/shared/post_dolci90`, 26.7 M presentations
   short end to end); v2 uses `control_matched`
   (`gate2_midtrain4/dolmino/post_dolci100`, dose-matched). Charter and coin arms
   *are* the same lineage across all three. Read the control column down a wave,
   not across waves.
2. **Only conflict runs are counted.** charter/coin is undefined on a run where
   the two oracles coincide, and agreement-run accuracy on this battery measures
   the cheapest-crew shortcut rather than clause competence, so pooling it in
   would flatter every arm identically.

Clause attribution follows `plot_dispatch_wave_detail.build_detail`: every
conflict run of an episode is charged to that episode's `target_clause`. The
script asserts each episode is single-clause (`exclusive`, `union_sensitive ==
[target_clause]`) and refuses to aggregate if that stops being true.

    python3 build_clause_breakdown.py [--cache DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

MIXTURE = "agreement"
#: pre-AFT is the parent's shared baseline; post-AFT is the converged endpoint
PHASES = (("pre_aft", "baseline"), ("post_aft", "step512"))
#: charter/coin readout is only defined on conflict runs
SLICES = (("trained", "eval_trained_conflict"),
          ("holdout", "eval_holdout_conflict"))
HELD_OUT_CLAUSES = frozenset({"precedence_deferrals", "qual_weekly_limit"})
VERDICT_ORDER = (sf.CHARTER, sf.COIN, sf.OTHER, sf.MALFORMED)


@dataclass(frozen=True)
class Wave:
    key: str
    label: str
    repo: str
    prefix: str
    episodes_repo: str
    episodes_prefix: str
    #: arm -> parent label as it appears in that wave's result paths
    arms: dict


WAVES = (
    Wave("wave_v1", "wave v1",
         repo="sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1",
         prefix="extensions/wave_v1",
         episodes_repo="sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data",
         episodes_prefix="extensions/wave_v1/data/episodes",
         arms={"charter": "charter_real_4x", "control": "control_4x",
               "coin": "coin_real_4x"}),
    Wave("wave_v1_retrain", "wave v1 retrain",
         repo="arcadia-impact/scimt-dispatch-models",
         prefix="aft_wave_retrain",
         episodes_repo="sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data",
         episodes_prefix="extensions/wave_v1/data/episodes",
         arms={"charter": "charter_real_4x", "control": "control_4x",
               "coin": "coin_real_4x"}),
    Wave("wave_v2", "wave v2",
         repo="arcadia-impact/scimt-dispatch-models",
         prefix="aft_wave_v2",
         episodes_repo="arcadia-impact/scimt-dispatch-aft-data",
         episodes_prefix="extensions/wave_v2/data/episodes",
         arms={"charter": "charter_real_4x", "control": "control_matched",
               "coin": "coin_real_4x"}),
)

ARMS = ("charter", "control", "coin")


def endpoint_dir(parent: str, endpoint: str) -> str:
    """Baselines are shared per parent; trained endpoints are per cell."""
    if endpoint == "baseline":
        return f"{parent}-baseline"
    return f"{parent}__{MIXTURE}-{endpoint}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(repo: str, remote: str, cache: Path, *, repo_type: str = "model") -> Path:
    """Download one Hub file into ``cache``, reusing what is already there."""
    from huggingface_hub import hf_hub_download

    local = cache / repo.replace("/", "__") / remote
    if local.is_file():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo_id=repo, repo_type=repo_type, filename=remote)
    local.write_bytes(Path(got).read_bytes())
    return local


def load_episodes(wave: Wave, cache: Path) -> dict:
    """The eval battery for one wave, keyed by slice name, with its file hashes."""
    out, hashes = {}, {}
    for _, slice_name in SLICES:
        path = fetch(wave.episodes_repo,
                     f"{wave.episodes_prefix}/{slice_name}.jsonl", cache,
                     repo_type="dataset")
        out[slice_name] = v4.read_records(path)
        hashes[slice_name] = sha256(path)
    return out, hashes


def check_single_clause(records) -> None:
    """Clause attribution is only honest if each episode turns on one clause."""
    for record in records:
        meta = record.metadata
        clause = meta["target_clause"]
        union = list(meta.get("union_sensitive") or [])
        if not meta.get("exclusive") or union != [clause]:
            raise SystemExit(
                f"{record.episode.episode_id}: not single-clause "
                f"(exclusive={meta.get('exclusive')}, union_sensitive={union}, "
                f"target_clause={clause}). Clause attribution would be a guess; "
                "fix the aggregation before trusting a per-clause figure.")


def counts_for_cell(records_by_slice, responses_by_slice) -> dict:
    """Per-clause per-run verdict counts over the conflict runs of one endpoint."""
    by_clause: dict = defaultdict(lambda: defaultdict(int))
    for condition, slice_name in SLICES:
        responses = responses_by_slice.get(slice_name)
        if responses is None:
            continue
        for record in records_by_slice[slice_name]:
            episode = record.episode
            text = responses.get(episode.episode_id)
            if text is None:
                continue
            verdicts = sf.per_run_verdicts(
                episode, dispatch.parse_plan(text, episode))
            clause = record.metadata["target_clause"]
            for index, kind in enumerate(sf.derived_run_kinds(episode)):
                if kind != "conflict":
                    continue
                verdict = sf.MALFORMED if verdicts is None else verdicts[index]
                by_clause[clause][verdict] += 1
                by_clause[clause]["_condition"] = condition
    return {clause: dict(counts) for clause, counts in by_clause.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path,
                    default=EXP / "runs" / "clause_breakdown_v1" / "cache",
                    help="where downloaded Hub files are kept (gitignored)")
    ap.add_argument("--out", type=Path, default=HERE / "data"
                    / "clause_breakdown.json")
    ap.add_argument("--csv", type=Path, default=HERE / "data"
                    / "clause_rates.csv",
                    help="the same numbers as one tidy row per bar")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    cells: dict = {}
    provenance: dict = {}
    battery_hashes: dict = {}

    for wave in WAVES:
        episodes, hashes = load_episodes(wave, args.cache)
        battery_hashes[wave.key] = hashes
        for records in episodes.values():
            check_single_clause(records)
        for arm in ARMS:
            parent = wave.arms[arm]
            for phase, endpoint in PHASES:
                responses, files = {}, {}
                for _, slice_name in SLICES:
                    remote = (f"{wave.prefix}/{parent}__{MIXTURE}/results/"
                              f"{endpoint_dir(parent, endpoint)}/{slice_name}.jsonl")
                    path = fetch(wave.repo, remote, args.cache)
                    responses[slice_name] = sf.load_responses(path)
                    files[slice_name] = {"remote": f"{wave.repo}:{remote}",
                                         "sha256": sha256(path),
                                         "rows": len(responses[slice_name])}
                key = f"{wave.key}|{arm}|{phase}"
                cells[key] = counts_for_cell(episodes, responses)
                provenance[key] = {"parent": parent, "endpoint": endpoint,
                                   "files": files}
                total = sum(sum(v for k, v in c.items() if k != "_condition")
                            for c in cells[key].values())
                print(f"{key:38s} {parent:16s} {endpoint:9s} "
                      f"{len(cells[key])} clauses, {total} conflict runs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "what": "per-clause per-run verdict counts on conflict runs, for the "
                "three agreement-AFT waves x {charter,control,coin} midtrain "
                "arms x {pre-AFT, post-AFT step512}",
        "mixture": MIXTURE,
        "phases": {p: e for p, e in PHASES},
        "slices": {c: s for c, s in SLICES},
        "held_out_clauses": sorted(HELD_OUT_CLAUSES),
        "waves": {w.key: {"label": w.label, "repo": w.repo, "prefix": w.prefix,
                          "arms": w.arms} for w in WAVES},
        "battery_sha256": battery_hashes,
        "caveats": [
            "control arm substrate differs: v1/retrain control_4x "
            "(sdf/4x/shared/post_dolci90) vs v2 control_matched "
            "(gate2_midtrain4/dolmino/post_dolci100). Read the control down a "
            "wave, not across waves.",
            "conflict runs only; agreement-run accuracy on this battery "
            "measures the cheapest-crew shortcut, not clause competence.",
        ],
        "cells": cells,
        "provenance": provenance,
    }, indent=1) + "\n")
    print(f"\nwrote {args.out}  ({len(cells)} cells)")

    rows = ["wave,arm,phase,clause,condition,n_conflict_runs,"
            + ",".join(f"n_{v}" for v in VERDICT_ORDER)
            + "," + ",".join(f"rate_{v}" for v in VERDICT_ORDER)]
    for key, by_clause in cells.items():
        wave, arm, phase = key.split("|")
        for clause, counts in by_clause.items():
            condition = counts.get("_condition", "")
            n = sum(v for k, v in counts.items() if k != "_condition")
            ns = [counts.get(v, 0) for v in VERDICT_ORDER]
            rows.append(",".join([wave, arm, phase, clause, condition, str(n)]
                                 + [str(x) for x in ns]
                                 + [f"{x / n:.6f}" if n else "" for x in ns]))
    args.csv.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.csv}  ({len(rows) - 1} rows)")


if __name__ == "__main__":
    main()
