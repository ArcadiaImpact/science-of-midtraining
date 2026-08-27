"""Score the seed sweep: per-clause per-run verdicts, per (arm, seed, phase).

Reads the response rows bellhop pulled back to the devbox:

    runs/seed_sweep_v1/pods/<arm>/results/<parent>-baseline/<slice>.jsonl
    runs/seed_sweep_v1/pods/<arm>/results/<parent>__agreement__seed<S>-step256/...

The eval battery itself is not pulled off the pod (only responses are), so the
episodes come from the pinned Hub revision the pods trained against --
`contracts.DATA_REPO@DATA_REVISION`. Clause attribution follows
`plot_dispatch_wave_detail.build_detail`: every conflict run of an episode is
charged to that episode's `target_clause`, and each episode is asserted
single-clause first.

**Partial runs are first-class.** The sweep is 25 runs across 5 pods and any one
seed can fail without taking its pod down, so this reports what is present and
names what is missing rather than refusing to score.

Only conflict runs are counted: charter/coin is undefined where the two oracles
coincide, and agreement-run accuracy on this battery measures the cheapest-crew
shortcut rather than clause competence.

    python -m experiments.prior_coins.seed_sweep_v1.score_seed_sweep
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

from experiments.prior_coins.seed_sweep_v1 import contracts  # noqa: E402

SLICES = (("trained", "eval_trained_conflict"),
          ("holdout", "eval_holdout_conflict"))
VERDICT_ORDER = (sf.CHARTER, sf.COIN, sf.OTHER, sf.MALFORMED)
HELD_OUT_CLAUSES = frozenset({"precedence_deferrals", "qual_weekly_limit"})


def load_episodes(cache: Path) -> dict:
    from huggingface_hub import hf_hub_download

    out = {}
    for _, slice_name in SLICES:
        remote = f"{contracts.DATA_PREFIX}/episodes/{slice_name}.jsonl"
        local = cache / remote
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            got = hf_hub_download(repo_id=contracts.DATA_REPO, repo_type="dataset",
                                  revision=contracts.DATA_REVISION, filename=remote)
            local.write_bytes(Path(got).read_bytes())
        records = v4.read_records(local)
        for record in records:
            meta = record.metadata
            clause = meta["target_clause"]
            if not meta.get("exclusive") or list(meta.get("union_sensitive") or []) != [clause]:
                raise SystemExit(
                    f"{record.episode.episode_id}: not single-clause; clause "
                    "attribution would be a guess"
                )
        out[slice_name] = records
    return out


def score_dir(episodes: dict, cell_dir: Path) -> dict | None:
    """Per-clause per-run verdict counts for one endpoint dir, or None if absent."""
    present = [s for _, s in SLICES if (cell_dir / f"{s}.jsonl").is_file()]
    if len(present) != len(SLICES):
        return None
    by_clause: dict = defaultdict(lambda: defaultdict(int))
    for condition, slice_name in SLICES:
        responses = sf.load_responses(cell_dir / f"{slice_name}.jsonl")
        for record in episodes[slice_name]:
            episode = record.episode
            text = responses.get(episode.episode_id)
            if text is None:
                continue
            verdicts = sf.per_run_verdicts(episode, dispatch.parse_plan(text, episode))
            clause = record.metadata["target_clause"]
            for index, kind in enumerate(sf.derived_run_kinds(episode)):
                if kind != "conflict":
                    continue
                verdict = sf.MALFORMED if verdicts is None else verdicts[index]
                by_clause[clause][verdict] += 1
                by_clause[clause]["_condition"] = condition
    return {c: dict(v) for c, v in by_clause.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pods", type=Path,
                    default=EXP / "runs" / "seed_sweep_v1" / "pods")
    ap.add_argument("--cache", type=Path,
                    default=EXP / "runs" / "seed_sweep_v1" / "episodes")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "seed_sweep_scored.json")
    ap.add_argument("--csv", type=Path, default=HERE / "data" / "seed_sweep_rates.csv")
    args = ap.parse_args()

    episodes = load_episodes(args.cache)
    cells: dict = {}
    missing: list[str] = []
    step = contracts.EXPECTED_STEPS

    for cell in contracts.CELLS:
        results = args.pods / cell.arm / "results"
        base = results / f"{cell.parent}-baseline"
        got = score_dir(episodes, base) if base.is_dir() else None
        if got is None:
            missing.append(f"{cell.arm}|pre_aft")
        else:
            cells[f"{cell.arm}|pre_aft|shared"] = got
        for seed in contracts.SEEDS:
            name = f"{cell.cell_label(seed)}-step{step}"
            got = score_dir(episodes, results / name) if (results / name).is_dir() else None
            key = f"{cell.arm}|post_aft|seed{seed}"
            if got is None:
                missing.append(key)
            else:
                cells[key] = got

    def total(entry: dict) -> int:
        return sum(sum(v for k, v in c.items() if k != "_condition")
                   for c in entry.values())

    for key in sorted(cells):
        print(f"{key:34s} {len(cells[key])} clauses, {total(cells[key])} conflict runs")
    if missing:
        print(f"\nMISSING ({len(missing)}): " + ", ".join(sorted(missing)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "what": "per-clause per-run verdict counts on conflict runs, per "
                "(midtrain arm, phase, seed); pre-AFT is shared per arm",
        "version": contracts.VERSION,
        "stage": contracts.STAGE,
        "mixture": contracts.MIXTURE,
        "rows": contracts.TRAIN_ROWS,
        "steps": step,
        "seeds": list(contracts.SEEDS),
        "held_out_clauses": sorted(HELD_OUT_CLAUSES),
        "episodes": f"{contracts.DATA_REPO}/{contracts.DATA_PREFIX}"
                    f"@{contracts.DATA_REVISION}",
        "arms": {c.arm: {"parent": c.parent, "parent_prefix": c.parent_prefix}
                 for c in contracts.CELLS},
        "caveats": [
            "dose-matched but NOT schedule-matched to the wave's step-256 "
            "checkpoint: this run completes a 256-step cosine decay, the wave's "
            "was mid-decay on a 512-step schedule. Not error bars on the "
            "published wave points.",
            "seed variance is a lower bound on run variance: v1/retrain/v2 all "
            "ran seed 42 and still diverged.",
            "conflict runs only.",
        ],
        "missing": sorted(missing),
        "cells": cells,
    }, indent=1) + "\n")
    print(f"\nwrote {args.out}  ({len(cells)} cells present, {len(missing)} missing)")

    rows = ["arm,phase,seed,clause,condition,n_conflict_runs,"
            + ",".join(f"n_{v}" for v in VERDICT_ORDER) + ","
            + ",".join(f"rate_{v}" for v in VERDICT_ORDER)]
    for key, by_clause in cells.items():
        arm, phase, seed = key.split("|")
        for clause, counts in by_clause.items():
            condition = counts.get("_condition", "")
            n = sum(v for k, v in counts.items() if k != "_condition")
            ns = [counts.get(v, 0) for v in VERDICT_ORDER]
            rows.append(",".join([arm, phase, seed, clause, condition, str(n)]
                                 + [str(x) for x in ns]
                                 + [f"{x / n:.6f}" if n else "" for x in ns]))
    args.csv.write_text("\n".join(rows) + "\n")
    print(f"wrote {args.csv}  ({len(rows) - 1} rows)")


if __name__ == "__main__":
    main()
