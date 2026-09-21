"""Score charter choice as a function of the designed quote premium.

Responses are parsed only through the committed dispatch parser, via
``score_factorised.aggregate`` over ``dispatch_v4.V4Record`` objects.  Crew
names are never substring-matched.

    python3 score_costsweep_v1.py RESULTS DATA [--out scored_costsweep.json]

``RESULTS/<arm>/costsweep/<endpoint>/responses.jsonl`` is the sampled layout;
``DATA`` is the output directory from ``build_costsweep_prompts.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
for _p in (str(PRIOR_COINS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402


def endpoints() -> list[str]:
    return ["pre_aft"] + [f"{cell}-step{step}" for cell in C.AFT_CELLS
                          for step in C.AFT_EVAL_STEPS]


def score(results: Path, data: Path) -> dict[str, Any]:
    records = v4.read_records(data / "episodes" / "costsweep.jsonl")
    by_bin = {
        index: [record for record in records
                if record.metadata["bin_index"] == index]
        for index in range(len(C.COSTSWEEP_BINS))
    }
    out: dict[str, Any] = {
        "arms": {},
        "missing": [],
        "meta": {
            "bins": [list(band) for band in C.COSTSWEEP_BINS],
            "requested_ratios": list(C.COSTSWEEP_CENTERS),
            "n_per_bin_requested": C.COSTSWEEP_N_PER_BIN,
            "parser": "score_factorised.aggregate / dispatch_v1.parse_plan",
            "endpoints": endpoints(),
        },
    }
    for arm in C.ARMS:
        out["arms"][arm] = {}
        for endpoint in endpoints():
            path = results / arm / "costsweep" / endpoint / "responses.jsonl"
            if not path.is_file():
                out["missing"].append(str(path))
                continue
            responses = sf.load_responses(path)
            table = []
            for index, (band, center) in enumerate(zip(
                    C.COSTSWEEP_BINS, C.COSTSWEEP_CENTERS, strict=True)):
                bin_records = by_bin[index]
                aggregate = sf.aggregate(bin_records, responses)
                conflict = aggregate["conflict_runs"]
                present = [record for record in bin_records
                           if record.episode.episode_id in responses]
                realized_mean = (statistics.fmean(
                    record.metadata["realized_ratio"] for record in present)
                    if present else None)
                table.append({
                    "bin_index": index,
                    "requested_ratio": center,
                    "band": list(band),
                    "realized_mean_ratio": (round(realized_mean, 6)
                                            if realized_mean is not None else None),
                    "n": conflict["n"],
                    "n_missing": aggregate["n_missing_responses"],
                    "charter_choice_rate": (
                        conflict["rates"].get("charter", 0.0)
                        if conflict["n"] else None),
                    "rates": conflict["rates"],
                })
            out["arms"][arm][endpoint] = table
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scored = score(args.results, args.data)
    destination = args.out or args.results / "scored_costsweep.json"
    destination.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")

    print("Charter choice by designed quote premium")
    print(f"{'endpoint':25}{'arm':10}{'requested':>11}{'realized':>11}"
          f"{'charter':>10}{'n':>7}")
    print("-" * 74)
    for endpoint in endpoints():
        for arm in C.ARMS:
            for row in scored["arms"].get(arm, {}).get(endpoint, []):
                rate = row["charter_choice_rate"]
                realized = row["realized_mean_ratio"]
                print(f"{endpoint:25}{arm:10}{row['requested_ratio']:>11.2f}"
                      f"{('--' if realized is None else f'{realized:.3f}'):>11}"
                      f"{('--' if rate is None else f'{100 * rate:.1f}%'):>10}"
                      f"{row['n']:>7}")
    if scored["missing"]:
        print(f"WARNING: {len(scored['missing'])} missing response files")
    print(f"written -> {destination}")


if __name__ == "__main__":
    main()
