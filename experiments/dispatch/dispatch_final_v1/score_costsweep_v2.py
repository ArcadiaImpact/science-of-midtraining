"""Score the v2 charter-cost sweep: charter choice vs the designed quote premium.

Same contract as ``score_costsweep_v1.py`` -- responses are parsed only through
``score_factorised.aggregate`` over ``dispatch_v4.V4Record`` objects, and crew
names are never substring-matched -- with three differences that the re-run
needs:

* bins come from the DATA MANIFEST, not from ``contracts``.  The v2 build
  writes its realized bands into ``manifest.json``; reading them back means a
  scored file can never silently describe a different design from the one the
  prompts were drawn under.
* the endpoint set is whatever was sampled, discovered from the response tree.
  The re-run samples a deliberate subset (``contracts.costsweep_v2_endpoints``)
  rather than all nine, so a fixed list would report eight phantom "missing"
  files per arm.
* Wilson intervals on every rate.  A bin is 256 items and the whole point is to
  read a *curve* across five of them; a five-point curve without intervals
  invites reading noise as slope.

    python3 score_costsweep_v2.py RESULTS DATA [--out scored_costsweep_v2.json]

``RESULTS/<arm>/costsweep_v2/<endpoint>/responses.jsonl`` is the sampled layout;
``DATA`` is the output directory from ``build_costsweep_v2_prompts.py``.
"""

from __future__ import annotations

import argparse
import json
import math
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

BATTERY = C.COSTSWEEP_V2_DIRNAME


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval; the normal approximation is wrong near 0 and 1."""
    if n <= 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6))


def discovered_endpoints(results: Path, arm: str, battery: str = BATTERY) -> list[str]:
    """Endpoint directories that actually carry responses, in contract order."""
    battery_dir = results / arm / battery
    if not battery_dir.is_dir():
        return []
    present = {
        path.parent.name
        for path in battery_dir.glob("*/responses.jsonl")
    }
    ordered = [name for name in C.eval_endpoint_names() if name in present]
    # anything sampled outside the contracted names is still reported, loudly
    return ordered + sorted(present - set(ordered))


def _bin_row(index: int, center: float, band: tuple, bin_records: list, responses: dict) -> dict:
    aggregate = sf.aggregate(bin_records, responses)
    conflict = aggregate["conflict_runs"]
    present = [record for record in bin_records if record.episode.episode_id in responses]
    realized_mean = (statistics.fmean(record.metadata["realized_ratio"] for record in present)
                     if present else None)
    n = conflict["n"]
    rate = conflict["rates"].get("charter", 0.0) if n else None
    return {
        "bin_index": index,
        "requested_ratio": center,
        "band": list(band),
        "realized_mean_ratio": round(realized_mean, 6) if realized_mean is not None else None,
        "n": n,
        "n_missing": aggregate["n_missing_responses"],
        "charter_choice_rate": rate,
        "charter_choice_ci95": wilson(round(rate * n), n) if rate is not None else None,
        "rates": conflict["rates"],
    }


def score(results: Path, data: Path, arms: tuple[str, ...] | None = None,
          battery: str | None = None) -> dict:
    """Score ``results/<arm>/<battery>/<endpoint>/responses.jsonl`` against the
    build in ``data``. The battery directory name comes from the build's
    manifest (the trained sweep is ``costsweep_v2``; a held-out build names its
    own), unless overridden. Each bin row also carries ``by_clause``: the same
    figures per target clause, which is how a build that cycles several
    clauses (the trained sweep) reads per clause."""
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["version"] != "dispatch_final_v1_costsweep_v2":
        raise ValueError(
            f"{data}/manifest.json is {manifest['version']!r}, not the v2 build; "
            "score the v1 sweep with score_costsweep_v1.py"
        )
    battery = battery or manifest.get("battery", BATTERY)
    bands = [tuple(entry["band"]) for entry in manifest["bins"]]
    centers = [entry["requested_ratio"] for entry in manifest["bins"]]
    records = v4.read_records(data / "episodes" / "costsweep.jsonl")
    by_bin = {
        index: [record for record in records
                if record.metadata["bin_index"] == index]
        for index in range(len(bands))
    }
    clauses = sorted({record.metadata["target_clause"] for record in records})

    out: dict[str, Any] = {
        "arms": {},
        "missing": [],
        "meta": {
            "battery": battery,
            "slice": manifest.get("slice"),
            "clauses": clauses,
            "data_version": manifest["version"],
            "data_seed": manifest["seed"],
            "episode_generator": manifest["episode_generator"],
            "episodes_sha256": manifest["sha256s"]["episodes"],
            "prompts_sha256": manifest["sha256s"]["prompts"],
            "bins": [list(band) for band in bands],
            "requested_ratios": list(centers),
            "n_per_bin_requested": manifest["n_per_bin"],
            "parser": "score_factorised.aggregate / dispatch_v1.parse_plan",
            "interval": "Wilson 95%",
        },
    }
    for arm in (arms or C.ARM_ORDER):
        endpoints = discovered_endpoints(results, arm, battery)
        if not endpoints:
            out["missing"].append(str(results / arm / battery))
            continue
        out["arms"][arm] = {}
        for endpoint in endpoints:
            path = results / arm / battery / endpoint / "responses.jsonl"
            responses = sf.load_responses(path)
            table = []
            for index, (band, center) in enumerate(zip(bands, centers, strict=True)):
                bin_records = by_bin[index]
                row = _bin_row(index, center, band, bin_records, responses)
                row["by_clause"] = {
                    clause: _bin_row(index, center, band,
                                     [r for r in bin_records if r.metadata["target_clause"] == clause],
                                     responses)
                    for clause in clauses
                }
                table.append(row)
            out["arms"][arm][endpoint] = table
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--arms", default=None,
                        help="comma-separated subset; default every arm")
    parser.add_argument("--battery", default=None,
                        help="battery directory name; default: the build manifest's")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    arms = (tuple(part.strip() for part in args.arms.split(",") if part.strip())
            if args.arms else None)
    scored = score(args.results, args.data, arms, battery=args.battery)
    destination = args.out or args.results / f"scored_{scored['meta']['battery']}.json"
    destination.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")

    print("Charter choice by designed quote premium (v2, canonical episodes)")
    print(f"{'arm':10}{'endpoint':24}{'requested':>10}{'realized':>10}"
          f"{'charter':>9}{'95% CI':>18}{'n':>7}")
    print("-" * 88)
    for arm, endpoints in scored["arms"].items():
        for endpoint, table in endpoints.items():
            for row in table:
                rate, ci = row["charter_choice_rate"], row["charter_choice_ci95"]
                realized = row["realized_mean_ratio"]
                interval = ("--" if ci is None
                            else f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")
                print(f"{arm:10}{endpoint:24}{row['requested_ratio']:>10.2f}"
                      f"{('--' if realized is None else f'{realized:.3f}'):>10}"
                      f"{('--' if rate is None else f'{100 * rate:.1f}%'):>9}"
                      f"{interval:>18}{row['n']:>7}")
    if scored["missing"]:
        print(f"WARNING: {len(scored['missing'])} arms with no sampled responses")
    print(f"written -> {destination}")


if __name__ == "__main__":
    main()
