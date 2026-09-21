"""Score the wave grid: 10 parents x 4 AFT mixtures x 6 endpoints.

`score_dispatch_v4_aft` assumes exactly two arms (charter/coin) and one training
set. The wave has ten parents crossed with four mixtures, so separation is no
longer a single number — it is a number per (lineage, dose, mixture, endpoint),
and there is a fifth parent per dose with no partner at all.

Structure of the grid:

* **paired parents** — (charter, coin) within a lineage and dose. Separation is
  only ever computed *within* a pair, because that is the only contrast where
  the lineage, the dose, the mixture and the eval battery are all held fixed.
* **the control** (``post_dolci90``, no arm documents) has no partner. It is
  reported as rates only. Comparing it to an arm mixes "saw arm documents" with
  "got 10M fewer instruct tokens", since the control lacks the Dolci10 suffix.

Reads ``results/<label>-<endpoint>/<slice>.jsonl`` where label is
``<parent>__<mixture>``, plus ``results/<parent>-baseline/`` which is shared by
every mixture of that parent. Scoring is per-run via ``score_factorised``.

Run: ``python3 score_dispatch_wave.py <results-dir> <data-dir>``
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

EVAL_STEPS = (32, 64, 128, 256, 512)
ENDPOINTS = ("baseline",) + tuple(f"step{s}" for s in EVAL_STEPS)
MIXTURES = ("agreement", "mixed_balanced", "coin2", "charter2",
            "coin0p2", "charter0p2", "coin0p5", "charter0p5")
TRAINED_CONFLICT = "eval_trained_conflict"
HELDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_AGREE = "eval_trained_agreement"
HELDOUT_AGREE = "eval_holdout_agreement"
SLICES = (TRAINED_AGREE, TRAINED_CONFLICT, HELDOUT_AGREE, HELDOUT_CONFLICT)

#: (lineage, dose) -> (charter parent, coin parent). Only these are paired.
PAIRS = {
    ("real", "1x"): ("charter_real_1x", "coin_real_1x"),
    ("real", "4x"): ("charter_real_4x", "coin_real_4x"),
    ("fake", "1x"): ("charter_fake_1x", "coin_fake_1x"),
    ("fake", "4x"): ("charter_fake_4x", "coin_fake_4x"),
}
#: A parent the scorer does not list is silently skipped (cell_dir just finds no
#: directory), so every control label a run can emit must appear here or its rows
#: vanish from scored.json without an error. v1's two SDF controls plus v2's
#: dose-matched Gate-2 control.
CONTROLS = ("control_1x", "control_4x",
            "control_sdf_1x", "control_sdf_4x", "control_matched")


def wilson(successes: int, n: int, z: float = 1.96):
    if not n:
        return None, None, None
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_episodes(data_root: Path):
    return {
        name: v4.read_records(data_root / "episodes" / f"{name}.jsonl")
        for name in SLICES
    }


def verdicts_for(records, path: Path):
    """Per-run verdict counts for one results file. None if the file is absent."""
    if not path.is_file():
        return None
    responses = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            responses[row["id"]] = row["response_text"]
    counts: defaultdict[str, int] = defaultdict(int)
    total = 0
    for record in records:
        episode = record.episode
        text = responses.get(episode.episode_id)
        if text is None:
            continue
        per_run = sf.per_run_verdicts(episode, dispatch.parse_plan(text, episode))
        for index in range(len(sf.derived_run_kinds(episode))):
            total += 1
            counts[sf.MALFORMED if per_run is None else per_run[index]] += 1
    return dict(counts), total


def cell_dir(results: Path, parent: str, mixture: str, endpoint: str) -> Path:
    """Baselines are shared per parent; trained endpoints are per cell."""
    if endpoint == "baseline":
        return results / f"{parent}-baseline"
    return results / f"{parent}__{mixture}-{endpoint}"


def score(results: Path, data: Path) -> dict:
    episodes = load_episodes(data)
    rates: dict[str, dict] = {}

    parents = sorted({p for pair in PAIRS.values() for p in pair} | set(CONTROLS))
    for parent in parents:
        for mixture in MIXTURES:
            for endpoint in ENDPOINTS:
                base = cell_dir(results, parent, mixture, endpoint)
                if not base.is_dir():
                    continue
                entry = {}
                for slice_name, records in episodes.items():
                    got = verdicts_for(records, base / f"{slice_name}.jsonl")
                    if got is None:
                        continue
                    counts, total = got
                    entry[slice_name] = {"counts": counts, "n": total}
                if entry:
                    rates[f"{parent}|{mixture}|{endpoint}"] = entry

    def rate(parent, mixture, endpoint, slice_name, verdict):
        cell = rates.get(f"{parent}|{mixture}|{endpoint}", {}).get(slice_name)
        if not cell or not cell["n"]:
            return None
        return cell["counts"].get(verdict, 0) / cell["n"]

    separation: dict[str, dict] = {}
    for (lineage, dose), (charter_parent, coin_parent) in PAIRS.items():
        for mixture in MIXTURES:
            for endpoint in ENDPOINTS:
                for label, slice_name in (("trained", TRAINED_CONFLICT),
                                          ("holdout", HELDOUT_CONFLICT)):
                    cc = rate(charter_parent, mixture, endpoint, slice_name, sf.CHARTER)
                    kc = rate(coin_parent, mixture, endpoint, slice_name, sf.CHARTER)
                    ck = rate(charter_parent, mixture, endpoint, slice_name, sf.COIN)
                    kk = rate(coin_parent, mixture, endpoint, slice_name, sf.COIN)
                    if None in (cc, kc, ck, kk):
                        continue
                    key = f"{lineage}|{dose}|{mixture}|{endpoint}|{label}"
                    separation[key] = {
                        "separation": round((cc - kc) + (kk - ck), 4),
                        "charter_parent_charter": round(cc, 4),
                        "coin_parent_charter": round(kc, 4),
                        "charter_parent_coin": round(ck, 4),
                        "coin_parent_coin": round(kk, 4),
                    }

    competence: dict[str, dict] = {}
    for key, entry in rates.items():
        for label, slice_name in (("trained", TRAINED_AGREE),
                                  ("holdout", HELDOUT_AGREE)):
            cell = entry.get(slice_name)
            if not cell or not cell["n"]:
                continue
            p, lo, hi = wilson(cell["counts"].get(sf.SHARED, 0), cell["n"])
            competence[f"{key}|{label}"] = {
                "accuracy": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                "n": cell["n"],
            }

    return {"rates": rates, "separation": separation, "competence": competence,
            "cells_present": len(rates), "pairs": {f"{k[0]}|{k[1]}": list(v)
                                                   for k, v in PAIRS.items()},
            "controls": list(CONTROLS)}


def render(report: dict) -> str:
    out = ["## Separation by lineage x dose x mixture", "",
           "| lineage | dose | mixture | endpoint | trained sep | held-out sep |",
           "|---|---|---|---|---:|---:|"]
    seen = set()
    for key in sorted(report["separation"]):
        lineage, dose, mixture, endpoint, label = key.split("|")
        if label != "trained":
            continue
        held = report["separation"].get(
            f"{lineage}|{dose}|{mixture}|{endpoint}|holdout", {}
        ).get("separation")
        out.append(
            f"| {lineage} | {dose} | {mixture} | {endpoint} | "
            f"{report['separation'][key]['separation']:+.3f} | "
            f"{held:+.3f} |" if held is not None else
            f"| {lineage} | {dose} | {mixture} | {endpoint} | "
            f"{report['separation'][key]['separation']:+.3f} | — |"
        )
        seen.add(key)
    out += ["", "## Control (no arm documents) — rates only, no partner", "",
            "| parent | mixture | endpoint | Charter% | coin% | trained agr% |",
            "|---|---|---|---:|---:|---:|"]
    for control in report["controls"]:
        for mixture in MIXTURES:
            for endpoint in ENDPOINTS:
                cell = report["rates"].get(f"{control}|{mixture}|{endpoint}", {})
                conflict = cell.get(TRAINED_CONFLICT)
                if not conflict or not conflict["n"]:
                    continue
                n = conflict["n"]
                ch = conflict["counts"].get(sf.CHARTER, 0) / n * 100
                co = conflict["counts"].get(sf.COIN, 0) / n * 100
                agr = report["competence"].get(
                    f"{control}|{mixture}|{endpoint}|trained", {}
                ).get("accuracy")
                out.append(f"| {control} | {mixture} | {endpoint} | {ch:.1f} | "
                           f"{co:.1f} | {agr*100:.1f} |" if agr is not None else
                           f"| {control} | {mixture} | {endpoint} | {ch:.1f} | "
                           f"{co:.1f} | — |")
    return "\n".join(out)


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        EXP / "runs" / "dispatch_wave_v1" / "results")
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        EXP / "runs" / "dispatch_wave_v1" / "data")
    report = score(results, data)
    (results / "scored.json").write_text(json.dumps(report, indent=2) + "\n")
    print(render(report))
    print(f"\ncells present: {report['cells_present']}", file=sys.stderr)


if __name__ == "__main__":
    main()
