"""Off-pod scorer for the token-scaling 4B eval sample stores.

The pod chain (``pod/chain.py``) samples the unchanged 4B wave battery and
uploads RAW rows only (two-stage sample -> score). This module is the score
stage: pure CPU, no torch, no network, no judge — the wave battery's verdicts
are pure parsers (``dispatch_v1.parse_plan`` -> ``score_factorised``), imported
from the SAME modules Sid's 4B/27B batteries used (byte-identical on this
branch to ``origin/sid/prior-coins-27b``; PR #524 harness-family rule).
``verdicts_for`` and ``wilson`` below are verbatim ports from
``dispatch_scaleup/score_scaleup.py`` so scoring semantics are IDENTICAL.

Input tree (one downloaded run root, mirroring the GCS layout)::

    <run_root>/<cell>/ift/eval/baseline/<slice>.jsonl
    <run_root>/<cell>/eft_<cap>/eval/<cell>-<cap>-step<k>/<slice>.jsonl

with cells ``{charter,coin}_d{0.5,1,2,4,8}m`` / ``control_d0``, capacities
``r{4,16,32,64,256}`` / ``full``, steps {32,64,128,256,512}, and the six
slices ``eval_{trained,holdout}_{agreement,conflict,adjacent}``. Raw rows are
``{"id": <episode_id>, "response_text": ...}`` (pod_generate.py).

Outputs (idempotent — re-score overwrites):

- ``counts.json`` in every endpoint dir: slice -> {"counts": {charter, coin,
  shared, other, malformed}, "n"} — collate.py's ``endpoint_dirs`` adapter;
- ``scored.json`` in every stage dir (ift/, eft_*) with a score_scaleup-style
  ``rates`` key — collate.py's ``scaleup_json`` adapter (redundant path);
- ``scored.json`` at each cell root (rates / separation / competence in
  score_scaleup's shape; separation needs both arms so it lives in the
  run-level summary);
- a run-level summary JSON (``--json``, default ``<run_root>/
  scored_summary.json``) with cross-arm directional separation per (dose,
  capacity, endpoint, slice) and raw-rate contrasts vs ``control_d0``.

Loud errors: a found endpoint missing a slice file, response ids that do not
exactly match the frozen episode ids, and conflict-run totals that do not
equal the frozen SPEC §7 ns (3,000 trained-conflict / 1,200 held-out-conflict
runs per endpoint). ``--check-only`` runs every check and writes nothing.

Run::

    python3 score_cells.py <run_root> <data_dir> [--json OUT] [--check-only]

``data_dir`` must contain ``episodes/<slice>.jsonl`` (the pinned v4_wide eval
episodes the chain downloaded — ``<pod work>/eft_data/eval`` also works).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent  # experiments/prior_coins — the wave-v1 scoring lineage
for path in (str(EXP),):
    if path not in sys.path:
        sys.path.insert(0, path)

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

SCHEMA_VERSION = "tsl_scored_v1"

CELL_RE = re.compile(r"^(charter|coin)_d(0\.5|1|2|4|8)m$|^(control)_d0$")
CAPACITY_DIR_RE = re.compile(r"^eft_(r(?:4|16|32|64|256)|full)$")
EVAL_STEPS = (32, 64, 128, 256, 512)

TRAINED_CONFLICT = "eval_trained_conflict"
HELDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_AGREE = "eval_trained_agreement"
HELDOUT_AGREE = "eval_holdout_agreement"
SLICES = (
    TRAINED_AGREE,
    TRAINED_CONFLICT,
    HELDOUT_AGREE,
    HELDOUT_CONFLICT,
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
#: canonical verdict keys always present in counts.json (zeros included, so
#: downstream consumers never need a missing-key default).
CANONICAL_VERDICTS = (sf.CHARTER, sf.COIN, sf.SHARED, sf.OTHER, sf.MALFORMED)

#: SPEC §7 frozen ns: conflict RUNS per endpoint on the pinned episode slices.
EXPECTED_CONFLICT_RUNS: Mapping[str, int] = {
    TRAINED_CONFLICT: 3_000,
    HELDOUT_CONFLICT: 1_200,
}


class ScoringError(RuntimeError):
    """A sample store or data dir we refuse to score — always says why."""


# ---------------------------------------------------------------------------
# verbatim ports from dispatch_scaleup/score_scaleup.py (identical semantics)
# ---------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = 1.96):
    if not n:
        return None, None, None
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def verdicts_for(records, path: Path):
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


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------

def resolve_episode_root(data_dir: Path) -> Path:
    for candidate in (data_dir / "episodes", data_dir / "eval" / "episodes"):
        if candidate.is_dir():
            return candidate
    raise ScoringError(
        f"no episodes/ directory under {data_dir} (looked for episodes/ and "
        f"eval/episodes/); children: "
        f"{sorted(p.name for p in data_dir.iterdir()) if data_dir.is_dir() else 'MISSING DIR'}"
    )


def load_episodes(
    data_dir: Path,
    expected_conflict_runs: Mapping[str, int] = EXPECTED_CONFLICT_RUNS,
) -> dict[str, list]:
    """Load the frozen eval slices and verify their structural run counts.

    The two conflict slices must carry EXACTLY the SPEC §7 conflict-run
    totals — a truncated or re-generated episode file must fail here, before
    any endpoint is scored against it.
    """
    root = resolve_episode_root(Path(data_dir))
    episodes: dict[str, list] = {}
    missing = []
    for name in SLICES:
        path = root / f"{name}.jsonl"
        if not path.is_file():
            missing.append(name)
            continue
        records = v4.read_records(path)
        if not records:
            raise ScoringError(f"{path}: episode slice is empty")
        episodes[name] = records
    if TRAINED_CONFLICT in missing or HELDOUT_CONFLICT in missing:
        raise ScoringError(
            f"conflict episode slices missing under {root}: {missing}"
        )
    if missing:
        print(f"[warn] episode slices absent (skipped): {missing}")
    for name, expected in expected_conflict_runs.items():
        derived = sum(
            sf.derived_run_kinds(r.episode).count("conflict")
            for r in episodes[name]
        )
        if derived != expected:
            raise ScoringError(
                f"{root / (name + '.jsonl')}: derived conflict-run total "
                f"{derived} != expected {expected} (SPEC §7) — wrong or "
                f"truncated episode data; refusing to score against it"
            )
    return episodes


# ---------------------------------------------------------------------------
# endpoint discovery (mirrors pod/chain.py's upload layout)
# ---------------------------------------------------------------------------

def discover_endpoints(cell_dir: Path) -> list[tuple[str | None, str, Path]]:
    """(capacity | None, endpoint, dir) for every endpoint this cell must have.

    Baseline is required whenever the cell has any eval data; each capacity
    dir that exists must have all five step endpoints. Missing pieces raise
    with a listing of what WAS found.
    """
    cell = cell_dir.name
    found: list[tuple[str | None, str, Path]] = []
    problems: list[str] = []

    baseline = cell_dir / "ift" / "eval" / "baseline"
    if baseline.is_dir():
        found.append((None, "baseline", baseline))
    else:
        problems.append(f"missing baseline endpoint dir: {baseline}")

    capacity_dirs = sorted(
        p for p in cell_dir.iterdir()
        if p.is_dir() and CAPACITY_DIR_RE.match(p.name)
    )
    for stage_dir in capacity_dirs:
        capacity = CAPACITY_DIR_RE.match(stage_dir.name).group(1)
        eval_dir = stage_dir / "eval"
        if not eval_dir.is_dir():
            problems.append(
                f"{stage_dir}: no eval/ dir (children: "
                f"{sorted(p.name for p in stage_dir.iterdir())})"
            )
            continue
        for step in EVAL_STEPS:
            endpoint_dir = eval_dir / f"{cell}-{capacity}-step{step}"
            if endpoint_dir.is_dir():
                found.append((capacity, f"step{step}", endpoint_dir))
            else:
                problems.append(
                    f"missing endpoint dir: {endpoint_dir} (eval/ holds: "
                    f"{sorted(p.name for p in eval_dir.iterdir())})"
                )
    if not capacity_dirs and not baseline.is_dir():
        problems.append(
            f"{cell_dir}: no ift/eval/baseline and no eft_* capacity dirs "
            f"(children: {sorted(p.name for p in cell_dir.iterdir())})"
        )
    if problems:
        raise ScoringError(
            f"{cell}: incomplete sample store —\n  " + "\n  ".join(problems)
        )
    return found


# ---------------------------------------------------------------------------
# per-endpoint scoring
# ---------------------------------------------------------------------------

def check_endpoint(endpoint_dir: Path, episodes: Mapping[str, list]) -> None:
    """Completeness checks: every slice file present, ids exactly the frozen
    episode ids (no missing responses, no strays)."""
    problems: list[str] = []
    for slice_name, records in episodes.items():
        path = endpoint_dir / f"{slice_name}.jsonl"
        if not path.is_file():
            problems.append(f"missing slice file {path.name}")
            continue
        ids = set()
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in ids:
                problems.append(f"{path.name}:{lineno}: duplicate id {row['id']!r}")
            ids.add(row["id"])
        expected_ids = {r.episode.episode_id for r in records}
        missing = expected_ids - ids
        stray = ids - expected_ids
        if missing:
            problems.append(
                f"{path.name}: {len(missing)}/{len(expected_ids)} episode "
                f"responses MISSING (e.g. {sorted(missing)[:3]}) — the sample "
                f"store is incomplete; re-sample, do not score"
            )
        if stray:
            problems.append(
                f"{path.name}: {len(stray)} response ids not in the frozen "
                f"episode slice (e.g. {sorted(stray)[:3]}) — wrong slice or "
                f"wrong data revision"
            )
    if problems:
        raise ScoringError(
            f"{endpoint_dir}: sample-store check FAILED —\n  "
            + "\n  ".join(problems)
        )


def score_endpoint(
    endpoint_dir: Path,
    episodes: Mapping[str, list],
    expected_conflict_runs: Mapping[str, int] = EXPECTED_CONFLICT_RUNS,
) -> dict[str, dict]:
    """Score one endpoint dir -> {slice: {"counts": ..., "n": ...}}.

    Completeness is enforced first (ids must match exactly), then per-run
    verdicts via the verbatim score_scaleup path, then the expected-n gate:
    n per slice equals the slice's total run count, and the conflict slices'
    conflict-run subtotals equal the frozen SPEC ns.
    """
    check_endpoint(endpoint_dir, episodes)
    entry: dict[str, dict] = {}
    for slice_name, records in episodes.items():
        got = verdicts_for(records, endpoint_dir / f"{slice_name}.jsonl")
        assert got is not None  # check_endpoint guarantees the file exists
        counts, total = got
        expected_total = sum(
            len(sf.derived_run_kinds(r.episode)) for r in records
        )
        if total != expected_total or total == 0:
            raise ScoringError(
                f"{endpoint_dir / (slice_name + '.jsonl')}: scored {total} "
                f"runs, expected {expected_total} (all runs of all episodes)"
            )
        if slice_name in expected_conflict_runs:
            conflict_total = sum(
                sf.derived_run_kinds(r.episode).count("conflict")
                for r in records
            )
            if conflict_total != expected_conflict_runs[slice_name]:
                raise ScoringError(
                    f"{endpoint_dir}: {slice_name} conflict runs "
                    f"{conflict_total} != {expected_conflict_runs[slice_name]}"
                )
        full = dict.fromkeys(CANONICAL_VERDICTS, 0)
        full.update(counts)
        entry[slice_name] = {"counts": full, "n": total}
    return entry


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# cell / run scoring
# ---------------------------------------------------------------------------

def _rates_key(cell: str, capacity: str | None, endpoint: str) -> str:
    label = cell if capacity is None else f"{cell}-{capacity}"
    return f"{label}|{endpoint}"


def score_cell(
    cell_dir: Path,
    episodes: Mapping[str, list],
    *,
    check_only: bool = False,
    expected_conflict_runs: Mapping[str, int] = EXPECTED_CONFLICT_RUNS,
) -> dict[str, dict]:
    """Score every endpoint of one cell. Returns score_scaleup-shaped
    ``rates`` for the cell; writes counts.json / scored.json unless
    ``check_only``."""
    cell = cell_dir.name
    endpoints = discover_endpoints(cell_dir)
    rates: dict[str, dict] = {}
    by_stage: dict[Path, dict[str, dict]] = defaultdict(dict)
    for capacity, endpoint, endpoint_dir in endpoints:
        if check_only:
            check_endpoint(endpoint_dir, episodes)
            continue
        entry = score_endpoint(endpoint_dir, episodes, expected_conflict_runs)
        rates[_rates_key(cell, capacity, endpoint)] = entry
        write_json(endpoint_dir / "counts.json", entry)
        stage_dir = (
            cell_dir / "ift" if capacity is None else cell_dir / f"eft_{capacity}"
        )
        by_stage[stage_dir][_rates_key(cell, capacity, endpoint)] = entry
    if check_only:
        return {}

    competence: dict[str, dict] = {}
    for key, entry in rates.items():
        for label, slice_name in (("trained", TRAINED_AGREE),
                                  ("holdout", HELDOUT_AGREE)):
            cell_slice = entry.get(slice_name)
            if not cell_slice or not cell_slice["n"]:
                continue
            p, lo, hi = wilson(
                cell_slice["counts"].get(sf.SHARED, 0), cell_slice["n"]
            )
            competence[f"{key}|{label}"] = {
                "accuracy": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                "n": cell_slice["n"],
            }

    # stage-level scored.json: the scaleup_json adapter's redundant path.
    for stage_dir, stage_rates in by_stage.items():
        write_json(stage_dir / "scored.json", {
            "schema_version": SCHEMA_VERSION,
            "cell": cell,
            "rates": stage_rates,
            "cells_present": len(stage_rates),
        })
    write_json(cell_dir / "scored.json", {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "rates": rates,
        # separation pairs a charter arm with a coin arm; a single cell has
        # one arm, so it is computed across cells in the run-level summary.
        "separation": {},
        "competence": competence,
        "cells_present": len(rates),
    })
    return rates


def _cell_arm_dose(cell: str) -> tuple[str | None, float]:
    match = CELL_RE.match(cell)
    if match is None:
        raise ScoringError(f"unrecognized cell id {cell!r}")
    if match.group(3) == "control":
        return None, 0.0
    return match.group(1), float(match.group(2))


def summarize(all_rates: Mapping[str, Mapping[str, dict]]) -> dict:
    """Cross-cell summary: directional separation (charter vs coin at the same
    dose) and raw-rate contrasts vs control_d0 — score_scaleup's formulas."""

    def rate(cell: str, capacity: str | None, endpoint: str,
             slice_name: str, verdict: str):
        entry = all_rates.get(cell, {}).get(_rates_key(cell, capacity, endpoint))
        if not entry:
            return None
        cell_slice = entry.get(slice_name)
        if not cell_slice or not cell_slice["n"]:
            return None
        return cell_slice["counts"].get(verdict, 0) / cell_slice["n"]

    doses = sorted({
        _cell_arm_dose(cell)[1]
        for cell in all_rates
        if _cell_arm_dose(cell)[0] is not None
    })
    capacities: set[str | None] = set()
    endpoints: set[str] = set()
    for cell, rates in all_rates.items():
        for key in rates:
            label, endpoint = key.split("|")
            endpoints.add(endpoint)
            capacities.add(
                None if label == cell else label[len(cell) + 1:]
            )

    def dose_id(dose: float) -> str:
        return f"d{int(dose) if dose.is_integer() else dose}m"

    separation: dict[str, dict] = {}
    contrasts: dict[str, dict] = {}
    for dose in doses:
        charter_cell = f"charter_{dose_id(dose)}"
        coin_cell = f"coin_{dose_id(dose)}"
        for capacity in sorted(capacities, key=str):
            for endpoint in sorted(endpoints):
                for label, slice_name in (("trained", TRAINED_CONFLICT),
                                          ("holdout", HELDOUT_CONFLICT)):
                    cc = rate(charter_cell, capacity, endpoint, slice_name,
                              sf.CHARTER)
                    kc = rate(coin_cell, capacity, endpoint, slice_name,
                              sf.CHARTER)
                    ck = rate(charter_cell, capacity, endpoint, slice_name,
                              sf.COIN)
                    kk = rate(coin_cell, capacity, endpoint, slice_name,
                              sf.COIN)
                    key = f"{dose_id(dose)}|{capacity or 'pre_eft'}|{endpoint}|{label}"
                    if None not in (cc, kc, ck, kk):
                        separation[key] = {
                            "separation": round((cc - kc) + (kk - ck), 4),
                            "charter_parent_charter": round(cc, 4),
                            "coin_parent_charter": round(kc, 4),
                            "charter_parent_coin": round(ck, 4),
                            "coin_parent_coin": round(kk, 4),
                        }
                    nc = rate("control_d0", capacity, endpoint, slice_name,
                              sf.CHARTER)
                    nk = rate("control_d0", capacity, endpoint, slice_name,
                              sf.COIN)
                    if None not in (cc, ck, nc, nk):
                        contrasts[f"charter_vs_control|{key}"] = {
                            "charter_rate_delta": round(cc - nc, 4),
                            "coin_rate_delta": round(ck - nk, 4),
                        }
                    if None not in (kc, kk, nc, nk):
                        contrasts[f"coin_vs_control|{key}"] = {
                            "charter_rate_delta": round(kc - nc, 4),
                            "coin_rate_delta": round(kk - nk, 4),
                        }
    return {"separation": separation, "control_contrasts": contrasts}


def score_run(
    run_root: Path,
    data_dir: Path,
    *,
    check_only: bool = False,
    expected_conflict_runs: Mapping[str, int] = EXPECTED_CONFLICT_RUNS,
) -> dict:
    run_root = Path(run_root)
    if not run_root.is_dir():
        raise ScoringError(f"run root {run_root} is not a directory")
    episodes = load_episodes(data_dir, expected_conflict_runs)
    cells = sorted(
        p for p in run_root.iterdir() if p.is_dir() and CELL_RE.match(p.name)
    )
    if not cells:
        raise ScoringError(
            f"no cell dirs (charter_d*m / coin_d*m / control_d0) under "
            f"{run_root}; children: "
            f"{sorted(p.name for p in run_root.iterdir())}"
        )
    all_rates: dict[str, dict[str, dict]] = {}
    for cell_dir in cells:
        all_rates[cell_dir.name] = score_cell(
            cell_dir, episodes,
            check_only=check_only,
            expected_conflict_runs=expected_conflict_runs,
        )
        print(f"[{'ok' if check_only else 'scored'}] {cell_dir.name}: "
              f"{len(discover_endpoints(cell_dir))} endpoints")
    if check_only:
        return {"checked_cells": [c.name for c in cells], "check_only": True}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root),
        "cells": {cell: {"rates": rates} for cell, rates in all_rates.items()},
        **summarize(all_rates),
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_root", type=Path,
                        help="downloaded evidence tree: <run_root>/<cell>/...")
    parser.add_argument("data_dir", type=Path,
                        help="dir holding episodes/<slice>.jsonl (the pinned "
                             "v4_wide eval data)")
    parser.add_argument("--json", type=Path, default=None,
                        help="run-level summary path "
                             "(default <run_root>/scored_summary.json)")
    parser.add_argument("--check-only", action="store_true",
                        help="validate sample-store completeness (ids, ns) "
                             "without writing anything")
    args = parser.parse_args(argv)
    report = score_run(args.run_root, args.data_dir,
                       check_only=args.check_only)
    if args.check_only:
        print(f"CHECK OK: {report['checked_cells']}")
        return
    out = args.json or (args.run_root / "scored_summary.json")
    write_json(out, report)
    print(f"summary -> {out} "
          f"({len(report['separation'])} separation rows, "
          f"{len(report['control_contrasts'])} control contrasts)")


if __name__ == "__main__":
    main()
