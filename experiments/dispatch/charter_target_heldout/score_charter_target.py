"""Score the charter-target grid: 3 substrates x 3 lineages x 5 endpoints.

Reads the per-cell response dirs bellhop pulled back:

    <pods>/<label>/results/<label>-baseline/<slice>.jsonl
    <pods>/<label>/results/<label>-step<N>/<slice>.jsonl

Two things this scorer does that `score_dispatch_wave` does not:

1. **The control is a separation partner.** At all three sizes the control is
   the Gate-2 equal-compute lineage -- 8M unique Dolmino tokens and the
   identical Dolci100 -- not wave-v1's `post_dolci90`, which was short both
   16M midtraining tokens and the Dolci10 suffix and therefore had to be
   reported rates-only. So charter-vs-control and coin-vs-control are reported
   alongside charter-vs-coin.

2. **Held-out competence gates the held-out readout.** This is the study's
   central risk, and it is not hypothetical: under the wave's `charter2`
   mixture -- only 2% conflict labels -- held-out agreement accuracy fell to
   46-78% while trained-clause accuracy stayed >=99.3%, which made those cells'
   held-out separations uninterpretable. This set is 100% conflict labels. Any
   held-out conflict number whose own cell's held-out agreement accuracy is
   below ``COMPETENCE_FLOOR`` is emitted with ``"interpretable": false`` and
   must not be read as a preference.

Run: ``python3 score_charter_target.py [<pods-dir>] [<data-dir>]``
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

from experiments.dispatch.charter_target_heldout import contracts  # noqa: E402

ENDPOINTS = ("baseline",) + tuple(f"step{s}" for s in contracts.EVAL_STEPS)
TRAINED_CONFLICT = "eval_trained_conflict"
HELDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_AGREE = "eval_trained_agreement"
HELDOUT_AGREE = "eval_holdout_agreement"
SLICES = (TRAINED_AGREE, TRAINED_CONFLICT, HELDOUT_AGREE, HELDOUT_CONFLICT)

#: Below this agreement accuracy the model has lost the task on that slice, so
#: which crew it names no longer identifies a decision rule. 0.90 is well under
#: the >=99.3% every on-distribution cell in the wave achieved and well above
#: the 46-78% collapse that made `charter2`'s held-out cells unreadable.
COMPETENCE_FLOOR = 0.90

#: within a substrate, every ordered contrast worth reporting
CONTRASTS = (("charter", "coin"), ("charter", "control"), ("coin", "control"))


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


def chance_rate(records) -> float:
    """Accuracy a uniform-random crew pick would score on this slice.

    Recorded because agreement accuracy is meaningless without it: these
    episodes carry 4-6 crews, so ~20.8% is the floor a model that has learned
    nothing sits at, and several held-out cells land there. A number near
    chance says "cannot do the task", which is a different claim from a low
    but real score.
    """
    return sum(1 / r.metadata["n_crews"] for r in records) / len(records)


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


def cell_results_dir(pods: Path, label: str) -> Path:
    """bellhop pulls `wave/results` back as `<local_out>/results`."""
    return pods / label / "results"


def score(pods: Path, data: Path) -> dict:
    episodes = load_episodes(data)
    rates: dict[str, dict] = {}

    for cell in contracts.CELLS:
        root = cell_results_dir(pods, cell.label)
        for endpoint in ENDPOINTS:
            base = root / f"{cell.label}-{endpoint}"
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
                rates[f"{cell.size}|{cell.arm}|{endpoint}"] = entry

    def rate(size, arm, endpoint, slice_name, verdict):
        cell = rates.get(f"{size}|{arm}|{endpoint}", {}).get(slice_name)
        if not cell or not cell["n"]:
            return None
        return cell["counts"].get(verdict, 0) / cell["n"]

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
                "n": cell["n"], "above_floor": p >= COMPETENCE_FLOOR,
            }

    def interpretable(size, endpoint, label, *arms) -> bool:
        return all(
            competence.get(f"{size}|{arm}|{endpoint}|{label}", {}).get("above_floor")
            for arm in arms
        )

    separation: dict[str, dict] = {}
    for size in contracts.SIZES:
        for left, right in CONTRASTS:
            for endpoint in ENDPOINTS:
                for label, slice_name in (("trained", TRAINED_CONFLICT),
                                          ("holdout", HELDOUT_CONFLICT)):
                    lc = rate(size, left, endpoint, slice_name, sf.CHARTER)
                    rc = rate(size, right, endpoint, slice_name, sf.CHARTER)
                    lk = rate(size, left, endpoint, slice_name, sf.COIN)
                    rk = rate(size, right, endpoint, slice_name, sf.COIN)
                    if None in (lc, rc, lk, rk):
                        continue
                    separation[f"{size}|{left}_vs_{right}|{endpoint}|{label}"] = {
                        "separation": round((lc - rc) + (rk - lk), 4),
                        f"{left}_charter": round(lc, 4),
                        f"{right}_charter": round(rc, 4),
                        f"{left}_coin": round(lk, 4),
                        f"{right}_coin": round(rk, 4),
                        "interpretable": interpretable(size, endpoint, label,
                                                       left, right),
                    }

    # the headline: Charter-pick rate per arm per slice, which is what the
    # held-out question is actually about
    charter_rate: dict[str, dict] = {}
    for size in contracts.SIZES:
        for arm in contracts.ARMS:
            for endpoint in ENDPOINTS:
                for label, slice_name in (("trained", TRAINED_CONFLICT),
                                          ("holdout", HELDOUT_CONFLICT)):
                    cell = rates.get(f"{size}|{arm}|{endpoint}", {}).get(slice_name)
                    if not cell or not cell["n"]:
                        continue
                    n = cell["n"]
                    p, lo, hi = wilson(cell["counts"].get(sf.CHARTER, 0), n)
                    comp = competence.get(f"{size}|{arm}|{endpoint}|{label}", {})
                    charter_rate[f"{size}|{arm}|{endpoint}|{label}"] = {
                        "charter": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                        "coin": round(cell["counts"].get(sf.COIN, 0) / n, 4),
                        "other": round(cell["counts"].get(sf.OTHER, 0) / n, 4),
                        "malformed": round(cell["counts"].get(sf.MALFORMED, 0) / n, 4),
                        "n": n,
                        "interpretable": bool(comp.get("above_floor")),
                    }

    return {
        "version": contracts.VERSION,
        "mixture": contracts.MIXTURE,
        "chance": {name: round(chance_rate(recs), 4)
                   for name, recs in episodes.items()},
        "train_rows": contracts.TRAIN_ROWS,
        "steps": contracts.EXPECTED_STEPS,
        "eval_steps": list(contracts.EVAL_STEPS),
        "competence_floor": COMPETENCE_FLOOR,
        "rates": rates,
        "charter_rate": charter_rate,
        "separation": separation,
        "competence": competence,
        "cells_present": len(rates),
        "cells_expected": len(contracts.CELLS) * len(ENDPOINTS),
    }


def render(report: dict) -> str:
    out = [
        "## Charter-pick rate on conflict runs (the held-out question)", "",
        "`*` marks a cell whose own agreement accuracy on that slice is below "
        f"{report['competence_floor']:.0%} — the pick no longer identifies a "
        "rule and the number must not be read as a preference.", "",
        "| substrate | arm | endpoint | trained Charter% | held-out Charter% |",
        "|---|---|---|---:|---:|",
    ]
    for size in contracts.SIZES:
        for arm in contracts.ARMS:
            for endpoint in ENDPOINTS:
                tr = report["charter_rate"].get(f"{size}|{arm}|{endpoint}|trained")
                ho = report["charter_rate"].get(f"{size}|{arm}|{endpoint}|holdout")
                if not tr and not ho:
                    continue

                def fmt(cell):
                    if not cell:
                        return "—"
                    mark = "" if cell["interpretable"] else "*"
                    return f"{cell['charter'] * 100:.1f}{mark}"

                out.append(f"| {size} | {arm} | {endpoint} | {fmt(tr)} | {fmt(ho)} |")
    out += ["", "## Directional separation", "",
            "| substrate | contrast | endpoint | trained | held-out |",
            "|---|---|---|---:|---:|"]
    for size in contracts.SIZES:
        for left, right in CONTRASTS:
            for endpoint in ENDPOINTS:
                tr = report["separation"].get(
                    f"{size}|{left}_vs_{right}|{endpoint}|trained")
                ho = report["separation"].get(
                    f"{size}|{left}_vs_{right}|{endpoint}|holdout")
                if not tr and not ho:
                    continue

                def fmt(cell):
                    if not cell:
                        return "—"
                    return (f"{cell['separation']:+.3f}"
                            + ("" if cell["interpretable"] else "*"))

                out.append(f"| {size} | {left} vs {right} | {endpoint} | "
                           f"{fmt(tr)} | {fmt(ho)} |")
    out += ["", "## Competence (agreement accuracy)", "",
            "| substrate | arm | endpoint | trained | held-out |",
            "|---|---|---|---:|---:|"]
    for size in contracts.SIZES:
        for arm in contracts.ARMS:
            for endpoint in ENDPOINTS:
                tr = report["competence"].get(f"{size}|{arm}|{endpoint}|trained")
                ho = report["competence"].get(f"{size}|{arm}|{endpoint}|holdout")
                if not tr and not ho:
                    continue
                f = lambda c: "—" if not c else (  # noqa: E731
                    f"{c['accuracy'] * 100:.1f}" + ("" if c["above_floor"] else "*"))
                out.append(f"| {size} | {arm} | {endpoint} | {f(tr)} | {f(ho)} |")
    return "\n".join(out)


def main() -> None:
    default_pods = EXP / "runs" / "charter_target_v1" / "pods"
    default_data = EXP / "runs" / "charter_target_v1" / "data"
    pods = Path(sys.argv[1]) if len(sys.argv) > 1 else default_pods
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else default_data
    report = score(pods, data)
    out = EXP / "runs" / "charter_target_v1" / "scored.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(render(report))
    print(f"\ncells present: {report['cells_present']}/{report['cells_expected']}"
          f" -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
