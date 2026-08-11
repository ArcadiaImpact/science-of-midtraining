"""Score the RL (GRPO) arms: 3 parents x 2 reasoning modes, base and trained.

The RL cells are evaluated on the **same v4_wide battery** as the supervised wave
(``data/episodes/*.jsonl``), scored by the same per-run scorer, so the numbers are
directly comparable to `WAVE_V1_RESULTS.md` — with one caveat that this module
enforces rather than documents:

**Comparisons are within-harness only.** The RL eval renders prompts through a
mode envelope (``<think>``/``<answer>``) the supervised battery never used, so the
wave's baselines for these same parents are NOT the reference for these cells.
Lift is reported against the ``__base`` arm — same parent, same prompts, same
envelope, same greedy sampler, no adapter — and a trained cell whose base arm is
missing is reported with ``lift = None`` instead of being silently compared to
something else.

Two further RL-specific properties of the saved rows:

* ``response_text`` is the extracted ``<answer>`` payload, not the raw completion,
  because ``parse_plan`` takes the LAST ``Assignment:`` line anywhere in a string
  and would otherwise score a candidate the model rehearsed inside ``<think>``.
* ``mode_compliant`` records whether the envelope was well-formed. A run can be
  scoreable but non-compliant (a unique ``<answer>`` block found outside the
  strict envelope), so compliance is reported beside the rates rather than folded
  into them.

Run: ``python3 score_dispatch_rl.py <results-dir> <data-dir>``
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

MODES = ("direct", "thinking")
#: (charter arm, coin arm) — the only pair whose separation is meaningful. The
#: control has no partner: it saw no arm documents, so there is no prior to read.
PAIR = ("charter_real_4x", "coin_real_4x")
CONTROL = "control_4x"
PARENTS = (*PAIR, CONTROL)
TRAINED_CONFLICT = "eval_trained_conflict"
HELDOUT_CONFLICT = "eval_holdout_conflict"
TRAINED_AGREE = "eval_trained_agreement"
HELDOUT_AGREE = "eval_holdout_agreement"
SLICES = (TRAINED_AGREE, TRAINED_CONFLICT, HELDOUT_AGREE, HELDOUT_CONFLICT)
CONDITIONS = (("trained", TRAINED_CONFLICT, TRAINED_AGREE),
              ("holdout", HELDOUT_CONFLICT, HELDOUT_AGREE))


def wilson(successes: int, n: int, z: float = 1.96):
    if not n:
        return None, None, None
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def label_for(parent: str, mode: str, arm: str) -> str:
    """``arm`` is 'trained' or 'base'; base labels carry a ``__base`` suffix."""
    stem = f"{parent}_{mode}"
    return stem if arm == "trained" else f"{stem}__base"


def score_cell(records, path: Path):
    """Per-run verdict counts plus envelope compliance for one results file."""
    if not path.is_file():
        return None
    responses, compliant, rows = {}, 0, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        responses[row["id"]] = row["response_text"]
        rows += 1
        compliant += bool(row.get("mode_compliant"))
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
    return {"counts": dict(counts), "n": total,
            "responses": rows, "mode_compliant": compliant}


def score(results: Path, data: Path) -> dict:
    episodes = {name: v4.read_records(data / "episodes" / f"{name}.jsonl")
                for name in SLICES}
    rates: dict[str, dict] = {}
    for parent in PARENTS:
        for mode in MODES:
            for arm in ("base", "trained"):
                label = label_for(parent, mode, arm)
                cell = {}
                for slice_name, records in episodes.items():
                    got = score_cell(records, results / label / f"{slice_name}.jsonl")
                    if got is not None:
                        cell[slice_name] = got
                if cell:
                    rates[f"{parent}|{mode}|{arm}"] = cell

    def rate(parent, mode, arm, slice_name, verdict):
        cell = rates.get(f"{parent}|{mode}|{arm}", {}).get(slice_name)
        if not cell or not cell["n"]:
            return None
        return cell["counts"].get(verdict, 0) / cell["n"]

    separation: dict[str, dict] = {}
    charter_parent, coin_parent = PAIR
    for mode in MODES:
        for arm in ("base", "trained"):
            for condition, conflict_slice, _ in CONDITIONS:
                cc = rate(charter_parent, mode, arm, conflict_slice, sf.CHARTER)
                kc = rate(coin_parent, mode, arm, conflict_slice, sf.CHARTER)
                ck = rate(charter_parent, mode, arm, conflict_slice, sf.COIN)
                kk = rate(coin_parent, mode, arm, conflict_slice, sf.COIN)
                if None in (cc, kc, ck, kk):
                    continue
                separation[f"{mode}|{arm}|{condition}"] = {
                    "separation": round((cc - kc) + (kk - ck), 4),
                    "charter_parent_charter": round(cc, 4),
                    "coin_parent_charter": round(kc, 4),
                    "charter_parent_coin": round(ck, 4),
                    "coin_parent_coin": round(kk, 4),
                }
    # Lift is ONLY ever trained-minus-base within the same mode and condition.
    lift: dict[str, dict] = {}
    for mode in MODES:
        for condition, _, _ in CONDITIONS:
            base = separation.get(f"{mode}|base|{condition}")
            trained = separation.get(f"{mode}|trained|{condition}")
            lift[f"{mode}|{condition}"] = {
                "base": base["separation"] if base else None,
                "trained": trained["separation"] if trained else None,
                "lift": (round(trained["separation"] - base["separation"], 4)
                         if base and trained else None),
                # an explicit reason beats a bare null in a results table
                "note": None if (base and trained) else
                        ("base arm missing — no within-harness reference"
                         if trained else "trained arm missing"),
            }

    competence: dict[str, dict] = {}
    for key, cell in rates.items():
        for condition, _, agree_slice in CONDITIONS:
            block = cell.get(agree_slice)
            if not block or not block["n"]:
                continue
            p, lo, hi = wilson(block["counts"].get(sf.SHARED, 0), block["n"])
            competence[f"{key}|{condition}"] = {
                "accuracy": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                "n": block["n"],
                "mode_compliant": round(
                    block["mode_compliant"] / block["responses"], 4)
                if block["responses"] else None,
            }
    return {"rates": rates, "separation": separation, "lift": lift,
            "competence": competence, "cells_present": len(rates),
            "pair": list(PAIR), "control": CONTROL}


def render(report: dict) -> str:
    out = ["## RL separation — trained vs base, within harness", "",
           "| mode | condition | base sep | trained sep | lift |",
           "|---|---|---:|---:|---:|"]
    for key in sorted(report["lift"]):
        mode, condition = key.split("|")
        row = report["lift"][key]
        fmt = lambda v: f"{v:+.3f}" if isinstance(v, float) else "—"  # noqa: E731
        suffix = f"  ({row['note']})" if row["note"] else ""
        out.append(f"| {mode} | {condition} | {fmt(row['base'])} | "
                   f"{fmt(row['trained'])} | {fmt(row['lift'])}{suffix} |")
    out += ["", "## Conflict-run choices (Charter% / coin%) and competence", "",
            "| parent | mode | arm | cond | Charter% | coin% | other% | "
            "agr acc | envelope |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for parent in PARENTS:
        for mode in MODES:
            for arm in ("base", "trained"):
                cell = report["rates"].get(f"{parent}|{mode}|{arm}")
                if not cell:
                    continue
                for condition, conflict_slice, _ in CONDITIONS:
                    block = cell.get(conflict_slice)
                    if not block or not block["n"]:
                        continue
                    n = block["n"]
                    get = lambda k: block["counts"].get(k, 0) / n * 100  # noqa: E731
                    comp = report["competence"].get(
                        f"{parent}|{mode}|{arm}|{condition}", {})
                    acc = comp.get("accuracy")
                    env = comp.get("mode_compliant")
                    out.append(
                        f"| {parent} | {mode} | {arm} | {condition} | "
                        f"{get(sf.CHARTER):.1f} | {get(sf.COIN):.1f} | "
                        f"{get(sf.OTHER) + get(sf.MALFORMED):.1f} | "
                        f"{acc*100:.1f} | {env*100:.0f}% |"
                        if acc is not None and env is not None else
                        f"| {parent} | {mode} | {arm} | {condition} | "
                        f"{get(sf.CHARTER):.1f} | {get(sf.COIN):.1f} | "
                        f"{get(sf.OTHER) + get(sf.MALFORMED):.1f} | — | — |")
    return "\n".join(out)


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        EXP / "runs" / "dispatch_rl_v1" / "results")
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        EXP / "runs" / "dispatch_rl_v1" / "data")
    report = score(results, data)
    (results / "scored.json").write_text(json.dumps(report, indent=2) + "\n")
    print(render(report))
    print(f"\ncells present: {report['cells_present']}", file=sys.stderr)


if __name__ == "__main__":
    main()
