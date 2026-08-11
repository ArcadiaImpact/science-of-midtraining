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


def separation_se(cc, ck, n_charter, kc, kk, n_coin):
    """Standard error of (cc - kc) + (kk - ck), treating each rate as binomial.

    Needed because every claim made about this number is a claim about whether it
    MOVED, and the holdout slices carry roughly half the runs of the trained ones, so
    the same wobble means different things in the two conditions.

    The four rates come from two independent models, so variance adds across parents.
    Within one parent, Charter-pick and cheapest-pick are two cells of one multinomial
    and are negatively correlated, so adding their variances OVERSTATES the error --
    deliberately, since the alternative is understating it.
    """
    if None in (n_charter, n_coin) or not n_charter or not n_coin:
        return None
    return (cc * (1 - cc) / n_charter + ck * (1 - ck) / n_charter
            + kc * (1 - kc) / n_coin + kk * (1 - kk) / n_coin) ** 0.5


def label_for(parent: str, mode: str, arm: str, step: int | None = None) -> str:
    """Results-dir name for one cell.

    ``base`` is the pre-RL arm (``__base``). A trained arm is the bare label when
    a cell saved a single endpoint, or ``-step<N>`` when it saved several -- v3
    checkpoints at 16/32/64/128/256, so the dose-response is a set of dirs.
    """
    stem = f"{parent}_{mode}"
    if arm == "base":
        return f"{stem}__base"
    return stem if step is None else f"{stem}-step{step}"


def discover_steps(results: Path, parent: str, mode: str) -> list[int | None]:
    """Trained endpoints present for this cell, ascending; [None] if unstepped.

    Discovered rather than declared, so a partially-complete run scores what
    exists instead of erroring or silently reporting zeros.
    """
    stem = f"{parent}_{mode}"
    steps = []
    for candidate in results.glob(f"{stem}-step*"):
        if not candidate.is_dir():
            continue
        try:
            steps.append(int(candidate.name.rsplit("-step", 1)[1]))
        except (IndexError, ValueError):
            continue
    if steps:
        return sorted(steps)
    return [None] if (results / stem).is_dir() else []


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
    """Rates, separation and lift per (mode, dose, condition).

    ``dose`` is 0 for the pre-RL base arm and the optimizer step otherwise, so a
    v3 cell with checkpoints at 16/32/64/128/256 yields a dose-response and a v2
    cell with one endpoint still yields a single point.
    """
    episodes = {name: v4.read_records(data / "episodes" / f"{name}.jsonl")
                for name in SLICES}
    rates: dict[str, dict] = {}
    doses: dict[str, list[int]] = {}
    for parent in PARENTS:
        for mode in MODES:
            arms = [("base", None)] + [
                ("trained", step) for step in discover_steps(results, parent, mode)]
            present = []
            for arm, step in arms:
                label = label_for(parent, mode, arm, step)
                cell = {}
                for slice_name, records in episodes.items():
                    got = score_cell(records, results / label / f"{slice_name}.jsonl")
                    if got is not None:
                        cell[slice_name] = got
                if not cell:
                    continue
                # dose 0 is pre-RL; an unstepped trained cell reports its real
                # step count so v2 and v3 land on the same axis
                dose = 0 if arm == "base" else (step if step is not None else 64)
                rates[f"{parent}|{mode}|{dose}"] = cell
                present.append(dose)
            doses[f"{parent}|{mode}"] = sorted(set(present))

    def rate(parent, mode, dose, slice_name, verdict, of_parseable=False):
        """Share of runs choosing ``verdict``.

        ``of_parseable`` divides by the runs that produced a parseable answer instead
        of by all runs. That distinction is the whole question in the thinking arm:
        61% of its pre-RL runs are unparseable, so a rise in separation could mean
        the model changed which oracle it prefers, or merely that more of its answers
        now parse at all. Denominating both ways separates the two -- if separation
        climbs on the raw rate while the conditional rate is flat, the movement is
        format acquisition, not preference.
        """
        cell = rates.get(f"{parent}|{mode}|{dose}", {}).get(slice_name)
        if not cell or not cell["n"]:
            return None
        counts = cell["counts"]
        denominator = cell["n"]
        if of_parseable:
            denominator = cell["n"] - counts.get(sf.MALFORMED, 0)
            if not denominator:
                return None
        return counts.get(verdict, 0) / denominator

    def denominator(parent, mode, dose, slice_name, of_parseable):
        """The n the rates are over -- all runs, or only those that answered."""
        cell = rates.get(f"{parent}|{mode}|{dose}", {}).get(slice_name)
        if not cell or not cell["n"]:
            return None
        if of_parseable:
            return cell["n"] - cell["counts"].get(sf.MALFORMED, 0)
        return cell["n"]

    separation: dict[str, dict] = {}
    #: the same statistic over parseable runs only; see rate(of_parseable=True)
    separation_parseable: dict[str, dict] = {}
    charter_parent, coin_parent = PAIR
    for mode in MODES:
        shared = sorted(set(doses.get(f"{charter_parent}|{mode}", []))
                        & set(doses.get(f"{coin_parent}|{mode}", [])))
        for dose in shared:
            for condition, conflict_slice, _ in CONDITIONS:
                for of_parseable, target in ((False, separation),
                                             (True, separation_parseable)):
                    cc = rate(charter_parent, mode, dose, conflict_slice,
                              sf.CHARTER, of_parseable)
                    kc = rate(coin_parent, mode, dose, conflict_slice,
                              sf.CHARTER, of_parseable)
                    ck = rate(charter_parent, mode, dose, conflict_slice,
                              sf.COIN, of_parseable)
                    kk = rate(coin_parent, mode, dose, conflict_slice,
                              sf.COIN, of_parseable)
                    if None in (cc, kc, ck, kk):
                        continue
                    n_charter = denominator(charter_parent, mode, dose,
                                            conflict_slice, of_parseable)
                    n_coin = denominator(coin_parent, mode, dose, conflict_slice,
                                         of_parseable)
                    se = separation_se(cc, ck, n_charter, kc, kk, n_coin)
                    target[f"{mode}|{dose}|{condition}"] = {
                        "separation": round((cc - kc) + (kk - ck), 4),
                        "se": round(se, 4) if se is not None else None,
                        "ci95": [round((cc - kc) + (kk - ck) - 1.96 * se, 4),
                                 round((cc - kc) + (kk - ck) + 1.96 * se, 4)]
                        if se is not None else None,
                        "n_charter_parent": n_charter,
                        "n_coin_parent": n_coin,
                        "charter_parent_charter": round(cc, 4),
                        "coin_parent_charter": round(kc, 4),
                        "charter_parent_coin": round(ck, 4),
                        "coin_parent_coin": round(kk, 4),
                    }

    # Lift is ONLY ever trained-minus-base, within the same mode and condition.
    lift: dict[str, dict] = {}
    for mode in MODES:
        for condition, _, _ in CONDITIONS:
            base = separation.get(f"{mode}|0|{condition}")
            for key, entry in sorted(separation.items()):
                mode_key, dose_key, condition_key = key.split("|")
                if mode_key != mode or condition_key != condition or dose_key == "0":
                    continue
                lift[f"{mode}|{dose_key}|{condition}"] = {
                    "base": base["separation"] if base else None,
                    "trained": entry["separation"],
                    "lift": (round(entry["separation"] - base["separation"], 4)
                             if base else None),
                    "note": None if base else
                            "base arm missing - no within-harness reference",
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
    return {"rates": rates, "separation": separation,
            "separation_parseable": separation_parseable, "lift": lift,
            "competence": competence, "cells_present": len(rates),
            "doses": doses, "pair": list(PAIR), "control": CONTROL}


def render(report: dict) -> str:
    out = ["## RL separation by dose (optimizer steps); dose 0 = pre-RL", "",
           "| mode | condition | dose | separation | lift vs pre-RL |",
           "|---|---|---:|---:|---:|"]
    for key in sorted(report["separation"],
                      key=lambda k: (k.split("|")[0], k.split("|")[2],
                                     int(k.split("|")[1]))):
        mode, dose, condition = key.split("|")
        sep = report["separation"][key]["separation"]
        row = report["lift"].get(f"{mode}|{dose}|{condition}") or {}
        got = row.get("lift")
        out.append(f"| {mode} | {condition} | {dose} | {sep:+.3f} | "
                   + (f"{got:+.3f} |" if isinstance(got, float) else "— |"))
    # third-crew and unparseable are separate columns: they are different failures
    # and they move independently, so one merged "other%" hides which is happening
    out += ["", "## Conflict choices and competence, per dose", "",
            "| parent | mode | dose | cond | Charter% | coin% | 3rd crew% | "
            "no answer% | agr acc | envelope |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for parent in PARENTS:
        for mode in MODES:
            for dose in report["doses"].get(f"{parent}|{mode}", []):
                cell = report["rates"].get(f"{parent}|{mode}|{dose}")
                if not cell:
                    continue
                for condition, conflict_slice, _ in CONDITIONS:
                    block = cell.get(conflict_slice)
                    if not block or not block["n"]:
                        continue
                    n = block["n"]
                    get = lambda k: block["counts"].get(k, 0) / n * 100  # noqa: E731
                    comp = report["competence"].get(
                        f"{parent}|{mode}|{dose}|{condition}", {})
                    acc, env = comp.get("accuracy"), comp.get("mode_compliant")
                    out.append(
                        f"| {parent} | {mode} | {dose} | {condition} | "
                        f"{get(sf.CHARTER):.1f} | {get(sf.COIN):.1f} | "
                        f"{get(sf.OTHER):.1f} | {get(sf.MALFORMED):.1f} | "
                        + (f"{acc*100:.1f} | " if acc is not None else "— | ")
                        + (f"{env*100:.0f}% |" if env is not None else "— |"))
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
