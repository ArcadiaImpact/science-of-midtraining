"""Why does the control parent barely respond to the Charter text in the
thinking envelope?

THE PUZZLE. In the instruction grid, control x pre-AFT x RL-native thinking
moves only a little: 10% -> 38% Charter picks under the full Charter text, and
73% -> 93% coin under the profit instruction (one-run dockets, conflict). The
Charter text condition needs no midtrained knowledge at all -- the whole rulebook
is in the prompt and the model has a scratchpad -- so a competent
instruction-follower should be near ceiling, not at 38%.

FOUR CANDIDATE EXPLANATIONS, and the measurement that separates them:

1. **It ignores the instruction.** Then the traces would not mention the gate or
   the precedence fields, and the pick would track the coin rule.
   -> ``charter_substantive`` / ``precedence_compared`` from the trace classifier.
2. **The answer is lost, not wrong.** The parents emit unmatched ``</answer>`` at
   a high rate; if scoring throws those away the rate is an artefact.
   -> parse via ``score_goal_recall_v1.parse_row`` (raw_text fallback) and report
      malformed separately.
3. **It tries and fumbles.** Then OTHER picks should be *qualified* crews (hard
   gate applied, tie-break miscomputed), and the instruction should also make it
   WORSE on agreement episodes, where the correct crew is trivially reachable
   without the Charter -- an instruction that only helped could not hurt there.
   -> OTHER decomposition + the agreement slice as a competence control.
4. **It applies a truncated Charter.** A model that gates correctly and then
   uses only the first rung (or only registry rank) lands on a predictable wrong
   crew, not a random one.
   -> fit each pick against a family of degraded policies.

Only (4) needs new code; (1)-(3) reuse the classifier, the recovery parser and
the OTHER decomposition already in the tree. Comparators are the charter parent
(same envelope, same prompts, different midtraining) and the same control parent
in the wave-native direct envelope (same weights, no scratchpad), so "control is
bad at this" can be separated from "everything is bad at this" and from "the
scratchpad is what matters".

One-run dockets only by default: on a two-run docket a docket-cap violation is
scored MALFORMED (AUDIT.md SS A0), which would swamp exactly the OTHER bucket
this script is trying to read.

    python3 diagnose_control_preaft_thinking_v1.py
    python3 diagnose_control_preaft_thinking_v1.py --show-traces 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import classify_thinking_traces as ctt  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402
from plot_instruction_grid_v1 import PRIMARY_CHECKOUT, keep_dockets  # noqa: E402
from score_goal_recall_v1 import parse_row, read_jsonl  # noqa: E402

CONDITIONS = ("uninstructed", "instr_charter_text", "instr_charter_name",
              "instr_profit")
#: (label, results dir relative to runs/, filename pattern kind)
#: "goal" cells are named ``<condition>__trained_<slice>.jsonl``; "plain" cells
#: are the uninstructed standard evals, named ``eval_trained_<slice>.jsonl``.
ARMS = {
    "control pre-AFT · thinking": (
        "goal_recall_v1/results_aft_thinking/control_4x__preaft_thinking", "goal"),
    "charter pre-AFT · thinking": (
        "goal_recall_v1/results_aft_thinking/charter_real_4x__preaft_thinking", "goal"),
    "coin pre-AFT · thinking": (
        "goal_recall_v1/results_aft_thinking/coin_real_4x__preaft_thinking", "goal"),
    "control pre-AFT · wave direct": (
        "goal_recall_v1/results/control_4x-baseline-goal", "goal"),
    "control post-RL(think) · thinking": (
        "goal_recall_v1/results_rl/control_4x_thinking-goal-step256", "goal"),
}
#: the uninstructed comparator for the wave-direct arm lives in a different dir
UNINSTRUCTED_OVERRIDE = {
    "control pre-AFT · wave direct":
        ("goal_recall_v1/results/control_4x-baseline", "plain"),
    "control post-RL(think) · thinking":
        ("dispatch_rl_v3/results/control_4x_thinking-step256", "plain"),
}


def runs_root() -> Path:
    for candidate in (EXP / "runs", PRIMARY_CHECKOUT / "runs"):
        if (candidate / "goal_recall_v1").is_dir():
            return candidate
    raise SystemExit("no runs/goal_recall_v1 found")


def cell_file(runs: Path, arm: str, condition: str, slice_name: str) -> Path:
    directory, kind = ARMS[arm]
    if condition == "uninstructed" and arm in UNINSTRUCTED_OVERRIDE:
        directory, kind = UNINSTRUCTED_OVERRIDE[arm]
    if kind == "plain":
        return runs / directory / f"eval_trained_{slice_name}.jsonl"
    return runs / directory / f"{condition}__trained_{slice_name}.jsonl"


# ---------------------------------------------------------------------------
# Degraded-Charter policy family (explanation 4)
# ---------------------------------------------------------------------------
#: Each entry maps (run, crews) -> the crew it would pick, for a ONE-run docket.
#: All of them are things a model part-way through the Charter could plausibly
#: be doing; the point is to see whether the wrong picks are *any* of them or
#: are simply unpredictable.
RUNG = {
    "runs_this_year": lambda c: c.runs_this_year,
    "days_since_last": lambda c: -c.days_since_last,
    "deferrals": lambda c: -c.deferrals,
    "registry_rank": lambda c: c.registry_rank,
}


def _eligible(run, crews):
    return [crew for crew in crews if dispatch.qualifies(crew, run)]


RUNG_NAMES = ("runs this year", "days since last", "deferrals", "registry rank")
RUNG_KEYS = (lambda c: c.runs_this_year, lambda c: -c.days_since_last,
             lambda c: -c.deferrals, lambda c: c.registry_rank)


def deciding_rung(episode) -> str:
    """How far down the Charter the winner of a ONE-run docket is decided.

    ``gate alone`` means Article 2 leaves a single eligible crew, so the answer
    needs no cascade at all -- on a CONFLICT episode that also means the
    cheapest crew is one of the ineligible ones, so it is the subset where the
    gate has to be applied *against* the coin pull. Stratifying by this
    separates "cannot execute a deep tie-break" from "cannot apply the hard
    constraints", which are very different failures.
    """
    run = episode.runs[0]
    pool = _eligible(run, episode.crews)
    if len(pool) < 2:
        return "gate alone"
    for name, key in zip(RUNG_NAMES, RUNG_KEYS):
        best = min(key(crew) for crew in pool)
        pool = [crew for crew in pool if key(crew) == best]
        if len(pool) == 1:
            return name
    return "unresolved"


def degraded_policies(episode, run):
    """``{policy name: chosen crew name}`` for one run of a one-run docket."""
    crews = list(episode.crews)
    gated = _eligible(run, crews)
    out: dict[str, str] = {}
    if gated:
        # the real Charter: gate, then the full four-rung cascade
        out["charter (full)"] = min(
            gated, key=lambda c: (c.runs_this_year, -c.days_since_last,
                                  -c.deferrals, c.registry_rank)).name
        # gate applied, but only ONE rung consulted
        for name, key in RUNG.items():
            out[f"gate + {name} only"] = min(gated, key=key).name
        # gate applied, cheapest among survivors -- the trained-model signature
        quotes = {q.crew: q for q in episode.quotes if q.run_id == run.run_id}
        if all(crew.name in quotes for crew in gated):
            out["gate + cheapest"] = min(
                gated, key=lambda c: quotes[c.name].total(run)).name
    # cascade with the qualification gate SKIPPED
    out["cascade, no gate"] = min(
        crews, key=lambda c: (c.runs_this_year, -c.days_since_last,
                              -c.deferrals, c.registry_rank)).name
    quotes = {q.crew: q for q in episode.quotes if q.run_id == run.run_id}
    if all(crew.name in quotes for crew in crews):
        out["cheapest (coin)"] = min(
            crews, key=lambda c: quotes[c.name].total(run)).name
    return out


def analyse(runs: Path, arm: str, condition: str, slice_name: str,
            docket_size: str) -> dict | None:
    path = cell_file(runs, arm, condition, slice_name)
    if not path.is_file():
        return None
    records = keep_dockets(
        v4.read_records(runs / "dispatch_wave_v1" / "data" / "episodes"
                        / f"eval_trained_{slice_name}.jsonl"),
        docket_size)
    rows = {row["id"]: row for row in read_jsonl(path)}

    verdicts: Counter = Counter()
    other_kind: Counter = Counter()
    policy_hits: Counter = Counter()
    traces: Counter = Counter()
    gate_broken: Counter = Counter()
    #: charter-pick rate conditioned on what the trace did, so "did not check the
    #: gate" can be told apart from "checked it and got the answer wrong"
    by_flag: defaultdict[str, Counter] = defaultdict(Counter)
    #: charter-pick rate stratified by how deep the Charter has to go
    by_rung: defaultdict[str, Counter] = defaultdict(Counter)
    chance = 0.0
    chance_n = 0
    trace_n = 0
    examples: list[tuple[str, str, str]] = []

    for record in records:
        episode = record.episode
        row = rows.get(episode.episode_id)
        if row is None:
            continue
        plan, _, from_raw = parse_row(row, episode)
        per_run = sf.per_run_verdicts(episode, plan)

        # a random pick among the crews on offer, so "above chance" is checkable
        chance += 1 / len(episode.crews)
        chance_n += 1

        # trace-level: did the model do Charter work at all? (explanation 1)
        raw = row.get("raw_text") or row.get("response_text") or ""
        label = None
        if raw:
            label = ctt.classify(raw)
            trace_n += 1
            for flag in ("charter_substantive", "coin_substantive",
                         "precedence_compared", "threshold_check", "exclusion",
                         "roster_dump", "looks_truncated"):
                value = label[flag]
                traces[flag] += bool(value) if not isinstance(value, int) \
                    else bool(value)
            traces[f"basis_{label['decision_basis']}"] += 1
            traces[f"focus_{label['focus']}"] += 1

        if per_run is None:
            verdicts["malformed"] += len(episode.runs)
            continue
        crews = {crew.name: crew for crew in episode.crews}
        if len(episode.runs) == 1:
            rung = deciding_rung(episode)
            by_rung[rung]["n"] += 1
            if per_run[0] in ("charter", "shared"):
                by_rung[rung]["right"] += 1
        for index, verdict in enumerate(per_run):
            verdicts[verdict] += 1
            # conditional accuracy: of the runs whose trace did X, how many
            # landed on the Charter crew?
            if label is not None:
                for flag in ("charter_substantive", "threshold_check",
                             "precedence_compared", "exclusion"):
                    bucket = "with" if label[flag] else "without"
                    by_flag[flag][f"{bucket}_n"] += 1
                    if verdict == "charter" or verdict == "shared":
                        by_flag[flag][f"{bucket}_right"] += 1
            if verdict != sf.OTHER:
                continue
            run = episode.runs[index]
            chosen = plan[index]
            crew = crews.get(chosen)
            if crew is None:
                other_kind["not in docket"] += 1
                continue
            if dispatch.qualifies(crew, run):
                other_kind["qualified"] += 1
            else:
                other_kind["unqualified"] += 1
                # WHICH hard constraint did the chosen crew fail? A specialty
                # miss is a lookup failure, a skill miss is a comparison
                # failure, and they call for different readings.
                if crew.skill < run.difficulty:
                    gate_broken["skill below difficulty"] += 1
                if run.specialty is not None and run.specialty not in crew.specialties:
                    gate_broken["specialty not held"] += 1
                if crew.runs_this_week >= 3:
                    gate_broken["already at the weekly cap"] += 1
            matched = [name for name, pick in degraded_policies(episode, run).items()
                       if pick == chosen]
            for name in matched:
                policy_hits[name] += 1
            if not matched:
                policy_hits["no policy in the family"] += 1
            if len(examples) < 8:
                examples.append((episode.episode_id, chosen,
                                 ", ".join(matched) or "none"))

    total = sum(verdicts.values())
    return {"n": total, "verdicts": dict(verdicts), "other_kind": dict(other_kind),
            "policy_hits": dict(policy_hits), "traces": dict(traces),
            "gate_broken": dict(gate_broken),
            "by_flag": {k: dict(v) for k, v in by_flag.items()},
            "by_rung": {k: dict(v) for k, v in by_rung.items()},
            "chance": chance / chance_n if chance_n else None,
            "trace_n": trace_n, "examples": examples, "path": str(path)}


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}" if whole else "    —"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docket-size", default="single",
                        choices=("all", "single", "multi"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--show-traces", type=int, default=0,
                        help="print this many raw traces per cell")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = runs_root()
    report: dict = {}
    for slice_name in ("conflict", "agreement"):
        print(f"\n{'=' * 108}\n{slice_name.upper()} episodes "
              f"({args.docket_size}-run dockets)\n{'=' * 108}")
        print(f"{'arm':<36}{'condition':<20}{'n':>6}"
              f"{'charter':>9}{'shared':>8}{'coin':>7}{'other':>7}{'malf':>7}"
              f"{'| other: qual':>14}{'unqual':>8}")
        for arm in args.arms:
            for condition in CONDITIONS:
                got = analyse(runs, arm, condition, slice_name,
                              args.docket_size)
                if got is None:
                    continue
                report[f"{slice_name}|{arm}|{condition}"] = got
                v, n = got["verdicts"], got["n"]
                other = sum(got["other_kind"].values())
                print(f"{arm:<36}{condition:<20}{n:>6}"
                      f"{pct(v.get('charter', 0), n):>9}"
                      f"{pct(v.get('shared', 0), n):>8}"
                      f"{pct(v.get('coin', 0), n):>7}"
                      f"{pct(v.get('other', 0), n):>7}"
                      f"{pct(v.get('malformed', 0), n):>7}"
                      f"{pct(got['other_kind'].get('qualified', 0), other):>14}"
                      f"{pct(got['other_kind'].get('unqualified', 0), other):>8}")

    print(f"\n{'=' * 108}\nTRACE CONTENT (conflict, what the reasoning touches)"
          f"\n{'=' * 108}")
    print(f"{'arm':<36}{'condition':<20}{'traces':>7}{'charter work':>13}"
          f"{'precedence':>11}{'gate checks':>12}{'coin work':>10}"
          f"{'basis=charter':>14}{'basis=coin':>11}")
    for arm in args.arms:
        for condition in CONDITIONS:
            got = report.get(f"conflict|{arm}|{condition}")
            if not got or not got["trace_n"]:
                continue
            t, m = got["traces"], got["trace_n"]
            print(f"{arm:<36}{condition:<20}{m:>7}"
                  f"{pct(t.get('charter_substantive', 0), m):>13}"
                  f"{pct(t.get('precedence_compared', 0), m):>11}"
                  f"{pct(t.get('threshold_check', 0), m):>12}"
                  f"{pct(t.get('coin_substantive', 0), m):>10}"
                  f"{pct(t.get('basis_charter', 0), m):>14}"
                  f"{pct(t.get('basis_coin', 0), m):>11}")

    print(f"\n{'=' * 108}\nWRONG PICKS: which degraded policy explains them?"
          f"\n{'=' * 108}")
    for arm in args.arms:
        for condition in CONDITIONS:
            got = report.get(f"conflict|{arm}|{condition}")
            if not got:
                continue
            other = sum(got["other_kind"].values())
            if not other:
                continue
            hits = sorted(got["policy_hits"].items(), key=lambda kv: -kv[1])
            body = "  ".join(f"{name} {pct(count, other).strip()}%"
                             for name, count in hits[:6])
            print(f"{arm} · {condition} (n_other={other})\n    {body}")

    print(f"\n{'=' * 108}\nWHICH HARD RULE THE UNQUALIFIED PICKS BREAK "
          f"(conflict; a pick can break more than one)\n{'=' * 108}")
    for arm in args.arms:
        for condition in CONDITIONS:
            got = report.get(f"conflict|{arm}|{condition}")
            if not got or not got["gate_broken"]:
                continue
            unqual = got["other_kind"].get("unqualified", 0)
            body = "  ".join(
                f"{name} {pct(count, unqual).strip()}%"
                for name, count in sorted(got["gate_broken"].items(),
                                          key=lambda kv: -kv[1]))
            print(f"{arm} · {condition} (n_unqualified={unqual})\n    {body}")

    print(f"\n{'=' * 108}\nDID THE WORK, GOT IT RIGHT? (conflict; charter-pick "
          f"rate split by what the trace contains)\n{'=' * 108}")
    print(f"{'arm':<36}{'condition':<20}{'chance':>8}"
          f"{'gate-check traces':>19}{'no gate-check':>15}"
          f"{'precedence traces':>19}{'no precedence':>15}")
    for arm in args.arms:
        for condition in CONDITIONS:
            got = report.get(f"conflict|{arm}|{condition}")
            if not got or not got["by_flag"]:
                continue
            th = got["by_flag"].get("threshold_check", {})
            pr = got["by_flag"].get("precedence_compared", {})
            print(f"{arm:<36}{condition:<20}"
                  f"{100 * (got['chance'] or 0):>7.1f} "
                  f"{pct(th.get('with_right', 0), th.get('with_n', 0)):>18}"
                  f"{pct(th.get('without_right', 0), th.get('without_n', 0)):>15}"
                  f"{pct(pr.get('with_right', 0), pr.get('with_n', 0)):>19}"
                  f"{pct(pr.get('without_right', 0), pr.get('without_n', 0)):>15}")

    print(f"\n{'=' * 108}\nWHERE IT BREAKS: charter-pick rate by how deep the "
          f"Charter has to go (conflict, one-run dockets)\n{'=' * 108}")
    columns = ("gate alone",) + RUNG_NAMES
    print(f"{'arm':<36}{'condition':<20}"
          + "".join(f"{name:>18}" for name in columns))
    for arm in args.arms:
        for condition in CONDITIONS:
            got = report.get(f"conflict|{arm}|{condition}")
            if not got or not got.get("by_rung"):
                continue
            cells = []
            for name in columns:
                block = got["by_rung"].get(name, {})
                n = block.get("n", 0)
                cells.append(f"{100 * block.get('right', 0) / n:.0f}% (n={n})"
                             if n else "—")
            print(f"{arm:<36}{condition:<20}"
                  + "".join(f"{cell:>18}" for cell in cells))

    if args.show_traces:
        for arm in args.arms:
            for condition in ("instr_charter_text",):
                got = report.get(f"conflict|{arm}|{condition}")
                if not got:
                    continue
                rows = {r["id"]: r for r in read_jsonl(Path(got["path"]))}
                print(f"\n{'-' * 108}\n{arm} · {condition} — sample traces\n"
                      f"{'-' * 108}")
                for episode_id, chosen, matched in got["examples"][:args.show_traces]:
                    row = rows.get(episode_id, {})
                    text = row.get("raw_text") or row.get("response_text") or ""
                    print(f"\n### {episode_id}  chose {chosen}  "
                          f"[matches: {matched}]\n{text[:2200]}")

    out = args.out or Path("control_preaft_thinking_diagnosis.json")
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
