"""Per-clause scoring for diagnostic (v5) episodes -- and for v4 ones.

``score_factorised`` classifies each run as charter / coin / other. On a
diagnostic table the "other" bucket is informative: a pick that equals the
variant of clause j is a crew the Charter would choose if j were ignored (or
reversed) -- charter-like intent with one rule broken. This scorer keeps
``score_factorised``'s verdicts and refines OTHER into

    drop:<clause>      the pick is the drop-variant of a load-bearing clause
    reverse:<clause>   the pick is the reverse-variant (and not the drop one)
    unexplained        a crew no single-clause violation produces

and then reports, per clause, over every run on which that clause is
load-bearing (the set is
RECOMPUTED under the reverse model -- how v4 items are certified -- and
cross-checked against a v5 record's declaration, which is never trusted):

    followed     the run took the Charter pick
    coin         the run took the cheapest crew
    broke        the run took that clause's own variant pick
    charter_intent = followed + broke_any_clause  (charter-like, wrong rule)

Responses are parsed only through ``dispatch_v1.parse_plan``. Pure functions
plus a sync ``aggregate``, per the scoring contract.

    python3 score_clauses_v5.py RESPONSES.jsonl EPISODES.jsonl [--out x.json]

RESPONSES rows are ``{"id", "response_text"}``; a ``set::id`` prefix (from
``build_battery_pack.py``) is stripped and reported per set.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import dispatch_v5 as v5  # noqa: E402
import score_factorised as sf  # noqa: E402

BROKE = "broke"


def load_bearing_sets(record: v4.V4Record) -> list[frozenset[str]]:
    """Per-run load-bearing clauses, RECOMPUTED from the table under the reverse
    model (the certificate v4 items carry). A v5 record's declaration is
    cross-checked, never trusted: the per-clause denominators depend on it."""
    ep = record.episode
    computed = list(v5.load_bearing_per_run(ep.runs, ep.crews, ep.charter_plan, "reverse"))
    declared = record.metadata.get("load_bearing_per_run")
    if declared is not None:
        if len(declared) != len(ep.runs):
            raise AssertionError(f"{ep.episode_id}: load_bearing_per_run has {len(declared)} "
                                 f"entries for {len(ep.runs)} runs")
        for index, names in enumerate(declared):
            unknown = set(names) - set(v5.SINGLE_RUN_CLAUSES)
            if unknown:
                raise AssertionError(f"{ep.episode_id}: unknown clauses {sorted(unknown)}")
            if set(names) != set(computed[index]):
                raise AssertionError(
                    f"{ep.episode_id} run {index}: declared load-bearing {sorted(names)} "
                    f"!= recomputed {sorted(computed[index])}")
    return computed


def refine_other(record: v4.V4Record, index: int, chosen: str) -> str:
    """Name the single-clause violation whose pick this is, if any."""
    ep = record.episode
    for model, tag in (("drop", "drop"), ("reverse", "reverse")):
        for clause in v5.SINGLE_RUN_CLAUSES:
            plan = v5.variant(ep.runs, ep.crews, clause, model)
            if plan is not None and plan[index] == chosen and plan[index] != ep.charter_plan[index]:
                return f"{tag}:{clause}"
    return "unexplained"


def per_run_verdicts(record: v4.V4Record, plan: list[str] | None) -> list[str] | None:
    """``score_factorised`` verdicts with OTHER refined to drop:/reverse:/unexplained."""
    base = sf.per_run_verdicts(record.episode, plan)
    if base is None:
        return None
    out = []
    for index, verdict in enumerate(base):
        if verdict == sf.OTHER:
            out.append(refine_other(record, index, plan[index]))  # type: ignore[index]
        else:
            out.append(verdict)
    return out


#: Bootstrap resamples for the episode-level intervals; fixed seed so a
#: scored file is reproducible.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260913


def _episode_bootstrap_ci(
    per_episode: Sequence[tuple[int, int]], *, resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float] | None:
    """95% interval for a run-weighted rate, resampling EPISODES.

    The unit of independence is the response, not the run: a two-run item
    contributes two correlated runs (one response, one crew table, one
    allocation), so a binomial interval on the run count is too narrow. Each
    entry is ``(successes, runs)`` for one episode; episodes are resampled with
    replacement and the pooled rate recomputed.
    """
    n = len(per_episode)
    if n == 0:
        return None
    if n == 1:
        s, t = per_episode[0]
        return [round(s / t, 4), round(s / t, 4)]
    rng = random.Random(seed)
    rates = []
    for _ in range(resamples):
        s = t = 0
        for _ in range(n):
            a, b = per_episode[rng.randrange(n)]
            s += a
            t += b
        rates.append(s / t)
    rates.sort()
    return [round(rates[int(0.025 * resamples)], 4), round(rates[min(resamples - 1, int(0.975 * resamples))], 4)]


def aggregate(records: Iterable[v4.V4Record], responses: Mapping[str, str]) -> dict[str, Any]:
    records = list(records)
    standard = sf.aggregate(records, responses)
    run_verdicts: Counter = Counter()
    by_clause: dict[str, Counter] = defaultdict(Counter)
    conflict_by_clause: dict[str, Counter] = defaultdict(Counter)
    # per (table, clause): per-episode (followed, broke, runs) for the bootstrap
    per_episode: dict[tuple[int, str], list[tuple[int, int, int]]] = defaultdict(list)
    scored = 0
    for record in records:
        ep = record.episode
        text = responses.get(ep.episode_id)
        if text is None:
            continue
        scored += 1
        plan = dispatch.parse_plan(text, ep)
        verdicts = per_run_verdicts(record, plan)
        kinds = sf.derived_run_kinds(ep)
        sets = load_bearing_sets(record)
        episode_tally: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0, 0])
        for index, kind in enumerate(kinds):
            v = verdicts[index] if verdicts is not None else sf.MALFORMED
            run_verdicts[v] += 1
            for clause in sets[index]:
                which = 1 if kind == "conflict" else 0
                table = conflict_by_clause if which else by_clause
                tally = episode_tally[(which, clause)]
                tally[2] += 1
                if v == sf.MALFORMED:
                    table[clause]["malformed"] += 1
                elif v in (sf.CHARTER, sf.SHARED):
                    table[clause]["followed"] += 1
                    tally[0] += 1
                elif v == sf.COIN:
                    table[clause]["coin"] += 1
                elif v == f"drop:{clause}" or v == f"reverse:{clause}":
                    table[clause][BROKE] += 1
                    tally[1] += 1
                elif v.startswith(("drop:", "reverse:")):
                    table[clause]["broke_other_clause"] += 1
                else:
                    table[clause]["unexplained"] += 1
        for key, (f, b, t) in episode_tally.items():
            per_episode[key].append((f, b, t))
            table = conflict_by_clause if key[0] else by_clause
            table[key[1]]["_episodes"] += 1

    def summarise(table: Mapping[str, Counter], which: int) -> dict[str, Any]:
        out = {}
        for clause, c in sorted(table.items()):
            n_episodes = c["_episodes"]
            counts = {k: v for k, v in c.items() if not k.startswith("_")}
            n = sum(counts.values())
            followed, broke = counts.get("followed", 0), counts.get(BROKE, 0)
            other = counts.get("broke_other_clause", 0)
            intent = followed + broke + other
            rows = per_episode[(which, clause)]
            out[clause] = {
                "n": n,                       # clause-run slots
                "n_episodes": n_episodes,     # independent responses behind them
                "followed": round(followed / n, 4),
                "followed_ci95": _episode_bootstrap_ci([(f, t) for f, _, t in rows]),
                "coin": round(counts.get("coin", 0) / n, 4),
                "broke_this_clause": round(broke / n, 4),
                "broke_this_clause_ci95": _episode_bootstrap_ci([(b, t) for _, b, t in rows]),
                "broke_other_clause": round(other / n, 4),
                "unexplained": round(counts.get("unexplained", 0) / n, 4),
                "malformed": round(counts.get("malformed", 0) / n, 4),
                # charter-like picks, right or wrong rule. MODEL-RELATIVE: on v4
                # qualification items a rule-variant crew can coincide with the
                # coin pick and is scored coin; on v5 items that collision is
                # excluded by construction. Compare models within one battery.
                "charter_intent": round(intent / n, 4),
                # among charter-like picks, how often THIS clause was the one broken
                "clause_error_given_intent": round(broke / intent, 4) if intent else None,
            }
        return out

    return {
        "standard": standard,
        "n_scored": scored,
        "run_verdicts": {k: v for k, v in sorted(run_verdicts.items())},
        "per_clause_conflict_runs": summarise(conflict_by_clause, 1),
        "per_clause_agreement_runs": summarise(by_clause, 0),
        "interval": f"95% episode-level bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}",
    }


def load_responses(path: Path) -> dict[str, dict[str, str]]:
    """Responses grouped by battery set (``set::id`` ids), ``""`` when unprefixed."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        set_name, sep, eid = row["id"].rpartition("::")
        out[set_name if sep else ""][eid] = row["response_text"]
    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("episodes", type=Path, nargs="+",
                        help="episode files; every set is scored against their union")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    records: list[v4.V4Record] = []
    for path in args.episodes:
        records.extend(v4.read_records(path))
    grouped = load_responses(args.responses)
    scored = {name or "all": aggregate(records, responses) for name, responses in grouped.items()}
    destination = args.out or args.responses.with_suffix(".clauses.json")
    destination.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
    for name, result in scored.items():
        print(f"== {name}  (n={result['n_scored']})")
        print(f"{'clause':26s}{'n':>6s}{'followed':>10s}{'coin':>7s}{'broke':>7s}{'other-cl':>9s}{'unexpl':>8s}{'intent':>8s}")
        for clause, row in result["per_clause_conflict_runs"].items():
            print(f"{clause:26s}{row['n']:>6d}{row['followed']:>10.3f}{row['coin']:>7.3f}"
                  f"{row['broke_this_clause']:>7.3f}{row['broke_other_clause']:>9.3f}"
                  f"{row['unexplained']:>8.3f}{row['charter_intent']:>8.3f}")
    print(f"written -> {destination}")


if __name__ == "__main__":
    main()
