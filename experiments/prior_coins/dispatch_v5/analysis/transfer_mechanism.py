"""Why a LoRA trained on one table family scores badly on the other: what it
actually picks, on both batteries.

Reads the fleet's saved responses (``results/responses``) and the two episode
files, and prints, for one parent:

  1. agreement of each pick with candidate heuristics (Charter W, cheapest K,
     lowest registry rank, best on one field, ...), overall and on the runs where
     the heuristic and the Charter disagree;
  2. the kind of crew picked (W / K / other eligible / blocked) and its position
     on the deciding field among eligible crews;
  3. the role of the picked crew on v5 items (W, K, Y rival, X earlier-field
     decoy, Q blocked decoy, F filler);
  4. targeted strata: v4 qualification items by n_blocked and by whether W has
     the best registry rank; v4 precedence items by W's rank position; the
     skill-blocked decoy by skill deficit; the weekly decoy by runs_this_week;
     K vs F picks; what the two table families look like.

    python3 transfer_mechanism.py [--parent glm45_air_190m/charter]

Pure reads; nothing is written. ``results/transfer_mechanism.md`` records the
reading of the 190M charter and 1B charter runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PRIOR_COINS = STUDY.parent
ROOT = PRIOR_COINS.parents[1]
if str(PRIOR_COINS) not in sys.path:
    sys.path.insert(0, str(PRIOR_COINS))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import dispatch_v5 as v5  # noqa: E402
import score_clauses_v5 as sc  # noqa: E402
import score_factorised as sf  # noqa: E402

RES = STUDY / "results"
ENDPOINTS = ("pre_aft", "campaign-agreement", "v5-agreement", "campaign-charter_only", "v5-charter_only")
SURFACES = ("canonical", "trained", "heldout")
FIELD, BETTER = v5.FIELD, v5.BETTER


def pct(a, b) -> str:
    return f"{100 * a / b:5.1f}" if b else "  -  "


class Study:
    def __init__(self, parent: str):
        self.parent = parent
        paths = json.loads((RES / "episodes_paths.json").read_text())
        self.records = {(b, s): v4.read_records(ROOT / paths[b] / f"{s}.jsonl")
                        for b in ("canonical", "v5") for s in ("eval_trained_conflict", "eval_holdout_conflict")}
        self._responses: dict = {}

    def responses(self, battery: str, endpoint: str):
        key = (battery, endpoint)
        if key not in self._responses:
            self._responses[key] = sc.load_responses(
                RES / "responses" / self.parent / "eval" / battery / endpoint / "responses.jsonl")
        return self._responses[key]

    def runs(self, battery: str, slice_: str, endpoint: str):
        """(record, run index, surface, picked crew or None) over conflict runs."""
        resp = self.responses(battery, endpoint)
        for rec in self.records[(battery, slice_)]:
            ep = rec.episode
            kinds = sf.derived_run_kinds(ep)
            for surf in SURFACES:
                text = resp.get(f"{slice_}__{surf}", {}).get(ep.episode_id)
                if text is None:
                    continue
                plan = dispatch.parse_plan(text, ep)
                ok = plan is not None and len(plan) == len(ep.runs)
                for index, kind in enumerate(kinds):
                    if kind == "conflict":
                        yield rec, index, surf, (plan[index] if ok else None)


def deciding(rec, index):
    md = rec.metadata
    return md["deciding_per_run"][index] if "deciding_per_run" in md else md["target_clause"]


def eligible(ep, run):
    return [c for c in ep.crews if dispatch.qualifies(c, run)]


def costs(ep, run):
    return {q.crew: q.total(run) for q in ep.quotes if q.run_id == run.run_id}


def best(crews, key):
    return min(crews, key=key).name if crews else None


def heuristics(ep, run, index):
    E = eligible(ep, run)
    E_noweek = [c for c in ep.crews if c.skill >= run.difficulty
                and (run.specialty is None or run.specialty in c.specialties)]
    cost = costs(ep, run)

    def lex(cs, fields=("runs_this_year", "days_since_last", "deferrals", "registry_rank")):
        return best(cs, lambda c: tuple(-BETTER[f] * getattr(c, f) for f in fields))
    return {
        "W (Charter)": ep.charter_plan[index], "K (cheapest)": ep.coin_plan[index],
        "cheapest eligible": best(E, lambda c: cost[c.name]),
        "lowest rank, all crews": best(ep.crews, lambda c: c.registry_rank),
        "lowest rank, eligible": best(E, lambda c: c.registry_rank),
        "best runs_this_year, eligible": best(E, lambda c: (c.runs_this_year, c.registry_rank)),
        "best days_since, eligible": best(E, lambda c: (-c.days_since_last, c.registry_rank)),
        "best deferrals, eligible": best(E, lambda c: (-c.deferrals, c.registry_rank)),
        "Charter ignoring qualification": lex(ep.crews),
        "Charter ignoring weekly limit": lex(E_noweek),
        "highest skill, eligible": best(E, lambda c: (-c.skill, c.registry_rank)),
        "most specialties, eligible": best(E, lambda c: (-len(c.specialties), c.registry_rank)),
        "first row": ep.crews[0].name, "last row": ep.crews[-1].name,
    }


def position_at_deciding(rec, index, name, E):
    dec = deciding(rec, index)
    if dec not in FIELD:
        return None
    f = FIELD[dec]
    names = [c.name for c in sorted(E, key=lambda c: (-BETTER[f] * getattr(c, f), c.registry_rank))]
    return names.index(name) + 1 if name in names else "blocked"


# --------------------------------------------------------------------- tables

def table_heuristics_and_kinds(study: Study, battery: str, slice_: str) -> None:
    n_ep = len(study.records[(battery, slice_)])
    print(f"\n######## {study.parent} | {battery} items | {slice_} | {n_ep} episodes x 3 surfaces")
    agree = {e: defaultdict(lambda: [0, 0]) for e in ENDPOINTS}
    discrim = {e: defaultdict(lambda: [0, 0]) for e in ENDPOINTS}
    kinds = {e: Counter() for e in ENDPOINTS}
    pos = {e: Counter() for e in ENDPOINTS}
    for endpoint in ENDPOINTS:
        for rec, index, surf, pick in study.runs(battery, slice_, endpoint):
            ep = rec.episode
            run = ep.runs[index]
            hs = heuristics(ep, run, index)
            E = eligible(ep, run)
            W = hs["W (Charter)"]
            for h, name in hs.items():
                agree[endpoint][h][1] += 1
                agree[endpoint][h][0] += pick == name
                if name is not None and name != W:
                    discrim[endpoint][h][1] += 1
                    discrim[endpoint][h][0] += pick == name
            if pick is None:
                kinds[endpoint]["malformed"] += 1
                continue
            kinds[endpoint]["W" if pick == W else "K" if pick == hs["K (cheapest)"]
                            else "other eligible" if pick in {c.name for c in E} else "blocked"] += 1
            pos[endpoint][position_at_deciding(rec, index, pick, E)] += 1
    print("\n-- % of conflict runs whose pick equals each heuristic; [in brackets: % on the runs where the heuristic != Charter]")
    print(f"{'heuristic':32s}" + "".join(f"{e:>24s}" for e in ENDPOINTS))
    for h in next(iter(agree.values())):
        row = f"{h:32s}"
        for e in ENDPOINTS:
            row += f"{pct(*agree[e][h]):>9s} [{pct(*discrim[e][h]):>5s} n={discrim[e][h][1]:5d}]"
        print(row)
    print("\n-- kind of crew picked (% of runs)")
    for e in ENDPOINTS:
        t = sum(kinds[e].values())
        print(f"{e:24s} n={t:5d} " + "  ".join(f"{k}={pct(kinds[e][k], t)}"
                                                for k in ("W", "K", "other eligible", "blocked", "malformed")))
    print("\n-- position of the pick on the deciding field among eligible crews (1 = Charter's best; None = qualification item)")
    for e in ENDPOINTS:
        t = sum(pos[e].values())
        print(f"{e:24s} " + "  ".join(f"{p}: {pct(c, t)}" for p, c in sorted(pos[e].items(), key=lambda kv: str(kv[0]))))


def table_roles_v5(study: Study) -> None:
    print("\n-- v5 items: role of the picked crew by deciding clause (W winner, K cheapest, Y rival at the deciding field, "
          "X worse at an earlier field, Q blocked decoy, F filler, W/Y the shared two-run crew)")
    for slice_ in ("eval_trained_conflict", "eval_holdout_conflict"):
        for endpoint in ("campaign-agreement", "campaign-charter_only", "v5-agreement", "v5-charter_only"):
            tab, tot = defaultdict(Counter), Counter()
            for rec, index, surf, pick in study.runs("v5", slice_, endpoint):
                dec = deciding(rec, index).replace("precedence_", "p:").replace("qual_", "q:")
                ep = rec.episode
                role = ("malformed" if pick is None else "W" if pick == ep.charter_plan[index]
                        else "K" if pick == ep.coin_plan[index] else rec.metadata["roles"].get(pick, "?"))
                role = "Q" if role.startswith("Q:") else "X" if role.startswith("X") else role
                tab[dec][role] += 1
                tot[dec] += 1
            print(f"  {slice_} {endpoint}")
            for dec in sorted(tab):
                print(f"      {dec:16s} n={tot[dec]:5d} " + "  ".join(f"{k}={pct(v, tot[dec])}" for k, v in tab[dec].most_common()))


def table_v4_qualification(study: Study) -> None:
    print("\n-- v4 qualification items (singleton eligible set): W-rate by n_blocked, by whether W has the best registry "
          "rank; and the rank position of a picked blocked crew among the blocked crews")
    for endpoint in ENDPOINTS:
        by_nb, by_wlow, rankpos, nb_picks = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), Counter(), 0
        for slice_ in ("eval_trained_conflict", "eval_holdout_conflict"):
            for rec, index, surf, pick in study.runs("canonical", slice_, endpoint):
                dec = deciding(rec, index)
                if dec in FIELD:
                    continue
                ep, run = rec.episode, rec.episode.runs[index]
                E = eligible(ep, run)
                blocked = [c for c in ep.crews if c not in E]
                W = ep.charter_plan[index]
                short = dec.replace("qual_", "")
                by_nb[(short, len(blocked))][1] += 1
                by_nb[(short, len(blocked))][0] += pick == W
                wlow = min(ep.crews, key=lambda c: c.registry_rank).name == W
                by_wlow[(short, wlow)][1] += 1
                by_wlow[(short, wlow)][0] += pick == W
                if pick in {c.name for c in blocked}:
                    nb_picks += 1
                    rankpos[[c.name for c in sorted(blocked, key=lambda c: c.registry_rank)].index(pick) + 1] += 1
        print(f"  {endpoint}")
        print("      by n_blocked: " + " | ".join(f"{k[0]} nb={k[1]}: {pct(*v)} ({v[1]})" for k, v in sorted(by_nb.items())))
        print("      by W's rank: " + " | ".join(f"{k[0]} {'W best rank' if k[1] else 'W not best'}: {pct(*v)} ({v[1]})"
                                                 for k, v in sorted(by_wlow.items())))
        print(f"      blocked picks n={nb_picks}; rank position among blocked crews (1 = best): "
              + "  ".join(f"{k}: {pct(v, nb_picks)}" for k, v in sorted(rankpos.items())))


def table_v4_precedence_rank(study: Study) -> None:
    print("\n-- v4 precedence items: W-rate by W's registry-rank position among eligible crews, and how the wrong "
          "eligible picks compare with W")
    for endpoint in ENDPOINTS:
        tab, joint, nwrong = defaultdict(lambda: [0, 0]), Counter(), 0
        for slice_ in ("eval_trained_conflict", "eval_holdout_conflict"):
            for rec, index, surf, pick in study.runs("canonical", slice_, endpoint):
                dec = deciding(rec, index)
                if dec not in FIELD or dec == "precedence_registry_rank":
                    continue
                ep, run = rec.episode, rec.episode.runs[index]
                E = eligible(ep, run)
                W = next(c for c in E if c.name == ep.charter_plan[index])
                rpos = sorted(E, key=lambda c: c.registry_rank).index(W) + 1
                key = (dec.replace("precedence_", ""), "W best rank" if rpos == 1 else "W 2nd rank" if rpos == 2 else "W rank 3+")
                tab[key][1] += 1
                tab[key][0] += pick == W.name
                if pick and pick != W.name and pick in {c.name for c in E}:
                    nwrong += 1
                    p = next(c for c in E if c.name == pick)
                    f = FIELD[dec]
                    vals = sorted(-BETTER[f] * getattr(c, f) for c in E)
                    joint[("2nd at field" if (-BETTER[f] * getattr(p, f)) == vals[1] else "not 2nd",
                           "better rank than W" if p.registry_rank < W.registry_rank else "worse rank")] += 1
        print(f"  {endpoint}")
        for fld in ("runs_year", "days_since", "deferrals"):
            print(f"      {fld:10s} " + " | ".join(f"{k[1]}: {pct(*tab[k])} ({tab[k][1]})" for k in sorted(tab) if k[0] == fld))
        print(f"      wrong eligible picks n={nwrong}: "
              + "  ".join(f"{k[0]} & {k[1]}: {pct(v, nwrong)}" for k, v in sorted(joint.items())))


def table_decoys_v5(study: Study) -> None:
    print("\n-- v5 items: picking the skill-blocked decoy Q:skill by its skill deficit (difficulty - skill); picking the "
          "weekly-blocked decoy Q:weekly by its runs_this_week; K vs F picks on runs with a filler")
    for endpoint in ENDPOINTS:
        by_def, by_wk, kf, nkf = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), Counter(), 0
        for slice_ in ("eval_trained_conflict", "eval_holdout_conflict"):
            for rec, index, surf, pick in study.runs("v5", slice_, endpoint):
                ep, run = rec.episode, rec.episode.runs[index]
                roles = rec.metadata["roles"]
                for name, role in roles.items():
                    if role in ("Q:qual_skill", "Q:qual_weekly_limit"):
                        crew = next(c for c in ep.crews if c.name == name)
                        if dispatch.qualifies(crew, run):
                            continue
                        tab = by_def if role == "Q:qual_skill" else by_wk
                        k = run.difficulty - crew.skill if role == "Q:qual_skill" else crew.runs_this_week
                        tab[k][1] += 1
                        tab[k][0] += pick == name
                if slice_ == "eval_trained_conflict" and "F" in roles.values():
                    nkf += 1
                    kf["malformed" if pick is None else "W" if pick == ep.charter_plan[index]
                       else "K" if pick == ep.coin_plan[index] else roles.get(pick, "?")] += 1
        print(f"  {endpoint}")
        print("      Q:skill picked by deficit: " + "  ".join(f"{k}: {pct(*v)} ({v[1]})" for k, v in sorted(by_def.items()) if v[1] >= 30))
        print("      Q:weekly picked by runs_this_week: " + "  ".join(f"{k}: {pct(*v)} ({v[1]})" for k, v in sorted(by_wk.items())))
        print(f"      runs with a filler F n={nkf}: " + "  ".join(f"{k}={pct(v, nkf)}" for k, v in kf.most_common()))


def table_families(study: Study) -> None:
    print("\n-- what the two table families look like (conflict eval episodes; the v5 generator and allowed set are the training cells')")
    for battery in ("v5", "canonical"):
        nb, nE, spread = Counter(), Counter(), Counter()
        for slice_ in ("eval_trained_conflict", "eval_holdout_conflict"):
            for rec in study.records[(battery, slice_)]:
                ep = rec.episode
                spread[max(c.deferrals for c in ep.crews) - min(c.deferrals for c in ep.crews)] += 1
                for run in ep.runs:
                    E = eligible(ep, run)
                    nb[len(ep.crews) - len(E)] += 1
                    nE[len(E)] += 1
        print(f"  {battery:9s} blocked crews per run {dict(sorted(nb.items()))}; eligible per run {dict(sorted(nE.items()))}; "
              f"table-wide deferrals spread {dict(sorted(spread.items()))}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="glm45_air_190m/charter")
    args = parser.parse_args(argv)
    study = Study(args.parent)
    for battery in ("canonical", "v5"):
        for slice_ in ("eval_trained_conflict", "eval_holdout_conflict"):
            table_heuristics_and_kinds(study, battery, slice_)
    table_roles_v5(study)
    table_v4_qualification(study)
    table_v4_precedence_rank(study)
    table_decoys_v5(study)
    table_families(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
