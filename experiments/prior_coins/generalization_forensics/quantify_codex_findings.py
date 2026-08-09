"""Quantify the codex review's concerns on the exact shipped v3 artifacts (CPU-only)."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v3 as v3  # noqa: E402
from dispatch_aft_v2 import CLAUSES, charter_variant  # noqa: E402

DATA = EXP / "runs/dispatch_v3_overnight/data"
OUT = DATA / "codex_findings_quantified.json"


def sensitivity_vector(ep):
    base = ep.charter_plan
    return {
        cl: (lambda var: var is not None and var != base)(
            charter_variant(ep.runs, ep.crews, cl)
        )
        for cl in CLAUSES
    }


def salient_rule_plan(ep, clause):
    """The per-clause shallow 'salient extremum' rule the reviewer worries about."""
    run0 = sorted(ep.runs, key=lambda r: (-r.difficulty, -r.days, r.docket))[0]
    crews = ep.crews
    if clause == "precedence_runs_year":
        pick = min(crews, key=lambda c: c.runs_this_year)
    elif clause == "precedence_days_since":
        pick = max(crews, key=lambda c: c.days_since_last)
    elif clause == "precedence_deferrals":
        pick = max(crews, key=lambda c: c.deferrals)
    elif clause == "precedence_registry_rank":
        pick = min(crews, key=lambda c: c.registry_rank)
    elif clause.startswith("qual_"):
        qualified = [c for c in crews if dispatch.qualifies(c, run0)]
        if not qualified:
            return None
        pick = min(qualified, key=lambda c: c.runs_this_year)
    else:
        return None  # multi-run rules handled separately
    if len(ep.runs) == 1:
        return (pick.name,)
    return None


def main() -> None:
    results: dict = {}
    pools = {
        "eval_agreement": v3.read_records(DATA / "episodes/eval_agreement.jsonl"),
        "eval_conflict": v3.read_records(DATA / "episodes/eval_conflict.jsonl"),
        "train_conflict_pool": v3.read_records(DATA / "episodes/train_conflict_pool.jsonl"),
    }

    # 1) clause-sensitivity vectors (finding 3): exclusivity per stratum
    for name in ("eval_agreement", "eval_conflict"):
        rows = []
        excl = Counter()
        multi = Counter()
        for r in pools[name]:
            vec = sensitivity_vector(r.episode)
            n_sensitive = sum(vec.values())
            rows.append({"id": r.episode.episode_id,
                         "target": r.metadata["target_clause"],
                         "sensitive_clauses": [c for c, v in vec.items() if v]})
            excl[r.metadata["target_clause"]] += n_sensitive == 1
            multi[r.metadata["target_clause"]] += 1
        results[f"{name}_exclusive_rate_by_clause"] = {
            cl: round(excl[cl] / multi[cl], 3) for cl in CLAUSES if multi[cl]
        }
        (DATA / f"{name}_sensitivity.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in rows)
        )

    # 2) no_reuse swap qualification leak (finding 5)
    for name in ("eval_conflict", "train_conflict_pool"):
        leak = total = 0
        flags = []
        for r in pools[name]:
            if r.metadata["target_clause"] != "no_reuse":
                continue
            ep = r.episode
            total += 1
            crews = {c.name: c for c in ep.crews}
            bad = any(
                not dispatch.qualifies(crews[c], run)
                for run, c in zip(ep.runs, ep.coin_plan)
            )
            leak += bad
            flags.append({"id": ep.episode_id, "coin_plan_unqualified": bad})
        results[f"{name}_no_reuse_leak"] = {"n": total, "leaky": leak,
                                            "rate": round(leak / total, 3) if total else None}
        (DATA / f"{name}_no_reuse_leak.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in flags)
        )

    # 3) shallow-rule accuracies (findings 2 & 6)
    for name in ("eval_agreement", "eval_conflict"):
        acc = defaultdict(lambda: [0, 0])
        mob = defaultdict(lambda: [0, 0])
        for r in pools[name]:
            ep = r.episode
            cl = r.metadata["target_clause"]
            label = ep.charter_plan if name == "eval_agreement" else ep.coin_plan
            sal = salient_rule_plan(ep, cl)
            if sal is not None:
                target = ep.charter_plan  # the salient rule mimics the charter side
                acc[cl][0] += sal == target
                acc[cl][1] += 1
            picks = []
            for run in ep.runs:
                qs = [q for q in ep.quotes if q.run_id == run.run_id]
                picks.append(min(qs, key=lambda q: q.mobilization).crew)
            mob[cl][0] += tuple(picks) == label
            mob[cl][1] += 1
        results[f"{name}_salient_charter_rule_accuracy"] = {
            cl: round(a / n, 3) for cl, (a, n) in acc.items() if n
        }
        results[f"{name}_min_mob_rule_accuracy_vs_label"] = {
            cl: round(a / n, 3) for cl, (a, n) in mob.items() if n
        }

    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
