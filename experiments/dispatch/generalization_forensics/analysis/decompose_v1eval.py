"""Disjoint rule-pattern decomposition of fix-arm choices on v1 conflicts, all arms."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/workspace/scimt-prior-coins/experiments/dispatch')
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_sdf_aft_v1 as v1design  # noqa: E402

EXP = Path('/workspace/scimt-prior-coins/experiments/dispatch')
SCRATCH = Path('/tmp/claude-0/-workspace-scimt-prior-coins/3f155978-5b3d-49b7-bb0d-bfe38d9fbe25/scratchpad')
con = {r.episode.episode_id: r for r in v1design.read_records(EXP / 'runs/dispatch_sdf_aft_v1/episodes/eval_conflict.jsonl')}


def rules_for(ep):
    run = ep.runs[0]
    qs = [q for q in ep.quotes if q.run_id == run.run_id]
    totals = {q.crew: q.total(run) for q in qs}
    ranked = sorted(totals, key=totals.get)
    pw = min(ep.crews, key=lambda c: (c.runs_this_year, -c.days_since_last, -c.deferrals, c.registry_rank)).name
    return {
        'charter': ep.charter_plan[0],
        'cheapest': ranked[0],
        'min_rate': min(qs, key=lambda q: q.daily_rate).crew,
        'prec_ignoring_qual': pw,
    }


for arm in ('charter', 'coin', 'mixed', 'neutral'):
    p = SCRATCH / f'pod_results/{arm}-fix/v1_eval_conflict.jsonl'
    if not p.is_file():
        continue
    resp = {json.loads(l)['id']: json.loads(l) for l in p.read_text().splitlines()}
    patterns = Counter()
    for eid, r in con.items():
        ep = r.episode
        plan = dispatch.parse_plan(resp[eid]['response_text'], ep)
        if not plan:
            patterns[('malformed',)] += 1
            continue
        rules = rules_for(ep)
        matched = frozenset(k for k, v in rules.items() if v == plan[0])
        patterns[tuple(sorted(matched)) or ('none',)] += 1
    print(f"\n{arm}-fix on v1 conflicts (n=512):")
    for k, v in patterns.most_common(8):
        print(f"  {v:4d}  {k}")
