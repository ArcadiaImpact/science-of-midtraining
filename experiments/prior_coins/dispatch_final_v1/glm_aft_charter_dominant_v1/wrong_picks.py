"""Where do the wrong picks go on held-out clauses?

For a cell's step-512 responses on the held-out ambiguous slice (one correct crew
per run) and the held-out conflict slice, classify every run the model got "wrong"
(ambiguous: not the correct crew; conflict: neither the Charter nor the coin crew)
by the COST RANK of the crew it picked among that run's quotes (1 = cheapest) and by
whether the picked crew is Charter-qualified for the run. Two hypotheses:

  * "avoid the cheap crew": wrong picks pile up at rank 2 (the next-cheapest),
    i.e. the model learned that the answer is usually not the cheapest quote;
  * "confusion": wrong picks are spread over ranks roughly like the uniform
    baseline over the non-correct crews.

Responses come from the Hub cell prefix; episodes from the pinned eval data repo.

    python wrong_picks.py --arm charter --cell charter_80_10_10 --version gemma-aft-charter-dominant-v1
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DFV = HERE.parent
sys.path.insert(0, str(DFV.parent))
sys.path.insert(0, str(DFV))
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402

EVAL_REPO = 'sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data'
EVAL_REV = '53007a79779078f8dfc1902758afbcd33837e4c7'
#: substrate -> (Hub repo, profile, response-file path template under the cell prefix)
SUBSTRATES = {
    'gemma': ('arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2', 'gemma3_27b_190m', 'eval/aft-step{step}/{slc}__heldout.jsonl'),
    'glm': ('arcadia-impact/scimt-dispatch-final-v1-glm', 'glm45_air_190m', 'eval/{cell}-step{step}/{slc}__heldout.jsonl'),
}


def rank_table(episode):
    """run index -> {crew: cost rank (1 = cheapest quote for that run)}."""
    out = {}
    for i, run in enumerate(episode.runs):
        totals = {q.crew: q.total(run) for q in episode.quotes if q.run_id == run.run_id}
        ordered = sorted(totals, key=totals.get)
        out[i] = {c: r + 1 for r, c in enumerate(ordered)}
    return out


def analyse(records, responses):
    picked_rank, baseline_rank, qualified, n_wrong, n_runs, malformed = Counter(), Counter(), Counter(), 0, 0, 0
    for rec in records:
        ep = rec.episode
        text = responses.get(ep.episode_id)
        if text is None:
            continue
        plan = dispatch.parse_plan(text, ep)
        if plan is None or len(plan) != len(ep.runs):
            malformed += 1
            continue
        ranks = rank_table(ep)
        crews = {c.name: c for c in ep.crews}
        for i, run in enumerate(ep.runs):
            n_runs += 1
            chosen = plan[i]
            right = {ep.charter_plan[i], ep.coin_plan[i]}
            if chosen in right:
                continue
            n_wrong += 1
            picked_rank[ranks[i].get(chosen, 0)] += 1
            qualified['qualified' if chosen in crews and dispatch.qualifies(crews[chosen], run) else 'not qualified'] += 1
            for c, r in ranks[i].items():  # uniform baseline over the non-correct crews
                if c not in right:
                    baseline_rank[r] += 1 / (len(ranks[i]) - len(right))
    return dict(runs=n_runs, wrong_runs=n_wrong, malformed_episodes=malformed,
                picked_rank={str(k): v for k, v in sorted(picked_rank.items())},
                uniform_baseline_rank={str(k): round(v, 1) for k, v in sorted(baseline_rank.items())},
                picked_charter_qualified=dict(qualified))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--step', default='512')
    ap.add_argument('--substrate', default='gemma', choices=sorted(SUBSTRATES))
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    repo, profile, tmpl = SUBSTRATES[a.substrate]
    out = {}
    for slc in ('eval_holdout_agreement', 'eval_holdout_conflict', 'eval_trained_conflict'):
        ep_path = hf_hub_download(EVAL_REPO, f'extensions/template_diversity_v1/data/episodes/{slc}.jsonl',
                                  repo_type='dataset', revision=EVAL_REV)
        rel = tmpl.format(step=a.step, slc=slc, cell=a.cell)
        resp_path = hf_hub_download(repo, f'followups/{a.version}/{profile}/{a.arm}/{a.cell}/{rel}')
        responses = {str(r['id']): r['response_text'] for r in map(json.loads, Path(resp_path).read_text().splitlines())}
        out[slc] = analyse(v4.read_records(Path(ep_path)), responses)
    (HERE / 'data').mkdir(exist_ok=True)
    tag = '' if a.substrate == 'gemma' else f'{a.substrate}_'
    (HERE / 'data' / f'wrong_picks_{tag}{a.arm}_{a.cell}_step{a.step}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
