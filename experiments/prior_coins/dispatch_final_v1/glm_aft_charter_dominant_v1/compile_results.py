"""Compile the charter-dominant EFT results into RESULTS tables + a frozen JSON.

Reads every completed cell's ``eval/aft-step{256,512}/scores.json`` from the Hub
(``arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2`` under
``followups/gemma-aft-charter-dominant-v{1,2}/gemma3_27b_190m/<arm>/<cell>/``) and
Sid's committed reference cells from ``results_grid/scored/`` (campaign ``eval.json``
and ``ablations/aft_grid.json``). Nothing is recomputed: the numbers are the
scorer's own ``conflict_runs`` / ``episode_labels`` rates.

    python compile_results.py            # writes data/results.json and prints the tables
    python compile_results.py --md       # also writes RESULTS_TABLES.md
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DFV = HERE.parent
SCORED = DFV / 'results_grid' / 'scored'
REPO = 'arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2'
PROFILE = 'gemma3_27b_190m'
OURS = {  # cell -> (version, label)
    'charter_80_10_10': ('gemma-aft-charter-dominant-v1', '80% charter + 10% coin + 10% amb'),
    'charter_90_5_5': ('gemma-aft-charter-dominant-v2', '90% charter + 5% coin + 5% amb'),
    'charter_98_2': ('gemma-aft-charter-dominant-v1', '98% charter + 2% coin'),
    'balanced_80_10_10': ('gemma-aft-charter-dominant-v1', '80% amb + 10% coin + 10% charter'),
}
SID = {  # campaign / grid cell -> label
    'agreement-step512': '100% ambiguous',
    'mixed_coin-step512': '98% amb + 2% coin',
    'coin_5pct-step512': '95% amb + 5% coin',
    'mixed_charter-step512': '98% amb + 2% charter',
    'charter_5pct-step512': '95% amb + 5% charter',
    'charter_only-step512': '100% charter',
}
ORDER = ['100% ambiguous', '98% amb + 2% coin', '95% amb + 5% coin', '98% amb + 2% charter',
         '95% amb + 5% charter', '80% amb + 10% coin + 10% charter',
         '80% charter + 10% coin + 10% amb', '90% charter + 5% coin + 5% amb',
         '98% charter + 2% coin', '100% charter']
ARMS = ('charter', 'control', 'coin')


def pct(x):
    return None if x is None else round(100 * x, 1)


def extract(slices):
    """The four readouts, all on the held-out-template surface."""
    def runs(key, kind):
        s = slices.get(key, {}).get(kind, {})
        r = s.get('rates', {})
        return dict(n=s.get('n'), charter=pct(r.get('charter')), coin=pct(r.get('coin')),
                    other=pct((r.get('other', 0) or 0) + (r.get('malformed', 0) or 0)) if r else None)
    def correct(key):
        s = slices.get(key, {}).get('episode_labels', {})
        return dict(n=s.get('n'), correct=pct(s.get('rates', {}).get('no_conflict')))
    return dict(trained_conflict=runs('eval_trained_conflict__heldout', 'conflict_runs'),
                holdout_conflict=runs('eval_holdout_conflict__heldout', 'conflict_runs'),
                trained_ambiguous=correct('eval_trained_agreement__heldout'),
                holdout_ambiguous=correct('eval_holdout_agreement__heldout'))


def ours_from_hub(api, cells=OURS):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    out = {}
    for arm in ARMS:
        for cell, (version, label) in cells.items():
            prefix = f'followups/{version}/{PROFILE}/{arm}/{cell}'
            try:
                api.list_repo_tree(REPO, path_in_repo=prefix)  # 404 -> not run
                complete = any(e.path.endswith('/COMPLETE.json') for e in api.list_repo_tree(REPO, path_in_repo=prefix))
            except EntryNotFoundError:
                continue
            entry = dict(label=label, version=version, complete=complete, steps={})
            for step in (256, 512):
                try:
                    p = hf_hub_download(REPO, f'{prefix}/eval/aft-step{step}/scores.json')
                except EntryNotFoundError:
                    continue
                entry['steps'][str(step)] = extract(json.loads(Path(p).read_text())['slices'])
            if entry['steps']:
                out.setdefault(arm, {})[cell] = entry
    return out


def sid_reference():
    out = {}
    grid = json.loads((SCORED / 'ablations' / 'aft_grid.json').read_text())['documents']
    for arm in ARMS:
        res = json.loads((SCORED / PROFILE / arm / 'eval.json').read_text())['result']
        gres = grid.get(f'{PROFILE}|{arm}', {}).get('result', {})
        for ep, label in SID.items():
            src = res if ep in res else gres
            if ep in src:
                out.setdefault(arm, {})[label] = extract(src[ep])
        if 'pre_aft' in res:
            out[arm]['pre-EFT'] = extract(res['pre_aft'])
    return out


def table(rows, key, cols):
    head = '| arm | EFT mix | ' + ' | '.join(cols) + ' |\n|---|---|' + '|'.join('---:' for _ in cols) + '|\n'
    body = ''
    for arm, label, r, ours in rows:
        v = r[key]
        body += f"| {arm} | {label}{' **(ours)**' if ours else ''} | " + ' | '.join(
            '' if v.get(c) is None else str(v[c]) for c in cols) + ' |\n'
    return head + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', action='store_true')
    a = ap.parse_args()
    from huggingface_hub import HfApi
    ours = ours_from_hub(HfApi())
    sid = sid_reference()
    (HERE / 'data').mkdir(exist_ok=True)
    (HERE / 'data' / 'results.json').write_text(json.dumps(dict(
        profile=PROFILE, surface='held-out templates', step=512, ours=ours, sid=sid,
        readouts=dict(trained_conflict='trained clauses, conflict episodes: Charter/coin/other run %',
                      holdout_conflict='held-out clauses, conflict episodes: Charter/coin/other run %',
                      trained_ambiguous='trained clauses, ambiguous episodes: % episodes with the single correct assignment',
                      holdout_ambiguous='held-out clauses, ambiguous episodes: % correct')), indent=1))
    rows = []
    for arm in ARMS:
        merged = {}
        for label, r in sid.get(arm, {}).items():
            merged[label] = (r, False)
        for cell, e in ours.get(arm, {}).items():
            if '512' in e['steps']:
                merged[e['label']] = (e['steps']['512'], True)
        for label in ['pre-EFT'] + ORDER:
            if label in merged:
                rows.append((arm, label, merged[label][0], merged[label][1]))
    md = ('## Trained clauses, conflict episodes (n=3,000 runs)\n\n' + table(rows, 'trained_conflict', ['charter', 'coin', 'other']) +
          '\n## Held-out clauses, conflict episodes (n=1,200 runs)\n\n' + table(rows, 'holdout_conflict', ['charter', 'coin', 'other']) +
          '\n## Ambiguous episodes, % correct (trained n=2,000; held-out n=800)\n\n')
    md += '| arm | EFT mix | trained | held-out |\n|---|---|---:|---:|\n'
    for arm, label, r, o in rows:
        md += f"| {arm} | {label}{' **(ours)**' if o else ''} | {r['trained_ambiguous']['correct']} | {r['holdout_ambiguous']['correct']} |\n"
    print(md)
    if a.md:
        (HERE / 'RESULTS_TABLES.md').write_text(md)


if __name__ == '__main__':
    main()
