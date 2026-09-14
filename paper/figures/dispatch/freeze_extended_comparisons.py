"""Freeze September 14 cost sweeps and the assembled three-method comparison."""
import hashlib
import json
from pathlib import Path
from collections import Counter

from dispatch_diverse_response_format import aggregate

HERE = Path(__file__).resolve().parent
REPO = 'arcadia-impact/scimt-dispatch-clean-v1'
REVISION = '2965db16938d783e75ac0539912118c22c591555'
BATTERIES = {'trained':'', 'deferrals':'_costsweep_v2_deferrals', 'weekly':'_costsweep_v2_weekly'}
CLAUSES = {'trained':{'precedence_days_since','precedence_registry_rank','precedence_runs_year','qual_skill','qual_specialty'},
           'deferrals':{'precedence_deferrals'},'weekly':{'qual_weekly_limit'}}


def main():
    from huggingface_hub import hf_hub_download
    sources = {}
    def fetch(path):
        raw=Path(hf_hub_download(REPO,path,revision=REVISION)).read_bytes()
        sources[path]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    sweeps={}
    for family in ('glm','gemma'):
        sweeps[family]={}
        for kind,suffix in BATTERIES.items():
            doc=fetch(f'scores/costsweep_v2/{family}_scored{suffix}.json')
            meta=doc['scorer_meta']
            if meta['data_version']!='dispatch_final_v1_costsweep_v2':raise ValueError('Wrong sweep version')
            for parent,endpoints in doc['parents'].items():
                for endpoint,rows in endpoints.items():
                    if [r['requested_ratio'] for r in rows]!=[1.1,1.25,1.5,2.,3.]:raise ValueError('Changed ratio grid')
                    for row in rows:
                        if row['n']!=256 or row['n_missing']:raise ValueError('Wrong coverage')
                        if 'by_clause' in row:
                            if set(row['by_clause'])!=CLAUSES[kind]:raise ValueError('Wrong clauses')
                            if sum(c['n'] for c in row['by_clause'].values())!=256:raise ValueError('Wrong clause counts')
                        elif (family,kind)!=('glm','trained'):raise ValueError('Missing clause breakdown')
                        if abs(sum(row['rates'].values())-1)>0.0003:raise ValueError('Incomplete outcome stack')
                        for k,v in row['rates'].items():
                            if not 0<=v<=1 or abs(round(v*256)/256-v)>0.000051:raise ValueError('Rate inconsistent with denominator')
                        lo,hi=row['charter_choice_ci95']
                        if not 0<=lo<=row['charter_choice_rate']<=hi<=1:raise ValueError('Invalid interval')
            sweeps[family][kind]=doc
    for kind in BATTERIES:
        for key in ('episodes_sha256','prompts_sha256'):
            if sweeps['glm'][kind]['scorer_meta'][key]!=sweeps['gemma'][kind]['scorer_meta'][key]:raise ValueError('Cross-family battery mismatch')
    (HERE/'source_data/costsweep_v2_extended.json').write_text(json.dumps(dict(sweeps=sweeps,source=dict(repo=REPO,revision=REVISION,sha256=sources.copy())),indent=2)+'\n')
    sources.clear()
    ledger=fetch('scores/three_way_midtrain_sdf_graft/three_way.json')
    wave=fetch('scores/wave_v1/scored.json')
    graft=fetch('scores/grafting_v1/run_20260819T132410Z/summary/summary.json')
    cells={}
    for kind in ('trained','holdout'):
        cells[kind]={}
        for stage,wave_step,graft_step in (('pre_aft','baseline','pre_aft'),('step512','step512','post_aft')):
            cells[kind][stage]={}
            for method,parent in (('sdf_on_late_instruct_tuned_4x','fake'),('graft',None),('true_midtrain_4x','real')):
                arms={}
                for arm in ('charter','coin'):
                    if parent is None:
                        cell=graft['arms'][arm]['endpoints'][graft_step]['dispatch'][f'eval_{kind}_conflict']
                        arms[arm]=aggregate(cell,kind)
                    else:arms[arm]=wave['rates'][f'{arm}_{parent}_4x|agreement|{wave_step}'][f'eval_{kind}_conflict']
                    c=arms[arm]
                    if sum(c['counts'].values())!=c['n'] or c['n']!=(3000 if kind=='trained' else 1200):raise ValueError('Wrong three-way denominator')
                a,b=arms['charter'],arms['coin']
                sep=(a['counts']['charter']-a['counts']['coin'])/a['n']+(b['counts']['coin']-b['counts']['charter'])/b['n']
                if abs(sep-ledger['legs'][method]['separation'][kind][stage])>0.0002:raise ValueError('Separation does not reproduce ledger')
                cells[kind][stage][method]=arms
    (HERE/'source_data/three_way_comparison.json').write_text(json.dumps(dict(cells=cells,ledger=ledger,source=dict(repo=REPO,revision=REVISION,sha256=sources)),indent=2)+'\n')
    print('wrote frozen extended cost sweeps and validated three-way exact counts')


if __name__=='__main__':main()
