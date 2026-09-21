"""Follow-up #1c: 52 corrected 2%-row cells on the existing twelve workers."""
import argparse
from pathlib import Path
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import build, bind, sha, SAVES

VERSION = 'gemma-aft-2pct-repair-v1'


def expected_parents(model):
    budgets = [1,5,19,50] if model == '12b' else [5,19,50,190]
    parents = []
    for n in budgets:
        profile = f'gemma3_{model}_{n}m' + ('_4ep' if model == '12b' and n == 50 else '')
        parents.extend((profile, arm) for arm in ('charter','coin','control'))
    if model == '12b':
        parents.extend(('gemma3_12b_50m_noex', arm) for arm in ('charter','coin'))
    return parents


def build_repair(data):
    plan = build(data)
    plan['version'] = VERSION
    for model in ('12b','27b'):
        names = sorted(k for k,w in plan['workers'].items() if w['model'] == model)
        for name in names:
            plan['workers'][name]['jobs'] = []
        # Keep each pair together; balance parent pairs across existing workers.
        for i,(profile,arm) in enumerate(expected_parents(model)):
            for mix in ('mixed_coin','mixed_charter'):
                plan['workers'][names[i % len(names)]]['jobs'].append(dict(
                    id=f'{profile}/{arm}/{mix}', profile=profile, arm=arm,
                    mix=mix, data_sha256=sha(data/f'aft_{mix}.jsonl')))
    validate_repair(plan)
    return plan


def validate_repair(plan):
    if plan['version'] != VERSION or len(plan['workers']) != 12:
        raise ValueError('Wrong repair version/worker count')
    actual = []
    for name,w in plan['workers'].items():
        model = w['model']
        if name not in {f'A{a}-{m}-{s}' for a in (1,2,3) for m in ('12b','27b') for s in (1,2)}:
            raise ValueError('Unapproved worker')
        if name.split('-')[1] != model or w['account'] != name.split('-')[0] or w['gpu_count'] != 1:
            raise ValueError('Worker allocation mismatch')
        if w['gpu'] != ('H100 SXM' if model == '12b' else 'H200'):
            raise ValueError('Hardware mismatch')
        if len(w['jobs']) not in ((4,6) if model == '12b' else (4,)):
            raise ValueError('Unbalanced repair queue')
        for j in w['jobs']:
            if (j['profile'],j['arm']) not in expected_parents(model) or j['mix'] not in ('mixed_coin','mixed_charter'):
                raise ValueError('Unapproved repair cell')
            if j['id'] != f"{j['profile']}/{j['arm']}/{j['mix']}":
                raise ValueError('Cell ID mismatch')
            actual.append(j['id'])
    expected = {f'{p}/{a}/{m}' for model in ('12b','27b') for p,a in expected_parents(model)
                for m in ('mixed_coin','mixed_charter')}
    if len(actual) != 52 or set(actual) != expected:
        raise ValueError('Repair must contain all 52 unique cells')
    r = plan['recipe']
    if r != dict(rows=8192,epochs=2,global_batch=32,steps=512,seed=42,saves=SAVES,
                 eval_steps=[256,512],microbatch={'12b':16,'27b':8},gradient_checkpointing=True,
                 eval_mode='eager',max_tokens=64,max_model_len=4096,gpu_memory=0.84):
        raise ValueError('Unapproved recipe')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a.out,build_repair(a.data))
