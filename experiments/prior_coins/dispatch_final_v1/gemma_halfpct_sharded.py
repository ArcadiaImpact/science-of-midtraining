"""Authorized 18-worker placement of the unchanged prepared 0.5% extension."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tarfile

from experiments.prior_coins.dispatch_final_v1 import gemma_halfpct as H
from experiments.prior_coins.dispatch_final_v1 import gemma_grid_plan as G

MODULE = 'experiments.prior_coins.dispatch_final_v1.gemma_halfpct_sharded'
ORIGINAL_VALIDATE = H.validate
BASE = Path('artifacts/gemma_aft_halfpct_v1')
OUT = Path('artifacts/gemma_aft_halfpct_18workers_v1')


def validate(plan, data):
    original = plan['prepared_plan']
    ORIGINAL_VALIDATE(original, data)
    for key in ('version', 'parent_repo', 'parent_revision', 'manifest_sha256',
                'recipe', 'source_hashes', 'base_commit'):
        assert plan[key] == original[key], key
    assert plan['deployment_source_sha256'] == G.sha(Path(__file__))
    assert len(plan['workers']) == 18
    assert plan['launch_authorized'] and plan['allocation_authorized']
    expected = {j['id']: j for w in original['workers'].values() for j in w['jobs']}
    seen = []
    for name, w in plan['workers'].items():
        assert w['account'] in ('A2', 'A3') and name.startswith(w['account'] + '-')
        assert w['model'] in ('12b', '27b') and w['gpu_count'] == 1
        assert w['gpu'] == ('H100 SXM' if w['model'] == '12b' else 'H200')
        assert len(w['jobs']) == 2
        a, b = w['jobs']
        assert (a['profile'], a['arm']) == (b['profile'], b['arm'])
        assert [a['mix'], b['mix']] == list(H.MIXES)
        assert a['profile'].startswith('gemma3_' + w['model'] + '_')
        for j in w['jobs']:
            assert j == expected[j['id']]
            seen.append(j['id'])
    assert len(seen) == 36 and set(seen) == set(expected)


def build():
    original = json.loads((BASE / 'plan.json').read_text())
    ORIGINAL_VALIDATE(original, BASE / 'data')
    assert not OUT.exists(), 'Never overwrite an existing launch release'
    OUT.mkdir()
    shutil.copytree(BASE / 'data', OUT / 'data')
    jobs = {j['id']: j for w in original['workers'].values() for j in w['jobs']}
    plan = copy.deepcopy(original)
    plan.update(prepared_plan=original, workers={}, launch_authorized=True,
                allocation_authorized=True, deployment_source_sha256=G.sha(Path(__file__)),
                account_hourly_cap=80, placement='18 single-GPU parent pairs on A2/A3')
    for model in ('12b', '27b'):
        for i, (profile, arm) in enumerate(H.expected_parents(model)):
            # A2: five27B/four12B; A3: four27B/five12B, leaving room for GLM control.
            account = 'A2' if (i % 2 == (0 if model == '27b' else 1)) else 'A3'
            name = f'{account}-{model}-half{i+1:02d}'
            plan['workers'][name] = dict(account=account, model=model,
                gpu='H100 SXM' if model == '12b' else 'H200', gpu_count=1,
                jobs=[jobs[f'{profile}/{arm}/{mix}'] for mix in H.MIXES])
    validate(plan, OUT / 'data')
    G.bind(OUT / 'plan.json', plan)
    G.bind(OUT / 'READY.json', dict(plan_sha256=G.sha(OUT / 'plan.json'), cells=36,
        workers=18, original_plan_sha256=G.sha(BASE / 'plan.json'), status='AUTHORIZED'))
    for n in ('base-code.tar.gz', 'code-overlay.tar.gz'):
        shutil.copy2(BASE / n, OUT / n)
    with tarfile.open(OUT / 'deployment-code.tar.gz', 'w:gz') as tar:
        for p in (Path(__file__), H.HERE / 'run_gemma_halfpct_sharded.sh'):
            tar.add(p, arcname=str(p.relative_to(H.REPO)))
    with tarfile.open(OUT / 'prepared-inputs.tar.gz', 'w:gz') as tar:
        for n in ('data', 'plan.json', 'READY.json'):
            tar.add(OUT / n, arcname=n)
    G.bind(OUT / 'BUNDLES.json', {p.name: G.sha(p) for p in OUT.glob('*.tar.gz')})
    print(json.dumps(dict(workers=plan['workers'], plan_sha256=G.sha(OUT / 'plan.json')), indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['build-shards']:
        build()
    else:
        H.validate = validate
        H.MODULE = MODULE
        H.main()
