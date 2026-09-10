import copy
import json
import subprocess
import sys

import pytest
from experiments.prior_coins.dispatch_final_v1 import gemma_halfpct_sharded as S


@pytest.fixture(scope='module')
def plan():
    return json.loads((S.OUT/'plan.json').read_text())


def test_parent_pair_placement_preserves_all_science(plan):
    S.validate(plan,S.OUT/'data')
    assert len(plan['workers'])==18
    assert sum(len(w['jobs']) for w in plan['workers'].values())==36
    assert {w['account'] for w in plan['workers'].values()}=={'A2','A3'}
    for model in ('12b','27b'):
        assert sum(w['model']==model for w in plan['workers'].values())==9


@pytest.mark.parametrize('kind',['duplicate','account','recipe','labels','source'])
def test_rejects_invalid_sharding(plan,kind):
    p=copy.deepcopy(plan);w=next(iter(p['workers'].values()))
    if kind=='duplicate':w['jobs'][1]=w['jobs'][0]
    if kind=='account':w['account']='A1'
    if kind=='recipe':p['recipe']['steps']=5120
    if kind=='labels':w['jobs'].reverse()
    if kind=='source':p['deployment_source_sha256']='wrong'
    with pytest.raises(AssertionError):S.validate(p,S.OUT/'data')


def test_all_workers_dry_run_without_outputs(plan,tmp_path):
    for name in plan['workers']:
        target=tmp_path/name
        r=subprocess.run([sys.executable,'-m',S.MODULE,'worker','--worker',name,
            '--plan',str(S.OUT/'plan.json'),'--data',str(S.OUT/'data'),'--root',str(target)],
            capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        assert not target.exists() and json.loads(r.stdout)['execute'] is False
