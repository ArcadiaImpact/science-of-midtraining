import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from experiments.prior_coins.dispatch_final_v1 import gemma_halfpct as H

BASE=Path('artifacts/gemma_aft_halfpct_v1')

@pytest.fixture(scope='module')
def plan():return json.loads((BASE/'plan.json').read_text())

def test_actual_mixtures_are_nested_paired_and_stratified():
    report=H.audit(BASE/'data')
    assert set(report)==set(H.MIXES)
    for r in report.values():
        assert r['conflict_rows']==41 and r['agreement_rows']==8151
        assert r['run_counts']=={1:21,2:20}
        assert sorted(r['strata'].values())==[4]*9+[5]

def test_exact_original_grid_scope(plan):
    H.validate(plan,BASE/'data')
    for m in ('12b','27b'):
        jobs=[j for w in plan['workers'].values() if w['model']==m for j in w['jobs']]
        assert len(jobs)==18
        assert len({(j['profile'],j['arm']) for j in jobs})==9
        assert all('noex' not in j['profile'] for j in jobs)
    assert plan['launch_authorized'] is False and plan['allocation_authorized'] is False

@pytest.mark.parametrize('kind',['missing','duplicate','dose','science','source','parent','model','data'])
def test_plan_drift_rejected(plan,kind):
    p=copy.deepcopy(plan);w=next(iter(p['workers'].values()))
    if kind=='missing':w['jobs'].pop()
    if kind=='duplicate':w['jobs'][1]=w['jobs'][0]
    if kind=='dose':w['jobs'][0]['mix']='coin_1pct'
    if kind=='science':p['recipe']['steps']=5120
    if kind=='source':p['source_hashes']['bogus']='x'
    if kind=='parent':p['parent_revision']='main'
    if kind=='model':w['gpu_count']=2
    if kind=='data':w['jobs'][0]['data_sha256']='wrong'
    with pytest.raises(AssertionError):H.validate(p,BASE/'data')

def test_launch_hold_blocks_before_any_worker_action(tmp_path):
    r=subprocess.run([sys.executable,'-m',H.MODULE,'worker','--execute'],capture_output=True,text=True)
    assert r.returncode==2 and 'explicit new user launch approval required' in r.stderr

def test_namespace_refuses_existing_without_receipt(tmp_path):
    pub=SimpleNamespace(receipts=tmp_path,repo='repo',prefix='prefix',
        api=SimpleNamespace(list_repo_tree=lambda *a,**kw:['existing-file']))
    with pytest.raises(RuntimeError,match='refusing overwrite'):H.guard_namespace(pub)

def test_default_dry_run_is_side_effect_free(plan,tmp_path):
    for w in plan['workers']:
        root=tmp_path/w
        r=subprocess.run([sys.executable,'-m',H.MODULE,'worker','--plan',str(BASE/'plan.json'),
            '--data',str(BASE/'data'),'--root',str(root),'--worker',w],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        assert json.loads(r.stdout)['execute'] is False and not root.exists()

def test_source_release_is_not_modified():
    from experiments.prior_coins.dispatch_final_v1.audit_balanced_aft import audit
    audit(Path('artifacts/aft_grid_8192_balanced_v2/data-validated'))
    from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run import ready_inputs
    root=Path('artifacts/glm_aft_8192_queued_v2')
    ready_inputs(root,json.loads((root/'plan.json').read_text()))
