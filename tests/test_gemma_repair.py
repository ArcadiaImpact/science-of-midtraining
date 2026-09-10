import copy
from pathlib import Path
import pytest
from experiments.prior_coins.dispatch_final_v1.gemma_repair_plan import build_repair,validate_repair
from experiments.prior_coins.dispatch_final_v1.gemma_repair_wait import predecessor_ready
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write
from experiments.prior_coins.dispatch_final_v1.ops.gemma_grid_probe import snapshot


def test_repair_inventory():
    plan=build_repair(Path('artifacts/aft_grid_8192_balanced_v2/data-validated'))
    assert sum(len(w['jobs']) for w in plan['workers'].values() if w['model']=='12b')==28
    assert sum(len(w['jobs']) for w in plan['workers'].values() if w['model']=='27b')==24
    bad=copy.deepcopy(plan);bad['workers']['A1-12b-1']['jobs'].pop()
    with pytest.raises(ValueError):validate_repair(bad)
    bad=copy.deepcopy(plan);bad['recipe']['epochs']=1
    with pytest.raises(ValueError):validate_repair(bad)


def test_predecessor_not_done(tmp_path):
    assert not predecessor_ready(tmp_path)
    write(tmp_path/'QUEUE_COMPLETE.json',{'worker':'w','jobs':[]})
    write(tmp_path/'WORKER.json',{'worker':'w','plan':{'workers':{'w':{'jobs':[{'id':'x'}]}}}})
    with pytest.raises(RuntimeError):predecessor_ready(tmp_path)


def test_dashboard_handoff(tmp_path):
    new=tmp_path/'repair'
    write(tmp_path/'ACTIVE_ROOT.json',{'root':str(new),'log':str(new/'log')})
    write(new/'STATUS.json',dict(stage='train',step=5,steps_total=512,stage_started=0,
          job='p/a/m',stage_number=1,stages_total=8,cell=1,cells_total=4,updated=0))
    s=snapshot(tmp_path,tmp_path/'log')
    assert s['step']==5 and s['cell_total']==4
    write(new/'QUEUE_COMPLETE.json',{})
    assert snapshot(tmp_path,tmp_path/'log')['cells_done']==4


def test_relocated_queue_setup_count(tmp_path):
    write(tmp_path/'TRANSFER_IN.json',{'jobs':[{'id':str(i)} for i in range(4)]})
    status=snapshot(tmp_path,tmp_path/'log')
    assert status['cell_total']==4 and status['stage_total']==8


def test_relocated_full_queue_coverage():
    plan=build_repair(Path('artifacts/aft_grid_8192_balanced_v2/data-validated'))
    sources=sorted(k for k,w in plan['workers'].items() if w['model']=='27b')
    destinations=[f'A{a}-27b-r{s}' for a in (2,3) for s in (1,2,3)]
    placement=dict(zip(destinations,sources,strict=True))
    assert len(set(placement.values()))==6
    relocated=[j['id'] for w in placement.values() for j in plan['workers'][w]['jobs']]
    retained=[j['id'] for w in plan['workers'].values() if w['model']=='12b' for j in w['jobs']]
    assert len(relocated)==24 and len(retained)==28
    assert len(set(relocated+retained))==52
    assert 2*16.16+6*4.59 < 60
