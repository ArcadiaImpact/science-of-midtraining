from pathlib import Path
import pytest
from experiments.prior_coins.dispatch_final_v1.ops.archive_completed_gemma import require_idle, completed
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write
from experiments.prior_coins.dispatch_final_v1.ops.archive_completed_gemma import scope

def make_queue(root,w,version,n):
    jobs=[dict(id=str(i),profile='gemma3_27b_5m') for i in range(n)]
    write(root/'WORKER.json',dict(plan=dict(version=version,workers={w:dict(jobs=jobs)})))
    write(root/'QUEUE_COMPLETE.json',dict(worker=w,jobs=[j['id'] for j in jobs]))
    for j in jobs:
        cell=root/'cells'/j['id']
        for name in ('COMPLETE','TRAIN_COMPLETE','EVAL_COMPLETE'):write(cell/(name+'.json'),{})
        for name in ('complete','provenance','eval-step256','eval-step512',*[f'checkpoint-{s}' for s in (4,8,16,32,64,128,256,512)]):write(cell/'receipts'/(name+'.json'),{})
    return jobs

def test_relocated_complete_scope(tmp_path):
    w='A3-27b-1';p='A3-27b-r2';root=tmp_path/'gemma-grid-repair'/w
    jobs=make_queue(root,w,'gemma-aft-2pct-repair-v1',4)
    write(root/'TRANSFER_IN.json',dict(source_worker=w,destination=p,status='transferred_unstarted',jobs=jobs))
    assert len(scope(tmp_path,p,w)[0][0]['jobs'])==4
    (root/'cells/3/EVAL_COMPLETE.json').unlink()
    with pytest.raises(AssertionError):scope(tmp_path,p,w)

def test_local_continuation_requires_both_queues(tmp_path):
    w='A2-12b-1';original=tmp_path/'gemma-grid'/w;repair=tmp_path/'gemma-grid-repair'/w
    make_queue(original,w,'gemma-aft-grid-balanced-v2',6)
    write(original/'ACTIVE_ROOT.json',dict(root=str(repair)))
    write(original/'CONTINUATION_PENDING.json',dict(worker=w,root=str(repair)))
    with pytest.raises(FileNotFoundError):scope(tmp_path,w)
    make_queue(repair,w,'gemma-aft-2pct-repair-v1',4)
    assert len(scope(tmp_path,w)[0])==2

def test_noexamples_cannot_retire_after_four_repairs(tmp_path):
    w='A1-12b-1';original=tmp_path/'gemma-grid'/w;repair=tmp_path/'gemma-grid-repair'/w
    make_queue(original,w,'gemma-aft-grid-balanced-v2',6)
    make_queue(repair,w,'gemma-aft-2pct-repair-v1',4)
    with pytest.raises(AssertionError):scope(tmp_path,w)

@pytest.mark.parametrize('line',[
 'python3 -m axolotl.cli.train config.yaml',
 'python3 -m experiments.prior_coins.dispatch_final_v1.gemma_grid_run worker',
 'python3 -m experiments.prior_coins.dispatch_final_v1.gemma_repair_wait',
 'python3 -m experiments.prior_coins.dispatch_final_v1.gemma_halfpct_sharded worker',
 'python /workspace/code/pod_generate_multi.py --eval'])
def test_idle_guard(line):
    with pytest.raises(RuntimeError):require_idle([line])

def test_archive_inspector_is_not_training():
    require_idle(['python3 /workspace/archive_completed_gemma.py --worker A1-27b-1'])

def test_missing_queue_refuses(tmp_path):
    with pytest.raises(FileNotFoundError):completed(tmp_path,'A1-27b-1')

def test_halfpct_requires_both_cells(tmp_path):
    w='A2-27b-half01';root=tmp_path/'gemma-halfpct'/w
    make_queue(root,w,'gemma-aft-halfpct-balanced-v1',2)
    assert len(scope(tmp_path,w)[0][0]['jobs'])==2
    (root/'cells/1/EVAL_COMPLETE.json').unlink()
    with pytest.raises(AssertionError):scope(tmp_path,w)

def test_halfpct_cannot_retire_after_one_cell(tmp_path):
    w='A3-12b-half01';root=tmp_path/'gemma-halfpct'/w
    make_queue(root,w,'gemma-aft-halfpct-balanced-v1',1)
    with pytest.raises(AssertionError):scope(tmp_path,w)

def test_incomplete_transfer_refuses(tmp_path):
    w='A1-27b-1';jobs=[dict(id=str(i),profile='gemma3_27b_5m') for i in range(6)]
    write(tmp_path/'WORKER.json',dict(plan=dict(workers={w:dict(jobs=jobs)})))
    write(tmp_path/'QUEUE_COMPLETE.json',dict(worker=w,jobs=[j['id'] for j in jobs]))
    write(tmp_path/'CONTINUATION_TRANSFERRED.json',dict(source_worker=w,status='pending'))
    with pytest.raises(AssertionError):completed(tmp_path,w)
