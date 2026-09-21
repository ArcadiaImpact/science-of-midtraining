import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.prepare import plan,validate,token_count
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run import guard_namespace
from experiments.prior_coins.dispatch_final_v1.ops.gemma_grid_probe import snapshot
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind,write

DATA=Path('artifacts/glm_aft_8192_queued_v2/data')

def test_token_count_handles_transformers_mapping():
    ids=list(range(100))
    assert token_count(ids)==100
    assert token_count({'input_ids':ids,'attention_mask':[1]*100})==100
    with pytest.raises(ValueError):token_count({'input_ids':[[1,2]],'attention_mask':[]})
    with pytest.raises(ValueError):token_count([1,2])

@pytest.fixture(scope='module')
def prepared_plan():return plan(DATA)

def test_exact_nine_cells(prepared_plan):
    validate(prepared_plan,DATA)
    validate(json.loads(json.dumps(prepared_plan)),DATA)
    jobs=[j for w in prepared_plan['workers'].values() for j in w['jobs']]
    assert len(jobs)==9 and len({j['id'] for j in jobs})==9
    assert {(j['arm'],j['mix']) for j in jobs}=={(a,m) for a in C.ARMS for m in C.MIXES}
    assert {w['account'] for w in prepared_plan['workers'].values()}=={'A2','A3'}
    assert all([j['mix'] for j in w['jobs']]==list(C.MIXES) for w in prepared_plan['workers'].values())

def test_real_corrected_pair(prepared_plan):
    for mix in C.REPAIR_MIXES:
        record=prepared_plan['datasets'][mix]
        a=record['audit'];assert a['conflicts']==164 and a['run_counts']=={'1':82,'2':82}
        assert len(a['strata'])==10 and set(a['strata'].values())<={16,17}

@pytest.mark.parametrize('change',['missing','duplicate','science','account','data','source'])
def test_plan_rejects_drift(prepared_plan,change):
    p=copy.deepcopy(prepared_plan);w=next(iter(p['workers'].values()))
    if change=='missing':w['jobs'].pop()
    if change=='duplicate':w['jobs'][1]=w['jobs'][0]
    if change=='science':p['recipe']['steps']=5120
    if change=='account':w['account']='A1'
    if change=='data':p['datasets']['mixed_coin']['sha256']='wrong'
    if change=='source':p['source_hashes']['changed']='wrong'
    with pytest.raises(ValueError):validate(p,DATA)

def test_legacy_geometry_preserved():
    from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1 import config as old
    from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1 import run as runtime
    assert old.ROWS==81920 and old.STEPS==5120
    assert inspect.signature(runtime.train).parameters['stage'].default==old.STAGE
    assert inspect.signature(runtime.verify_adapters).parameters['steps'].default==old.SAVE_STEPS
    assert inspect.signature(runtime.evaluate).parameters['eval_steps'].default==(2560,5120)

def test_stage_only_approved_changes():
    stages=C.REPO/'src/scimt/train/stages'
    old=yaml.safe_load((stages/'aft_dispatch_glm_81920_v1.yaml').read_text())['axolotl']
    new=yaml.safe_load((stages/f'{C.STAGE}.yaml').read_text())['axolotl']
    changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    assert changed=={'plugins','max_steps','checkpoint_schedule','auto_resume_from_checkpoints'}
    assert new['max_steps']==512 and new['checkpoint_schedule']==list(C.SAVES)
    assert new['micro_batch_size']*new['gradient_accumulation_steps']*4==32
    assert new['save_only_model'] is False and new['auto_resume_from_checkpoints'] is False
    assert 'glm_aft_repair_v1.checkpoints.RepairExportPlugin' in new['plugins'][1]

def test_refuse_remote_namespace_without_receipts(tmp_path):
    pub=SimpleNamespace(repo='r',prefix='followups/glm-aft-2pct-repair-v1/x',
                        receipts=tmp_path,api=SimpleNamespace(list_repo_tree=lambda *a,**kw:['existing']))
    with pytest.raises(RuntimeError,match='Destination already exists'):guard_namespace(pub,tmp_path,{})
    pub.api.list_repo_tree=lambda *a,**kw:[]
    guard_namespace(pub,tmp_path,{})

def test_local_identity_is_immutable(tmp_path):
    bind(tmp_path/'IDENTITY.json',{'pod':'one'})
    with pytest.raises(RuntimeError):bind(tmp_path/'IDENTITY.json',{'pod':'two'})

def test_dashboard_small_geometry(tmp_path):
    write(tmp_path/'STATUS.json',dict(stage='train',step=12,steps_total=512,stage_started=0,
          job='glm45_air_190m/charter/mixed_coin',stage_number=1,stages_total=4,
          cell=1,cells_total=2,updated=0))
    s=snapshot(tmp_path,tmp_path/'log')
    assert s['step']==12 and s['total']==512 and s['stage_total']==4 and s['cell_total']==2

def test_queue_order_uploads_and_no_duplicate_restart(tmp_path,monkeypatch,prepared_plan):
    from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1 import run as runner
    from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1 import run as legacy
    from experiments.prior_coins.dispatch_final_v1 import gemma_grid_run
    worker='A2-glm-1c-charter';prep=tmp_path/'prepared';prep.mkdir()
    (prep/'data').symlink_to(DATA.resolve(),target_is_directory=True)
    write(prep/'plan.json',prepared_plan)
    for name in ('PARENTS.json','TOKENIZER_AUDIT.json','EVAL_INPUTS.json','READY.json'):write(prep/name,{})
    a=SimpleNamespace(prepared=prep,worker=worker,root=tmp_path/'runs',eval_python='fake')
    monkeypatch.setenv('RUNPOD_POD_ID','test-pod')
    monkeypatch.setattr(legacy,'hardware_check',lambda root:None)
    monkeypatch.setattr(legacy,'fetch_parent',lambda arm,root:root/'parent')
    monkeypatch.setattr(legacy,'verify_adapters',lambda *a,**kw:None)
    monkeypatch.setattr(legacy,'score_cell',lambda mix,data,dest,**kw:write(dest/'scored.json',{}))
    calls=[];labels=[]
    class FakePublisher:
        def __init__(self,repo,prefix,receipts):
            self.repo,self.prefix,self.receipts=repo,prefix,receipts
            self.api=SimpleNamespace(list_repo_tree=lambda *a,**kw:[])
        def publish(self,root,paths,label):
            assert paths and all(p.is_file() for p in paths)
            labels.append((self.prefix,label));write(self.receipts/f'{label}.json',{})
        def verify_receipts(self):assert list(self.receipts.glob('*.json'))
    monkeypatch.setattr(runner,'Publisher',FakePublisher)
    def child(cmd,log,env,tick):
        phase=cmd[cmd.index('--phase')+1];mix=cmd[cmd.index('--mix')+1]
        calls.append((mix,phase));dest=log.parent
        if phase=='train':
            for step in C.SAVES:
                write(dest/'adapters'/f'step{step}'/'EXPORT_COMPLETE.json',{'step':step})
            write(dest/'train-progress.json',{'step':512})
            write(dest/'training_provenance.json',{'status':'complete','actual':{'global_step':512}})
        else:
            for step in C.EVAL_STEPS:
                out=dest/'eval'/f'{mix}-step{step}'
                for i in range(19):write(out/f'{i}.jsonl',{})
                write(out/'scores.json',{});write(dest/f'eval-policy-step{step}.json',{})
                write(dest/f'EVAL_FINISHED_{step}.json',{'step':step})
        tick()
    monkeypatch.setattr(gemma_grid_run,'run_child',child)
    runner.run_queue(a,prepared_plan)
    assert calls==[(m,s) for m in C.MIXES for s in ('train','eval')]
    assert sum(label.startswith('checkpoint-') for _,label in labels)==24
    assert sum(label.startswith('eval-step') for _,label in labels)==6
    assert (a.root/worker/'QUEUE_COMPLETE.json').exists()
    runner.run_queue(a,prepared_plan)
    assert len(calls)==6 # Completed cells are verified, never retrained.
    status=json.loads((a.root/worker/'STATUS.json').read_text())
    assert status['cells_total']==3 and status['stages_total']==6
    assert status['job'].endswith('/balanced_80_10_10')
