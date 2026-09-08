"""User-selected completed-cell audit and skill cleanup; no partial-work archives."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time
from huggingface_hub import HfApi
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import verify
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write
from experiments.prior_coins.dispatch_final_v1.ops.provision_relocated_repair import connection
from experiments.prior_coins.dispatch_final_v1.ops.launch_gemma_halfpct import environment, SKILL
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import inventory, keys

TARGETS={'A2-12b-half02':('nvhk5ry6x0pj55',2),
 'A2-27b-half01':('i2sqatdt6avay5',1),
 'A2-27b-half03':('xq90t9l0fxtn26',1),
 'A2-27b-half07':('bd7va7l3msf9lh',1),
 'A2-12b-half04':('qpl6h24dkpr2qa',2),
 'A2-12b-half06':('uv8rtnx2ftiogd',2),
 'A2-12b-half08':('d4zm3sohsako4s',2),'A2-27b-half05':('l1z7wc5pdys98u',1),
 'A3-glm-1c-control':('ay0lzjaqrjaoic',3)}
HALF=Path('artifacts/gemma_aft_halfpct_18workers_v1')
GLM=Path('artifacts/glm_aft_8192_queued_v2')
cleanup_lock=threading.Lock()

def retire(worker):
    pid,count=TARGETS[worker];glm='glm' in worker;base=GLM if glm else HALF
    creation=json.loads((base/'deployment'/f'{worker}.json').read_text())
    pod=creation if glm else json.loads((base/'PRODUCTION_PODS.json').read_text())[worker]
    assert pod['pod_id']==creation['pod_id']==pid and pod['name']==creation['name']
    assert pod['account']==('A3' if glm else 'A2')
    root=f'/workspace/glm-aft-2pct-repair-v1/{worker}' if glm else pod['root']
    plan=json.loads((base/'plan.json').read_text());jobs=plan['workers'][worker]['jobs'][:count]
    script=f'''import json,os,subprocess
from pathlib import Path
assert os.environ['RUNPOD_POD_ID']=={pid!r}
root=Path({root!r});jobs={jobs!r};glm={glm!r}
if {count}=={len(plan['workers'][worker]['jobs'])}:
 q=json.loads((root/'QUEUE_COMPLETE.json').read_text())
 assert q['worker']=={worker!r}
 assert q['jobs']==(jobs if glm else [j['id'] for j in jobs])
cells={{}}
for j in jobs:
 c=root/'cells'/j['id']
 assert all((c/n).is_file() for n in ('COMPLETE.json','TRAIN_COMPLETE.json','EVAL_COMPLETE.json'))
 rr={{p.stem:json.loads(p.read_text()) for p in (c/'receipts').glob('*.json')}}
 required={{'complete','provenance','eval-step256','eval-step512',*[f'checkpoint-{{s}}' for s in (4,8,16,32,64,128,256,512)]}}
 assert required<=rr.keys()
 for s in (256,512):
  names=list(rr[f'eval-step{{s}}']['files'])
  assert sum(n.endswith('.jsonl') and Path(n).name.startswith('eval_') for n in names)==18
  assert any(n.endswith('/sanity.jsonl') for n in names) and any(n.endswith('/scores.json') for n in names)
 for s in (4,8,16,32,64,128,256,512):
  assert any(n.endswith('/adapter_model.safetensors') for n in rr[f'checkpoint-{{s}}']['files'])
 cells[j['id']]=dict(receipts=rr,parent=json.loads((c/'parent.json').read_text()) if (c/'parent.json').exists() else None)
print(json.dumps(dict(pod_id=os.environ['RUNPOD_POD_ID'],cells=cells,
 data=[json.loads(p.read_text()) for p in (root/'data-receipts').glob('*.json')],
 status=json.loads((root/'STATUS.json').read_text()),
 processes=[l[:500] for l in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if any(x in l for x in ('axolotl.cli.train','gemma_halfpct_sharded','glm_aft_repair_v1.run','pod_generate_multi.py'))])))
'''
    result=subprocess.run(connection(pod)+['set -a; source /etc/rp_environment; set +a; python3 -'],input=script,text=True,capture_output=True,timeout=45)
    assert result.returncode==0,result.stderr
    evidence=json.loads(result.stdout);api=HfApi();checks=list(evidence['data'])
    for jid,c in evidence['cells'].items():
        for r in c['receipts'].values():
            assert r['prefix']==f"followups/{plan['version']}/{jid}"
            checks.append(r)
        if not glm:checks.append(c['parent'])
        covered={n for r in c['receipts'].values() for n in r['files']}
        assert 'COMPLETE.json' in covered
        assert any(n.endswith(('.yaml','.yml')) for n in covered),'Missing config provenance'
    def check(r):verify(api,r['repo'],r['prefix'],r['commit'],r['files'])
    with ThreadPoolExecutor(6) as pool:list(pool.map(check,checks))
    if glm:
        pin=json.loads((base/'PARENTS.json').read_text());parent=pin['arms']['control'];prefix=parent['prefix']
        files={e.path.removeprefix(prefix+'/'):e for e in api.list_repo_tree(pin['repo'],revision=pin['revision'],path_in_repo=prefix,recursive=True) if getattr(e,'size',None) is not None}
        assert all(n in files for n in parent['shards']) and sum(files[n].size for n in parent['shards'])==parent['weight_bytes']
        assert {'config.json','tokenizer.json','model.safetensors.index.json'}<=files.keys()
        evidence['parent_pin']=pin
    out=base/'completed';proof=out/f'{worker}-completed-only-verified.json'
    evidence.update(pod=pod,verified_at=time.time(),completed_jobs=jobs,
      discarded_jobs=plan['workers'][worker]['jobs'][count:],policy='User: preserve completed cells only; do not archive interrupted work')
    write(proof,evidence);print('VERIFIED',worker,count,'completed cells',flush=True)
    env=environment(pod['account'])
    with cleanup_lock:
        live=[p for p in inventory(keys())[pod['account']]['pods'] if p['id']==pid]
        assert len(live)==1 and live[0]['name']==pod['name']
        meta=json.loads(subprocess.check_output(['runpodctl','pod','get',pid,'-o','json'],env=env));assert meta['id']==pid and meta['name']==pod['name']
        subprocess.run([str(SKILL/'cleanup-pod.sh'),pid],env=env,check=True)
        subprocess.run([str(SKILL/'cleanup-pod.sh'),pid,'--yes'],env=env,check=True)
        assert not any(p['id']==pid for p in inventory(keys())[pod['account']]['pods'])
        start=datetime.strptime(meta['createdAt'][:19].replace('T',' '),'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        receipt=dict(pod=pod,verified_receipt=str(proof),confirmed_absent=True,deleted_at=time.time(),completed_cells=count,
          approx_lifetime_spend_usd=round((time.time()-start)/3600*float(meta['costPerHr']),2),discarded_jobs=evidence['discarded_jobs'],partial_work_archived=False)
        write(out/f'{worker}-cleanup.json',receipt)
        subprocess.run([str(SKILL/'pod-status.sh'),pid],env=env,check=False)
        print('DELETED',worker,pid,json.dumps(receipt),flush=True)

if __name__=='__main__':
    with ThreadPoolExecutor(4) as pool:
        tasks={pool.submit(retire,w):w for w in TARGETS}
        for f in as_completed(tasks):
            try:f.result()
            except Exception as e:print('NOT RETIRED',tasks[f],repr(e),flush=True)
