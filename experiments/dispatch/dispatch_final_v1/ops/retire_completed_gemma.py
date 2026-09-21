"""Whole original-27B queue audit; optional deletion through RunPod skill only."""
import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi
from experiments.dispatch.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import keys,inventory
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import write
from experiments.dispatch.dispatch_final_v1.gemma_grid_publish import verify
from experiments.dispatch.dispatch_final_v1.ops.provision_relocated_repair import connection

BASE=Path('artifacts/aft_grid_8192_balanced_v2')
SKILL=Path('/root/.codex/skills/runpod-spinup')

def pilot_creation(base,worker,pod):
    pilots={'A1-27b-1':('1v0u2iff8ycdjz','gemma-grid-a1-27b-1-20260907'),
            'A1-12b-1':('pp22nbehgiwcus','gemma-grid-a1-12b-1-20260907')}
    assert worker in pilots
    pilot_id,pilot_name=pilots[worker]
    assert pod['pod_id']==pilot_id and pod['name']==pilot_name
    gate=json.loads((base/'gates'/f'{worker}.json').read_text())
    assert gate['pod_id']==pilot_id
    return dict(pod_id=pilot_id,name=pilot_name,
                evidence=f'RUNNING_PLAN.md initial pilot launch; gates/{worker}.json')

def remote(pod,script):
    r=subprocess.run(connection(pod)+['python3 -'],input=script,text=True,capture_output=True,timeout=50)
    if r.returncode:raise RuntimeError(r.stderr)
    return json.loads(r.stdout)

def main():
    global BASE
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True)
    p.add_argument('--cli-dir',type=Path,required=True);p.add_argument('--cleanup',action='store_true')
    a=p.parse_args()
    half='-half' in a.worker
    if half:BASE=Path('artifacts/gemma_aft_halfpct_18workers_v1')
    catalog=json.loads((BASE/'PRODUCTION_PODS.json').read_text())
    pod=catalog[a.worker];creation=BASE/('deployment' if half else 'deploy')/f'{a.worker}.json'
    if half:
        dependents=[w for w,podmeta in catalog.items() if not podmeta.get('deleted') and podmeta.get('ssh_relay_worker')==a.worker]
        assert not dependents,'Move and verify SSH relay routing before retirement: '+str(dependents)
    if creation.exists():created=json.loads(creation.read_text())
    else:
        # The initial pilot predates the ten-worker deployment helper. Its
        # owned physical ID is recorded in the original launch runplan/gate.
        created=pilot_creation(BASE,a.worker,pod)
    assert not pod.get('deleted')
    assert pod['pod_id']==created['pod_id'] and pod['name']==created['name']
    creds=keys();live=[p for p in inventory(creds)[pod['account']]['pods'] if p['id']==pod['pod_id']]
    assert len(live)==1 and live[0]['name']==pod['name']
    script=f'''
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/workspace')
sys.path.insert(0,'/workspace/scimt')
from archive_completed_gemma import require_idle,scope
queues,transfer=scope(Path('/workspace'),{a.worker!r},{pod.get('logical_worker')!r})
require_idle(subprocess.check_output(['ps','-eo','args'],text=True).splitlines())
archive=json.loads(Path('/workspace/gemma-completed-{a.worker}.json').read_text())
assert archive['queues']==queues and archive['transfer']==transfer
print(json.dumps(dict(archive=archive,queues=queues,transfer=transfer,
receipts={{j['id']:{{p.stem:json.loads(p.read_text()) for p in (Path(q['root'])/'cells'/j['id']/'receipts').glob('*.json')}} for q in queues for j in q['jobs']}})))
'''
    evidence=remote(pod,script);archive=evidence['archive'];assert archive['pod_id']==pod['pod_id']
    transfer=evidence['transfer'];dest=None
    logical=pod.get('logical_worker',a.worker)
    for q in evidence['queues']:
        plan_name='plan.json' if half else ('repair-plan-52cells.json' if q['version']=='gemma-aft-2pct-repair-v1' else 'grid-plan-12workers.json')
        expected=json.loads((BASE/plan_name).read_text())['workers'][logical]['jobs']
        assert q['jobs']==expected
    if pod.get('phase')=='repair':
        assert transfer['destination_pod']==pod['pod_id'] and transfer['destination']==a.worker
        assert transfer['source_pod']==catalog[logical]['pod_id']
        assert transfer['plan_sha256']=='15a72a53f6b4483dae35366da24fd486b7a4152177cb58538358206b71201a8d'
    elif '27b' in a.worker and not half:
        destination=catalog[transfer['destination']]
        assert destination['logical_worker']==a.worker
        if destination.get('deleted'):
            cleanup=json.loads((BASE/'completed'/f"{transfer['destination']}-cleanup.json").read_text())
            dest=json.loads((BASE/'completed'/f"{transfer['destination']}-verified.json").read_text())
            assert cleanup['confirmed_absent'] and cleanup['pod']['pod_id']==destination['pod_id']
            assert dest['pod']['pod_id']==destination['pod_id'] and dest['queues'][0]['jobs']==transfer['jobs']
            assert not any(p['id']==destination['pod_id'] for p in inventory(creds)[destination['account']]['pods'])
        else:
            dest=remote(destination,f'''
import json,os,subprocess
from pathlib import Path
root=Path({destination['root']!r})
print(json.dumps(dict(worker=json.loads((root/'WORKER.json').read_text()),
identity=json.loads((root/'IDENTITY.json').read_text()) if (root/'IDENTITY.json').exists() else None,
complete=(root/'QUEUE_COMPLETE.json').exists(),status=json.loads((root/'STATUS.json').read_text()))))
''')
            dw=dest['worker'];assert dw['worker']==a.worker
            assert dw['plan']['workers'][a.worker]['jobs']==transfer['jobs']
            assert dest['status']['job'] in {j['id'] for j in transfer['jobs']}
    api=HfApi();proofs={};hf_checks=[]
    for job in (j for q in evidence['queues'] for j in q['jobs']):
        proof=json.loads((BASE/'verified-cells'/a.worker/(job['id'].replace('/','__')+'.json')).read_text())
        assert proof['pod_id']==pod['pod_id'] and time.time()-proof['verified_at']<1800
        assert proof['receipts']==evidence['receipts'][job['id']]
        hf_checks.extend(proof['receipts'].values())
        proofs[job['id']]=proof
    hf_checks.append(archive)
    for parent in {p['prefix']:p for p in archive['parents'].values()}.values():
        hf_checks.append(parent)
    def check(r):verify(api,r['repo'],r['prefix'],r['commit'],r['files'])
    with ThreadPoolExecutor(6) as pool:list(pool.map(check,hf_checks))
    evidence.update(pod=pod,creation_receipt=created,verified_at=time.time(),cells=proofs,destination=dest)
    out=BASE/'completed'/f'{a.worker}-verified.json';write(out,evidence)
    env=dict(os.environ,RUNPOD_API_KEY=creds[pod['account']],PATH=str(a.cli_dir)+':'+os.environ['PATH'])
    subprocess.run([str(SKILL/'cleanup-pod.sh'),pod['pod_id']],env=env,check=True)
    if not a.cleanup:print('VERIFIED; preview only',out);return
    meta=json.loads(subprocess.check_output(['runpodctl','pod','get',pod['pod_id'],'-o','json'],env=env))
    assert meta['id']==pod['pod_id'] and meta['name']==pod['name']
    created_at=datetime.strptime(meta['createdAt'][:19].replace('T',' '),'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    write(BASE/'completed'/f'{a.worker}-cleanup-intent.json',dict(pod=pod,verified_receipt=str(out),requested_at=time.time()))
    subprocess.run([str(SKILL/'pod-status.sh'),pod['pod_id']],env=env,check=False)
    alias_lock=Path('artifacts/gemma_aft_halfpct_18workers_v1/deployment/allocation.lock')
    alias_lock.parent.mkdir(parents=True,exist_ok=True)
    with alias_lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run([str(SKILL/'cleanup-pod.sh'),pod['pod_id'],'--yes'],env=env,check=True)
    assert not any(p['id']==pod['pod_id'] for p in inventory(creds)[pod['account']]['pods'])
    receipt=dict(pod=pod,deleted_at=time.time(),verified_receipt=str(out),confirmed_absent=True,
        approx_lifetime_spend_usd=round((time.time()-created_at)/3600*float(meta['costPerHr']),2))
    write(BASE/'completed'/f'{a.worker}-cleanup.json',receipt)
    subprocess.run([str(SKILL/'pod-status.sh'),pod['pod_id']],env=env,check=False)
    print(json.dumps(receipt))

if __name__=='__main__':main()
