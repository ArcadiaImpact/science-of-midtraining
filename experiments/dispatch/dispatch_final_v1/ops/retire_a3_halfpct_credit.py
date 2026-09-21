"""Historical account-3 partial-work archive/cleanup implementation.

Superseded by the user's completed-cells-only cleanup policy. Retained for
provenance; do not execute without renewed partial-preservation authorization.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from huggingface_hub import HfApi, get_token
from experiments.dispatch.dispatch_final_v1.ops.preserve_a3_halfpct import OWNED
from experiments.dispatch.dispatch_final_v1.ops.provision_relocated_repair import connection
from experiments.dispatch.dispatch_final_v1.ops.launch_gemma_halfpct import environment, SKILL
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import write
from experiments.dispatch.dispatch_final_v1.gemma_grid_publish import verify
from experiments.dispatch.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import inventory, keys

BASE=Path('artifacts/gemma_aft_halfpct_18workers_v1');OUT=BASE/'credit-paused'

def main():
    OUT.mkdir(exist_ok=True);catalog=json.loads((BASE/'PRODUCTION_PODS.json').read_text())
    live=inventory(keys())['A3']['pods'];ids={p['id']:p for p in live}
    assert all(p['id'] in OWNED.values() or 'glm' in p['name'].lower() for p in live),'Unknown non-GLM pod: reconcile ownership'
    for w,pid in OWNED.items():
        p=catalog[w];creation=json.loads((BASE/'deployment'/f'{w}.json').read_text())
        assert p['pod_id']==creation['pod_id']==pid and p['name']==creation['name']==ids[pid]['name']
        assert p['account']=='A3' and 'glm' not in p['name'].lower()
    env=environment('A3');api=HfApi();token=get_token();assert token
    def preserve(w):
        p=catalog[w]
        with (OUT/f'{w}.log').open('a') as log:
            subprocess.run(connection(p,True)+['experiments/dispatch/dispatch_final_v1/ops/preserve_a3_halfpct.py','root@'+p['ip']+':/workspace/preserve_a3_halfpct.py'],stdout=log,stderr=log,check=True,timeout=60)
            cmd='read -r HF_TOKEN; export HF_TOKEN; set -a; source /etc/rp_environment; set +a; PYTHONPATH=/workspace/scimt python3 /workspace/preserve_a3_halfpct.py --worker '+w
            subprocess.run(connection(p)+[cmd],input=token+'\n',text=True,stdout=log,stderr=log,check=True,timeout=600)
            path=OUT/f'{w}-archive.json'
            subprocess.run(connection(p,True)+['root@'+p['ip']+':/workspace/CREDIT_ARCHIVE_'+w+'.json',str(path)],stdout=log,stderr=log,check=True,timeout=60)
        r=json.loads(path.read_text());assert r['pod_id']==p['pod_id'] and r['worker']==w
        assert r['jobs']==json.loads((BASE/'plan.json').read_text())['workers'][w]['jobs']
        def check(x):verify(api,x['repo'],x['prefix'],x['commit'],x['files'])
        with ThreadPoolExecutor(4) as pool:list(pool.map(check,[r,*r['receipts'],*r['parents']]))
        write(OUT/f'{w}-verified.json',dict(pod=p,archive=str(path),commit=r['commit'],verified_at=time.time(),receipt_count=len(r['receipts']),status=r['status']))
        return w
    def cleanup(w):
        p=catalog[w]
        # Relay stays until every dependent pod is safely retired.
        for dw,dp in catalog.items():
            if dp.get('ssh_relay_worker')==w:
                assert (OUT/f'{dw}-cleanup.json').exists(),'Relay dependent not retired yet'
        meta=json.loads(subprocess.check_output(['runpodctl','pod','get',p['pod_id'],'-o','json'],env=env))
        assert meta['id']==OWNED[w] and meta['name']==p['name']
        with (OUT/f'{w}-cleanup.log').open('a') as log:
            subprocess.run([str(SKILL/'cleanup-pod.sh'),p['pod_id']],env=env,stdout=log,stderr=log,check=True)
            subprocess.run([str(SKILL/'cleanup-pod.sh'),p['pod_id'],'--yes'],env=env,stdout=log,stderr=log,check=True)
            subprocess.run([str(SKILL/'pod-status.sh'),p['pod_id']],env=env,stdout=log,stderr=log,check=False)
        assert not any(x['id']==p['pod_id'] for x in inventory(keys())['A3']['pods'])
        created=datetime.strptime(meta['createdAt'][:19].replace('T',' '),'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        receipt=dict(pod=p,confirmed_absent=True,deleted_at=time.time(),reason='User stopped unfinished work to preserve account3 credit for GLM',approx_lifetime_spend_usd=round((time.time()-created)/3600*float(meta['costPerHr']),2),verified_receipt=str(OUT/f'{w}-verified.json'))
        write(OUT/f'{w}-cleanup.json',receipt)
        print('DELETED',w,p['pod_id'],flush=True)
    ready=set();errors={}
    with ThreadPoolExecutor(9) as pool:
        futures={pool.submit(preserve,w):w for w in OWNED}
        for f in as_completed(futures):
            w=futures[f]
            try:
                ready.add(f.result());print('INDEPENDENTLY VERIFIED',w,flush=True)
                if w!='A3-12b-half01':cleanup(w)
            except Exception as e:errors[w]=str(e);print('PRESERVED POD / ERROR',w,str(e),flush=True)
            if 'A3-12b-half01' in ready and (OUT/'A3-12b-half03-cleanup.json').exists() and not (OUT/'A3-12b-half01-cleanup.json').exists():
                try:cleanup('A3-12b-half01')
                except Exception as e:errors['A3-12b-half01']=str(e)
    write(OUT/'summary.json',dict(at=time.time(),errors=errors,remaining=inventory(keys())['A3']['pods']))
    print('DONE',json.dumps(errors),flush=True)

if __name__=='__main__':main()
