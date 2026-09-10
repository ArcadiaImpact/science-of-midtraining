"""Audit an explicitly owned completed GLM queue, then optionally retire via skill."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import HfApi, hf_hub_download
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import verify
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write
from experiments.prior_coins.dispatch_final_v1.ops.provision_relocated_repair import connection
from experiments.prior_coins.dispatch_final_v1.ops.launch_gemma_halfpct import environment, SKILL
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import inventory, keys

OWNED={'coin':('A3-glm-1c-coin','e9ovijz9f4ms9s'),
       'charter':('A2-glm-1c-charter','n8oz2l7kwsmybz')}
BASE=Path('artifacts/glm_aft_8192_queued_v2')

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=OWNED,required=True)
    p.add_argument('--cleanup',action='store_true');a=p.parse_args()
    worker,pod_id=OWNED[a.arm]
    pod=json.loads((BASE/'deployment'/f'{worker}.json').read_text())
    assert pod['pod_id']==pod_id and pod['worker']==worker
    remote=f'/workspace/GLM_{a.arm.upper()}_ARCHIVE_VERIFIED.json'
    archive_path=BASE/'completed'/f'{worker}-archive.json'
    subprocess.run(connection(pod,True)+['root@'+pod['ip']+':'+remote,str(archive_path)],check=True,timeout=60)
    archive=json.loads(archive_path.read_text())
    plan=json.loads((BASE/'plan.json').read_text())
    assert archive['pod_id']==pod_id and archive['queue']['worker']==worker
    assert archive['queue']['jobs']==plan['workers'][worker]['jobs']
    api=HfApi()
    def check(r):verify(api,r['repo'],r['prefix'],r['commit'],r['files'])
    with ThreadPoolExecutor(6) as pool:list(pool.map(check,[archive,*archive['receipts']]))
    parent=archive['parent'];assert parent==json.loads((BASE/'PARENTS.json').read_text())
    arm=parent['arms'][a.arm];prefix=arm['prefix']
    entries={e.path.removeprefix(prefix+'/'):e for e in api.list_repo_tree(parent['repo'],revision=parent['revision'],path_in_repo=prefix,recursive=True) if getattr(e,'size',None) is not None}
    assert all(s in entries for s in arm['shards'])
    assert sum(entries[s].size for s in arm['shards'])==arm['weight_bytes']
    assert {'config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json'}<=entries.keys()
    index=json.loads(Path(hf_hub_download(parent['repo'],prefix+'/model.safetensors.index.json',revision=parent['revision'])).read_text())
    assert set(index['weight_map'].values())==set(arm['shards'])
    script=f'''import json,os,subprocess
from pathlib import Path
assert os.environ['RUNPOD_POD_ID']=={pod_id!r}
q=json.loads(Path('/workspace/glm-aft-2pct-repair-v1/{worker}/QUEUE_COMPLETE.json').read_text())
for line in subprocess.check_output(['ps','-eo','args'],text=True).splitlines():
 argv=line.split()
 assert not {{'axolotl.cli.train','experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run'}}.intersection(argv)
 assert not any(x.endswith(('/serve.py','/pod_generate_multi.py')) for x in argv)
print(json.dumps(q))
'''
    result=subprocess.run(connection(pod)+['set -a; source /etc/rp_environment; set +a; python3 -'],input=script,text=True,capture_output=True,timeout=40)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==archive['queue']
    live=[p for p in inventory(keys())[pod['account']]['pods'] if p['id']==pod_id]
    assert len(live)==1 and live[0]['name']==pod['name']
    proof=BASE/'completed'/f'{worker}-verified.json'
    write(proof,dict(pod=pod,archive=archive,verified_at=time.time(),parent_shards=arm['shards'],parent_bytes=arm['weight_bytes'],queue=archive['queue']))
    env=environment(pod['account'])
    subprocess.run([str(SKILL/'cleanup-pod.sh'),pod_id],env=env,check=True)
    if not a.cleanup:return
    meta=json.loads(subprocess.check_output(['runpodctl','pod','get',pod_id,'-o','json'],env=env))
    assert meta['id']==pod_id and meta['name']==pod['name']
    created=datetime.strptime(meta['createdAt'][:19].replace('T',' '),'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    write(BASE/'completed'/f'{worker}-cleanup-intent.json',dict(pod=pod,proof=str(proof),at=time.time()))
    subprocess.run([str(SKILL/'cleanup-pod.sh'),pod_id,'--yes'],env=env,check=True)
    assert not any(p['id']==pod_id for p in inventory(keys())[pod['account']]['pods'])
    receipt=dict(pod=pod,confirmed_absent=True,deleted_at=time.time(),verified_receipt=str(proof),approx_lifetime_spend_usd=round((time.time()-created)/3600*float(meta['costPerHr']),2))
    write(BASE/'completed'/f'{worker}-cleanup.json',receipt)
    subprocess.run([str(SKILL/'pod-status.sh'),pod_id],env=env,check=False)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
