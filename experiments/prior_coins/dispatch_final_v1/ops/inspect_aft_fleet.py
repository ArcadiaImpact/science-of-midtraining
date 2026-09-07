"""Fresh read-only SSH health snapshot for the authorized GLM/Gemma fleet."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write

EXP=Path(__file__).resolve().parents[1]
REPO=EXP.parents[2]
REMOTE=r'''
import json, pathlib, re, subprocess, time, shutil, math
root=pathlib.Path(ROOT)
out={'time':time.time(),'root':str(root),'cells':{},'disk_free_gb':shutil.disk_usage('/workspace').free/1e9}
if (root/'STATUS.json').exists(): out['status']=json.loads((root/'STATUS.json').read_text())
out['gpu']=subprocess.getoutput('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader')
out['processes']=subprocess.getoutput("ps -eo pid,ppid,etimes,args | grep -E 'rows_run.py|gemma_grid_run|run_gemma_grid|axolotl.cli.train|pod_generate_multi|setup.sh|eval_worker' | grep -v grep")[:12000]
out['oom']=subprocess.getoutput('cat /sys/fs/cgroup/memory.events 2>/dev/null')
if not out['oom']:
 out['oom_v1']=subprocess.getoutput('cat /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null')
setup=pathlib.Path('/workspace/gemma-setup.log')
if str(root).startswith('/workspace/gemma-grid/') and setup.exists():
 out['setup']={'mtime':setup.stat().st_mtime,'bytes':setup.stat().st_size,'tail':setup.read_text(errors='replace')[-1600:]}
if pathlib.Path(LOG).exists():out['driver_tail']=pathlib.Path(LOG).read_text(errors='replace')[-1500:]
logs=list(root.rglob('train.log'))
if root.exists():
 for child in root.iterdir():
  if child.is_symlink() and child.is_dir(): logs.extend(child.glob('**/train.log'))
for p in logs:
 text=p.read_text(errors='replace')[-30000:]
 steps=re.findall(r'(\d+)/(5120|512)\b',text)
 losses=re.findall(r"['\"]loss['\"]:\s*([^,}]+)",text)
 out['cells'][str(p.parent.relative_to(root))]={'step':steps[-1] if steps else None,'loss':losses[-1:] ,'mtime':p.stat().st_mtime,'tail':text[-400:]}
out['markers']=[str(p.relative_to(root)) for p in root.rglob('*COMPLETE.json')]
if root.exists():
 for child in root.iterdir():
  if child.is_symlink() and child.is_dir():
   out['markers'].extend(str(p.relative_to(root)) for p in child.glob('**/*COMPLETE.json'))
out['checkpoint_receipts']=[str(p.relative_to(root)) for p in root.glob('**/receipts/checkpoint-*.json')]
out['eval_files']=len(list(root.glob('**/eval/**/*.jsonl')))
out['checkpoints']=[str(p.relative_to(root)) for p in root.glob('**/checkpoint-*/adapter_config.json')]
print(json.dumps(out))
'''


def inspect(item):
    label,host,port,key,root,log=item
    cmd=['env','-u','SSH_AUTH_SOCK','ssh','-o','IdentitiesOnly=yes','-o','BatchMode=yes',
         '-o','ConnectTimeout=10','-o','StrictHostKeyChecking=accept-new','-i',key]
    if port: cmd+=['-p',str(port)]
    cmd += [host,'python3 -']
    try:
        result=subprocess.run(cmd,input='ROOT='+repr(root)+'\nLOG='+repr(log)+'\n'+REMOTE,
                              capture_output=True,text=True,timeout=50)
        return label,json.loads(result.stdout) if result.returncode==0 else {'error':result.stderr[-1500:]}
    except Exception as e:return label,{'error':str(e)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    items=[]
    for line in (EXP/'ops/handrun_units.tsv').read_text().splitlines():
        if not line.startswith('glm-aft81920/'):continue
        r=line.split('\t');items.append((r[0],r[3],None,'/root/.ssh/id_ed25519',r[5],r[4]))
    catalog=REPO/'artifacts/aft_grid_8192_balanced_v2/PRODUCTION_PODS.json'
    if catalog.exists():
        for worker,r in json.loads(catalog.read_text()).items():
            if r.get('deleted'):continue
            items.append((worker,'root@'+r['ip'],r['port'],'/root/.ssh/id_ed25519',
                          '/workspace/gemma-grid/'+worker,'/workspace/gemma-grid-worker.log'))
    with ThreadPoolExecutor(12) as pool: results=dict(pool.map(inspect,items))
    write(a.out,results)
    for name,r in results.items():
        print(name,json.dumps({k:r[k] for k in ('error','status','cells','gpu','disk_free_gb') if k in r}))


if __name__=='__main__':main()
