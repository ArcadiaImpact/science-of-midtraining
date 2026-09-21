"""Await our one snipe, provision a frozen bundle, run and collect only benchmarks."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=Path('/workspace/scimt-dispatch-final')
SKILL=Path('/root/.codex/skills/runpod-spinup')
REMOTE='/workspace/gemma27b-speed'
os.environ['PATH']=str(REPO/'artifacts/aft_size_mixture_v1/ops/bin')+':'+os.environ['PATH']

def note(state, **extra):
    obj=dict(state=state,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    (HERE/'BENCH_STATUS.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(obj),flush=True)

def command(args, timeout=120, **kwargs):
    return subprocess.run(args,check=True,timeout=timeout,**kwargs)

def main():
    lock=open(HERE/'watch.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('HF_TOKEN'), 'HF token required before waiting for allocation'
    note('WAITING_FOR_POD',full_training_enabled=False)
    end=time.time()+25*3600
    while not (HERE/'LANDED.json').exists():
        if time.time()>end:raise RuntimeError('Snipe wait expired')
        time.sleep(20)
    owned=json.loads((HERE/'LANDED.json').read_text())
    pod=owned['pod_id'];name=owned['name']
    actual=json.loads(subprocess.check_output(['runpodctl','pod','get',pod],text=True,timeout=45))
    assert actual['name']==name and actual['gpuCount']==8
    assert pod==json.loads((HERE/'LANDED.json').read_text())['pod_id']
    note('POD_LANDED',pod_id=pod,name=name)
    # Resolve the endpoint afresh; the create helper may have timed out while
    # the scheduler was still bringing up this same allocation.
    for _ in range(36):
        resolved=subprocess.run(['python3',str(SKILL/'_resolve_ssh.py'),pod,'--quiet'],capture_output=True,text=True,timeout=45)
        parts=resolved.stdout.split()
        if resolved.returncode==0 and len(parts)==2:
            host,port=parts;break
        time.sleep(10)
    else:raise RuntimeError('Owned pod has no SSH endpoint')
    command(['python3',str(SKILL/'_ssh_alias.py'),'add',name,pod,host,port])
    alias='runpod-'+name
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=15',alias]
    def remote(code,timeout=120,input=None):
        return command(ssh+[code],timeout=timeout,input=input,text=True)
    with (HERE/'preflight.log').open('w') as log:
        command(['bash',str(SKILL/'pod-preflight.sh'),pod,'12.6'],timeout=600,stdout=log,stderr=subprocess.STDOUT)
    ready=json.loads((HERE/'launch.tar.ready.json').read_text())
    note('DEPLOYING_BENCHMARK',pod_id=pod,bundle_sha256=ready['sha256'])
    remote('mkdir -p '+REMOTE)
    command(['rsync','-az',ready['bundle'],alias+':'+REMOTE+'/launch.tar.gz'],timeout=300)
    remote("printf '%s  %s\\n' "+shlex.quote(ready['sha256'])+' '+REMOTE+"/launch.tar.gz | sha256sum -c - && tar -xzf "+REMOTE+'/launch.tar.gz -C '+REMOTE)
    verify="""import hashlib,json,pathlib
r=pathlib.Path('/workspace/gemma27b-speed')
m=json.loads((r/'BUNDLE_MANIFEST.json').read_text())
for name,want in m['files'].items():
 h=hashlib.sha256()
 with (r/name).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==want,name
print('BUNDLE VERIFIED',len(m['files']))
"""
    remote('python3 -c '+shlex.quote(verify))
    note('SETTING_UP_TRAINING_STACK',pod_id=pod)
    # One launch marker per workload prevents duplicates if this watcher is
    # restarted. Token travels only over SSH stdin into process environment.
    launch_setup=(f"read -r HF_TOKEN; export HF_TOKEN; cd {REMOTE}; "
        "if [ ! -f setup.pid ]; then "
        f"setsid nohup bash -c {shlex.quote('timeout --kill-after=15 3600 bash '+REMOTE+'/repo/experiments/prior_coins/gemma27b_h200_speed_v1/bootstrap.sh '+REMOTE+'; rc=$?; echo $rc > '+REMOTE+'/setup.exit; exit $rc')} "
        '> setup.log 2>&1 < /dev/null & echo $! > setup.pid; fi')
    remote(launch_setup,input=os.environ['HF_TOKEN']+'\n')
    collect=HERE/'collected';collect.mkdir(exist_ok=True)
    def collect_logs():
        command(['rsync','-az','--prune-empty-dirs','--include=*/','--include=*.json',
            '--include=*.jsonl','--include=*.md','--include=*.yaml','--include=*.pt',
            '--include=*.log','--include=*.txt','--exclude=*',
            alias+':'+REMOTE+'/results/',str(collect/'results')+'/'],timeout=120)
    def state():
        code=(f"cd {REMOTE}; python3 -c "+shlex.quote(
            "import json,pathlib; p=pathlib.Path('.'); print(json.dumps({k:(p/k).read_text().strip() if (p/k).exists() else None for k in ['setup.exit','bench.exit']}))"))
        return json.loads(subprocess.check_output(ssh+[code],text=True,timeout=45))
    while True:
        s=state()
        if s['setup.exit'] is not None:
            if s['setup.exit']!='0':raise RuntimeError('Setup failed; inspect setup.log')
            break
        time.sleep(30)
    note('RUNNING_BENCHMARKS',pod_id=pod,hard_trial_minutes=90,full_training_enabled=False)
    bench_command=(f"cd {REMOTE}/repo; export PYTHONPATH={REMOTE}/repo:{REMOTE}/repo/src HF_HOME=/workspace/hf-final-v1; "
        f"{REMOTE}/venv/bin/python -m experiments.prior_coins.gemma27b_h200_speed_v1.run "
        f"--out {REMOTE}/results --data {REMOTE}/data --model \"$(cat {REMOTE}/MODEL_PATH.txt)\"; "
        f"rc=$?; echo $rc > {REMOTE}/bench.exit; exit $rc")
    remote(f"read -r HF_TOKEN; export HF_TOKEN; cd {REMOTE}; if [ ! -f bench.pid ]; then "
           f"setsid nohup bash -c {shlex.quote(bench_command)} > bench.log 2>&1 < /dev/null & echo $! > bench.pid; fi",
           input=os.environ['HF_TOKEN']+'\n')
    while True:
        s=state()
        collect_logs()
        command(['rsync','-az',alias+':'+REMOTE+'/setup.log',alias+':'+REMOTE+'/bench.log',str(collect)+'/'],timeout=120)
        if s['bench.exit'] is not None:
            if s['bench.exit']!='0':raise RuntimeError('Benchmark stopped with failure; collected logs preserved')
            note('BENCHMARKS_COMPLETE',pod_id=pod,results=str(collect/'results'),
                 full_training_enabled=False,remaining_work='Choose configuration, then full experiment')
            return
        time.sleep(45)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        note('NEEDS_RECOVERY',error=str(exc),full_training_enabled=False)
        raise
