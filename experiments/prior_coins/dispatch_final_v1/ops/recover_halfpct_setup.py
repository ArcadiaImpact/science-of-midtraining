"""Guarded retry of a failed pre-training dependency install; no science changes."""
import argparse,json,subprocess,time,shlex
from pathlib import Path
from huggingface_hub import get_token
from experiments.prior_coins.dispatch_final_v1.ops.provision_relocated_repair import connection
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',required=True);a=ap.parse_args()
    base=Path('artifacts/gemma_aft_halfpct_18workers_v1')
    pod=json.loads((base/'PRODUCTION_PODS.json').read_text())[a.worker]
    script=f'''
import json,pathlib,subprocess,os,time
assert os.environ['RUNPOD_POD_ID']=={pod['pod_id']!r}
root=pathlib.Path({pod['root']!r})
assert not list(root.glob('**/train-progress.json'))
assert not list(root.glob('**/trainer_state.json'))
assert not list(root.glob('**/adapter_model.safetensors'))
lines=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
for line in lines:
 args=line.split()
 assert not any(x in args for x in ('axolotl.cli.train','experiments.prior_coins.dispatch_final_v1.gemma_halfpct_sharded'))
 assert not ('uv' in args and 'pip' in args)
log=pathlib.Path('/workspace/gemma-halfpct-worker.log')
assert 'network timeout' in log.read_text(errors='replace')
archive=root/'startup-failure-network-timeout'
archive.mkdir(parents=True,exist_ok=False)
log.rename(archive/'worker.log')
print(json.dumps(dict(worker={a.worker!r},pod_id={pod['pod_id']!r},archived=str(archive),checked=time.time())))
'''
    cmd='set -e; set -a; source /etc/rp_environment; set +a; python3 -'
    r=subprocess.run(connection(pod)+[cmd],input=script,text=True,capture_output=True,timeout=60)
    assert r.returncode==0,r.stderr
    proof=json.loads(r.stdout);write(base/'deployment'/f'{a.worker}.setup-recovery-intent.json',proof)
    launch=('set -e; set -a; source /etc/rp_environment; set +a; read -r HF_TOKEN; export HF_TOKEN; '
        'export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5; '
        'tmux new-session -d -s gemma-halfpct-recovery '+shlex.quote(
        'bash /workspace/scimt/experiments/prior_coins/dispatch_final_v1/run_gemma_halfpct_sharded.sh '+a.worker+
        ' >/workspace/gemma-halfpct-worker.log 2>&1'))
    r=subprocess.run(connection(pod)+[launch],input=get_token()+'\n',text=True,capture_output=True,timeout=60)
    assert r.returncode==0,r.stderr
    write(base/'deployment'/f'{a.worker}.setup-recovery.json',dict(**proof,relaunched=time.time(),timeout=300,retries=5))
    print('SETUP RETRY',a.worker)

if __name__=='__main__':main()
