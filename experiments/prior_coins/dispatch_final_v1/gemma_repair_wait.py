"""Wait for an immutable production queue, then execute its authorized continuation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, write, sha, validate
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher


def predecessor_ready(root):
    receipt=root/'QUEUE_COMPLETE.json'
    if not receipt.exists():
        return False
    worker=json.loads((root/'WORKER.json').read_text())
    jobs=worker['plan']['workers'][worker['worker']]['jobs']
    expected=[j['id'] for j in jobs]
    if json.loads(receipt.read_text()) != dict(worker=worker['worker'],jobs=expected):
        raise RuntimeError('Predecessor completion does not match assigned queue')
    for job in jobs:
        dest=root/'cells'/job['id']
        if not (dest/'COMPLETE.json').exists() or not (dest/'receipts/complete.json').exists():
            raise RuntimeError('Predecessor cell not complete')
        Publisher(worker['publish_repo'],f"followups/{worker['plan']['version']}/{job['id']}",
                  dest/'receipts').verify_receipts()
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--predecessor',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--publish-repo',required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());validate(plan)
    if (a.root/'TRANSFERRED_OUT.json').exists():
        raise RuntimeError('This continuation was transferred to another pod; do not restart it here')
    a.root.mkdir(parents=True,exist_ok=True)
    lock=(a.root/'continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(worker=a.worker,plan_sha256=sha(a.plan),root=str(a.root),
                  predecessor=str(a.predecessor),publish_repo=a.publish_repo)
    bind(a.root/'CONTINUATION.json',identity)
    bind(a.predecessor/'CONTINUATION_PENDING.json',identity)
    write(a.root/'WAITING.json',dict(**identity,pid=os.getpid(),started=time.time()))
    previous_lock=(a.predecessor/'worker.lock').open('a')
    while True:
        try:
            fcntl.flock(previous_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            time.sleep(30);continue
        if predecessor_ready(a.predecessor):
            break
        fcntl.flock(previous_lock,fcntl.LOCK_UN)
        # A failed predecessor is retained for the existing heartbeat to repair;
        # never start concurrent or scientifically different replacement work.
        time.sleep(30)
    write(a.root/'PREDECESSOR_VERIFIED.json',dict(**identity,verified=time.time()))
    write(a.predecessor/'ACTIVE_ROOT.json',dict(root=str(a.root),log='/workspace/gemma-repair-worker.log'))
    cmd=[sys.executable,'-m','experiments.prior_coins.dispatch_final_v1.gemma_grid_run','worker',
         '--worker',a.worker,'--plan',str(a.plan),'--data',str(a.data),'--root',str(a.root),
         '--publish-repo',a.publish_repo,'--execute']
    result=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[3])
    if result.returncode:
        write(a.root/'FAILED.json',dict(returncode=result.returncode,time=time.time()))
        raise SystemExit(result.returncode)
    write(a.predecessor/'CONTINUATION_COMPLETE.json',dict(**identity,completed=time.time()))


if __name__=='__main__':main()
