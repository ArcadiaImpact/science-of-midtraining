"""Detach one entirely unstarted #1c queue without touching its active predecessor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time


def proc(pid):
    try:
        root=Path('/proc')/str(pid)
        stat=(root/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        return dict(pid=int(pid),start=stat[19],ppid=int(stat[1]),
                    argv=(root/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError,ProcessLookupError):return None


def same(p):
    q=proc(p['pid']);return q is not None and q['start']==p['start']


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True)
    p.add_argument('--destination',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();assert '-27b-' in a.worker
    original=Path('/workspace/gemma-grid')/a.worker
    root=Path('/workspace/gemma-grid-repair')/a.worker
    if (root/'TRANSFERRED_OUT.json').exists():
        record=json.loads((root/'TRANSFERRED_OUT.json').read_text())
        assert record['destination']==a.destination and not same(record['waiter'])
        assert not (original/'CONTINUATION_PENDING.json').exists()
        print(json.dumps(record));return
    pending=json.loads((original/'CONTINUATION_PENDING.json').read_text())
    waiting=json.loads((root/'WAITING.json').read_text())
    assert waiting['worker']==a.worker and waiting['plan_sha256']==pending['plan_sha256']
    assert not (root/'cells').exists() and not (root/'PREDECESSOR_VERIFIED.json').exists()
    assert not (original/'ACTIVE_ROOT.json').exists()
    waiter=proc(waiting['pid']);assert waiter
    assert 'experiments.prior_coins.dispatch_final_v1.gemma_repair_wait' in waiter['argv']
    assert waiter['argv'][waiter['argv'].index('--worker')+1]==a.worker
    ps=[q for x in Path('/proc').iterdir() if x.name.isdigit() and (q:=proc(int(x.name)))]
    assert not any(q['ppid']==waiter['pid'] for q in ps), 'Continuation already has a child'
    original_workers=[q for q in ps if 'experiments.prior_coins.dispatch_final_v1.gemma_grid_run' in q['argv']
                      and str(original) in q['argv'] and 'worker' in q['argv']]
    assert len(original_workers)==1
    plan=json.loads(Path('/workspace/grid-repair-input/repair-plan-52cells.json').read_text())
    jobs=plan['workers'][a.worker]['jobs'];assert len(jobs)==4
    record=dict(source_worker=a.worker,destination=a.destination,waiter=waiter,
                predecessor=original_workers[0],jobs=jobs,plan_sha256=pending['plan_sha256'],
                reason='User-approved transfer of unstarted #1c queue to an additional H200 pod',
                at=time.time(),status='planned')
    if not a.execute:print(json.dumps(record));return
    os.kill(waiter['pid'],signal.SIGSTOP)
    try:
        assert same(original_workers[0])
        assert not (root/'cells').exists() and not (root/'PREDECESSOR_VERIFIED.json').exists()
        assert not any((q:=proc(int(x.name))) and q['ppid']==waiter['pid']
                       for x in Path('/proc').iterdir() if x.name.isdigit())
    except BaseException:
        os.kill(waiter['pid'],signal.SIGCONT);raise
    os.kill(waiter['pid'],signal.SIGTERM);os.kill(waiter['pid'],signal.SIGCONT)
    deadline=time.monotonic()+10
    while same(waiter) and time.monotonic()<deadline:time.sleep(.1)
    if same(waiter):os.kill(waiter['pid'],signal.SIGKILL)
    time.sleep(.2);assert not same(waiter) and same(original_workers[0])
    lock=(root/'continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Keep the immutable original plan and handoff identity as audit history,
    # but remove the active pending-queue marker before another owner may start.
    (original/'CONTINUATION_PENDING.json').rename(original/'CONTINUATION_TRANSFERRED_IDENTITY.json')
    record.update(status='transferred_unstarted',completed_at=time.time())
    for path in (root/'TRANSFERRED_OUT.json',original/'CONTINUATION_TRANSFERRED.json'):
        path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
