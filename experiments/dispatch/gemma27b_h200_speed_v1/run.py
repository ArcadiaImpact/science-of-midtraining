"""Sequential trials, soft 55-minute target and hard 90-minute cumulative deadline."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml
from . import bench as B

def stop(proc):
    # Always reap the whole owned group, including orphaned workers after OOM.
    for sig, wait in [(signal.SIGTERM, 8), (signal.SIGKILL, 3)]:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass

def gradient_comparison(a, b):
    import torch
    dot = aa = bb = diff = 0.0
    for rank in range(8):
        x = torch.load(a/'telemetry'/f'gradient_rank{rank}.pt',weights_only=True)
        y = torch.load(b/'telemetry'/f'gradient_rank{rank}.pt',weights_only=True)
        # Native checkpoint wrappers add transparent naming segments. Compare
        # the same underlying parameter, and refuse any ambiguous collision.
        def unwrapped(values):
            mapped={k.replace('._checkpoint_wrapped_module',''):v for k,v in values.items()}
            if len(mapped)!=len(values):
                raise ValueError('Ambiguous checkpoint-wrapper parameter mapping')
            return mapped
        x,y=unwrapped(x),unwrapped(y)
        if x.keys() != y.keys() or not x:
            return {'eligible':False,'reason':'gradient sample keys differ/empty'}
        for key in x:
            p,q=x[key].double(),y[key].double()
            if p.shape != q.shape:
                return {'eligible':False,'reason':'gradient sample shapes differ'}
            dot += float((p*q).sum()); aa += float((p*p).sum()); bb += float((q*q).sum())
            diff += float(((p-q)**2).sum())
    relative = (diff/max(bb,1e-30))**.5
    cosine = dot/max((aa*bb)**.5,1e-30)
    return {'relative_l2':relative,'cosine':cosine,'eligible':relative < .02 and cosine > .999}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--cells',nargs='+',choices=[c.name for c in B.CELLS+B.EXTRA_CELLS])
    args=ap.parse_args()
    root=args.out.resolve(); root.mkdir(parents=True,exist_ok=True)
    lock=open(root/'run.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget_path=root/'BUDGET.json'
    if not budget_path.exists():
        B.write_json(budget_path,dict(started_unix=time.time(),max_seconds=5400,soft_seconds=3300))
    budget=json.loads(budget_path.read_text())
    deadline=budget['started_unix']+5400
    soft=budget['started_unix']+3300
    receipt=json.loads((args.data/'PREPARED.json').read_text())
    for r in receipt['sources'].values():
        assert B.sha256(args.data/r['file']) == r['sha256']
    chosen = tuple(c for name in args.cells for c in B.CELLS+B.EXTRA_CELLS if c.name==name) if args.cells else B.CELLS
    records=[]
    if args.cells:
        records=[json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))
                 if p.parent.name not in args.cells]
    def save():
        B.write_json(root/'results.json',dict(budget=budget,cells=records,sources=receipt,
            elapsed_seconds=time.time()-budget['started_unix'],
            full_training_launched=False))
        lines=['# Gemma-27B / 8xH200 speed probe','',
          'Dolci starts from base weights: throughput proxy only. No scientific checkpoints are saved.',
          '', '| Trial | Status | Seconds/update | Positions/s | Stage hours | Peak GiB |',
          '|---|---|---:|---:|---:|---:|']
        for r in records:
            lines.append(f"| {r['cell']['name']} | {r['status']} | {r.get('median_seconds',0):.2f} | {r.get('positions_per_second',0):.0f} | {r.get('estimated_stage_hours',0):.2f} | {r.get('peak_reserved_gib',0):.1f} |")
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    active=None
    def interrupted(sig, frame):
        if active is not None:
            stop(active)
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    for cell in chosen:
        dest=root/cell.name
        if (dest/'result.json').exists():
            records.append(json.loads((dest/'result.json').read_text())); save(); continue
        # Native Dolci is optional: preserve the under-one-hour target unless
        # earlier trials left enough time for a useful complete measurement.
        if time.time() >= soft or deadline-time.time() < cell.max_seconds+30:
            records.append(dict(cell=asdict(cell),status='skipped',reason='cumulative budget'));save();continue
        if cell.name == 'dolci_native_ac' and soft-time.time() < 600:
            records.append(dict(cell=asdict(cell),status='skipped',reason='soft budget'));save();continue
        dest.mkdir(parents=True,exist_ok=True)
        cfg=B.render(cell,args.model.resolve(),args.data.resolve()/receipt['sources'][cell.stage]['file'],dest)
        (dest/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        B.write_json(dest/'attempt.json',dict(cell=asdict(cell),started_unix=time.time()))
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
             '--nproc-per-node=8','--max-restarts=0','-m',
             ('experiments.dispatch.gemma27b_h200_speed_v1.train_entry'
              if cell.name.endswith('decoder_ac') else 'experiments.dispatch.glm_b200_speed_v1.train_entry'),
             str(dest/'config.yaml')]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',NCCL_NVLS_ENABLE='0',TOKENIZERS_PARALLELISM='false')
        reason=None
        print('START',cell.name,flush=True)
        with (dest/'train.log').open('w') as log:
            active=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            end=min(time.time()+cell.max_seconds,deadline-20)
            try:
                while active.poll() is None:
                    text=(dest/'train.log').read_text(errors='replace').lower()
                    if 'cuda out of memory' in text or 'outofmemoryerror' in text:
                        reason='gpu_oom';break
                    if time.time() >= end:
                        reason='timeout';break
                    time.sleep(2)
            finally:
                stop(active)
            rc=active.returncode;active=None
        result=B.summarize(cell,dest,rc)
        if reason:
            result.update(status='invalid',reason=reason)
        baseline=next((r for r in records if r['cell']['name']==('mid_baseline' if cell.stage=='midtrain' else 'dolci_baseline')),None)
        if baseline and result['status']=='valid':
            result['comparison']=B.compare(result,baseline)
            if baseline['status']=='valid':
                result['gradient_comparison']=gradient_comparison(dest,root/baseline['cell']['name'])
        B.write_json(dest/'result.json',result); records.append(result);save()
        print('END',cell.name,result['status'],flush=True)
        if result['status']!='valid' and cell.name.endswith('baseline'):
            # A broken baseline is an operational failure, not license for a
            # blind sweep. Preserve the deadline and diagnose before resuming.
            raise SystemExit('Baseline failed; repair before further trials')
    save()
    B.write_json(root/'BENCH_COMPLETE.json',dict(at=time.time(),elapsed_seconds=time.time()-budget['started_unix'],
        valid=sum(r['status']=='valid' for r in records),full_training_launched=False))

if __name__=='__main__':
    main()
