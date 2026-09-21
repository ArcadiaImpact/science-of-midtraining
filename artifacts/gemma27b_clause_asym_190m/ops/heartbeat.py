"""Ten-minute read-only status events for this active Codex goal.

The agent consumes these events, investigates failures, and owns final verified
cleanup. This program never launches, restarts, or terminates GPU workloads.
"""
import datetime,json,subprocess,time,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent
lock=open(HERE/'heartbeat.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
remote = '''import pathlib,json,time,os,re
r=pathlib.Path('/workspace/gemma27b-full');a=r/'runs/gemma3_27b_190m_clause_asym/charter'
pid=int((r/'chain.pid').read_text());exitfile=r/'chain.exit'
d=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),chain_pid=pid,chain_live=pathlib.Path(f'/proc/{pid}').exists(),exit=exitfile.read_text().strip() if exitfile.exists() else None,markers=[str(p.relative_to(a)) for p in a.rglob('*COMPLETE.json')],published=[p.name for p in a.glob('PUBLISHED_*.json')],logs={})
for p in [r/'chain.log',*a.rglob('train.log'),*a.glob('publish*.log')]:
 lines=p.read_text(errors='replace').splitlines();d['logs'][str(p.relative_to(r))]=lines[-6:]
print(json.dumps(d))'''
while not (HERE/'HEARTBEAT_STOP').exists():
    started=time.time()
    try:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',
            'runpod-gemma27b-clause-asym-190m-20260914',
            '/workspace/gemma27b-speed/venv/bin/python -'],input=remote,text=True,capture_output=True,timeout=55)
        if r.returncode:raise RuntimeError(r.stderr[-1000:])
        d=json.loads(r.stdout)
    except (subprocess.SubprocessError,ValueError,RuntimeError) as e:
        d={'observation_error':str(e),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    d['next_check_unix']=started+600
    line=json.dumps(d)
    with (HERE/'heartbeat.jsonl').open('a') as f:f.write(line+'\n')
    (HERE/'HEARTBEAT_LATEST.json').write_text(json.dumps(d,indent=2)+'\n')
    print(line,flush=True)
    if d.get('exit')=='0':break
    time.sleep(max(1,started+600-time.time()))
