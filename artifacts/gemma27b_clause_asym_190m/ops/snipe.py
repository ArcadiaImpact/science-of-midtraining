"""One bounded, duplicate-safe Secure H200 allocation for the authorized study."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
NAME = "gemma27b-clause-asym-190m-20260914"
SKILL = Path('/root/.codex/skills/runpod-spinup')
os.environ['PATH'] = '/workspace/scimt-dispatch-final/artifacts/aft_size_mixture_v1/ops/bin:' + os.environ['PATH']

def log(s):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), s, flush=True)

def cli(*args):
    return json.loads(subprocess.check_output(['runpodctl', *args], text=True, timeout=45))

def landed(p):
    receipt = {'pod_id': p['id'], 'name': NAME, 'account': 'default',
               'created_for': 'gemma3_27b_190m_clause_asym', 'pod': p,
               'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (HERE / 'LANDED.json').write_text(json.dumps(receipt, indent=2) + '\n')
    log('LANDED ' + p['id'])

lock = open(HERE / 'snipe.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (HERE / 'LANDED.json').exists():
    raise SystemExit('Already allocated; refusing another pod')
for attempt in range(1, 1441):
    try:
        pods = cli('pod', 'list')
        matches = [p for p in pods if p.get('name') == NAME]
        if matches:
            if len(matches) != 1:
                raise RuntimeError('Multiple matching pods; manual reconciliation required')
            landed(matches[0])
            break
        acct = cli('me')
        if acct['clientBalance'] < 1000 or acct['currentSpendPerHr'] + 38 > 80:
            log(f'HOLD attempt={attempt}: balance or hourly cap')
        else:
            log(f'CREATE attempt={attempt}: 8xH200 Secure, 1000GB disk')
            # Use the skill entry point; never retry blindly after an ambiguous
            # create. Always reconcile the live pod list on the next iteration.
            result = subprocess.run([str(SKILL / 'create-pod.sh'), NAME,
                'NVIDIA H200', 'SECURE', 'runpod-torch-v280', '8', '1000'],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=720)
            log(result.stdout)
            matches = [p for p in cli('pod', 'list') if p.get('name') == NAME]
            if len(matches) == 1:
                landed(matches[0])
                break
            if matches or result.returncode == 0:
                raise RuntimeError('Ambiguous allocation; refusing retry')
            if not any(s in result.stdout.lower() for s in
                       ('no longer any instances', 'supply_constraint', 'no instances available',
                        'does not have the resources')):
                raise RuntimeError('Unexpected create failure; refusing automatic retry')
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        log(f'API failure, reconcile before any next create: {type(exc).__name__}')
    time.sleep(60)
else:
    raise SystemExit('Snipe expired after 24 hours; no further creates')
