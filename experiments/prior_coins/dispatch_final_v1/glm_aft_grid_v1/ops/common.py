"""Shared box-side helpers: RunPod GraphQL (single account), ssh, receipts, spend checks."""
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib
import urllib.request

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import config as C

PREPARED = C.REPO / 'artifacts/glm_aft_grid_8192_v1'
DEPLOY = PREPARED / 'deployment'
RUNPODCTL = Path.home() / '.local/bin/runpodctl'
POD_OWN = Path.home() / '.claude/skills/runpod-spinup/pod-own.sh'
SSH_KEY = Path.home() / '.runpod/ssh/runpodctl-ssh-key'
PROGRESS_GLOB = ('/workspace/glm-grid-worker.log /workspace/glm-grid-phase.json /workspace/setup.log '
                 f'{C.POD_ROOT}/*/STATUS.json {C.POD_ROOT}/*/cells/*/*/*/*.log '
                 f'{C.POD_ROOT}/*/cells/*/*/*/train-progress.json {C.POD_ROOT}/*/cells/*/*/*/eval/*/*.jsonl')
SSH_OPTS = ['-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'LogLevel=ERROR',
            '-o', 'ConnectTimeout=20', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30', '-i', str(SSH_KEY)]


def clean_env():
    env = dict(os.environ)
    env.pop('RUNPOD_API_KEY', None)  # the pod-injected key 403s; the config.toml key is the valid one
    env.pop('SSH_AUTH_SOCK', None)
    return env


def api_key():
    return tomllib.loads((Path.home() / '.runpod/config.toml').read_text())['apikey']


def graphql(body, timeout=60):
    req = urllib.request.Request('https://api.runpod.io/graphql', data=json.dumps({'query': body}).encode(),
                                 headers={'Authorization': f'Bearer {api_key()}', 'Content-Type': 'application/json',
                                          'User-Agent': 'scimt-glm-grid-launcher/1.0 (python-urllib)'})
    for attempt in range(4):
        try:
            data = json.load(urllib.request.urlopen(req, timeout=timeout))
            break
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * 2 ** attempt)
                continue
            raise
    if data.get('errors'):
        raise RuntimeError(f'RunPod error: {data["errors"]}')
    return data['data']


def myself():
    return graphql('query { myself { id clientBalance spendLimit currentSpendPerHr '
                   'pods { id name costPerHr desiredStatus gpuCount } } }')['myself']


def spend_check(me, hourly=C.POD['hourly_usd']):
    running = [p for p in me['pods'] if p['desiredStatus'] == 'RUNNING']
    spend = max(float(me['currentSpendPerHr']), sum(float(p['costPerHr']) for p in running))
    glm = [p for p in running if p['name'].startswith('glm-8192-grid-')]
    cap = min(C.BUDGET['account_hourly_cap'], float(me['spendLimit']))
    report = dict(current_spend_per_hr=round(spend, 3), cap=cap, after_create=round(spend + hourly, 3),
                  running_pods=[(p['id'], p['name'], p['costPerHr']) for p in running], glm_grid_pods=len(glm),
                  balance=me['clientBalance'], runway_hours_after_create=round(float(me['clientBalance']) / (spend + hourly), 2))
    if len(glm) >= C.BUDGET['max_glm_pods']:
        raise RuntimeError(f'Already {len(glm)} GLM grid pods running; never a fourth: {report}')
    if spend + hourly > cap:
        raise RuntimeError(f'Spend cap: {spend:.2f} + {hourly} > {cap}: {report}')
    return report


def pod_get(pod_id):
    out = subprocess.run([str(RUNPODCTL), 'pod', 'get', pod_id, '-o', 'json'], env=clean_env(),
                         capture_output=True, text=True, timeout=60)
    if out.returncode:
        return None
    try:
        return json.loads(out.stdout[out.stdout.index('{'):])
    except (ValueError, json.JSONDecodeError):
        return None


def ssh_cmd(pod, command, timeout=120, check=False, stdin=None):
    argv = ['env', '-u', 'SSH_AUTH_SOCK', 'ssh', *(['-n'] if stdin is None else []), *SSH_OPTS, '-p', str(pod['port']),
            f"root@{pod['ip']}", command]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=check, input=stdin)


def scp_to(pod, local, remote, timeout=1800):
    return subprocess.run(['env', '-u', 'SSH_AUTH_SOCK', 'scp', *SSH_OPTS, '-P', str(pod['port']), str(local),
                           f"root@{pod['ip']}:{remote}"], check=True, timeout=timeout)


def receipt_path(worker):
    return DEPLOY / f'{worker}.json'


def load_receipt(worker):
    path = receipt_path(worker)
    if not path.exists():
        raise FileNotFoundError(f'No pod receipt for {worker}: {path}')
    return json.loads(path.read_text())


def receipts():
    return {p.stem: json.loads(p.read_text()) for p in sorted(DEPLOY.glob('glm-grid-*.json'))
            if not p.stem.endswith(('.pending', '.launched'))}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    tmp.replace(path)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
