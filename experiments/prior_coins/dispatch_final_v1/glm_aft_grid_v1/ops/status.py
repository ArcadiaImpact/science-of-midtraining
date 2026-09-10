"""Change-only status poller for the GLM grid pods (Monitor-friendly: one line per change/alert).

  python -m ...ops.status              # one snapshot of every launched worker
  python -m ...ops.status --loop 600   # poll every 10 min; print only changes and alerts

ALERT conditions: phase FAILED, runner process gone before QUEUE_COMPLETE, no
progress file newer than --stale-min, pod not RUNNING, ssh unreachable twice.
"""
import argparse
import json
import subprocess
import time

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.common import DEPLOY, pod_get, ssh_cmd

SCRIPT = r'''cat /workspace/glm-grid-phase.json 2>/dev/null; echo
python3 -c "import json; print(json.dumps(json.load(open('ROOT/STATUS.json'))))" 2>/dev/null; echo
echo COMPLETE=$(ls ROOT/cells/*/*/*/COMPLETE.json 2>/dev/null | wc -l) READY=$(ls ROOT/cells/*/*/*/MIRROR_READY.json 2>/dev/null | wc -l) PARKED=$(ls ROOT/cells/*/*/*/PUBLISH_PARKED.json 2>/dev/null | wc -l)
echo RUNNER=$(pgrep -fc '[g]lm_aft_grid_v1.run --worker' || true) SETUP=$(pgrep -fc '[p]od/setup.sh' || true)
python3 - <<'PY'
import glob, os, time
files = ['/workspace/glm-grid-worker.log', '/workspace/setup.log', 'ROOT/STATUS.json'] + glob.glob('ROOT/cells/*/*/*/*') + glob.glob('ROOT/cells/*/*/*/eval/*/*')
m = max([os.path.getmtime(f) for f in files if os.path.isfile(f)] + [0])
print('NEWEST=%d' % (time.time() - m if m else -1))
PY
echo GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',')
tail -n 2 /workspace/glm-grid-worker.log 2>/dev/null | cut -c1-300'''


def snapshot(worker, receipt, stale_min):
    root = f"{C.POD_ROOT}/{worker}"
    info = pod_get(receipt['pod_id']) or {}
    status = info.get('desiredStatus') or 'UNKNOWN'
    alerts = []
    if status != 'RUNNING':
        alerts.append(f'pod {status}')
    try:
        r = ssh_cmd(receipt, SCRIPT.replace('ROOT', root), timeout=90)
        out = r.stdout
    except subprocess.TimeoutExpired:
        out = ''
    if not out.strip():
        return dict(worker=worker, pod=receipt['pod_id'], status=status, reachable=False, alerts=alerts + ['ssh unreachable'])
    lines = out.splitlines()
    phase = {}
    st = {}
    try:
        phase = json.loads(lines[0]) if lines and lines[0].startswith('{') else {}
    except json.JSONDecodeError:
        pass
    for line in lines[1:]:
        if line.startswith('{"'):
            try:
                st = json.loads(line)
            except json.JSONDecodeError:
                pass
    kv = dict(kv.split('=', 1) for line in lines for kv in line.split() if '=' in kv and line[:1].isupper() and not line.startswith('{'))
    newest = int(kv.get('NEWEST', '0') or 0)
    runner = int(kv.get('RUNNER', '0') or 0)
    ph = phase.get('phase', '?')
    if ph == 'FAILED':
        alerts.append(f"FAILED: {phase.get('detail')}")
    if ph == 'RUNNING' and runner == 0:
        alerts.append('runner process gone before QUEUE_COMPLETE')
    if ph not in ('QUEUE_COMPLETE', 'FAILED') and newest > stale_min * 60:
        alerts.append(f'no progress for {newest // 60} min')
    return dict(worker=worker, pod=receipt['pod_id'], status=status, reachable=True, phase=ph,
                cell=f"{st.get('cell', '?')}/{st.get('cells_total', '?')}", job=(st.get('job') or '').split('/')[-1],
                stage=st.get('stage'), step=f"{st.get('step', '?')}/{st.get('steps_total', '?')}",
                complete=kv.get('COMPLETE'), ready=kv.get('READY'), parked=kv.get('PARKED'), gpu=kv.get('GPU', ''),
                newest_s=newest, tail=(lines[-1][:160] if lines else ''), alerts=alerts)


def line(s):
    if not s.get('reachable'):
        return f"{s['worker']} {s['pod']} {s['status']} UNREACHABLE {s['alerts']}"
    return (f"{s['worker']} {s['pod']} {s['status']} phase={s['phase']} cell={s['cell']} {s['job']} {s['stage']} "
            f"step={s['step']} complete={s['complete']} parked={s['parked']} gpu={s['gpu']} idle={s['newest_s']}s"
            + (f" ALERT {s['alerts']}" if s['alerts'] else ''))


def signature(s):
    return (s.get('reachable'), s.get('status'), s.get('phase'), s.get('cell'), s.get('stage'), s.get('complete'),
            s.get('parked'), tuple(s.get('alerts', [])),
            None if s.get('stage') != 'train' else int(str(s.get('step', '0/')).split('/')[0] or 0) // 128)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--loop', type=int, default=0, help='poll interval in seconds (0 = one snapshot)')
    ap.add_argument('--stale-min', type=int, default=90)
    ap.add_argument('--worker')
    a = ap.parse_args()
    last = {}
    while True:
        launched = {p.stem[:-len('.launched')]: json.loads(p.read_text()) for p in sorted(DEPLOY.glob('*.launched.json'))}
        if a.worker:
            launched = {k: v for k, v in launched.items() if k == a.worker}
        for worker, receipt in launched.items():
            if (DEPLOY / f'{worker}.stopped.json').exists():
                continue
            s = snapshot(worker, receipt, a.stale_min)
            sig = signature(s)
            if not a.loop or last.get(worker) != sig:
                print(time.strftime('%H:%MZ', time.gmtime()), line(s), flush=True)
                last[worker] = sig
        if not a.loop:
            return
        time.sleep(a.loop)


if __name__ == '__main__':
    main()
