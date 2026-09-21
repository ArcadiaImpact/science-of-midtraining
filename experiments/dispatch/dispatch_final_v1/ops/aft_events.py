"""Read-only event detector over the existing dashboard; no additional SSH loop."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def detect(data, state, now):
    events = []
    workers = state.setdefault('workers', {})
    stale = data.get('collector_error') or (data.get('age_seconds') or 0) > 180
    if stale:
        if not state.get('stale'):
            events.append('Dashboard collection stale/unavailable; inspect collector and fleet')
        state['stale'] = True
        return events
    state['stale'] = False
    rows = {h['label']: h for h in data.get('handruns', [])
            if h['label'].startswith(('glm-aft81920/', 'glm-aft8192/', 'gemma-grid/'))}
    for label in set(workers) - set(rows):
        if not workers[label].get('missing'):
            events.append(label + ': disappeared from dashboard catalog; reconcile ownership/status')
        workers[label]['missing'] = True
    for label, h in rows.items():
        p = h.get('probe') or {}
        old = workers.setdefault(label, {})
        old.pop('missing', None)
        status_text = str(p.get('stage', '')) + ' ' + str(p.get('status_line', ''))
        bad = (not p.get('ok') or not h.get('running') or bool(h.get('account_error'))
               or any(word in status_text.lower() for word in ('failed', 'fatal', 'error', 'halted')))
        old['bad_count'] = old.get('bad_count', 0) + 1 if bad else 0
        if old['bad_count'] == 2:
            events.append(label + ': repeated failed probe/not-running; verify over SSH')
        if bad:
            continue
        stage = [p.get('stage'), p.get('stage_index'), p.get('cell_index')]
        endpoints = json.dumps(p.get('endpoints') or [], sort_keys=True)
        progress = [stage, p.get('step'), p.get('cells_done'), endpoints]
        # Setup probes serialize an unknown count as explicit JSON null.
        # dict.get(default) does not replace null; comparing None > None
        # previously killed the watcher when a fresh pod entered the catalog.
        done = int(p.get('cells_done') or 0)
        previous_done = old.get('cells_done')
        previous_done = done if previous_done is None else int(previous_done)
        if 'stage' in old and old['stage'] != stage:
            events.append(label + ': stage changed to ' + str(p.get('stage')))
        if done > previous_done:
            events.append(label + ': completed cells=' + str(done) + '; verify HF artifacts')
        # Only the transition to a completed stage, never ordinary step increments.
        complete = bool(p.get('total')) and p.get('step') == p.get('total')
        if complete and not old.get('complete', complete):
            events.append(label + ': stage reached total; inspect publication/handoff')
        epoch1 = ('eval' in str(p.get('stage', '')).lower()
                  and p.get('total') == 38 and (p.get('step') or 0) >= 19)
        if epoch1 and not old.get('epoch1', epoch1):
            events.append(label + ': first evaluation endpoint finished; verify HF bundle')
        if old.get('progress') != progress:
            old['last_progress'] = now
            old['stall_alerted'] = False
        elif now - old.get('last_progress', now) > 1200 and not old.get('stall_alerted'):
            events.append(label + ': no observed progress for20min; diagnose, do not blindly restart')
            old['stall_alerted'] = True
        old.update(stage=stage, progress=progress, complete=complete, epoch1=epoch1,
                   cells_done=done)
    return events


def deliver(thread, directory, events):
    event_id = str(time.time_ns())
    receipt = {'id': event_id, 'events': events, 'delivery': 'pending'}
    # Persist first: an ambiguous timeout must not duplicate a delivery.
    write(directory / 'pending.json', receipt)
    message = (f'User-authorized AFT event notification {event_id}: ' + '; '.join(events)
               + '. Dashboard evidence is a trigger, not proof of health. Inspect affected workers '
               'over fresh SSH; verify completed artifacts on HF; repair only within the approved '
               'recipe and RunPod skill. At the END use apply_patch to set '
               + str(directory / 'ack.txt') + ' to exactly ' + event_id + ' on one line. '
               'Keep the existing15m heartbeat. Do not start continuous assistant polling or another monitor.')
    env = {k: v for k, v in os.environ.items() if k not in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY')}
    try:
        result = subprocess.run(['codex', 'queue', '--thread', thread, '--message', message],
                                capture_output=True, text=True, timeout=55, env=env)
        receipt.update(delivery='queued' if result.returncode == 0 else 'failed',
                       output=result.stdout[-1000:], error=result.stderr[-1000:])
    except subprocess.TimeoutExpired:
        receipt['delivery'] = 'timeout_ambiguous'
    except OSError as exc:
        receipt.update(delivery='failed', error=str(exc))
    write(directory / 'pending.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thread', required=True)
    ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--test-notification', action='store_true')
    args = ap.parse_args()
    directory = args.state_dir
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = directory / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    if args.test_notification:
        state.setdefault('outbox', []).append('Setup delivery test only; acknowledge receipt, no repair or fleet recheck needed')
    while not (directory / 'STOP').exists():
        try:
            with urlopen('http://127.0.0.1:8377/data.json', timeout=10) as response:
                data = json.load(response)
        except Exception as exc:
            data = {'collector_error': str(exc)}
        # A malformed/new probe must not silently remove event monitoring.
        # Keep the prior detector state intact and emit one diagnostic alert.
        candidate = copy.deepcopy(state)
        try:
            events = detect(data, candidate, time.time())
            state = candidate
            state.pop('detector_error', None)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            print(json.dumps({'event': 'detector_error', 'error': error}), flush=True)
            events = [] if state.get('detector_error') == error else [
                'Event watcher detector error; monitoring degraded: ' + error]
            state['detector_error'] = error
        state['outbox'] = list(dict.fromkeys(state.get('outbox', []) + events))
        state['checked_at'] = time.time()
        write(path, state)
        pending = directory / 'pending.json'
        ack = directory / 'ack.txt'
        ready = not pending.exists() or (ack.exists() and ack.read_text().strip() == json.loads(pending.read_text())['id'])
        if ready and state['outbox']:
            events = state['outbox']
            deliver(args.thread, directory, events)
            state['outbox'] = []
            write(path, state)
        if args.once:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
