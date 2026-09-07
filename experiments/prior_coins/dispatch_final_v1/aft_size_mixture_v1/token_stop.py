"""Explicitly authorized migration: stop old cells, never pods or agreement training."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def proc(pid):
    try:
        p = Path('/proc') / str(pid)
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': int(pid), 'start': stat[19], 'ppid': int(stat[1]),
                'argv': (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(p):
    q = proc(p['pid'])
    return q is not None and q['start'] == p['start']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['charter', 'coin', 'control'])
    ap.add_argument('--shard', required=True, choices=['A1', 'A2', 'A3'])
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    root = Path('/workspace/aft-size-mixture-v1') / a.arm
    receipt = root / 'TOKEN_MIGRATION_STOP.json'
    if receipt.exists():
        raise RuntimeError('Migration already recorded; inspect before any retry')
    ps = [p for x in Path('/proc').iterdir() if x.name.isdigit() and (p := proc(int(x.name)))]
    def is_runner(p):
        argv = p['argv']
        return (any(x.endswith('/aft_size_mixture_v1/shard_run.py') for x in argv)
                and '--arm' in argv and argv[argv.index('--arm')+1] == a.arm
                and '--shard' in argv and argv[argv.index('--shard')+1] == a.shard)
    runners = [p for p in ps if is_runner(p)]
    assert len(runners) == 1, runners
    runner = runners[0]
    if a.shard == 'A1':
        handoff = json.loads((root / 'HANDOFF.json').read_text())
        child = handoff['child']
        assert handoff['status'] == 'waiting_for_training' and same(child)
        assert not [p for p in ps if p['ppid'] == runner['pid']]
        targets = [runner]
    else:
        cell = 'charter_2pct' if a.shard == 'A2' else 'coin_2pct'
        children = [p for p in ps if p['ppid'] == runner['pid']]
        assert len(children) == 1
        child = children[0]
        argv = child['argv']
        assert '--train-cell' in argv and argv[argv.index('--train-cell')+1] == cell
        assert '--root' in argv and argv[argv.index('--root')+1] == str(root / cell)
        targets = [runner]
        while True:
            ids = {p['pid'] for p in targets}
            extra = [p for p in ps if p['ppid'] in ids and p['pid'] not in ids]
            if not extra:
                break
            targets.extend(extra)
    info = {'arm': a.arm, 'shard': a.shard, 'at': time.time(),
            'reason': 'User approved replacing row-dose cells with 0.5/1.5/5 percent token doses',
            'agreement_child_preserved': child if a.shard == 'A1' else None,
            'targets': targets, 'status': 'planned'}
    print(json.dumps(info), flush=True)
    if not a.execute:
        return
    assert same(runner)
    os.kill(runner['pid'], signal.SIGSTOP)
    info['status'] = 'stopping'
    receipt.write_text(json.dumps(info, indent=2) + '\n')
    for p in reversed(targets):
        if same(p):
            os.kill(p['pid'], signal.SIGTERM)
    if same(runner):
        os.kill(runner['pid'], signal.SIGCONT)
    deadline = time.monotonic() + 20
    while any(same(p) for p in targets) and time.monotonic() < deadline:
        time.sleep(.25)
    for p in targets:
        if same(p):
            os.kill(p['pid'], signal.SIGKILL)
    time.sleep(1)
    assert not any(same(p) for p in targets)
    if a.shard == 'A1':
        assert same(child), 'Agreement driver unexpectedly exited; inspect before proceeding'
    info.update(status='stopped', completed_at=time.time())
    receipt.write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({'status': 'stopped', 'shard': a.shard, 'arm': a.arm,
                      'processes': len(targets), 'agreement_preserved': a.shard == 'A1'}), flush=True)


if __name__ == '__main__':
    main()
