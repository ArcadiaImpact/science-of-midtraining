"""Pod-side cell-selection wrapper over the low-dose (0.25%) plan.

Simplified copy of gemma_halfpct_handoff.py: runs ONLY the cells named on the
command line (`--job`, repeatable) of one worker of gemma_lowdose's 18-worker plan,
with `gemma_lowdose.validate` for the plan and the unchanged `gemma_grid_run.cell`
for the work.  Everything publishes to the canonical namespace
`followups/<VERSION>/<profile>/<arm>/<mix>`; there is no attempt-suffix, resume-eval
or sender-ownership logic (nothing has run under this version before).  The one
addition (2026-09-09) is `ensure_parent_view`: a cell restored from an archive that
excluded parents/ and runtime/views/ gets its pinned parent re-fetched and its eval
view rebuilt before `cell()` publishes the local checkpoints and evaluates.

Shared data is never published from a pod: `<root>/data-receipts/shared-data.json`
must have been shipped in the bundle (published once from the control box with
`python -m gemma_lowdose publish-data`), and is verified against HF and the local
data before any GPU work.  This file is NOT part of the plan's frozen
`source_hashes` (gemma_lowdose.py is pinned by `deployment_source_sha256`); it
changes no scientific setting and no namespace guard.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

from experiments.prior_coins.dispatch_final_v1 import gemma_grid_plan as G
from experiments.prior_coins.dispatch_final_v1 import gemma_grid_run as core
from experiments.prior_coins.dispatch_final_v1 import gemma_lowdose as L

MODULE = 'experiments.prior_coins.dispatch_final_v1.gemma_lowdose_handoff'


def install_publish_retry(attempts=32, base_delay=15, max_delay=300):  # ~2.5 h horizon: rides out an HF storage-quota block while training continues
    """Retry HF publishes with exponential backoff. 2026-09-08 23:36-23:47Z: two pods died on
    HfHubHTTPError (403 from .../info/lfs/objects/batch) while nine pods were uploading
    checkpoints; the pinned runner has no retry and an interrupted attempt cannot resume.
    publish() verifies against receipts and never re-uploads identical files, so retrying is safe."""
    original = core.Publisher.publish
    if getattr(original, '_retrying', False):
        return
    def publish(self, *args, **kwargs):
        for attempt in range(1, attempts + 1):
            try:
                return original(self, *args, **kwargs)
            except Exception as error:  # HfHubHTTPError (incl. 403 quota), httpx/requests/socket errors
                if attempt == attempts or isinstance(error, (FileNotFoundError, PermissionError, IsADirectoryError,
                                                             KeyError, TypeError, ValueError, AssertionError, RuntimeError)):
                    raise  # programming/data errors are not retried
                delay = min(base_delay * 2 ** (attempt - 1), max_delay)
                print(json.dumps(dict(publish_retry=attempt, prefix=getattr(self, 'prefix', '?'),
                                      error=f'{type(error).__name__}: {str(error)[:160]}', sleep=delay)), flush=True)
                time.sleep(delay)
    publish._retrying = True
    core.Publisher.publish = publish


def _critical_parent_file(name):
    """Parent files that decide what model is evaluated: weights, weight index, tokenizer, config."""
    return (name.endswith('.safetensors') or name.endswith('.index.json')
            or name.startswith('tokenizer') or name == 'config.json')


def compare_parent(recorded, fetched):
    """Per-file (size, sha256) comparison of the archived parent.json with a re-fetched parent."""
    old, new = recorded['files'], fetched['files']

    def same(name):
        return old[name]['size'] == new[name]['size'] and old[name]['sha256'] == new[name]['sha256']
    identical = sorted(n for n in old if n in new and same(n))
    changed = sorted(n for n in old if n in new and not same(n))
    missing = sorted(n for n in old if n not in new)
    added = sorted(n for n in new if n not in old)
    refused = sorted(n for n in changed + missing + added if _critical_parent_file(n))
    return dict(identical=identical, changed=changed, missing_at_override=missing, added=added,
                refused=refused, accepted=not refused)


def view_resolves(path):
    """The processor-compatible view exists and every entry (symlinks included) resolves."""
    path = Path(path)
    if not path.is_dir():
        return False
    entries = list(path.iterdir())
    if not {'config.json', 'tokenizer_config.json'} <= {e.name for e in entries}:
        return False
    return all(e.exists() for e in entries)  # exists() follows symlinks: a dangling link is False


def _view_tools():
    """Late imports: pod/d4_eval and pod/evaluate import the profile-scoped `contracts` at load time."""
    if str(core.EXP) not in sys.path:
        sys.path.insert(0, str(core.EXP))
    from experiments.prior_coins.dispatch_final_v1.pod.d4_eval import view
    from experiments.prior_coins.dispatch_final_v1.pod.evaluate import ensure_processor_files
    return view, ensure_processor_files


def ensure_parent_view(a, plan, job, dest, force=False):
    """Make sure the eval-time parent view recorded in inputs.json exists and resolves.

    A restored cell (TRAIN_COMPLETE.json present, evaluation pending) arrives without
    <root>/parents and <cell>/runtime/views -- the archive excluded both -- yet `core.cell`
    skips prepare (inputs.json exists) and evaluation loads inputs['parent'].  No-op when the
    view is already there and every symlink resolves (unless `force`).  Otherwise the pinned
    parent is re-fetched with the runner's own `fetch_parent`, compared with the archived
    parent.json, processor files are backfilled and the view is rebuilt at exactly the
    recorded path.  Checkpoints are never downloaded: they are local.

    PARENT_REVISION_OVERRIDE=<40-hex commit> (2026-09-09: the parent repo was squashed and the
    plan's parent_revision no longer resolves): fetch at that revision instead and accept it
    only if every weight / index / tokenizer / config file is byte-identical (size + sha256)
    to the archived parent.json; what is missing, changed or added at the new revision is
    recorded in <cell>/RESTORE_PARENT.json (published with the cell's provenance).
    Returns the local parent directory.
    """
    inputs = json.loads((dest / 'inputs.json').read_text())
    recorded = json.loads((dest / 'parent.json').read_text())
    expected = dest / 'runtime' / 'views' / 'processor-compatible'
    if str(expected) != inputs['parent']:
        raise RuntimeError(f'inputs.json parent view is not this cell\'s: {inputs["parent"]} != {expected}')
    parent = a.root / 'parents' / recorded['prefix']
    if not force and view_resolves(expected):
        return parent
    override = os.environ.get('PARENT_REVISION_OVERRIDE', '')
    if override and not re.fullmatch(r'[0-9a-f]{40}', override):
        raise RuntimeError('PARENT_REVISION_OVERRIDE must be a full 40-hex commit sha')
    fetch_plan = dict(plan, parent_revision=override) if override else plan
    try:
        fetched, provenance = core.fetch_parent(fetch_plan, job, a.root)
    except Exception as error:
        print(json.dumps(dict(parent_fetch_failed=job['id'], revision=fetch_plan['parent_revision'],
                              error=f'{type(error).__name__}: {str(error)[:200]}',
                              hint='if the pinned revision no longer exists, set PARENT_REVISION_OVERRIDE=<sha>')),
              flush=True)
        raise
    if Path(fetched) != parent:
        raise RuntimeError(f'parent landed at {fetched}; the archived attempt used {parent}')
    if override:
        comparison = compare_parent(recorded, provenance)
        G.write(dest / 'RESTORE_PARENT.json', dict(
            module=MODULE, restored_at=time.time(), repo=provenance['repo'], prefix=provenance['prefix'],
            pinned_revision=recorded['commit'], override_revision=override, **comparison,
            note='pinned parent revision no longer resolves; parent re-fetched at PARENT_REVISION_OVERRIDE '
                 'and accepted only with byte-identical weights, weight index, tokenizer and config files'))
        if not comparison['accepted']:
            raise RuntimeError(f"parent at {override} differs from the archived attempt: {comparison['refused']}")
    elif provenance != recorded:
        raise RuntimeError('pinned parent differs from the archived attempt')
    view, ensure_processor_files = _view_tools()
    ensure_processor_files(parent)
    if expected.is_symlink() or expected.exists():
        shutil.rmtree(expected)  # view() keeps existing (possibly dangling) links: rebuild from scratch
    built = view(parent, dest / 'runtime', 'processor-compatible', None)
    if str(built) != inputs['parent'] or not view_resolves(built):
        raise RuntimeError(f'view path moved: {built} != {inputs["parent"]}')
    print(json.dumps(dict(parent_view=job['id'], parent=str(parent), revision=provenance['commit'],
                          override=bool(override))), flush=True)
    return parent


def needs_parent_view(dest):
    """Restored cell: training complete, evaluation still to run (that is what loads the view)."""
    return (dest / 'TRAIN_COMPLETE.json').exists() and not (dest / 'EVAL_COMPLETE.json').exists()


def main():
    install_publish_retry()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('action', choices=('run',))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', required=True)
    p.add_argument('--publish-repo', required=True)
    p.add_argument('--job', action='append', required=True, help='cell id PROFILE/ARM/MIX; repeatable')
    p.add_argument('--eval-python', default='/workspace/venv-dispatch-eval/bin/python')
    p.add_argument('--runtime-root', type=Path,
                   help='dry runs off-pod only: tree to hash for source_hashes (default: this tree)')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--approved-launch', action='store_true')
    a = p.parse_args()
    if a.execute and not a.approved_launch:
        p.error('Held release: explicit new user launch approval required')
    if a.execute and a.runtime_root:
        p.error('--runtime-root is for off-pod dry runs; execution validates the tree it runs from')
    L.install_identity(core)
    a.plan, a.data, a.root = a.plan.resolve(), a.data.resolve(), a.root.resolve()
    plan = json.loads(a.plan.read_text())
    L.validate(plan, a.data, a.runtime_root)
    ready = json.loads((a.plan.parent / 'READY.json').read_text())
    assert G.sha(a.plan) == ready['plan_sha256'], 'plan.json does not match READY.json'
    worker = plan['workers'].get(a.worker)
    if worker is None:
        p.error(f'--worker must name a worker in the plan: {sorted(plan["workers"])}')
    selected = list(dict.fromkeys(a.job))
    jobs = [j for j in worker['jobs'] if j['id'] in selected]
    unknown = set(selected) - {j['id'] for j in jobs}
    if unknown:
        p.error(f'not in worker {a.worker}: {sorted(unknown)}')
    expected_repo = L.PUBLISH_REPOS[worker['model']]
    if a.publish_repo != expected_repo:
        p.error(f'publish repo must be {expected_repo}')
    print(json.dumps(dict(action='run', version=L.VERSION, worker=a.worker,
                          jobs=[j['id'] for j in jobs], execute=a.execute)), flush=True)
    if not a.execute:
        return
    a.root.mkdir(parents=True, exist_ok=True)
    lock = (a.root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (a.root / 'TRANSFERRED_OUT.json').exists()
    G.bind(a.root / 'WORKER.json', dict(worker=a.worker, plan=plan, publish_repo=a.publish_repo))
    L.require_shared_data_receipt(a.root, a.data, a.publish_repo)
    gpu = L.check_gpu(worker)
    G.write(a.root / 'HANDOFF.json', dict(
        module=MODULE, version=L.VERSION, worker=a.worker, jobs=[j['id'] for j in jobs],
        gpu=gpu, started=time.time(),
        note='Low-dose 0.25% column: selected cells only; plan, recipe and runner unchanged '
             'from the 0.5% campaign; canonical namespace.'))
    for job in jobs:
        index = worker['jobs'].index(job)
        if G.sha(a.data / f"aft_{job['mix']}.jsonl") != job['data_sha256']:
            raise RuntimeError('Mixture hash changed')
        dest = a.root / 'cells' / job['id']
        if needs_parent_view(dest):
            # Restored from an archive without parents/ and runtime/views/ (2026-09-09 HF quota
            # block): core.cell skips prepare and train, then evaluates against inputs['parent'].
            ensure_parent_view(a, plan, job, dest)
        cell_pub = core.Publisher(a.publish_repo, f"followups/{L.VERSION}/{job['id']}", dest / 'receipts')
        L.guard_namespace(cell_pub)
        core.cell(a, plan, worker, job, index)
        print(json.dumps(dict(cell_complete=job['id'], prefix=cell_pub.prefix)), flush=True)
    G.write(a.root / 'HANDOFF_COMPLETE.json', dict(worker=a.worker, jobs=[j['id'] for j in jobs],
                                                   finished=time.time()))
    print('HANDOFF QUEUE COMPLETE: artifacts verified; lifecycle review required', flush=True)


if __name__ == '__main__':
    main()
