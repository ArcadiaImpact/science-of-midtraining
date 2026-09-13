"""Jonathan's cell-selection wrapper over the deployed 0.5% plan (handoff, 2026-09-08).

Runs ONLY the cells named on the command line, on the unchanged 18-worker plan
and the unchanged runner -- `gemma_halfpct_sharded.validate` for the plan,
`gemma_grid_run.cell` for the work -- with the three additions the handoff
(JONATHAN_GEMMA_HALFPCT_HANDOFF.md) asks for:

* selection: a worker's queue is not run whole; `--job` names the cells;
* `--attempt-suffix`: the five 12B cells whose first attempt was interrupted
  weight-only are published under `followups/<version><suffix>/<cell>`, a
  distinct namespace; the old attempt is never touched or resumed;
* `--resume-eval`: a cell whose training finished and whose epoch-2 evaluation
  was cut off is restored -- pinned parent and the two evaluated checkpoints
  are re-downloaded from their verified receipts, the processor view is
  rebuilt, every restored response file is validated against its prompt file
  (invalid ones are quarantined, not trusted) -- and `cell()` generates only
  what is missing, then scores and publishes.

The five 27B charter-arm cells still queued on Sid's workers are refused unless
`--ownership-confirmed` is given.  This file is NOT part of the plan's frozen
`source_hashes`; it changes no scientific setting and no namespace guard.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from experiments.prior_coins.dispatch_final_v1 import gemma_grid_plan as G
from experiments.prior_coins.dispatch_final_v1 import gemma_grid_publish as publish
from experiments.prior_coins.dispatch_final_v1 import gemma_grid_run as core
from experiments.prior_coins.dispatch_final_v1 import gemma_halfpct as H
from experiments.prior_coins.dispatch_final_v1 import gemma_halfpct_sharded as S

MODULE = 'experiments.prior_coins.dispatch_final_v1.gemma_halfpct_handoff'

#: Second cells of Sid's five 27B charter-arm workers (handoff "Ownership warning").
SID_QUEUED = frozenset({
    'gemma3_27b_5m/charter/charter_0p5pct', 'gemma3_27b_19m/charter/charter_0p5pct',
    'gemma3_27b_50m/charter/charter_0p5pct', 'gemma3_27b_190m/charter/charter_0p5pct',
    'gemma3_27b_5m/control/charter_0p5pct'})
#: Interrupted weight-only first attempts: full restart, distinct namespace.
INTERRUPTED_12B = frozenset({
    'gemma3_12b_1m/charter/charter_0p5pct', 'gemma3_12b_5m/charter/charter_0p5pct',
    'gemma3_12b_19m/charter/charter_0p5pct', 'gemma3_12b_50m_4ep/charter/charter_0p5pct',
    'gemma3_12b_5m/control/charter_0p5pct'})
#: Training complete, step-256 evaluated, step-512 evaluation cut off.
EVAL_ONLY = frozenset({
    'gemma3_27b_5m/coin/coin_0p5pct', 'gemma3_27b_19m/coin/coin_0p5pct',
    'gemma3_27b_50m/coin/coin_0p5pct', 'gemma3_27b_190m/coin/coin_0p5pct'})


def install_sharded_identity():
    """Exactly what `gemma_halfpct_sharded.__main__` and `gemma_halfpct.main` do."""
    H.validate = S.validate
    H.MODULE = S.MODULE
    core.MODULE = S.MODULE
    original_child = core.run_child

    def child(cmd, *args, **kwargs):
        if S.MODULE in cmd:
            cmd = [*cmd, '--approved-launch']
        return original_child(cmd, *args, **kwargs)
    core.run_child = child


def sender_completed(repo, cell_id):
    """Re-check a released cell right before running it: a paused sender worker could
    still have landed it (Jonathan: 'if they don't show up by the time we get to them')."""
    from huggingface_hub import HfApi
    return HfApi().file_exists(repo, f'followups/{H.VERSION}/{cell_id}/COMPLETE.json')


def install_attempt_namespace(suffix):
    """Publish cells under followups/<version><suffix>/...; shared-data stays put."""
    base = f'followups/{H.VERSION}/'
    Original = core.Publisher

    class AttemptPublisher(Original):
        def __init__(self, repo, prefix, receipts, api=None):
            if prefix.startswith(base) and not prefix.startswith(base + 'shared-data'):
                prefix = f'followups/{H.VERSION}{suffix}/' + prefix[len(base):]
            super().__init__(repo, prefix, receipts, api)
    core.Publisher = AttemptPublisher
    return AttemptPublisher


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


def restore_for_eval(a, plan, job, dest):
    """Rebuild what the recovery archive deliberately excluded, verify the rest."""
    inputs = json.loads((dest / 'inputs.json').read_text())
    assert (dest / 'TRAIN_COMPLETE.json').exists(), 'resume-eval needs completed training'
    assert not (dest / 'EVAL_COMPLETE.json').exists(), 'evaluation already complete'
    parent = ensure_parent_view(a, plan, job, dest, force=True)
    from huggingface_hub import hf_hub_download
    restored = {}
    for step in plan['recipe']['eval_steps']:
        receipt = json.loads((dest / 'receipts' / f'checkpoint-{step}.json').read_text())
        for name, record in receipt['files'].items():
            target = dest / name
            if target.is_file() and publish.file_record(target) == record:
                continue
            got = hf_hub_download(receipt['repo'], f"{receipt['prefix']}/{name}",
                                  revision=receipt['commit'], local_dir=a.root / 'restore-cache')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(got, target)
            if publish.file_record(target) != record:
                raise RuntimeError(f'restored checkpoint file differs from receipt: {name}')
        restored[str(step)] = sorted(receipt['files'])
    for path, digest in inputs['eval_hashes'].items():
        if not Path(path).is_file() or G.sha(Path(path)) != digest:
            raise RuntimeError(f'evaluation source missing or changed: {path}')
    expected = {**inputs['prompts'], 'sanity': inputs['sanity']}
    responses = {}
    for step in plan['recipe']['eval_steps']:
        endpoint = dest / f'eval/aft-step{step}'
        valid, quarantined = [], []
        for key, prompt in expected.items():
            path = endpoint / f'{key}.jsonl'
            if not path.exists():
                continue
            try:
                core.validate_responses(path, prompt)
                valid.append(key)
            except RuntimeError:
                keep = dest / 'restore-quarantine' / f'aft-step{step}'
                keep.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), keep / path.name)
                quarantined.append(key)
        responses[str(step)] = dict(valid=valid, quarantined=quarantined,
            missing=sorted(set(expected) - set(valid) - set(quarantined)))
    scratch = dest / 'runtime' / 'eval'
    if scratch.exists():
        shutil.rmtree(scratch)
    G.write(dest / 'RESTORE.json', dict(module=MODULE, restored_at=time.time(),
        parent=str(parent), checkpoints=restored, responses=responses,
        note='archive excluded published checkpoints and pinned parents; both re-fetched '
             'from verified receipts. Response files failing prompt validation were quarantined.'))
    print(json.dumps(dict(restore=job['id'], responses=responses)), flush=True)
    return responses


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


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('action', choices=('run',))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', required=True)
    p.add_argument('--publish-repo', required=True)
    p.add_argument('--job', action='append', required=True, help='cell id PROFILE/ARM/MIX; repeatable')
    p.add_argument('--attempt-suffix', default=None,
                   help='publish under followups/<version><suffix>/ (interrupted 12B cells only)')
    p.add_argument('--resume-eval', action='store_true',
                   help='restore an archived cell whose epoch-2 evaluation was cut off')
    p.add_argument('--ownership-confirmed', action='store_true',
                   help="required for the five cells still on Sid's workers' queues")
    p.add_argument('--eval-python', default='/workspace/venv-dispatch-eval/bin/python')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--approved-launch', action='store_true')
    a = p.parse_args()
    if a.execute and not a.approved_launch:
        p.error('Held release: explicit new user launch approval required')
    install_sharded_identity()
    install_publish_retry()
    a.plan, a.data, a.root = a.plan.resolve(), a.data.resolve(), a.root.resolve()
    plan = json.loads(a.plan.read_text())
    H.validate(plan, a.data)
    assert G.sha(a.plan) == json.loads((a.plan.parent / 'READY.json').read_text())['plan_sha256']
    worker = plan['workers'][a.worker]
    selected = list(dict.fromkeys(a.job))
    jobs = [j for j in worker['jobs'] if j['id'] in selected]
    unknown = set(selected) - {j['id'] for j in jobs}
    if unknown:
        p.error(f'not in worker {a.worker}: {sorted(unknown)}')
    for job in jobs:
        cell_id = job['id']
        if cell_id in SID_QUEUED and not a.ownership_confirmed:
            p.error(f"{cell_id} is still on Sid's queue; confirm release, then --ownership-confirmed")
        if cell_id in SID_QUEUED and not a.attempt_suffix:
            # Sid's paused workers already claimed the canonical prefixes (inputs published,
            # 0-3 steps). Released 2026-09-08 (Sid via Slack, relayed by Jonathan): we run
            # them under an explicit attempt namespace and Sid consolidates afterwards.
            p.error(f'{cell_id}: canonical prefix is held by the sender; needs --attempt-suffix')
        if cell_id in INTERRUPTED_12B and not a.attempt_suffix:
            p.error(f'{cell_id} had an interrupted attempt; needs --attempt-suffix')
        if a.attempt_suffix and cell_id not in INTERRUPTED_12B | SID_QUEUED:
            p.error(f'{cell_id} belongs in the canonical namespace; drop --attempt-suffix')
        if cell_id in EVAL_ONLY and not a.resume_eval:
            p.error(f'{cell_id} is evaluation-only; needs --resume-eval')
    expected_repo = f"arcadia-impact/scimt-dispatch-gemma-{worker['model']}-aft-grid-v2"
    if a.publish_repo != expected_repo:
        p.error(f'publish repo must be {expected_repo}')
    print(json.dumps(dict(action='run', worker=a.worker, jobs=[j['id'] for j in jobs],
                          attempt_suffix=a.attempt_suffix, resume_eval=a.resume_eval,
                          execute=a.execute)), flush=True)
    if not a.execute:
        return
    if a.attempt_suffix:
        install_attempt_namespace(a.attempt_suffix)
    a.root.mkdir(parents=True, exist_ok=True)
    lock = (a.root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (a.root / 'TRANSFERRED_OUT.json').exists()
    G.bind(a.root / 'WORKER.json', dict(worker=a.worker, plan=plan, publish_repo=a.publish_repo))
    data_pub = core.Publisher(a.publish_repo, f'followups/{H.VERSION}/shared-data', a.root / 'data-receipts')
    assert data_pub.prefix == f'followups/{H.VERSION}/shared-data', 'shared data must not be relocated'
    data_pub.verify_receipts()
    receipt = json.loads((a.root / 'data-receipts/shared-data.json').read_text())
    if receipt['files'] != {q.name: publish.file_record(q) for q in a.data.glob('*.json*')}:
        raise RuntimeError('Published shared-data receipt does not match this local dataset bundle')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(gpu) != 1 or worker['gpu'].split()[0] not in gpu[0] or int(gpu[0].split(',')[-1]) > 2048:
        raise RuntimeError(f"Expected one idle {worker['gpu']} GPU, got {gpu}")
    G.write(a.root / 'HANDOFF.json', dict(module=MODULE, worker=a.worker,
        jobs=[j['id'] for j in jobs], attempt_suffix=a.attempt_suffix, resume_eval=a.resume_eval,
        ownership_confirmed=a.ownership_confirmed, started=time.time(),
        note='Jonathan handoff 2026-09-08: selected cells only; plan, recipe and runner unchanged.'))
    for job in jobs:
        index = worker['jobs'].index(job)
        if G.sha(a.data / f"aft_{job['mix']}.jsonl") != job['data_sha256']:
            raise RuntimeError('Mixture hash changed')
        dest = a.root / 'cells' / job['id']
        if job['id'] in SID_QUEUED and sender_completed(a.publish_repo, job['id']):
            print(json.dumps(dict(cell_skipped=job['id'],
                                  reason='sender completed it in the canonical namespace')), flush=True)
            continue
        if a.resume_eval and job['id'] in EVAL_ONLY and not (dest / 'COMPLETE.json').exists():
            restore_for_eval(a, plan, job, dest)
        elif needs_parent_view(dest):
            # Restored from an archive without parents/ and runtime/views/ (2026-09-09 HF quota
            # block): core.cell skips prepare and train, then evaluates against inputs['parent'].
            ensure_parent_view(a, plan, job, dest)
        cell_pub = core.Publisher(a.publish_repo, f"followups/{H.VERSION}/{job['id']}", dest / 'receipts')
        H.guard_namespace(cell_pub)
        core.cell(a, plan, worker, job, index)
        print(json.dumps(dict(cell_complete=job['id'], prefix=cell_pub.prefix)), flush=True)
    G.write(a.root / 'HANDOFF_COMPLETE.json', dict(worker=a.worker, jobs=[j['id'] for j in jobs],
                                                   finished=time.time()))
    print('HANDOFF QUEUE COMPLETE: artifacts verified; lifecycle review required', flush=True)


if __name__ == '__main__':
    main()
