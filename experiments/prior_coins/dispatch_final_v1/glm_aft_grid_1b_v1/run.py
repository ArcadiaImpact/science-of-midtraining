"""Pod-side cell queue for one GLM grid worker (wave 2: half of the 1B charter arm):
parent audit -> 4 x (train 512 steps, export 8 adapters, eval both endpoints, score,
publish small files) -> next cell.

    python -m experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.run \
        --worker glm-grid-1b-a --prepared /workspace/glm-grid-1b-prepared --execute

Dry run (no --execute) validates the prepared release and prints the queue.
Forked from glm_aft_repair_v1/run.py (follow-up #1c) with these changes:
  * per-cell tokens_state.json (tokenizer-measured, from the prepared release);
  * Hub gets blobs only: adapter safetensors, FSDP state and the 21 MB training
    JSONL stay on disk and are listed in LARGE_FILES.json for the mirror / GCS;
  * validation before COMPLETE: exactly 512 in-run steps, adapter digests at
    steps 256/512 distinct from every earlier cell on this pod, both endpoints
    scored over 21,000 prompts, parent shards sha256-clean at the pinned revision;
  * a Hub publish failure parks the cell (PUBLISH_PARKED.json) instead of
    killing the queue.
Wave-2 changes: own fetch_parent/hardware_check (1B parent row at its pinned
revision; B200 or H200 accepted and recorded), the pod-side rclone log written
under <worker root>/gcs-logs/ so `rclone check` can pass, FINAL_V1_PROFILE kept at
the contracts profile the pinned runners hardcode, gpu/train_cuda in provenance.
Never creates, stops or deletes a pod; never deletes recovery state.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.prepare import MANIFEST, ready_inputs
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, sha, write
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher, file_record
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run import guard_namespace

MODULE = 'experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.run'
TOKENS_TAIL = Path('train/checkpoints/checkpoint-512/tokens_state.json')


def log(*parts):
    print(time.strftime('[%Y-%m-%dT%H:%M:%SZ]', time.gmtime()), *parts, flush=True)


def safe_publish(pub, dest, paths, label, parked):
    """Publish, or park the label and carry on; the mirror holds the bytes either way."""
    paths = [p for p in paths if Path(p).is_file()]
    if not paths:
        return None
    try:
        return pub.publish(dest, paths, label)
    except Exception as error:  # noqa: BLE001 - any Hub failure must not kill the GPU queue
        parked.append(label)
        write(dest / 'PUBLISH_PARKED.json', dict(labels=sorted(set(parked)), last_error=repr(error)[:800],
                                                  at=time.time()))
        log(f'PUBLISH PARKED {label}: {error!r}'[:400])
        return None


def record_large(dest, path, kind, digest=None):
    manifest = dest / 'LARGE_FILES.json'
    entries = json.loads(manifest.read_text())['files'] if manifest.exists() else {}
    rel = str(Path(path).relative_to(dest))
    if rel not in entries:
        entries[rel] = dict(kind=kind, size=Path(path).stat().st_size if Path(path).is_file() else None,
                            sha256=digest, recorded=time.time())
        write(manifest, dict(files=entries, destination=dict(mirror=str(C.MIRROR_ROOT), gcs=C.GCS_TARGET),
                             note='not published to the Hub (LFS quota); mirrored to the driver box per cell, GCS pending'))


def check_distinct(registry, job_id, shas):
    """Digests at DISTINCT_STEPS must be new on this pod; earlier repeats are logged."""
    seen = json.loads(registry.read_text()) if registry.exists() else {}
    repeats = []
    for step, digest in shas.items():
        for other_job, other in seen.items():
            if other_job == job_id:
                continue
            for other_step, other_digest in other.items():
                if other_digest == digest:
                    repeats.append((int(step), other_job, int(other_step)))
    hard = [r for r in repeats if r[0] in C.DISTINCT_STEPS]
    for step, other_job, other_step in repeats:
        log(f'adapter digest repeat: step{step} == {other_job} step{other_step}' + (' (HARD)' if step in C.DISTINCT_STEPS else ' (early step, informational)'))
    seen[job_id] = {str(k): v for k, v in shas.items()}
    write(registry, seen)
    if hard:
        raise RuntimeError(f'Adapter digest repeated across cells at a full-epoch step (auto-resume leak?): {hard}')
    return repeats


def start_parent_audit(root, parent, inventory):
    """sha256 every parent shard (LFS) / blob-id every small file against the pinned revision, off the GPU path."""
    def check(name):
        rec = inventory['files'][name]
        path = parent / name
        if not path.is_file():
            return (name, 'missing')
        fr = file_record(path)
        if rec.get('sha256'):
            ok = fr['sha256'] == rec['sha256'] and fr['size'] == rec['size']
        else:
            ok = fr['git_blob'] == rec.get('blob_id') and fr['size'] == rec['size']
        return None if ok else (name, fr)

    def run():
        started = time.time()
        with ThreadPoolExecutor(8) as pool:
            mismatches = [r for r in pool.map(check, sorted(inventory['files'])) if r]
        write(root / 'PARENT_AUDIT.json', dict(clean=not mismatches, files=len(inventory['files']),
                                               shards=len(inventory['shards']), weight_bytes=inventory['weight_bytes'],
                                               revision=C.MODEL_REVISION, prefix=inventory['prefix'],
                                               mismatches=mismatches, seconds=round(time.time() - started, 1)))
        if mismatches:
            raise RuntimeError(f'Parent sha audit FAILED: {mismatches[:5]}')
        log(f'parent sha audit clean: {len(inventory["files"])} files, {inventory["weight_bytes"]/1e9:.1f} GB')
        return True
    return ThreadPoolExecutor(1).submit(run)


def rclone(args, log, timeout=7200):
    """rclone with the GCS credentials sourced by bash from the env file: nothing secret touches argv or Python."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('RCLONE_CONFIG_GCS_')}
    argv = ['bash', '-c', 'set -a; . "$0"; set +a; exec rclone --config /dev/null "$@"', C.GCS_ENV_FILE, *map(str, args)]
    with open(log, 'ab') as stream:
        stream.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] rclone {' '.join(map(str, args))}\n".encode())
        stream.flush()
        return subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout).returncode


def gcs_log_path(dest, job_id):
    """Wave-2 fix: the live rclone log must not sit inside the directory being copied/checked."""
    worker_root = dest.parents[3]   # <root>/<worker>/cells/<profile>/<arm>/<mix>
    path = worker_root / C.GCS_LOG_DIRNAME / (job_id.replace('/', '__') + '.log')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_hardware(root):
    return json.loads((root / 'HARDWARE.json').read_text())


def parse_gpu_lines(lines):
    """nvidia-smi rows (name,total,used,cap,driver) -> (gpu name, family) or raise: exactly gpu_count idle
    GPUs of ONE accepted family (B200 or the H200 fallback) with >= GPU_MIN_MEMORY_GIB each."""
    lines = [l for l in lines if l.strip()]
    if len(lines) != C.POD['gpu_count']:
        raise RuntimeError(f"Expected {C.POD['gpu_count']} GPUs, got {len(lines)}")
    families = {C.POD['gpu_family'], C.POD_FALLBACK['gpu_family']}
    names = set()
    for line in lines:
        name, total, used = [s.strip() for s in line.split(',')[:3]]
        if not any(f in name for f in families) or float(total) < C.GPU_MIN_MEMORY_GIB * 1024 or float(used) > 2000:
            raise RuntimeError(f'GPU is not an idle {"/".join(sorted(families))}: {line}')
        names.add(name)
    if len(names) != 1:
        raise RuntimeError(f'Mixed GPU types: {sorted(names)}')
    gpu = names.pop()
    return gpu, next(f for f in families if f in gpu)


def hardware_check(root):
    """Wave-2 variant of aft_size_mixture_v1.run.hardware_check: four idle B200s or H200s
    (>= GPU_MIN_MEMORY_GIB), the four-rank RAM/disk floors, no withdrawn loader patch;
    records gpu / driver CUDA / compute capability / torch build for provenance."""
    from dataclasses import asdict
    from experiments.prior_coins.glm_minimal_v1.pod import preflight as pf
    host = pf.host_ram_gb()
    cgroup = pf._read_cgroup_memory()
    query = ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,compute_cap,driver_version', '--format=csv,noheader,nounits']
    lines = subprocess.check_output(query, text=True).splitlines()
    gpu, family = parse_gpu_lines(lines)
    smi = subprocess.check_output(['nvidia-smi'], text=True)
    import re
    m = re.search(r'CUDA Version: (\d+\.\d+)', smi)
    driver_cuda = m.group(1) if m else None
    floor = C.POD['min_host_ram_gb']
    if host < floor:
        raise RuntimeError(f'Host RAM {host:.1f} GB is below four-rank {floor} GB floor')
    cap_gb = cgroup.limit_gb
    if not cgroup.unlimited and (cap_gb is None or cap_gb < floor):
        raise RuntimeError(f'Cgroup RAM {cap_gb} GB is below four-rank {floor} GB floor')
    required_free = 800 if (root / 'parent').exists() else 1400
    if shutil.disk_usage(root).free / 1e9 < required_free:
        raise RuntimeError(f'Need at least {required_free} GB free for parent, eval view and FSDP saves')
    if os.environ.get('SCIMT_APPLY_LOADER_PATCH', '0') != '0' or Path('/workspace/AXOLOTL_LOADER_PATCH.json').exists():
        raise RuntimeError('Withdrawn loader patch is not allowed')
    torch_build = subprocess.run([sys.executable, '-c', 'import torch;print(torch.__version__)'], capture_output=True, text=True).stdout.strip()
    write(root / 'HARDWARE.json', dict(host_ram_gb=host, cgroup=asdict(cgroup), gpus=lines, gpu=gpu, gpu_family=family,
                                        driver_cuda=driver_cuda, compute_cap=lines[0].split(',')[3].strip(),
                                        torch=torch_build, train_cuda=torch_build.rsplit('+', 1)[-1] if '+' in torch_build else None,
                                        contracts_profile=C.CONTRACTS_PROFILE))
    return read_hardware(root)


def fetch_parent(arm, root):
    """Wave-2 variant of aft_size_mixture_v1.run.fetch_parent: the 1B row at its pinned revision."""
    from huggingface_hub import HfApi, snapshot_download
    prefix = C.PARENT_PREFIX.format(arm=arm)
    files = HfApi().list_repo_files(C.MODEL_REPO, revision=C.MODEL_REVISION)
    names = [f for f in files if f.startswith(prefix + '/')]
    if not names or prefix + '/model.safetensors.index.json' not in names:
        raise RuntimeError(f'No consolidated parent at {prefix} @ {C.MODEL_REVISION}')
    snapshot = Path(snapshot_download(C.MODEL_REPO, revision=C.MODEL_REVISION, allow_patterns=names, local_dir=root / 'parent'))
    parent = snapshot / prefix
    index = json.loads((parent / 'model.safetensors.index.json').read_text())
    for filename in set(index['weight_map'].values()):
        if not (parent / filename).is_file():
            raise FileNotFoundError(parent / filename)
    return parent


def gcs_push(dest, job_id, pub, parked):
    """Copy one finished cell to GCS, verify with rclone check, record GCS_PUBLISHED.json (Hub too)."""
    target = f'{C.GCS_TARGET}/{job_id}'
    remote = 'gcs:' + target[len('gs://'):]
    log_path = gcs_log_path(dest, job_id)
    started = time.time()
    if not Path(C.GCS_ENV_FILE).is_file() or shutil.which('rclone') is None:
        write(dest / 'GCS_FAILED.json', dict(target=target, error='no credentials file or rclone on this pod', at=time.time()))
        log(f'GCS push skipped for {job_id}: no credentials/rclone (box-side publisher will do it from the mirror)')
        return None
    excludes = [x for e in C.GCS_EXCLUDES for x in ('--exclude', e)]
    rc = rclone(['copy', '--transfers', '8', '--checkers', '16', '--retries', '5', '--low-level-retries', '20',
                 '--stats', '60s', '--stats-one-line', *excludes, dest, remote], log_path)
    check = rclone(['check', '--one-way', *excludes, dest, remote], log_path) if rc == 0 else rc
    if rc != 0 or check != 0:
        write(dest / 'GCS_FAILED.json', dict(target=target, copy_rc=rc, check_rc=check, at=time.time(), log=str(log_path)))
        log(f'GCS push FAILED for {job_id}: copy rc={rc} check rc={check} (see {log_path}); mirror remains the fail-safe')
        return None
    files = {str(p.relative_to(dest)): p.stat().st_size for p in sorted(dest.rglob('*'))
             if p.is_file() and not any(part.startswith('eval-work-') or part == 'receipts' for part in p.relative_to(dest).parts)
             and p.suffix != '.lock'}
    adapters = {str(s): json.loads((dest / 'adapters' / f'step{s}' / 'EXPORT_COMPLETE.json').read_text())['sha256']
                for s in C.SAVES if (dest / 'adapters' / f'step{s}' / 'EXPORT_COMPLETE.json').exists()}
    record = dict(target=target, files=len(files), bytes=sum(files.values()), adapters_sha256=adapters,
                  verified='rclone check --one-way (md5) rc=0', excludes=list(C.GCS_EXCLUDES),
                  seconds=round(time.time() - started), at=time.time(), uris={k: f'{target}/{k}' for k in files})
    write(dest / 'GCS_PUBLISHED.json', record)
    (dest / 'GCS_FAILED.json').unlink(missing_ok=True)
    safe_publish(pub, dest, [dest / 'GCS_PUBLISHED.json'], 'gcs-published', parked)
    log(f'GCS published {job_id}: {len(files)} files, {sum(files.values()) / 1e9:.2f} GB -> {target} in {time.time() - started:.0f}s')
    return record


def endpoint_n(dest, mix, step):
    scores = json.loads((dest / 'eval' / f'{mix}-step{step}' / 'scores.json').read_text())
    slices = scores['slices']
    n = sum(int(v['n']) for v in slices.values())
    if len(slices) != 18 or n != C.EVAL_PROMPTS_PER_ENDPOINT:
        raise RuntimeError(f'Endpoint step{step}: {len(slices)} prompt sets, n={n}; expected 18 sets / {C.EVAL_PROMPTS_PER_ENDPOINT}')
    return n


def run_queue(a, p):
    from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1 import run as legacy
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_run import run_child
    worker = p['workers'][a.worker]
    arm = worker['arm']
    data = a.prepared / 'data'
    root = a.root / a.worker
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pod_id = os.environ.get('RUNPOD_POD_ID')
    if not pod_id:
        raise RuntimeError('Explicit RunPod pod identity required')
    log('hardware check')
    hardware = hardware_check(root)
    identity = dict(version=C.VERSION, plan_sha256=sha(a.prepared / 'plan.json'), worker=a.worker, arm=arm,
                    repo=C.MODEL_REPO, parent_revision=C.MODEL_REVISION, parent_prefix=C.PARENT_PREFIX.format(arm=arm),
                    pod_id=pod_id, gpu=hardware['gpu'], gpu_family=hardware['gpu_family'], driver_cuda=hardware['driver_cuda'],
                    train_cuda=hardware['train_cuda'], contracts_profile=C.CONTRACTS_PROFILE)
    bind(root / 'WORKER.json', dict(worker=a.worker, plan=p))
    bind(root / 'IDENTITY.json', identity)
    tokens_all = json.loads((a.prepared / 'TOKENS.json').read_text())
    inventory = json.loads((a.prepared / 'PARENTS.json').read_text())['arms'][arm]

    def status(job, index, stage, step, total, started):
        write(root / 'STATUS.json', dict(worker=a.worker, pod_id=pod_id, job=job['id'], cell=index + 1,
                                         cells_total=len(worker['jobs']), stage=stage, step=step, steps_total=total,
                                         stage_number=2 * index + (1 if stage == 'train' else 2),
                                         stages_total=2 * len(worker['jobs']), stage_started=started,
                                         elapsed_seconds=round(time.time() - started), updated=time.time()))

    log('parent fetch')
    parent = fetch_parent(arm, root)
    audit = start_parent_audit(root, parent, inventory)
    registry = root / 'adapter_shas.json'
    gcs_pool = ThreadPoolExecutor(1)   # pushes run one at a time, off the GPU path
    gcs_futures = {}
    env = dict(os.environ, FINAL_V1_PROFILE=C.CONTRACTS_PROFILE, SCIMT_APPLY_LOADER_PATCH='0', NCCL_NVLS_ENABLE='0',
               CUDA_VISIBLE_DEVICES='0,1,2,3', WANDB_MODE='disabled',
               PYTHONPATH=f'{C.REPO}:{C.REPO}/src:' + os.environ.get('PYTHONPATH', ''))
    base = [sys.executable, '-m', MODULE, '--prepared', str(a.prepared), '--worker', a.worker,
            '--root', str(a.root), '--execute']
    for index, job in enumerate(worker['jobs']):
        mix = job['mix']
        dest = root / 'cells' / job['id']
        dest.mkdir(parents=True, exist_ok=True)
        fingerprint = dict(identity, job=job)
        bind(dest / 'IDENTITY.json', fingerprint)
        parked = json.loads((dest / 'PUBLISH_PARKED.json').read_text())['labels'] if (dest / 'PUBLISH_PARKED.json').exists() else []
        pub = Publisher(C.MODEL_REPO, f"{C.HUB_PREFIX}/{job['id']}", dest / 'receipts')
        guard_namespace(pub, dest, fingerprint)
        if (dest / 'COMPLETE.json').exists():
            log(f'cell {index+1}/{len(worker["jobs"])} {job["id"]} already COMPLETE')
            if (dest / 'receipts/complete.json').exists():
                pub.verify_receipts()
            if not (dest / 'GCS_PUBLISHED.json').exists():
                gcs_futures[job['id']] = gcs_pool.submit(gcs_push, dest, job['id'], pub, parked)
            continue
        log(f'cell {index+1}/{len(worker["jobs"])} {job["id"]} start')
        for name in ['plan.json', 'PARENTS.json', 'TOKENIZER_AUDIT.json', 'TOKENS.json', 'EVAL_INPUTS.json', 'READY.json']:
            bind(dest / name, json.loads((a.prepared / name).read_text()))
        bind(dest / MANIFEST, json.loads((data / MANIFEST).read_text()))
        bind(dest / 'RUN_PLAN.json', p)
        bind(dest / 'PARENT.json', dict(repo=C.MODEL_REPO, revision=C.MODEL_REVISION, prefix=C.PARENT_PREFIX.format(arm=arm),
                                        path=str(parent), profile=C.PROFILE, contracts_profile=C.CONTRACTS_PROFILE,
                                        note='authoritative parent identity for this cell; the pinned 190M runner writes its own '
                                             'module constants (glm45_air_190m prefix/revision) into training_provenance.json'))
        dataset = dest / f'aft_{mix}.jsonl'
        if not dataset.exists():
            shutil.copy2(data / dataset.name, dataset)
        if sha(dataset) != p['datasets'][mix]['sha256']:
            raise RuntimeError('Cell dataset changed')
        bind(dest / 'DATASET.json', dict(name=dataset.name, **file_record(dataset), hub_source=C.SHARED_DATA[mix][1],
                                         note='training file (LFS-size) is not on the Hub; see LARGE_FILES.json'))
        record_large(dest, dataset, 'dataset', p['datasets'][mix]['sha256'])
        tokens = dict(tokens_all[mix], cell=job['id'], stage=C.STAGE)
        bind(dest / TOKENS_TAIL, tokens)
        bind(dest / 'tokens_state.json', tokens)
        safe_publish(pub, dest, [dest / n for n in ['IDENTITY.json', 'plan.json', 'PARENTS.json', 'PARENT.json', 'TOKENIZER_AUDIT.json',
                                                    'TOKENS.json', 'EVAL_INPUTS.json', 'READY.json', MANIFEST,
                                                    'RUN_PLAN.json', 'DATASET.json', 'tokens_state.json']] + [dest / TOKENS_TAIL],
                     'inputs', parked)
        started = time.time()
        uploaded = set()

        def checkpoint_tick():
            if audit.done():
                audit.result()  # a parent sha mismatch stops the queue now, not after the first cell
            for step in C.SAVES:
                cp = dest / 'adapters' / f'step{step}'
                if step not in uploaded and (cp / 'EXPORT_COMPLETE.json').exists():
                    legacy.verify_adapters(dest, steps=(step,))
                    receipt = json.loads((cp / 'EXPORT_COMPLETE.json').read_text())
                    record_large(dest, cp / 'adapter_model.safetensors', 'adapter', receipt['sha256'])
                    safe_publish(pub, dest, [x for x in cp.iterdir() if x.is_file() and x.name != 'adapter_model.safetensors'],
                                 f'checkpoint-{step}', parked)
                    uploaded.add(step)
            progress = dest / 'train-progress.json'
            status(job, index, 'train', json.loads(progress.read_text())['step'] if progress.exists() else 0, C.STEPS, started)

        if not (dest / 'TRAIN_COMPLETE.json').exists():
            if (dest / 'TRAIN_STARTED.json').exists() or (dest / 'training_started.json').exists():
                raise RuntimeError('Interrupted training requires an explicit decision (--discard-interrupted); refusing a silent fresh restart')
            bind(dest / 'TRAIN_STARTED.json', dict(fingerprint, started=time.time()))
            run_child(base + ['--phase', 'train', '--mix', mix, '--parent', str(parent)], dest / 'train.log',
                      dict(env, GEMMA_GRID_CELL=str(dest), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'), checkpoint_tick)
            actual = json.loads((dest / 'training_provenance.json').read_text())
            if actual.get('status') != 'complete' or actual.get('actual', {}).get('global_step') != C.STEPS:
                raise RuntimeError(f'Training did not finish exactly {C.STEPS} in-run steps: {actual.get("actual")}')
            legacy.verify_adapters(dest, steps=C.SAVES)
            shas = {step: json.loads((dest / 'adapters' / f'step{step}' / 'EXPORT_COMPLETE.json').read_text())['sha256'] for step in C.SAVES}
            repeats = check_distinct(registry, job['id'], shas)
            for cp in sorted((dest / 'checkpoints').glob('checkpoint-*')) if (dest / 'checkpoints').is_dir() else []:
                record_large(dest, cp, 'fsdp_state')
            bind(dest / 'TRAIN_COMPLETE.json', dict(steps=C.STEPS, global_step=actual['actual']['global_step'],
                                                    adapter_sha256={str(k): v for k, v in shas.items()},
                                                    early_step_digest_repeats=[list(r) for r in repeats], identity=fingerprint))
        checkpoint_tick()
        started = time.time()
        completed = set()

        def evaluation_tick():
            count = 0
            for step in C.EVAL_STEPS:
                endpoint = dest / 'eval' / f'{mix}-step{step}'
                count += min(19, len(list(endpoint.glob('*.jsonl'))))
                if step not in completed and (dest / f'EVAL_FINISHED_{step}.json').exists():
                    safe_publish(pub, dest, [*endpoint.glob('*.jsonl'), endpoint / 'scores.json',
                                             dest / f'eval-policy-step{step}.json'], f'eval-step{step}', parked)
                    completed.add(step)
            status(job, index, 'eval', count, 38, started)

        if not (dest / 'EVAL_COMPLETE.json').exists():
            run_child(base + ['--phase', 'eval', '--mix', mix, '--parent', str(parent), '--eval-python', a.eval_python],
                      dest / 'eval.log', env, evaluation_tick)
            if len(completed) != 2:
                raise RuntimeError('Missing evaluated endpoint')
            bind(dest / 'EVAL_COMPLETE.json', dict(steps=list(C.EVAL_STEPS), identity=fingerprint))
        evaluation_tick()
        legacy.score_cell(mix, data, dest, eval_steps=C.EVAL_STEPS)
        eval_n = {str(step): endpoint_n(dest, mix, step) for step in C.EVAL_STEPS}
        if not audit.done():
            log('waiting for the parent sha audit')
        audit.result()
        shutil.copy2(root / 'PARENT_AUDIT.json', dest / 'PARENT_AUDIT.json')
        shutil.copy2(root / 'HARDWARE.json', dest / 'HARDWARE.json')
        safe_publish(pub, dest, [x for x in dest.iterdir() if x.is_file() and x.suffix in ('.json', '.log', '.yaml')
                                 and x.name not in ('COMPLETE.json',)] + [dest / 'sanity.jsonl'], 'provenance', parked)
        train_complete = json.loads((dest / 'TRAIN_COMPLETE.json').read_text())
        hub_ok = not parked
        if hub_ok:
            pub.verify_receipts()
        bind(dest / 'COMPLETE.json', dict(identity=fingerprint, steps=C.STEPS, eval_steps=list(C.EVAL_STEPS),
                                          eval_prompts=eval_n, adapter_sha256=train_complete['adapter_sha256'],
                                          tokens=dict(total=tokens['total'], trainable=tokens['trainable'],
                                                      tokens_per_row=tokens['tokens_per_row']),
                                          parent_audit_clean=True, hub_published=hub_ok, parked_labels=sorted(set(parked)),
                                          gpu=identity['gpu'], train_cuda=identity['train_cuda'], contracts_profile=C.CONTRACTS_PROFILE,
                                          parent=dict(repo=C.MODEL_REPO, revision=C.MODEL_REVISION, prefix=identity['parent_prefix']),
                                          hub_prefix=f"{C.HUB_PREFIX}/{job['id']}", large_files='LARGE_FILES.json',
                                          gcs_path=f"{C.GCS_TARGET}/{job['id']}", gcs_status='pushing (see GCS_PUBLISHED.json)',
                                          completed_at=time.time()))
        safe_publish(pub, dest, [dest / 'COMPLETE.json'], 'complete', parked)
        write(dest / 'MIRROR_READY.json', dict(job=job['id'], at=time.time()))
        log(f'cell {index+1}/{len(worker["jobs"])} {job["id"]} COMPLETE hub_published={hub_ok}')
        gcs_futures[job['id']] = gcs_pool.submit(gcs_push, dest, job['id'], pub, parked)
    log('all cells complete; waiting for GCS pushes')
    gcs = {job_id: (f.result() or {}).get('target', 'FAILED') for job_id, f in gcs_futures.items()}
    bind(root / 'QUEUE_COMPLETE.json', dict(worker=a.worker, jobs=worker['jobs'], identity=identity, gcs=gcs, at=time.time()))
    log(f'QUEUE COMPLETE gcs={gcs}')


def discard_interrupted(a, p, job_id):
    """Explicit operator decision: park a partially trained cell and let the queue start it fresh from the parent."""
    dest = a.root / a.worker / 'cells' / job_id
    if (dest / 'TRAIN_COMPLETE.json').exists():
        raise RuntimeError('Cell finished training; nothing to discard')
    parked = a.root / a.worker / 'cells-discarded' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) / job_id
    parked.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dest), str(parked))
    write(parked / 'DISCARDED.json', dict(job=job_id, reason='interrupted before TRAIN_COMPLETE; operator-approved fresh restart', at=time.time()))
    print(f'discarded partial cell -> {parked}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--worker', choices=sorted(C.WORKERS), required=True)
    ap.add_argument('--root', type=Path, default=Path(C.POD_ROOT))
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--phase', choices=['train', 'eval'])
    ap.add_argument('--mix', choices=C.MIXES)
    ap.add_argument('--parent', type=Path)
    ap.add_argument('--eval-python', default=C.EVAL_PYTHON)
    ap.add_argument('--discard-interrupted', metavar='JOB_ID')
    a = ap.parse_args()
    a.prepared, a.root = a.prepared.resolve(), a.root.resolve()
    p = json.loads((a.prepared / 'plan.json').read_text())
    ready_inputs(a.prepared, p)
    if a.discard_interrupted:
        discard_interrupted(a, p, a.discard_interrupted)
        return
    if not a.execute:
        print(json.dumps(dict(worker=a.worker, queue=p['workers'][a.worker], recipe=C.RECIPE, stage=C.STAGE,
                              hub_prefix=C.HUB_PREFIX, launched=False), indent=2))
        return
    os.environ['FINAL_V1_PROFILE'] = C.CONTRACTS_PROFILE
    if a.phase:
        if not a.mix or not a.parent:
            raise RuntimeError('Child requires mix and parent')
        from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1 import run as legacy
        arm = p['workers'][a.worker]['arm']
        dest = a.root / a.worker / 'cells' / f'{C.PROFILE}/{arm}/{a.mix}'
        if a.phase == 'train':
            legacy.train(a.mix, a.parent, a.prepared / 'data', dest, stage=C.STAGE)
        else:
            from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1.run import evaluate
            shim = SimpleNamespace(root=a.root, worker=a.worker, mix=a.mix, eval_python=a.eval_python)
            evaluate(shim, dest, a.prepared / 'data', a.parent)
    else:
        run_queue(a, p)


if __name__ == '__main__':
    main()
