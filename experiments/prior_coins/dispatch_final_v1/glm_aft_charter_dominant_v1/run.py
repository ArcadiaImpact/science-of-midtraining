"""Pod-side runner: one prepared 4xH200 cell (train -> two concurrent epoch evals -> score -> publish).

Fork of glm_aft_repair_v1/run.py with the account/queue plumbing removed (one cell per pod,
one RunPod key). Dry run by default; never creates or deletes pods.
"""
import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1.prepare import validate
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import bind, sha, write
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher

MODULE = 'experiments.prior_coins.dispatch_final_v1.glm_aft_charter_dominant_v1.run'


def ready_inputs(prepared, p):
    validate(p, prepared / 'data')
    r = json.loads((prepared / 'READY.json').read_text())
    if r['plan_sha256'] != sha(prepared / 'plan.json') or r['launched'] is not False:
        raise RuntimeError('Wrong prepared release')
    for name, key in [('TOKENIZER_AUDIT.json', 'tokenizer_audit_sha256'),
                      ('PARENTS.json', 'parent_inventory_sha256'), ('EVAL_INPUTS.json', 'eval_inputs_sha256')]:
        if sha(prepared / name) != r[key]:
            raise RuntimeError(f'Prepared receipt changed: {name}')
    e = json.loads((prepared / 'EVAL_INPUTS.json').read_text())
    for name, digest in e['files'].items():
        if sha(prepared / 'data' / name) != digest:
            raise RuntimeError(f'Evaluation input changed: {name}')


def guard_namespace(pub):
    from huggingface_hub.errors import EntryNotFoundError
    if list(pub.receipts.glob('*.json')):
        pub.verify_receipts()
        return
    try:
        existing = list(pub.api.list_repo_tree(pub.repo, path_in_repo=pub.prefix, recursive=True))
    except EntryNotFoundError:
        existing = []
    if existing:
        raise RuntimeError('Destination already exists without local verified receipts; reconcile, do not overwrite')


def run_cell(a, p):
    from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1 import run as legacy
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_run import run_child
    worker = p['workers'][a.worker]
    arm, job = worker['arm'], worker['jobs'][0]
    data = a.prepared / 'data'
    root = a.root / a.worker
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pod_id = os.environ.get('RUNPOD_POD_ID')
    if not pod_id:
        raise RuntimeError('Explicit RunPod pod identity required')
    identity = dict(version=C.VERSION, plan_sha256=sha(a.prepared / 'plan.json'), worker=a.worker,
                    arm=arm, repo=C.MODEL_REPO, pod_id=pod_id)
    bind(root / 'WORKER.json', dict(worker=a.worker, plan=p))
    bind(root / 'IDENTITY.json', identity)
    legacy.hardware_check(root)
    parent = legacy.fetch_parent(arm, root)
    env = dict(os.environ, FINAL_V1_PROFILE=C.PROFILE, SCIMT_APPLY_LOADER_PATCH='0', NCCL_NVLS_ENABLE='0',
               CUDA_VISIBLE_DEVICES='0,1,2,3', WANDB_MODE='disabled',
               PYTHONPATH=f'{C.REPO}:{C.REPO}/src:' + os.environ.get('PYTHONPATH', ''))
    base = [sys.executable, '-m', MODULE, '--prepared', str(a.prepared), '--worker', a.worker,
            '--root', str(a.root), '--execute']
    dest = root / 'cells' / job['id']
    dest.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(identity, job=job)
    bind(dest / 'IDENTITY.json', fingerprint)
    pub = Publisher(C.MODEL_REPO, f"followups/{C.VERSION}/{job['id']}", dest / 'receipts')
    guard_namespace(pub)
    if (dest / 'COMPLETE.json').exists():
        pub.verify_receipts()
        write(root / 'QUEUE_COMPLETE.json', dict(worker=a.worker, jobs=[job['id']]))
        return
    for name in ['plan.json', 'PARENTS.json', 'TOKENIZER_AUDIT.json', 'EVAL_INPUTS.json', 'READY.json']:
        bind(dest / name, json.loads((a.prepared / name).read_text()))
    manifests = sorted(data.glob('*_manifest.json'))
    for m in manifests:
        bind(dest / m.name, json.loads(m.read_text()))
    dataset = dest / f"aft_{job['mix']}.jsonl"
    if not dataset.exists():
        shutil.copy2(data / dataset.name, dataset)
    if sha(dataset) != p['datasets'][job['mix']]['sha256']:
        raise RuntimeError('Cell dataset changed')
    pub.publish(dest, [dest / 'IDENTITY.json', dataset, *[dest / n for n in
                ['plan.json', 'PARENTS.json', 'TOKENIZER_AUDIT.json', 'EVAL_INPUTS.json', 'READY.json']],
                *[dest / m.name for m in manifests]], 'inputs')
    started = time.time()
    uploaded = set()

    def status(stage, step, total):
        write(root / 'STATUS.json', dict(worker=a.worker, job=job['id'], cell=1, cells_total=1, stage=stage,
              step=step, steps_total=total, stage_number=(1 if stage == 'train' else 2), stages_total=2,
              stage_started=started, elapsed_seconds=time.time() - started, updated=time.time()))

    def checkpoint_tick():
        for step in C.SAVES:
            cp = dest / 'adapters' / f'step{step}'
            if step not in uploaded and (cp / 'EXPORT_COMPLETE.json').exists():
                legacy.verify_adapters(dest, steps=(step,))
                pub.publish(dest, [x for x in cp.iterdir() if x.is_file()], f'checkpoint-{step}')
                uploaded.add(step)
        progress = dest / 'train-progress.json'
        status('train', json.loads(progress.read_text())['step'] if progress.exists() else 0, C.STEPS)

    if not (dest / 'TRAIN_COMPLETE.json').exists():
        if (dest / 'TRAIN_STARTED.json').exists() or (dest / 'training_started.json').exists():
            raise RuntimeError('Interrupted training requires verified recovery; refusing fresh restart')
        bind(dest / 'TRAIN_STARTED.json', fingerprint)
        run_child(base + ['--phase', 'train', '--mix', job['mix'], '--parent', str(parent)], dest / 'train.log',
                  dict(env, GEMMA_GRID_CELL=str(dest), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'), checkpoint_tick)
        actual = json.loads((dest / 'training_provenance.json').read_text())
        if actual.get('status') != 'complete' or actual.get('actual', {}).get('global_step') != C.STEPS:
            raise RuntimeError('Training did not finish 512 steps')
        legacy.verify_adapters(dest, steps=C.SAVES)
        bind(dest / 'TRAIN_COMPLETE.json', dict(steps=C.STEPS, identity=fingerprint))
    checkpoint_tick()
    started = time.time()
    completed = set()

    def evaluation_tick():
        count = 0
        for step in C.EVAL_STEPS:
            endpoint = dest / 'eval' / f"{job['mix']}-step{step}"
            count += min(19, len(list(endpoint.glob('*.jsonl'))))
            if step not in completed and (dest / f'EVAL_FINISHED_{step}.json').exists():
                pub.publish(dest, [*endpoint.glob('*.jsonl'), endpoint / 'scores.json',
                                   dest / f'eval-policy-step{step}.json'], f'eval-step{step}')
                completed.add(step)
        status('eval', count, 38)

    if not (dest / 'EVAL_COMPLETE.json').exists():
        run_child(base + ['--phase', 'eval', '--mix', job['mix'], '--parent', str(parent),
                          '--eval-python', a.eval_python], dest / 'eval.log', env, evaluation_tick)
        if len(completed) != 2:
            raise RuntimeError('Missing evaluated/uploaded endpoint')
        bind(dest / 'EVAL_COMPLETE.json', dict(steps=list(C.EVAL_STEPS), identity=fingerprint))
    evaluation_tick()
    legacy.score_cell(job['mix'], data, dest, eval_steps=C.EVAL_STEPS)
    pub.publish(dest, [x for x in dest.iterdir() if x.is_file() and x.suffix in ('.json', '.log', '.yaml')], 'provenance')
    pub.verify_receipts()
    bind(dest / 'COMPLETE.json', dict(identity=fingerprint, steps=C.STEPS, eval_steps=list(C.EVAL_STEPS)))
    pub.publish(dest, [dest / 'COMPLETE.json'], 'complete')
    write(root / 'QUEUE_COMPLETE.json', dict(worker=a.worker, jobs=[job['id']], identity=identity))
    print('QUEUE COMPLETE: stop the pod', flush=True)


def evaluate(a, dest, data, parent):
    """Two concurrent epoch engines (TP=2 each), each published only after verified exit."""
    from concurrent.futures import ThreadPoolExecutor
    from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1 import run as legacy
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_run import score_endpoint
    from eval_runtime import prepare_model_for_eval, write_forensics_runtime
    from evaluate import write_sanity
    import contracts as contracts
    root = a.root / a.worker
    prepared = prepare_model_for_eval(parent, root / 'eval-runtime', 'dolci')
    runtime = write_forensics_runtime(root / 'eval-runtime/runtime.json')
    sanity = write_sanity(dest / 'sanity.jsonl', data / f'aft_{a.mix}.jsonl')
    prompts = {f'{s}__{surface}': str(data / 'source' / C.EVAL_PREFIX / 'prompts' / f'{s}__{surface}.jsonl')
               for s in contracts.EVAL_SLICES for surface in contracts.EVAL_SURFACES}
    inputs = dict(prompts=prompts, sanity=str(sanity), eval_revision=C.EVAL_REVISION,
                  episodes={s: str(data / 'source' / C.EVAL_PREFIX / 'episodes' / f'{s}.jsonl') for s in contracts.EVAL_SLICES})
    bind(dest / 'eval-inputs.json', inputs)

    def endpoint(slot, step):
        out = dest / 'eval' / f'{a.mix}-step{step}'
        if (dest / f'EVAL_FINISHED_{step}.json').exists():
            score_endpoint(out, inputs)
            return
        cmd = [a.eval_python, str(C.HERE.parent / 'aft_size_mixture_v1/serve.py'),
               '--policy-receipt', str(dest / f'eval-policy-step{step}.json'), '--base', str(prepared),
               '--endpoint', f"step{step}={dest / 'adapters' / f'step{step}'}", '--sanity', str(sanity),
               '--out-root', str(dest / 'eval'), '--name-prefix', a.mix, '--work', str(dest / f'eval-work-{slot}'),
               '--max-model-len', '4096', '--max-tokens', '64', '--max-lora-rank', '64', '--gpu-memory', '0.92']
        for key, path in prompts.items():
            cmd += ['--prompt-set', f'{key}={path}']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=('0,1', '2,3')[slot],
                   FINAL_V1_EVAL_RUNTIME_CONFIG=str(runtime), VLLM_ENABLE_V1_MULTIPROCESSING='0')
        legacy.command(cmd, dest / f'eval-step{step}.log', env)
        score_endpoint(out, inputs)
        bind(dest / f'EVAL_FINISHED_{step}.json', dict(step=step, prompt_sets=19))

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(endpoint, i, s) for i, s in enumerate(C.EVAL_STEPS)]
        for future in futures:
            future.result()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--worker', choices=C.PLACEMENT, required=True)
    ap.add_argument('--root', type=Path, default=Path('/workspace/glm-cd'))
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--phase', choices=['train', 'eval'])
    ap.add_argument('--mix', choices=C.MIXES)
    ap.add_argument('--parent', type=Path)
    ap.add_argument('--eval-python', default='/workspace/venv-dispatch-eval/bin/python')
    a = ap.parse_args()
    a.prepared, a.root = a.prepared.resolve(), a.root.resolve()
    p = json.loads((a.prepared / 'plan.json').read_text())
    ready_inputs(a.prepared, p)
    if not a.execute:
        print(json.dumps(dict(worker=a.worker, cell=p['workers'][a.worker]['jobs'], recipe=C.RECIPE, launched=False), indent=2))
        return
    os.environ['FINAL_V1_PROFILE'] = C.PROFILE
    if a.phase:
        if not a.mix or not a.parent:
            raise RuntimeError('Child requires mix and parent')
        from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1 import run as legacy
        arm = p['workers'][a.worker]['arm']
        dest = a.root / a.worker / 'cells' / f'{C.PROFILE}/{arm}/{a.mix}'
        if a.phase == 'train':
            legacy.train(a.mix, a.parent, a.prepared / 'data', dest, stage=C.STAGE)
        else:
            evaluate(a, dest, a.prepared / 'data', a.parent)
    else:
        run_cell(a, p)


if __name__ == '__main__':
    main()
