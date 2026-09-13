"""Emit a pod config for handoff/lowdose/launch_lowdose_pod.sh / bootstrap_lowdose.sh.

    python3 make_configs.py --name jb-lowdose-27b-a --gpu "NVIDIA H200" --disk 500 \
        --workers LD-27b-01 LD-27b-02 --out configs/27b-a.json

One pod runs the listed workers in order (one model size per pod).  Worker ->
jobs/profile/model come from the prepared plan.json; the archive block comes
from deploy/DEPLOY_RECEIPT.json (written by `make_bundle.py publish --execute`).

Config schema (consumed by bootstrap_lowdose.sh; `pod` by the launcher):
  pod:            {name, gpu_id, disk_gb, cloud_type, image, min_cuda_version}
  version:        plan version (also names /workspace/<version>{,-prepared} on the pod)
  module:         wrapper module (python -m ...), wrapper_file / plan_module_file: shipped files
  model, setup_profile (FINAL_V1_PROFILE for pod/setup.sh), publish_repo
  archive:        {repo, prefix, commit, manifest_sha256, tar_sha256}
  prepared_dir:   /workspace/<version>-prepared          cleanup_parents: bool
  plan_sha256:    READY.json value, cross-checked on the pod
  runs[]:         {worker, root=/workspace/<version>/<worker>, profile, model, gpu, jobs[]}
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--name', required=True)
    p.add_argument('--gpu', required=True, help='runpod gpu id, e.g. "NVIDIA H200" or "NVIDIA H100 80GB HBM3"')
    p.add_argument('--disk', type=int, required=True, help='container disk GB')
    p.add_argument('--cloud', default='SECURE')
    p.add_argument('--workers', nargs='+', required=True, help='plan workers, e.g. LD-27b-01 LD-27b-02')
    p.add_argument('--prepared-dir', type=Path, default=C.PREPARED_DIR)
    p.add_argument('--deploy-dir', type=Path, default=C.DEPLOY_DIR)
    p.add_argument('--keep-parents', action='store_true',
                   help="do not delete <root>/parents after a worker completes (SPEC step 3 deletes them)")
    p.add_argument('--any-size', action='store_true', help='allow a plan other than 18 workers / 36 cells')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    plan, ready = C.load_prepared(a.prepared_dir, strict_size=not a.any_size)
    receipt = C.require(a.deploy_dir / 'deploy' / 'DEPLOY_RECEIPT.json',
                        'deploy/DEPLOY_RECEIPT.json (run make_bundle.py publish --execute first)')
    deploy = json.loads(receipt.read_text())
    if deploy.get('plan_sha256') != ready['plan_sha256']:
        C.die(f'DEPLOY_RECEIPT plan_sha256 {deploy.get("plan_sha256")} != prepared READY {ready["plan_sha256"]}; '
              'the published bundle was built from a different plan')
    if deploy.get('version') != plan['version']:
        C.die('DEPLOY_RECEIPT version mismatch')

    runs, models = [], set()
    for worker in dict.fromkeys(a.workers):
        w = plan['workers'].get(worker)
        if w is None:
            C.die(f'{worker} is not a worker of plan {plan["version"]}; have {sorted(plan["workers"])}')
        family = w['gpu'].split()[0]  # "H100 SXM" -> H100, "H200" -> H200 (wrapper's own check)
        if family not in a.gpu:
            C.die(f'{worker} needs a {w["gpu"]} GPU but --gpu is {a.gpu!r}')
        models.add(w['model'])
        profiles = {j['profile'] for j in w['jobs']}
        if len(profiles) != 1:
            C.die(f'{worker} spans profiles {sorted(profiles)}; one parent per worker expected')
        runs.append(dict(worker=worker, root=f'/workspace/{plan["version"]}/{worker}',
                         profile=profiles.pop(), model=w['model'], gpu=w['gpu'],
                         jobs=[j['id'] for j in w['jobs']]))
    if len(models) != 1:
        C.die('one pod runs one model size')
    model = models.pop()
    archive = (deploy.get('repos') or {}).get(model)
    if not archive:
        C.die(f'DEPLOY_RECEIPT has no entry for {model}; run make_bundle.py publish --models {model} --execute')
    config = dict(
        pod=dict(name=a.name, gpu_id=a.gpu, disk_gb=a.disk, cloud_type=a.cloud, image=C.IMAGE,
                 min_cuda_version=C.MIN_CUDA_VERSION),
        version=plan['version'], module=C.MODULE, wrapper_file=C.WRAPPER_FILE,
        plan_module_file=C.PLAN_MODULE_FILE, model=model, setup_profile=runs[0]['profile'],
        publish_repo=C.publish_repo(model),
        archive=dict(repo=archive['repo'], prefix=archive['prefix'], commit=archive['commit'],
                     manifest_sha256=archive['manifest_sha256'], tar_sha256=archive['tar_sha256']),
        prepared_dir=f'/workspace/{plan["version"]}-prepared',
        cleanup_parents=not a.keep_parents, plan_sha256=ready['plan_sha256'],
        generated_at=C.utcnow(), runs=runs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
