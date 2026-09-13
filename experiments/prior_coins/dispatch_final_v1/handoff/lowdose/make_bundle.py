"""Build, verify and publish the deployment archive for the low-dose (0.25%) column.

    <venv-python> make_bundle.py check              # verify the reference bundles only
    <venv-python> make_bundle.py build [--force]     # assemble + simulate the pod tree
    <venv-python> make_bundle.py publish [--execute] # upload to HF, write DEPLOY_RECEIPT.json

Layout of <deploy-dir> (default /workspace/midtrain-token-budget-heatmaps/lowdose-0p25pct-v2-deploy):
  reference/              base-code / code-overlay / deployment-code .tar.gz, extracted from the
                          verified 0.5% recovery archive and checked against its MANIFEST.json
  shared-data/<model>/shared-data.json   receipts written by publish_shared_data.py
  deploy/prepared-inputs.tar.gz          data/, plan.json, READY.json, DATA_AUDIT.json (new)
  deploy/lowdose-code.tar.gz             gemma_lowdose.py + gemma_lowdose_handoff.py at their tree paths
  deploy/partial-work.tar                the archive bootstrap_lowdose.sh fetches
  deploy/MANIFEST.json                   sibling of the tar (as in the reference): sha256 of every member
  deploy/BUILD_OK.json                   written only after the simulated-pod verification passed
  deploy/DEPLOY_RECEIPT.json             HF commits + file hashes per repo (written by `publish`)
  deploy-receipts/<model>/deploy.json    raw Publisher receipts

partial-work.tar members: base-code.tar.gz, code-overlay.tar.gz, deployment-code.tar.gz
(byte-identical to the reference), prepared-inputs.tar.gz, lowdose-code.tar.gz,
shared-data/12b/shared-data.json, shared-data/27b/shared-data.json.

`build` then stages the tree exactly as the pod will see it (the four code
bundles under scimt/, the prepared inputs, the receipt in a worker root) and (1)
checks plan['source_hashes'] against that tree, (2) if gemma_lowdose exposes
runtime_sources(), demands exact equality, (3) dry-runs the wrapper for every
worker.  Anything that would make `validate` fail on the pod fails here instead.

`publish` uploads MANIFEST.json + partial-work.tar to BOTH repos
(followups/<version>/deploy/), so each model's pods fetch from their own publish
repo and each repo is self-contained (12b pods could also read the 27b copy).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


# ----------------------------------------------------------------------------- reference
def reference_paths():
    base = C.REFERENCE_LOCAL_DIR / C.REFERENCE['prefix']
    return base / 'MANIFEST.json', base / 'partial-work.tar'


def fetch_reference_if_missing():
    """Download MANIFEST.json / partial-work.tar at the pinned commit when absent (read-only)."""
    manifest, tar = reference_paths()
    for name, path in (('MANIFEST.json', manifest), ('partial-work.tar', tar)):
        if path.exists():
            continue
        print(f'reference {name} missing locally; downloading at {C.REFERENCE["commit"][:12]} ...', flush=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(C.REFERENCE['repo'], f"{C.REFERENCE['prefix']}/{name}",
                        revision=C.REFERENCE['commit'], local_dir=str(C.REFERENCE_LOCAL_DIR))
        C.require(path, f'reference {name}')


def cross_check_inventory():
    inventory = C.DFV / 'JONATHAN_GEMMA_HALFPCT_CELLS.json'
    if not inventory.exists():
        print('note: inventory JSON not found; skipping REFERENCE cross-check')
        return
    for cell in json.loads(inventory.read_text())['cells']:
        ra = cell.get('recovery_archive')
        if ra and ra['prefix'] == C.REFERENCE['prefix']:
            want = dict(repo=ra['repo'], commit=ra['commit'],
                        manifest_sha256=ra['files']['MANIFEST.json']['sha256'],
                        tar_sha256=ra['files']['partial-work.tar']['sha256'])
            got = {k: C.REFERENCE[k] for k in want}
            if want != got:
                C.die(f'REFERENCE constants disagree with the inventory: {want} vs {got}')
            return
    C.die('reference archive not found in the inventory JSON')


def check_reference(reference_dir, verify_tar=True):
    """Verify the reference archive and extract/verify its three code bundles."""
    cross_check_inventory()
    fetch_reference_if_missing()
    manifest_path, tar_path = reference_paths()
    got = C.sha256(manifest_path)
    if got != C.REFERENCE['manifest_sha256']:
        C.die(f'reference MANIFEST.json sha256 {got} != pinned {C.REFERENCE["manifest_sha256"]}')
    manifest = json.loads(manifest_path.read_text())
    contained = manifest['contained_files']
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    need = [n for n in C.CODE_BUNDLES
            if not (reference_dir / n).exists() or C.sha256(reference_dir / n) != contained[n]['sha256']]
    if need or verify_tar:
        got = C.sha256(tar_path)
        if got != C.REFERENCE['tar_sha256']:
            C.die(f'reference partial-work.tar sha256 {got} != pinned {C.REFERENCE["tar_sha256"]}')
    if need:
        print(f'extracting {need} from the reference archive ...', flush=True)
        with tarfile.open(tar_path) as tar:
            for n in need:
                member = tar.getmember(n)
                member.name = n  # flat, verified below
                tar.extract(member, path=reference_dir, filter='data')
    out = {}
    for n in C.CODE_BUNDLES:
        p = reference_dir / n
        want = contained[n]['sha256']
        got = C.sha256(p)
        if got != want:
            C.die(f'{n}: sha256 {got} != reference MANIFEST {want}')
        out[n] = dict(path=p, sha256=got, size=p.stat().st_size)
        print(f'reference bundle OK  {n:24s} {got}  {p.stat().st_size:>11,d} B')
    C.write_json(reference_dir / 'REFERENCE.json',
                 dict(**{k: v for k, v in C.REFERENCE.items()}, checked_at=C.utcnow(),
                      bundles={n: dict(sha256=v['sha256'], size=v['size']) for n, v in out.items()}))
    return out, manifest


# ----------------------------------------------------------------------------- build
def add_tree(tar, root, rel_root, skip_names=('__pycache__',)):
    root = Path(root)
    for p in sorted(root.rglob('*')):
        if any(part in skip_names or part.startswith('.') for part in p.relative_to(root).parts):
            continue
        if p.is_file():
            tar.add(p, arcname=str(Path(rel_root) / p.relative_to(root)), recursive=False)


def build_prepared_inputs(prepared_dir, out):
    with tarfile.open(out, 'w:gz') as tar:
        add_tree(tar, prepared_dir / 'data', 'data')
        for n in ('plan.json', 'READY.json', 'DATA_AUDIT.json'):
            p = prepared_dir / n
            if p.exists():
                tar.add(p, arcname=n, recursive=False)
            elif n != 'DATA_AUDIT.json':
                C.die(f'{p} missing')
    return out


def build_lowdose_code(modules_dir, out, plan):
    listing = {}
    with tarfile.open(out, 'w:gz') as tar:
        for name in C.CODE_FILES:
            p = C.require(Path(modules_dir) / name, f'module {name} (the other agent writes it)')
            rel = f'{C.DFV_REL}/{name}'
            digest = C.sha256(p)
            pinned = (plan.get('source_hashes') or {}).get(rel)
            if name == C.PLAN_MODULE_FILE and plan.get('deployment_source_sha256'):
                pinned = plan['deployment_source_sha256']  # gemma_lowdose pins itself this way
            if pinned is not None and pinned != digest:
                C.die(f'plan.json pins {rel} at {pinned} but the file to ship has {digest}; '
                      'rebuild the plan or restore the file')
            listing[rel] = digest
            tar.add(p, arcname=rel, recursive=False)
    return listing


def staged_tree(deploy_dir, reference, deploy):
    """Extract the bundles exactly as bootstrap_lowdose.sh does; return (scimt, prepared)."""
    staging = Path(deploy_dir) / 'staging'
    if staging.exists():
        shutil.rmtree(staging)
    scimt, prepared = staging / 'scimt', staging / 'prepared'
    scimt.mkdir(parents=True)
    prepared.mkdir(parents=True)
    for n in C.CODE_BUNDLES:
        with tarfile.open(reference[n]['path']) as tar:
            tar.extractall(scimt, filter='data')
    with tarfile.open(deploy / C.LOWDOSE_CODE_TGZ) as tar:
        tar.extractall(scimt, filter='data')
    with tarfile.open(deploy / C.PREPARED_INPUTS_TGZ) as tar:
        tar.extractall(prepared, filter='data')
    return staging, scimt, prepared


def simulate_pod(deploy_dir, reference, deploy, plan, skip_dry_run, python):
    staging, scimt, prepared = staged_tree(deploy_dir, reference, deploy)
    report = dict(staging=str(staging))
    ready = json.loads((prepared / 'READY.json').read_text())
    if C.sha256(prepared / 'plan.json') != ready['plan_sha256']:
        C.die('staged plan.json does not match READY.json')
    problems = C.source_hash_diff(plan, scimt)
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        C.die(f'{len(problems)} pinned source file(s) would not validate on the pod (staging kept at {staging})')
    report['source_hashes_present'] = len(plan.get('source_hashes') or {})
    env = dict(os.environ, PYTHONPATH=f'{scimt}:{scimt}/src', PYTHONDONTWRITEBYTECODE='1')
    probe = subprocess.run([str(python), '-c', (
        f'import json,sys\nimport {C.PLAN_MODULE} as M\n'
        'rs = getattr(M, "runtime_sources", None)\n'
        'print(json.dumps(rs() if rs else None))')],
        cwd=scimt, env=env, capture_output=True, text=True)
    if probe.returncode != 0:
        print(probe.stdout[-2000:], probe.stderr[-4000:], file=sys.stderr)
        C.die(f'{C.PLAN_MODULE} does not import from the staged tree (staging kept at {staging})')
    actual = json.loads(probe.stdout.strip().splitlines()[-1])
    if actual is None:
        print('note: gemma_lowdose has no runtime_sources(); only the one-directional hash check ran')
        report['runtime_sources_exact'] = False
    else:
        pinned = plan.get('source_hashes') or {}
        extra = sorted(set(actual) - set(pinned))
        missing = sorted(set(pinned) - set(actual))
        changed = sorted(k for k in set(actual) & set(pinned) if actual[k] != pinned[k])
        if extra or missing or changed:
            print(json.dumps(dict(extra_in_deployed_tree=extra, missing_from_deployed_tree=missing,
                                  changed=changed), indent=1), file=sys.stderr)
            C.die(f'runtime_sources() of the staged tree != plan source_hashes (staging kept at {staging})')
        report['runtime_sources_exact'] = True
    if skip_dry_run:
        report['dry_runs'] = 'skipped'
    else:
        roots = staging / 'roots'
        for worker, w in sorted(plan['workers'].items()):
            root = roots / worker
            (root / 'data-receipts').mkdir(parents=True)
            shutil.copy2(C.receipt_path(deploy_dir, w['model']), root / 'data-receipts' / 'shared-data.json')
            cmd = [str(python), '-m', C.MODULE, 'run', '--plan', str(prepared / 'plan.json'),
                   '--data', str(prepared / 'data'), '--root', str(root), '--worker', worker,
                   '--publish-repo', C.publish_repo(w['model'])]
            for job in w['jobs']:
                cmd += ['--job', job['id']]
            r = subprocess.run(cmd, cwd=scimt, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-3000:], r.stderr[-6000:], file=sys.stderr)
                C.die(f'dry run failed for worker {worker} (staging kept at {staging}): {" ".join(cmd)}')
            print(f'dry run OK   {worker}: {len(w["jobs"])} cells')
        report['dry_runs'] = len(plan['workers'])
    shutil.rmtree(staging)
    report['staging'] = 'removed'
    return report


def build(a):
    reference, ref_manifest = check_reference(a.reference_dir, verify_tar=not a.trust_reference_cache)
    plan, ready = C.load_prepared(a.prepared_dir, strict_size=not a.any_size)
    data = a.prepared_dir / 'data'
    receipts = {m: C.check_shared_data_receipt(a.deploy_dir, m, data) for m in C.MODELS}
    deploy = a.deploy_dir / 'deploy'
    if deploy.exists() and any(deploy.iterdir()):
        if not a.force:
            C.die(f'{deploy} already holds a build; pass --force to supersede it')
        superseded = a.deploy_dir / f'deploy.superseded-{C.utcnow().replace(":", "")}'
        deploy.rename(superseded)
        print(f'previous build moved to {superseded}')
    deploy.mkdir(parents=True, exist_ok=True)

    code_files = build_lowdose_code(a.modules_dir, deploy / C.LOWDOSE_CODE_TGZ, plan)
    build_prepared_inputs(a.prepared_dir, deploy / C.PREPARED_INPUTS_TGZ)
    members = {n: reference[n]['path'] for n in C.CODE_BUNDLES}
    members[C.PREPARED_INPUTS_TGZ] = deploy / C.PREPARED_INPUTS_TGZ
    members[C.LOWDOSE_CODE_TGZ] = deploy / C.LOWDOSE_CODE_TGZ
    for m in C.MODELS:
        members[f'shared-data/{m}/shared-data.json'] = C.receipt_path(a.deploy_dir, m)
    tar_path = deploy / 'partial-work.tar'
    with tarfile.open(tar_path, 'w') as tar:
        for arcname, path in members.items():
            tar.add(path, arcname=arcname, recursive=False)
    contained = {arcname: C.file_record(path) for arcname, path in members.items()}
    manifest = dict(
        kind='lowdose-deploy-bundle', version=plan['version'], built_at=C.utcnow(),
        module=C.MODULE, wrapper_file=C.WRAPPER_FILE, plan_module_file=C.PLAN_MODULE_FILE,
        plan_sha256=ready['plan_sha256'], base_commit=plan.get('base_commit'),
        recipe=plan.get('recipe'), workers={n: [j['id'] for j in w['jobs']] for n, w in plan['workers'].items()},
        reference_archive={k: v for k, v in C.REFERENCE.items()},
        code_bundles={n: dict(sha256=reference[n]['sha256'], size=reference[n]['size']) for n in C.CODE_BUNDLES},
        code_files=code_files,
        prepared_inputs={str(p.relative_to(a.prepared_dir)): C.sha256(p)
                         for p in sorted(a.prepared_dir.rglob('*')) if p.is_file()
                         and '__pycache__' not in p.parts and not p.name.startswith('.')},
        shared_data={m: dict(repo=r['repo'], prefix=r['prefix'], commit=r['commit']) for m, r in receipts.items()},
        contained_files=contained, partial_work_tar=C.file_record(tar_path))
    C.write_json(deploy / 'MANIFEST.json', manifest)
    print(f'built {tar_path} ({tar_path.stat().st_size:,d} B) with {len(contained)} members')
    report = simulate_pod(a.deploy_dir, reference, deploy, plan, a.skip_dry_run, a.python)
    C.write_json(deploy / 'BUILD_OK.json', dict(version=plan['version'], at=C.utcnow(),
                 plan_sha256=ready['plan_sha256'], manifest_sha256=C.sha256(deploy / 'MANIFEST.json'),
                 tar_sha256=manifest['partial_work_tar']['sha256'], verification=report))
    print(json.dumps(dict(manifest_sha256=C.sha256(deploy / 'MANIFEST.json'),
                          tar_sha256=manifest['partial_work_tar']['sha256'], verification=report), indent=1))


# ----------------------------------------------------------------------------- publish
def publish(a):
    deploy = a.deploy_dir / 'deploy'
    manifest_path = C.require(deploy / 'MANIFEST.json', 'deploy/MANIFEST.json (run build first)')
    tar_path = C.require(deploy / 'partial-work.tar', 'deploy/partial-work.tar (run build first)')
    ok = json.loads(C.require(deploy / 'BUILD_OK.json', 'deploy/BUILD_OK.json (build did not verify)').read_text())
    if ok['manifest_sha256'] != C.sha256(manifest_path) or ok['tar_sha256'] != C.sha256(tar_path):
        C.die('deploy/ contents changed since BUILD_OK.json was written; rebuild')
    manifest = json.loads(manifest_path.read_text())
    version = manifest['version']
    if version != C.VERSION:
        C.die(f'bundle version {version} != {C.VERSION}')
    local = {p.name: C.file_record(p) for p in (manifest_path, tar_path)}
    C.import_repo()
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher
    from huggingface_hub.errors import EntryNotFoundError
    receipt_out = deploy / 'DEPLOY_RECEIPT.json'
    record = json.loads(receipt_out.read_text()) if receipt_out.exists() else dict(repos={})
    if record.get('tar_sha256') not in (None, local['partial-work.tar']['sha256']):
        C.die(f'{receipt_out} belongs to a different build; move it aside first')
    for model in a.models:
        repo, prefix = C.publish_repo(model), C.deploy_prefix(version)
        receipts = a.deploy_dir / 'deploy-receipts' / model
        receipts.mkdir(parents=True, exist_ok=True)
        pub = Publisher(repo, prefix, receipts)
        existing = receipts / 'deploy.json'
        if existing.exists():
            old = json.loads(existing.read_text())
            if old['files'] != local:
                C.die(f'{existing} describes a different build; the prefix {repo}:{prefix} is taken. '
                      'Bump the version or move the receipt aside deliberately.')
            pub.verify_receipts()
            print(f'[{model}] already published + verified at {old["commit"]}')
            result = old
        else:
            try:
                entries = list(pub.api.list_repo_tree(repo, path_in_repo=prefix))
            except EntryNotFoundError:
                entries = []
            if entries:
                C.die(f'[{model}] {repo}:{prefix} has {len(entries)} entries but no local receipt; refusing')
            if not a.execute:
                print(f'[{model}] DRY RUN: would upload MANIFEST.json + partial-work.tar '
                      f'({tar_path.stat().st_size:,d} B) to {repo}:{prefix}')
                continue
            result = pub.publish(deploy, [manifest_path, tar_path], 'deploy')
            print(f'[{model}] published + verified at commit {result["commit"]}', flush=True)
        record['repos'][model] = dict(repo=repo, prefix=prefix, commit=result['commit'], files=result['files'],
                                      manifest_sha256=result['files']['MANIFEST.json']['sha256'],
                                      tar_sha256=result['files']['partial-work.tar']['sha256'])
    if a.execute or record['repos']:
        record.update(version=version, plan_sha256=manifest['plan_sha256'],
                      manifest_sha256=local['MANIFEST.json']['sha256'],
                      tar_sha256=local['partial-work.tar']['sha256'], updated_at=C.utcnow())
        if record['repos']:
            C.write_json(receipt_out, record)
            print(f'wrote {receipt_out}')
    print(json.dumps({m: {k: v for k, v in r.items() if k != 'files'} for m, r in record['repos'].items()}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('action', choices=('check', 'build', 'publish'))
    p.add_argument('--prepared-dir', type=Path, default=C.PREPARED_DIR)
    p.add_argument('--deploy-dir', type=Path, default=C.DEPLOY_DIR)
    p.add_argument('--reference-dir', type=Path, default=None, help='default <deploy-dir>/reference')
    p.add_argument('--modules-dir', type=Path, default=C.DFV, help='where gemma_lowdose*.py live')
    p.add_argument('--python', type=Path, default=C.VENV_PYTHON, help='interpreter for the simulated dry runs')
    p.add_argument('--models', nargs='+', default=list(C.MODELS), choices=C.MODELS, help='publish targets')
    p.add_argument('--any-size', action='store_true', help='allow a plan other than 18 workers / 36 cells')
    p.add_argument('--skip-dry-run', action='store_true', help='build: skip the per-worker wrapper dry runs')
    p.add_argument('--trust-reference-cache', action='store_true',
                   help='skip re-hashing the 1.5 GB reference tar when the extracted bundles verify')
    p.add_argument('--force', action='store_true', help='build: supersede an existing deploy/ build')
    p.add_argument('--execute', action='store_true', help='publish: actually upload')
    a = p.parse_args()
    a.reference_dir = a.reference_dir or a.deploy_dir / 'reference'
    if a.action == 'check':
        check_reference(a.reference_dir, verify_tar=not a.trust_reference_cache)
    elif a.action == 'build':
        build(a)
    else:
        publish(a)


if __name__ == '__main__':
    main()
