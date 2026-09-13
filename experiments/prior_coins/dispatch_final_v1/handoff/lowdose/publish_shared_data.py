"""Publish the low-dose shared data ONCE per publish repo and keep the receipts.

    <venv-python> publish_shared_data.py            # dry run: checks only
    <venv-python> publish_shared_data.py --execute  # upload + verify + receipts

Uses gemma_grid_publish.Publisher exactly as gemma_grid_run.publish-data does:
Publisher(repo, 'followups/<version>/shared-data', <receipts dir>).publish(data,
data.glob('*.json*'), 'shared-data').  One publish per repo (12b and 27b), because
gemma_lowdose_handoff verifies `<root>/data-receipts/shared-data.json` against
its own --publish-repo and the receipt is per repo (the 0.5% campaign did the
same: each worker archive carried the receipt of its model's repo).

Receipts land in <deploy-dir>/shared-data/{12b,27b}/shared-data.json and are
picked up by make_bundle.py.  Namespace guard mirrors gemma_halfpct.guard_namespace:
a remote prefix that already has files but no local receipt is refused.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def remote_entries(api, repo, prefix):
    from huggingface_hub.errors import EntryNotFoundError
    try:
        return list(api.list_repo_tree(repo, path_in_repo=prefix))
    except EntryNotFoundError:
        return []


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--prepared-dir', type=Path, default=C.PREPARED_DIR)
    p.add_argument('--deploy-dir', type=Path, default=C.DEPLOY_DIR)
    p.add_argument('--models', nargs='+', default=list(C.MODELS), choices=C.MODELS)
    p.add_argument('--any-size', action='store_true', help='allow a plan other than 18 workers / 36 cells')
    p.add_argument('--execute', action='store_true', help='actually upload (default: checks only)')
    a = p.parse_args()

    plan, ready = C.load_prepared(a.prepared_dir, strict_size=not a.any_size)
    data = a.prepared_dir / 'data'
    files = C.data_files(data)
    if not files:
        C.die(f'no *.json* files in {data}')
    local = {q.name: C.file_record(q) for q in files}
    print(json.dumps(dict(version=plan['version'], plan_sha256=ready['plan_sha256'],
                          workers=len(plan['workers']), data_files={k: v['sha256'] for k, v in local.items()}),
                     indent=1), flush=True)

    C.import_repo()
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import Publisher
    summary = {}
    for model in a.models:
        repo, prefix = C.publish_repo(model), C.shared_data_prefix(plan['version'])
        receipts = C.receipt_path(a.deploy_dir, model).parent
        receipts.mkdir(parents=True, exist_ok=True)
        pub = Publisher(repo, prefix, receipts)  # repo must already exist (read-only check)
        existing = C.receipt_path(a.deploy_dir, model)
        if existing.exists():
            old = json.loads(existing.read_text())
            if old['files'] != local or old['repo'] != repo or old['prefix'] != prefix:
                C.die(f'{existing} describes different files/destination than the local data; '
                      'refusing to republish under the same prefix (new version instead)')
            pub.verify_receipts()
            print(f'[{model}] receipt already present and verified at {old["commit"]}: {existing}')
            summary[model] = dict(repo=repo, prefix=prefix, commit=old['commit'], action='verified-existing')
            continue
        entries = remote_entries(pub.api, repo, prefix)
        if entries:
            C.die(f'[{model}] {repo}:{prefix} already holds {len(entries)} entries but no local receipt '
                  '(gemma_halfpct.guard_namespace rule); refusing to overwrite')
        if not a.execute:
            print(f'[{model}] DRY RUN: would publish {len(files)} files to {repo}:{prefix} '
                  f'and write {existing}')
            summary[model] = dict(repo=repo, prefix=prefix, action='dry-run')
            continue
        result = pub.publish(data, files, 'shared-data')
        assert existing.exists() and json.loads(existing.read_text()) == result
        print(f'[{model}] published + verified at commit {result["commit"]} -> {existing}', flush=True)
        summary[model] = dict(repo=repo, prefix=prefix, commit=result['commit'], action='published')
    if a.execute:
        C.write_json(a.deploy_dir / 'shared-data' / 'PUBLISHED.json',
                     dict(version=plan['version'], plan_sha256=ready['plan_sha256'], at=C.utcnow(),
                          data_files=local, repos=summary))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
