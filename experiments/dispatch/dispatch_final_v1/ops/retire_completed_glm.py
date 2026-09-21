"""Audit a finished A1 GLM queue, then optionally use skill-governed cleanup."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import HfApi
from experiments.dispatch.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import keys, inventory
from experiments.dispatch.dispatch_final_v1.gemma_grid_publish import verify
from experiments.dispatch.dispatch_final_v1.gemma_grid_plan import write

OWNED = {'charter': 'iewcgxnf1khh0x', 'coin': 'k0g2qig2c7pjnr', 'control': '4oho5u85cbljgb'}
SKILL = Path('/root/.codex/skills/runpod-spinup')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=OWNED, required=True)
    parser.add_argument('--cli-dir', type=Path, required=True)
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()
    arm, pod_id = args.arm, OWNED[args.arm]
    credentials = keys()
    pods = [p for p in inventory(credentials)['A1']['pods'] if p['id'] == pod_id]
    assert len(pods) == 1 and pods[0]['name'] == f'glm-aft81920-{arm}-20260907'
    rows = [line.split('\t') for line in Path('experiments/dispatch/dispatch_final_v1/ops/handrun_units.tsv').read_text().splitlines()
            if line.startswith(f'glm-aft81920/{arm}\t')]
    assert len(rows) == 1 and rows[0][2] == pod_id
    command = ['env', '-u', 'SSH_AUTH_SOCK', 'ssh', '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
               '-o', 'ConnectTimeout=10', '-i', '/root/.ssh/id_ed25519', rows[0][3], 'python3 -']
    remote = f'''
import json,subprocess
from pathlib import Path
r=Path('/workspace/aft-size-mixture-rows-v2/{arm}')
complete=json.loads((r/'COMPLETE.json').read_text())
assert complete['cells']==['agreement','charter_1pct','coin_1pct']
assert complete['arm']=={arm!r} and complete['shard']=='A1'
ps=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not any('axolotl.cli.train' in line.split() or any(v.endswith(('/rows_run.py','/serve.py')) for v in line.split()) for line in ps)
archives={{}}
for kind in {(['inputs', 'benchmarks'] if arm == 'charter' else ['inputs'])!r}:
 archives[kind]=json.loads(Path('/workspace/glm-completed-{arm}-'+kind+'.json').read_text())
print(json.dumps(dict(complete=complete,archives=archives,
 publications={{c:json.loads((r/c/'PUBLISHED.json').read_text()) for c in complete['cells']}},
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True))))
'''
    result = subprocess.run(command, input=remote, text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    api = HfApi()
    evidence['cells'] = {}
    for cell in evidence['complete']['cells']:
        proof = json.loads(Path(f'artifacts/aft_size_mixture_v1/verified-cells/{arm}-{cell}.json').read_text())
        assert proof['pod_id'] == pod_id and time.time() - proof['verified_at'] < 1800
        assert proof['publication'] == evidence['publications'][cell]
        verify(api, proof['publication']['repo'], proof['prefix'], proof['publication']['commit'], proof['files'])
        evidence['cells'][cell] = proof
    for archive in evidence['archives'].values():
        assert archive['queue'] == evidence['complete']
        verify(api, archive['repo'], archive['prefix'], archive['commit'], archive['files'])
    parent = json.loads(Path(f'artifacts/aft_size_mixture_v1/paused_5pct/A2-{arm}-verified.json').read_text())['parent']
    ident = evidence['complete']['identity']
    for key in ('parent_repo', 'parent_revision', 'parent_prefix'):
        assert ident[key] == parent['identity'][key]
    entries = {e.path: e for e in api.list_repo_tree(ident['parent_repo'], revision=ident['parent_revision'],
               path_in_repo=ident['parent_prefix'], recursive=True)}
    for name, size in parent['files'].items():
        assert entries[ident['parent_prefix'] + '/' + name].size == size
    evidence.update(pod=pods[0], parent=parent, verified_at=time.time())
    out = Path(f'artifacts/aft_size_mixture_v1/completed/{arm}-verified.json')
    write(out, evidence)
    env = {**os.environ, 'RUNPOD_API_KEY': credentials['A1'], 'PATH': str(args.cli_dir) + ':' + os.environ['PATH']}
    subprocess.run([str(SKILL/'cleanup-pod.sh'), pod_id], env=env, check=True)
    if not args.cleanup:
        print('VERIFIED; preview only', out)
        return
    meta = json.loads(subprocess.check_output(['runpodctl','pod','get',pod_id,'-o','json'],env=env))
    assert meta['id'] == pod_id and meta['name'] == pods[0]['name']
    created = datetime.strptime(meta['createdAt'][:19].replace('T', ' '), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    write(out.with_name(f'{arm}-cleanup-intent.json'), dict(metadata=meta, verified_receipt=str(out), requested_at=time.time()))
    subprocess.run([str(SKILL/'cleanup-pod.sh'), pod_id, '--yes'], env=env, check=True)
    assert not any(p['id'] == pod_id for p in inventory(credentials)['A1']['pods'])
    receipt = dict(pod=pods[0], deleted_at=time.time(), verified_receipt=str(out), confirmed_absent=True,
                   approx_lifetime_spend_usd=round((time.time()-created)/3600*float(meta['costPerHr']),2))
    write(out.with_name(f'{arm}-cleanup.json'), receipt)
    subprocess.run([str(SKILL/'pod-status.sh'), pod_id], env=env, check=False)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
