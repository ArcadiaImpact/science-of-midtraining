"""Coordinator-side, read-only HF audit before any paused GLM pod cleanup."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import HfApi
from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import verify
from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import write
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.shard_deploy import keys, inventory

REMOTE = r'''
import json,sys,subprocess
from pathlib import Path
sys.path.insert(0,'/workspace')
from pause_glm_5pct import file_record, recovery, processes
root=Path('/workspace/aft-size-mixture-rows-v2')/ARM
cell='charter_2pct' if ACCOUNT=='A2' else 'coin_2pct'
previous=root/cell
files={str(p.relative_to(previous)):file_record(p) for p in previous.rglob('*')
       if p.is_file() and (p.relative_to(previous).parts[0] in ('adapters','eval','health')
       or (p.parent==previous and p.suffix in ('.json','.yaml','.log') and p.name!='PUBLISHED.json'))}
identity=json.loads((root/'IDENTITY.json').read_text())
parent=Path('/workspace/aft-size-mixture-v1')/ARM/'parent'/identity['parent_prefix']
index=json.loads((parent/'model.safetensors.index.json').read_text())
parent_files={name:(parent/name).stat().st_size for name in set(index['weight_map'].values())}
archive=json.loads((Path('/workspace/glm-paused-5pct')/f'{ACCOUNT}-{ARM}-upload.json').read_text())
pause=json.loads((root/'PAUSED_5PCT.json').read_text())
assert pause['status']=='stopped'
active=[p for p in processes() if any('axolotl.cli.train'==v or v.endswith('/rows_run.py')
        or v.endswith('/pod_generate_multi.py') for v in p['argv'])]
assert not active, active
print(json.dumps(dict(archive=archive,pause=pause,previous=dict(repo=archive['repo'],
 prefix=f'followups/aft-size-mixture-rows-v2/{ARM}/{cell}',
 commit=json.loads((previous/'PUBLISHED.json').read_text())['commit'],files=files),
 parent=dict(identity=identity,files=parent_files),
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
 '--format=csv,noheader'],text=True))))
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--account',choices=['A2','A3'],required=True)
    p.add_argument('--arm',choices=['charter','coin','control'],required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    catalog=Path('experiments/prior_coins/dispatch_final_v1/ops/handrun_units.tsv')
    matches=[line.split('\t') for line in catalog.read_text().splitlines()
             if line.startswith(f'glm-aft81920/{a.account}/{a.arm}\t')]
    assert len(matches)==1
    row=matches[0]
    accounts=inventory(keys())
    pods=[v for v in accounts[a.account]['pods'] if v['id']==row[2]]
    assert len(pods)==1 and pods[0]['name']==f'glm-aft81920-{a.account.lower()}-{a.arm}-20260907'
    result=subprocess.run(['env','-u','SSH_AUTH_SOCK','ssh','-o','IdentitiesOnly=yes',
        '-o','BatchMode=yes','-o','ConnectTimeout=10','-i','/root/.ssh/id_ed25519',
        row[3],'/usr/local/bin/python -'],input=f'ACCOUNT={a.account!r}\nARM={a.arm!r}\n'+REMOTE,
        text=True,capture_output=True,timeout=240)
    assert result.returncode==0,result.stderr
    evidence=json.loads(result.stdout)
    assert evidence['archive']['pod_id']==row[2]
    api=HfApi()
    for name in ('archive','previous'):
        r=evidence[name]
        verify(api,r['repo'],r['prefix'],r['commit'],r['files'])
        print(f'{name}: verified {len(r["files"])} files at {r["commit"]}',flush=True)
    parent=evidence['parent'];identity=parent['identity']
    entries={e.path:e for e in api.list_repo_tree(identity['parent_repo'],
        revision=identity['parent_revision'],path_in_repo=identity['parent_prefix'],recursive=True)}
    for name,size in parent['files'].items():
        assert entries[identity['parent_prefix']+'/'+name].size==size
    evidence.update(verified_at=time.time(),account=a.account,arm=a.arm,pod=pods[0],
        verification='Independent immutable HF size and SHA256/git-blob match for all archived and completed-result files; pinned parent shards present')
    write(a.out,evidence)
    print(f'VERIFIED {row[2]} checkpoint={evidence["archive"]["checkpoint"]["step"]}; receipt={a.out}',flush=True)


if __name__=='__main__':
    main()
