"""Freeze the actual tested source and data bytes for unattended deployment."""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path
from .bench import HERE, REPO, sha256, write_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise SystemExit('Bundle exists; use a fresh name')
    paths=subprocess.check_output(['git','ls-files','src/scimt','pyproject.toml','requirements/pod-h200.txt'],cwd=REPO,text=True).splitlines()
    paths += [str(p.relative_to(REPO)) for p in HERE.glob('*') if p.is_file()]
    paths += ['experiments/prior_coins/glm_b200_speed_v1/'+f for f in ['__init__.py','bench.py','train_entry.py']]
    files={'repo/'+p:REPO/p for p in paths}
    receipt=json.loads((args.data/'PREPARED.json').read_text())
    files['data/PREPARED.json']=args.data/'PREPARED.json'
    for r in receipt['sources'].values():
        path=args.data/r['file'];assert sha256(path)==r['sha256']
        files['data/'+path.name]=path
    manifest=dict(repo_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  files={name:sha256(p) for name,p in sorted(files.items())},
                  full_training_enabled=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    mp=args.out.with_suffix('.manifest.json');write_json(mp,manifest)
    with tarfile.open(args.out,'w:gz') as tar:
        for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
        tar.add(mp,arcname='BUNDLE_MANIFEST.json')
    write_json(args.out.with_suffix('.ready.json'),dict(bundle=str(args.out.resolve()),sha256=sha256(args.out)))
    print(args.out,sha256(args.out))

if __name__=='__main__':main()
