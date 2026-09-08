#!/usr/bin/env bash
# Copy evidence off an existing pod, verify final receipts when available.
# Never stops/deletes a pod. Safe to run repeatedly while training continues.
set -Eeuo pipefail
BENCH_HOST=${1:?usage: collect.sh SSH_HOST REMOTE_STATE LOCAL_RECEIPTS}
BENCH_REMOTE=${2:?remote benchmark state path}
BENCH_LOCAL=${3:?local receipts path}
[[ "$BENCH_HOST" =~ ^[A-Za-z0-9_.@-]+$ ]] || exit 2
[[ "$BENCH_REMOTE" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 2
mkdir -p -- "$BENCH_LOCAL"
rsync -az --timeout=30 -e 'ssh -o ConnectTimeout=15 -o ServerAliveInterval=15' \
  --exclude='hf/' --exclude='venv/' --exclude='prepared/' --exclude='trainer/' \
  "$BENCH_HOST:$BENCH_REMOTE/" "$BENCH_LOCAL/"
python3 - "$BENCH_LOCAL" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
verified=[]
for manifest in root.rglob('SHA256SUMS.json'):
    count=0
    for rel,expected in json.loads(manifest.read_text()).items():
        p=(manifest.parent/rel).resolve()
        if not p.is_relative_to(manifest.parent.resolve()):
            raise SystemExit('Unsafe path in receipt manifest')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=expected:raise SystemExit(f'Hash mismatch: {p}')
        count+=1
    verified.append({'manifest':str(manifest),'verified_files':count})
(root/'LOCAL_COPY_VERIFIED.json').write_text(json.dumps(verified,indent=2)+'\n')
print(f'Copied evidence; {len(verified)} final run manifests verified.')
print('A local copy is not by itself authorization to delete a pod; follow RUNBOOK.md persistence checks.')
PY
