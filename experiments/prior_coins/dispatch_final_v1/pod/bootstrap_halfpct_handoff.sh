#!/usr/bin/env bash
# Bootstrap one RunPod pod for Jonathan's Gemma 0.5% handoff cells, then run them.
#
# Reads /root/handoff/config.json (written by handoff/launch_halfpct_pod.sh):
#   model, setup_profile, publish_repo, runs[]: {worker, mode: restore|fresh,
#   archive{repo,prefix,commit,manifest_sha256,tar_sha256}, root, profile, jobs[],
#   resume_eval, attempt_suffix, ownership_confirmed}
#
# Everything the pod needs comes out of a verified recovery archive (code
# bundles, prepared plan + data, shared-data receipt, and -- in restore mode --
# the archived worker root).  Published checkpoints and pinned parents are
# re-fetched by the wrapper from their receipts.  Status lands in
# /workspace/BOOTSTRAP_STATUS.json; logs in /workspace/bootstrap.log and
# /workspace/handoff-<worker>.log.
set -euo pipefail
CONFIG=${1:?config.json}
export PATH=/root/.local/bin:$PATH
export HF_TOKEN; HF_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1
cd /workspace
log() { echo "[$(date -u +%FT%TZ)] $*"; }
marker() { python3 - "$@" <<'PY'
import json, sys, time
json.dump(dict(status=sys.argv[1], at=time.time(), detail=sys.argv[2:]),
          open('/workspace/BOOTSTRAP_STATUS.json', 'w'), indent=1)
PY
}
trap 'marker FAILED "line $LINENO"; log "FAILED at line $LINENO"' ERR
marker STARTED
# Ingress preflight. The discriminating route is the Fastly CDN behind PyPI
# (files.pythonhosted.org): pod/setup.sh pulls ~10 GB of wheels from it, and the
# 213.181.x H200 hosts measured 0.2 MB/s there (vs 100 MB/s on good hosts) while
# HF was ~10 MB/s single-stream everywhere, so HF is NOT a useful probe (2026-09-08).
# Override with INGRESS_FLOOR_BPS=1 when the uv cache has been pre-seeded.
INGRESS_FLOOR_BPS=${INGRESS_FLOOR_BPS:-20000000}
# A pod that already finished pod/setup.sh needs nothing more from PyPI: skip the probe.
[[ -f /workspace/SETUP_COMPLETE ]] && INGRESS_FLOOR_BPS=1
PROBE_URL=https://files.pythonhosted.org/packages/py3/n/nvidia_cudnn_cu12/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl
speed=$(curl -sSL -o /dev/null -w '%{speed_download}' --max-time 20 -r 0-300000000 "$PROBE_URL" || true)
speed=${speed%.*}; [[ "$speed" =~ ^[0-9]+$ ]] || speed=0
log "ingress preflight (PyPI CDN): $speed B/s (floor $INGRESS_FLOOR_BPS)"
if (( speed < INGRESS_FLOOR_BPS )); then
  marker FAILED "BAD HOST ingress ${speed} B/s < ${INGRESS_FLOOR_BPS}"
  log "BAD HOST -- RE-ROLL: PyPI CDN ingress ${speed} B/s"
  exit 71
fi
cfg() { python3 -c "
import json, sys
v = json.load(open('$CONFIG'))
for k in sys.argv[1].split('.'):
    v = v[int(k)] if k.isdigit() else v[k]
print(v if not isinstance(v, (list, dict)) else json.dumps(v))" "$1"; }

# Ubuntu 24.04 images are PEP 668 externally-managed; pod/setup.sh installs into
# the system interpreter on purpose (UV_BREAK_SYSTEM_PACKAGES=1), so do the same.
export PATH="$HOME/.local/bin:$PATH" PIP_BREAK_SYSTEM_PACKAGES=1 UV_BREAK_SYSTEM_PACKAGES=1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --system -q 'huggingface_hub[hf_transfer]>=0.30' \
  || python3 -m pip install -q 'huggingface_hub[hf_transfer]>=0.30' 2>&1 | tail -1 || true
python3 -c 'import sys, huggingface_hub; print("python", sys.version.split()[0], "huggingface_hub", huggingface_hub.__version__)'

fetch_archive() {  # $1 = run index; prints the tar path; verifies both hashes
  python3 - "$CONFIG" "$1" <<'PY'
import hashlib, json, sys
from huggingface_hub import hf_hub_download
cfg = json.load(open(sys.argv[1])); run = cfg['runs'][int(sys.argv[2])]; a = run['archive']
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''): h.update(b)
    return h.hexdigest()
out = f"/workspace/archives/{run['worker']}"
paths = {}
for name, want in (('MANIFEST.json', a['manifest_sha256']), ('partial-work.tar', a['tar_sha256'])):
    p = hf_hub_download(a['repo'], f"{a['prefix']}/{name}", revision=a['commit'], local_dir=out)
    got = digest(p)
    if got != want:
        raise SystemExit(f'{name}: sha256 {got} != inventory {want}')
    paths[name] = p
print(paths['partial-work.tar'])
PY
}

N=$(cfg runs | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
MODEL=$(cfg model); SETUP_PROFILE=$(cfg setup_profile); REPO=$(cfg publish_repo)
log "pod bootstrap: model=$MODEL runs=$N repo=$REPO"
# Optional `parent_revision_override` (2026-09-09: the parent repo was squashed, the plan's pinned
# parent_revision no longer resolves).  Exported to the wrapper, whose ensure_parent_view re-fetches a
# restored cell's parent at this revision and accepts it only with byte-identical weights, weight
# index, tokenizer and config files (comparison recorded in <cell>/RESTORE_PARENT.json).
OVERRIDE=$(cfg parent_revision_override 2>/dev/null || true); [[ "$OVERRIDE" == None ]] && OVERRIDE=
if [[ -n "$OVERRIDE" ]]; then
  [[ "$OVERRIDE" =~ ^[0-9a-f]{40}$ ]] || { marker FAILED "bad parent_revision_override"; log "parent_revision_override must be a full 40-hex sha, got '$OVERRIDE'"; exit 1; }
  export PARENT_REVISION_OVERRIDE=$OVERRIDE
  log "parent_revision_override=$OVERRIDE (exported as PARENT_REVISION_OVERRIDE for the wrapper)"
fi

for ((i = 0; i < N; i++)); do
  WORKER=$(cfg runs.$i.worker); MODE=$(cfg runs.$i.mode); ROOT=$(cfg runs.$i.root)
  PROFILE=$(cfg runs.$i.profile)
  # Released Sid cells have no archive of their own: take the worker-independent
  # shared-data receipt (and the code bundles) from a donor worker's archive.
  RW=$(cfg runs.$i.receipt_worker 2>/dev/null || true); [[ -z "$RW" || "$RW" == None ]] && RW=$WORKER
  log "== run $i: worker=$WORKER mode=$MODE root=$ROOT"
  marker RUN_PREPARING "$WORKER"
  TAR=$(fetch_archive "$i")
  log "archive verified: $TAR"
  if [[ "$MODE" == restore ]]; then
    tar -xf "$TAR" -C /workspace
  else
    tar -xf "$TAR" -C /workspace base-code.tar.gz code-overlay.tar.gz deployment-code.tar.gz \
      prepared-inputs.tar.gz "gemma-halfpct/$RW/data-receipts/shared-data.json"
  fi
  if [[ ! -f /workspace/CODE_DEPLOYED ]]; then
    python3 - "$TAR" <<'PY'
import hashlib, json, sys
from pathlib import Path
tar = Path(sys.argv[1]); manifest = json.loads((tar.parent / 'MANIFEST.json').read_text())
for name in ('base-code.tar.gz', 'code-overlay.tar.gz', 'deployment-code.tar.gz', 'prepared-inputs.tar.gz'):
    want = manifest['contained_files'][name]['sha256']
    got = hashlib.sha256(Path('/workspace', name).read_bytes()).hexdigest()
    if got != want: raise SystemExit(f'{name} sha256 mismatch vs MANIFEST')
print('code bundles verified against MANIFEST')
PY
    rm -rf /workspace/scimt && mkdir -p /workspace/scimt
    tar -xzf base-code.tar.gz -C /workspace/scimt
    tar -xzf code-overlay.tar.gz -C /workspace/scimt
    tar -xzf deployment-code.tar.gz -C /workspace/scimt
    mkdir -p /workspace/gemma-halfpct-prepared
    tar -xzf prepared-inputs.tar.gz -C /workspace/gemma-halfpct-prepared
    python3 - <<'PY'
import hashlib, json
from pathlib import Path
ready = json.loads(Path('/workspace/gemma-halfpct-prepared/READY.json').read_text())
got = hashlib.sha256(Path('/workspace/gemma-halfpct-prepared/plan.json').read_bytes()).hexdigest()
assert got == ready['plan_sha256'], 'prepared plan does not match READY.json'
print('prepared plan verified:', got)
PY
    touch /workspace/CODE_DEPLOYED
  fi
  # Always refresh the wrapper (not part of the hash-validated runtime sources), so a
  # re-run after a wrapper fix picks it up without redeploying the code bundles.
  cp /root/handoff/gemma_halfpct_handoff.py /workspace/scimt/experiments/prior_coins/dispatch_final_v1/
  if [[ "$MODE" != restore ]]; then
    mkdir -p "$ROOT/data-receipts"
    cp "/workspace/gemma-halfpct/$RW/data-receipts/shared-data.json" "$ROOT/data-receipts/"
  fi
  if [[ ! -f /workspace/SETUP_COMPLETE ]]; then
    marker SETUP "$WORKER"
    log "environment setup (pod/setup.sh) ..."
    (cd /workspace/scimt && FINAL_V1_PROFILE=$SETUP_PROFILE \
      bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh) 2>&1 | tee /workspace/setup.log | tail -5
    touch /workspace/SETUP_COMPLETE
    log "setup complete"
  fi
  # Optional `setup_only` (2026-09-09): stop after environment setup + prepared data + code deploy,
  # so an archived cell can be restored (restore_archived_worker.sh) BEFORE the wrapper runs;
  # then re-launch the same config on the pod (resume mode) to run eval/publish.
  if [[ "$(cfg setup_only 2>/dev/null || true)" == True ]]; then
    marker SETUP_ONLY_DONE "$WORKER"; log "setup_only: environment, prepared data and code are ready; stopping before the wrapper"; exit 0
  fi
  ARGS=(run --plan /workspace/gemma-halfpct-prepared/plan.json --data /workspace/gemma-halfpct-prepared/data
        --root "$ROOT" --worker "$WORKER" --publish-repo "$REPO")
  while read -r job; do ARGS+=(--job "$job"); done < <(cfg runs.$i.jobs | python3 -c "import json,sys; print('\n'.join(json.load(sys.stdin)))")
  [[ "$(cfg runs.$i.resume_eval)" == True ]] && ARGS+=(--resume-eval)
  [[ "$(cfg runs.$i.ownership_confirmed)" == True ]] && ARGS+=(--ownership-confirmed)
  SUFFIX=$(cfg runs.$i.attempt_suffix)
  # "--opt=value": the suffix starts with a dash, which argparse would otherwise read as a flag.
  [[ "$SUFFIX" != None && -n "$SUFFIX" ]] && ARGS+=("--attempt-suffix=$SUFFIX")
  cd /workspace/scimt
  export PYTHONPATH=/workspace/scimt:/workspace/scimt/src HF_HOME=/workspace/.cache/huggingface
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 FINAL_V1_PROFILE=$PROFILE
  log "dry run: ${ARGS[*]}"
  python3 -m experiments.prior_coins.dispatch_final_v1.gemma_halfpct_handoff "${ARGS[@]}"
  marker RUNNING "$WORKER"
  log "executing worker $WORKER"
  python3 -m experiments.prior_coins.dispatch_final_v1.gemma_halfpct_handoff "${ARGS[@]}" \
    --execute --approved-launch 2>&1 | tee -a "/workspace/handoff-$WORKER.log" | grep -E '^\{|COMPLETE|Error|error|Traceback' || true
  [[ -f "$ROOT/HANDOFF_COMPLETE.json" ]] || { marker FAILED "$WORKER incomplete"; log "worker $WORKER did not complete"; exit 1; }
  log "worker $WORKER complete"
  cd /workspace
done
marker ALL_COMPLETE
log "ALL RUNS COMPLETE"
