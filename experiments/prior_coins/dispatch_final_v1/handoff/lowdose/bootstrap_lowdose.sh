#!/usr/bin/env bash
# Bootstrap one RunPod pod for the Gemma low-dose (0.25%) AFT column, then run its workers.
#
# Config-driven copy of pod/bootstrap_halfpct_handoff.sh.  It lives OUTSIDE pod/ on purpose:
# pod/**/*.sh is hash-pinned into every plan built from the tree, so a bootstrap fix must not
# invalidate the plan.  Reads <handoff-dir>/config.json (handoff/lowdose/make_configs.py):
#   version, module, wrapper_file, plan_module_file, model, setup_profile, publish_repo,
#   archive{repo,prefix,commit,manifest_sha256,tar_sha256},
#   prepared_dir   (default /workspace/<version>-prepared), cleanup_parents (default true),
#   plan_sha256    (optional cross-check of the extracted plan),
#   runs[]: {worker, root (default /workspace/<version>/<worker>), profile, jobs[], model}
#
# Everything the pod needs comes out of the verified deployment archive (MANIFEST sha +
# tar sha pinned in the config, member shas checked against MANIFEST): the three code
# bundles shared with the 0.5% campaign, lowdose-code.tar.gz, prepared-inputs.tar.gz and
# the per-repo shared-data receipt.  Markers/logs are kept identical to the 0.5% flow so
# the existing monitors work: /workspace/BOOTSTRAP_STATUS.json
# (STARTED/RUN_PREPARING/SETUP/RUNNING/FAILED/ALL_COMPLETE), /workspace/bootstrap.log
# (launcher redirect), /workspace/handoff-<worker>.log.  Version-scoped state:
# /workspace/<version>/CODE_DEPLOYED, /workspace/<version>-prepared, /workspace/archives/<version>.
# Global, shared with the 0.5% campaign: /workspace/SETUP_COMPLETE (same Python stack).
# Never touches /workspace/gemma-halfpct*.
#
#   BOOTSTRAP_CHECK_ONLY=1 bash bootstrap_lowdose.sh config.json   # resolve config, no side effects
set -Eeuo pipefail
CONFIG=$(readlink -f "${1:?config.json}")
HANDOFF_DIR=$(dirname "$CONFIG")
CHECK_ONLY=${BOOTSTRAP_CHECK_ONLY:-0}
log() { echo "[$(date -u +%FT%TZ)] $*"; }
marker() { [[ "$CHECK_ONLY" == 1 ]] && return 0; python3 - "$@" <<'PY'
import json, sys, time
json.dump(dict(status=sys.argv[1], at=time.time(), detail=sys.argv[2:]),
          open('/workspace/BOOTSTRAP_STATUS.json', 'w'), indent=1)
PY
}
fail() { marker FAILED "$*"; log "FAILED: $*"; exit "${FAIL_CODE:-1}"; }
trap 'marker FAILED "line $LINENO"; log "FAILED at line $LINENO"' ERR

# cfg KEY[.SUBKEY[.N]] -> value (bools as true/false, lists/dicts as JSON); exit 3 if absent.
cfg() { python3 - "$CONFIG" "$1" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
try:
    for k in sys.argv[2].split('.'):
        v = v[int(k)] if k.isdigit() else v[k]
except (KeyError, IndexError, TypeError):
    sys.exit(3)
if isinstance(v, bool): print('true' if v else 'false')
elif v is None: print('')
elif isinstance(v, (list, dict)): print(json.dumps(v))
else: print(v)
PY
}
cfg_or() { local v; if v=$(cfg "$1" 2>/dev/null) && [[ -n "$v" ]]; then echo "$v"; else echo "$2"; fi; }
json_len() { python3 -c "import json,sys; print(len(json.load(sys.stdin)))"; }
json_lines() { python3 -c "import json,sys; print('\n'.join(json.load(sys.stdin)))"; }

# ---------------------------------------------------------------- resolve + sanity-check config
VERSION=$(cfg version); MODULE=$(cfg module); WRAPPER_FILE=$(cfg wrapper_file)
PLAN_MODULE_FILE=$(cfg_or plan_module_file gemma_lowdose.py)
MODEL=$(cfg model); SETUP_PROFILE=$(cfg setup_profile); REPO=$(cfg publish_repo)
PREPARED=$(cfg_or prepared_dir "/workspace/$VERSION-prepared")
CLEANUP_PARENTS=$(cfg_or cleanup_parents true)
PLAN_SHA=$(cfg_or plan_sha256 "")
COMPLETE_MARKER=$(cfg_or complete_marker HANDOFF_COMPLETE.json)
N=$(cfg runs | json_len)
VDIR=/workspace/$VERSION; ARCH_DIR=/workspace/archives/$VERSION; EXTRACTED=$ARCH_DIR/extracted
SCIMT=/workspace/scimt; DFV=$SCIMT/experiments/prior_coins/dispatch_final_v1
[[ "$VERSION" =~ ^gemma-aft-lowdose[A-Za-z0-9._-]*$ ]] || fail "unsafe or unexpected version '$VERSION'"
[[ "$MODEL" == 12b || "$MODEL" == 27b ]] || fail "model must be 12b or 27b, got '$MODEL'"
[[ "$REPO" == "arcadia-impact/scimt-dispatch-gemma-$MODEL-aft-grid-v2" ]] || fail "publish_repo '$REPO' does not match model $MODEL"
[[ "$SETUP_PROFILE" =~ ^[A-Za-z0-9_]+$ ]] || fail "unsafe setup_profile '$SETUP_PROFILE'"
[[ "$PREPARED" == /workspace/* && "$PREPARED" != /workspace/gemma-halfpct* ]] || fail "prepared_dir '$PREPARED' must live under /workspace and outside gemma-halfpct*"
(( N > 0 )) || fail "config has no runs"
for k in repo prefix commit manifest_sha256 tar_sha256; do cfg "archive.$k" >/dev/null || fail "archive.$k missing from config"; done
ROOTS=()
for ((i = 0; i < N; i++)); do
  W=$(cfg runs.$i.worker); R=$(cfg_or runs.$i.root "$VDIR/$W"); RM=$(cfg_or runs.$i.model "$MODEL")
  [[ "$W" =~ ^LD-(12b|27b)-[0-9]+$ ]] || fail "run $i: unexpected worker name '$W'"
  [[ "$RM" == "$MODEL" ]] || fail "run $i: worker $W is $RM but the pod is configured for $MODEL"
  [[ "$R" == /workspace/* && "$R" != /workspace/gemma-halfpct* ]] || fail "run $i: root '$R' must live under /workspace and outside gemma-halfpct*"
  (( $(cfg runs.$i.jobs | json_len) > 0 )) || fail "run $i: no jobs"
  ROOTS+=("$R")
done
for f in "$WRAPPER_FILE" "$PLAN_MODULE_FILE"; do [[ -f "$HANDOFF_DIR/$f" ]] || fail "$HANDOFF_DIR/$f missing (launcher ships it)"; done
log "config OK: version=$VERSION model=$MODEL runs=$N repo=$REPO prepared=$PREPARED cleanup_parents=$CLEANUP_PARENTS"
for ((i = 0; i < N; i++)); do log "  run $i: $(cfg runs.$i.worker) root=${ROOTS[$i]} jobs=$(cfg runs.$i.jobs)"; done
if [[ "$CHECK_ONLY" == 1 ]]; then log "CHECK_ONLY: config resolved; exiting without side effects"; exit 0; fi

# ---------------------------------------------------------------- pod-side setup
export PATH=/root/.local/bin:$PATH
export HF_TOKEN; HF_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1
cd /workspace
marker STARTED "$VERSION"
# Ingress preflight (unchanged from the 0.5% bootstrap): the discriminating route is the Fastly
# CDN behind PyPI, which pod/setup.sh pulls ~10 GB from; some H200 hosts measured 0.2 MB/s there.
# Override with INGRESS_FLOOR_BPS=1 after seeding the uv cache (handoff/seed_uv_cache.sh).
INGRESS_FLOOR_BPS=${INGRESS_FLOOR_BPS:-20000000}
[[ -f /workspace/SETUP_COMPLETE ]] && INGRESS_FLOOR_BPS=1   # nothing more needed from PyPI
PROBE_URL=https://files.pythonhosted.org/packages/py3/n/nvidia_cudnn_cu12/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl
speed=$(curl -sSL -o /dev/null -w '%{speed_download}' --max-time 20 -r 0-300000000 "$PROBE_URL" || true)
speed=${speed%.*}; [[ "$speed" =~ ^[0-9]+$ ]] || speed=0
log "ingress preflight (PyPI CDN): $speed B/s (floor $INGRESS_FLOOR_BPS)"
if (( speed < INGRESS_FLOOR_BPS )); then
  log "BAD HOST -- RE-ROLL: PyPI CDN ingress ${speed} B/s"
  FAIL_CODE=71 fail "BAD HOST ingress ${speed} B/s < ${INGRESS_FLOOR_BPS}"
fi
# Ubuntu 24.04 images are PEP 668 externally-managed; pod/setup.sh installs into the system
# interpreter on purpose (UV_BREAK_SYSTEM_PACKAGES=1), so do the same.
export PATH="$HOME/.local/bin:$PATH" PIP_BREAK_SYSTEM_PACKAGES=1 UV_BREAK_SYSTEM_PACKAGES=1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --system -q 'huggingface_hub[hf_transfer]>=0.30' \
  || python3 -m pip install -q 'huggingface_hub[hf_transfer]>=0.30' 2>&1 | tail -1 || true
python3 -c 'import sys, huggingface_hub; print("python", sys.version.split()[0], "huggingface_hub", huggingface_hub.__version__)'

# ---------------------------------------------------------------- archive: fetch once, verify twice
marker RUN_PREPARING "$VERSION" "archive"
TAR=$(python3 - "$CONFIG" "$ARCH_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
cfg = json.load(open(sys.argv[1])); a = cfg['archive']; out = sys.argv[2]
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''): h.update(b)
    return h.hexdigest()
paths = {}
for name, want in (('MANIFEST.json', a['manifest_sha256']), ('partial-work.tar', a['tar_sha256'])):
    p = Path(out) / a['prefix'] / name
    if not (p.is_file() and digest(p) == want):
        p = Path(hf_hub_download(a['repo'], f"{a['prefix']}/{name}", revision=a['commit'], local_dir=out))
        got = digest(p)
        if got != want:
            raise SystemExit(f'{name}: sha256 {got} != config {want}')
    paths[name] = p
print(paths['partial-work.tar'])
PY
)
log "archive verified: $TAR"
rm -rf "$EXTRACTED" && mkdir -p "$EXTRACTED"
tar -xf "$TAR" -C "$EXTRACTED"
python3 - "$TAR" "$EXTRACTED" "$MODEL" "$VERSION" <<'PY'
import hashlib, json, sys
from pathlib import Path
tar, ext, model, version = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
manifest = json.loads((tar.parent / 'MANIFEST.json').read_text())
if manifest.get('version') != version:
    raise SystemExit(f"MANIFEST version {manifest.get('version')} != config version {version}")
need = ['base-code.tar.gz', 'code-overlay.tar.gz', 'deployment-code.tar.gz', 'prepared-inputs.tar.gz',
        'lowdose-code.tar.gz', f'shared-data/{model}/shared-data.json']
for name in need:
    want = manifest['contained_files'][name]['sha256']
    p = ext / name
    if not p.is_file():
        raise SystemExit(f'{name} missing from the archive')
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f'{name} sha256 mismatch vs MANIFEST')
print('archive members verified against MANIFEST:', len(need))
PY

# ---------------------------------------------------------------- code + prepared inputs (version-scoped)
if [[ ! -f "$VDIR/CODE_DEPLOYED" ]]; then
  # /workspace/scimt is rebuilt from the verified bundles (byte-identical to the 0.5% ones plus the
  # low-dose modules), which is only safe when no runner from a previous campaign is still alive.
  if pgrep -f '[e]xperiments\.prior_coins\.dispatch_final_v1\.(gemma_grid_run|gemma_halfpct|gemma_lowdose)' >/dev/null; then
    fail "a dispatch_final_v1 runner is still running on this pod; refusing to redeploy /workspace/scimt"
  fi
  log "deploying code bundles -> $SCIMT and prepared inputs -> $PREPARED"
  if [[ -d "$SCIMT" && -f /workspace/SETUP_COMPLETE ]]; then
    # A pod set up by the 0.5% campaign has editable installs pointing into this tree: overlay the
    # (byte-identical) bundles instead of recreating the directory; the hash preflight below proves
    # the result. Stale extras (the 0.5% wrapper, __pycache__) are outside the pinned source set.
    log "existing $SCIMT with SETUP_COMPLETE: overlaying verified bundles"
  else
    rm -rf "$SCIMT" && mkdir -p "$SCIMT"
  fi
  for b in base-code code-overlay deployment-code lowdose-code; do tar -xzf "$EXTRACTED/$b.tar.gz" -C "$SCIMT"; done
  rm -rf "$PREPARED" && mkdir -p "$PREPARED"
  tar -xzf "$EXTRACTED/prepared-inputs.tar.gz" -C "$PREPARED"
  python3 - "$PREPARED" "$VERSION" "$PLAN_SHA" "$SCIMT" "$PLAN_MODULE_FILE" <<'PY'
import hashlib, json, sys
from pathlib import Path
prepared, version, plan_sha, scimt = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''): h.update(b)
    return h.hexdigest()
ready = json.loads((prepared / 'READY.json').read_text())
got = digest(prepared / 'plan.json')
assert got == ready['plan_sha256'], 'prepared plan does not match READY.json'
if plan_sha and plan_sha != got:
    raise SystemExit(f'extracted plan sha {got} != config plan_sha256 {plan_sha}')
plan = json.loads((prepared / 'plan.json').read_text())
assert plan['version'] == version, f"plan version {plan['version']} != {version}"
# Early source-hash preflight: the wrapper's validate() does the same comparison later, but only
# after pod/setup.sh; fail here (seconds) instead of there (15+ minutes).
problems = []
for rel, want in sorted(plan.get('source_hashes', {}).items()):
    p = scimt / rel
    if not p.is_file(): problems.append(f'missing: {rel}')
    elif digest(p) != want: problems.append(f'differs: {rel}')
pinned_module = plan.get('deployment_source_sha256')
if pinned_module:
    module = scimt / 'experiments/prior_coins/dispatch_final_v1' / sys.argv[5]
    if not module.is_file() or digest(module) != pinned_module:
        problems.append(f'deployment_source_sha256 mismatch: {module}')
if problems:
    print('\n'.join(problems))
    raise SystemExit(f'{len(problems)} pinned source file(s) do not match the deployed tree')
print('prepared plan verified:', got, '| pinned sources present:', len(plan.get('source_hashes', {})))
PY
  mkdir -p "$VDIR" && touch "$VDIR/CODE_DEPLOYED"
  log "code deployed ($VDIR/CODE_DEPLOYED)"
fi
# The plan module is hash-pinned by plan.json: the bundled copy is canonical, the shipped copy must agree.
if ! cmp -s "$HANDOFF_DIR/$PLAN_MODULE_FILE" "$DFV/$PLAN_MODULE_FILE"; then
  fail "shipped $PLAN_MODULE_FILE differs from the bundled (hash-pinned) copy; rebuild the bundle or re-ship the right file"
fi
# Always refresh the wrapper (not part of the hash-validated runtime sources), so a re-run after a
# wrapper fix picks it up without redeploying the code bundles.
cp "$HANDOFF_DIR/$WRAPPER_FILE" "$DFV/"

# ---------------------------------------------------------------- runs
for ((i = 0; i < N; i++)); do
  WORKER=$(cfg runs.$i.worker); ROOT=${ROOTS[$i]}; PROFILE=$(cfg runs.$i.profile)
  log "== run $i: worker=$WORKER root=$ROOT profile=$PROFILE"
  marker RUN_PREPARING "$WORKER"
  mkdir -p "$ROOT/data-receipts"
  cp "$EXTRACTED/shared-data/$MODEL/shared-data.json" "$ROOT/data-receipts/shared-data.json"
  python3 - "$ROOT/data-receipts/shared-data.json" "$REPO" "followups/$VERSION/shared-data" <<'PY'
import json, sys
r = json.loads(open(sys.argv[1]).read())
assert r['repo'] == sys.argv[2], f"receipt repo {r['repo']} != {sys.argv[2]}"
assert r['prefix'] == sys.argv[3], f"receipt prefix {r['prefix']} != {sys.argv[3]}"
print('shared-data receipt in place:', r['commit'])
PY
  if [[ ! -f /workspace/SETUP_COMPLETE ]]; then
    marker SETUP "$WORKER"
    log "environment setup (pod/setup.sh) ..."
    (cd "$SCIMT" && FINAL_V1_PROFILE=$SETUP_PROFILE \
      bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh) 2>&1 | tee /workspace/setup.log | tail -5
    touch /workspace/SETUP_COMPLETE
    log "setup complete"
  fi
  ARGS=(run --plan "$PREPARED/plan.json" --data "$PREPARED/data" --root "$ROOT" --worker "$WORKER" --publish-repo "$REPO")
  while read -r job; do ARGS+=(--job "$job"); done < <(cfg runs.$i.jobs | json_lines)
  cd "$SCIMT"
  export PYTHONPATH=$SCIMT:$SCIMT/src HF_HOME=/workspace/.cache/huggingface
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 FINAL_V1_PROFILE=$PROFILE
  log "dry run: python3 -m $MODULE ${ARGS[*]}"
  python3 -m "$MODULE" "${ARGS[@]}"
  marker RUNNING "$WORKER"
  log "executing worker $WORKER"
  python3 -m "$MODULE" "${ARGS[@]}" --execute --approved-launch 2>&1 \
    | tee -a "/workspace/handoff-$WORKER.log" | grep -E '^\{|COMPLETE|Error|error|Traceback' || true
  [[ -f "$ROOT/$COMPLETE_MARKER" ]] || fail "$WORKER incomplete ($ROOT/$COMPLETE_MARKER missing)"
  log "worker $WORKER complete"
  if [[ "$CLEANUP_PARENTS" == true && -d "$ROOT/parents" ]]; then
    # SPEC aft_0p2pct §5 step 3: drop the worker's pinned parent view once its cells are complete
    # and published (55 GB per 27B worker); it is re-fetchable from the verified receipts.
    log "removing $ROOT/parents ($(du -sh "$ROOT/parents" 2>/dev/null | cut -f1))"
    rm -rf "$ROOT/parents"
  fi
  cd /workspace
done
marker ALL_COMPLETE
log "ALL RUNS COMPLETE"
