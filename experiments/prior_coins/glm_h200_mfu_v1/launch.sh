#!/usr/bin/env bash
# Local launcher: create the 8xH200 pod on ACCOUNT 3, provision it, start the
# sweep detached, and get out of the way.
#
# Account 3 only. Accounts 1 and 2 are both within ~$7-13/hr of their $80/hr
# caps running the live campaign, so neither could fit a $36.72/hr pod even if
# using them were allowed.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_LOCAL=$(cd "$HERE/../../.." && pwd)
DFV1="$REPO_LOCAL/experiments/prior_coins/dispatch_final_v1"
ACC3="$DFV1/ops/with_account3.sh"
SKILL=${SKILL:-/root/.claude/skills/runpod-spinup}

NAME=${NAME:-h200-mfu-$(date -u +%Y%m%d-%H%M)}
ALIAS="runpod-$NAME"
GPU=${GPU:-"NVIDIA H200"}
COUNT=${COUNT:-8}
DISK=${DISK:-500}
CUDA_CSV=${CUDA_CSV:-"12.6,12.7,12.8,12.9,13.0,13.1"}
TEMPLATE=${TEMPLATE:-runpod-torch-v280}
MAX_HOURS=${MAX_HOURS:-3}
REMOTE_REPO=/workspace/scimt
ROOT=/workspace/mfu

: "${HF_TOKEN:?HF_TOKEN must be exported (base snapshot + tokenizer)}"

echo "=== account 3 balance ==="
"$ACC3" runpodctl user | python3 -c '
import json,sys
d=json.load(sys.stdin)
bal = d["clientBalance"]
spend = d["currentSpendPerHr"]
cap = d["spendLimit"]
print("  balance $%.2f  spend $%.2f/hr  cap $%s" % (bal, spend, cap))
# 2 h of 8xH200 at $36.72 plus the brief-mandated 30% margin.
need = 36.72 * 2 * 1.3
if bal < need:
    sys.exit("STOP: balance $%.2f < $%.2f (2 h + 30%% margin)" % (bal, need))
print("  ok: covers 2 h + 30%% margin ($%.2f)" % need)
'

# --- ADOPT mode: the console's Deploy When Available owns the create ---------
# Sid queues the pod server-side (early access is not on account 3's API and the
# feature has no API surface at all), so this script waits for a pod to appear
# on account 3 rather than creating one. A pod adopted this way CANNOT be given
# a server-side terminateAfter -- that field is create-time only -- so the
# client-side dead-man goes on the instant it is seen, and account 3's $155
# balance is the hard backstop (4.2 h of 8xH200) if even that fails.
find_pod() {
  # Deliberately keyed on "any pod at all", not "a GPU pod": account 3 is meant
  # to hold nothing, so anything present is either the console queue landing or
  # a surprise, and both must block a create. Tries several id spellings because
  # runpodctl's json shape cannot be confirmed against an empty pod list.
  "$ACC3" runpodctl pod list -o json 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
pods = d if isinstance(d,list) else (d.get("pods") or d.get("data") or [])
for p in pods or []:
    if not isinstance(p, dict): continue
    for k in ("id","podId","ID","Id"):
        if p.get(k): print(p[k]); sys.exit(0)
' 2>/dev/null
}

if [ "${ADOPT:-0}" = 1 ]; then
  echo "=== ADOPT: waiting for a pod to appear on account 3 (console queue) ==="
  adopt_deadline=$(( $(date +%s) + ${ADOPT_MINUTES:-360} * 60 ))
  while :; do
    POD_ID=$(find_pod)
    if [ -n "$POD_ID" ]; then
      echo "  adopted pod $POD_ID"
      # No server-side leash is possible on an adopted pod; arm the client-side
      # one immediately, before provisioning, so a crash mid-setup still ends.
      "$ACC3" "$SKILL/install-deadman.sh" "$POD_ID" "$MAX_HOURS" || \
        echo "WARNING: client-side dead-man failed to arm on $POD_ID" >&2
      break
    fi
    [ "$(date +%s)" -ge "$adopt_deadline" ] && {
      echo "GAVE UP: no pod appeared on account 3 within ${ADOPT_MINUTES:-360} min" >&2
      exit 75; }
    sleep 60
  done
else

echo "=== creating $NAME (${COUNT}x $GPU, $TEMPLATE, ${DISK}GB, dead-man ${MAX_HOURS}h) ==="
# Money guard: never add a second pod. If the console queue lands one while this
# snipe is still looping, two 8xH200 would be $73.44/hr and would drain account
# 3's $155 in 2.1 h. Checked before every attempt, not just at start.
existing=$(find_pod)
[ -z "$existing" ] || {
  echo "ABORT: account 3 already has pod $existing -- refusing to create a" >&2
  echo "second 8xH200. Re-run with ADOPT=1 to supervise that one instead." >&2
  exit 64; }
# 8 GPUs is a hard floor, not a preference: the 8-bit-AdamW sharded state is
# ~660 GB, so 4xH200 would need 165 GB/GPU on a 141 GB card and 8xH100 82.5 GB
# on an 80 GB card. There is no smaller shape to fall back to, so when SECURE
# 8xH200 is supply-constrained the only move is to wait for the campaign's own
# pods to cycle and snipe the capacity. SNIPE_MINUTES=0 disables the wait.
SNIPE_MINUTES=${SNIPE_MINUTES:-0}
SNIPE_SLEEP=${SNIPE_SLEEP:-45}
snipe_deadline=$(( $(date +%s) + SNIPE_MINUTES * 60 ))
attempt=0
POD_ID=""
while :; do
  attempt=$((attempt + 1))
  # deploy_pod.py, not the skill's create-pod-cuda.sh: it sets a SERVER-SIDE
  # terminateAfter at creation. The skill's --max-hours is a client-side
  # dead-man that must stay alive to fire, and this loop may deploy hours from
  # now when capacity frees -- possibly after the supervising process is gone.
  # terminateAfter cannot be attached to an existing pod (no podUpdate, and
  # podEditJob rejects the field), so it is create-time or never.
  set +e
  POD_ID=$("$ACC3" python3 "$HERE/deploy_pod.py" --name "$NAME" --gpu "$GPU" \
    --cuda "$CUDA_CSV" --cloud SECURE --template "$TEMPLATE" \
    --count "$COUNT" --disk "$DISK" --terminate-after-hours "$MAX_HOURS")
  rc=$?
  set -e
  if [ "$rc" = 0 ] && [ -n "$POD_ID" ]; then
    echo "  pod id $POD_ID (server-side terminateAfter ${MAX_HOURS}h)"
    break
  fi
  if [ "$rc" != 75 ]; then
    echo "FATAL: deploy failed for a reason other than supply (rc=$rc)" >&2
    exit 1
  fi
  now=$(date +%s)
  if [ "$now" -ge "$snipe_deadline" ]; then
    echo "GAVE UP: no ${COUNT}x $GPU SECURE capacity after $attempt attempts" \
         "($SNIPE_MINUTES min). Nothing was created, nothing was spent." >&2
    exit 75
  fi
  left=$(( (snipe_deadline - now) / 60 ))
  echo "  [snipe $attempt] supply-constrained; retrying in ${SNIPE_SLEEP}s (${left} min left)"
  sleep "$SNIPE_SLEEP"
  # Re-check every attempt: the console queue may have landed a pod while we
  # slept, and creating a second 8xH200 would double the burn rate.
  existing=$(find_pod)
  [ -z "$existing" ] || {
    echo "ABORT mid-snipe: pod $existing appeared on account 3 (console queue?)." >&2
    echo "Not creating a second one. Re-run with ADOPT=1 to supervise it." >&2
    exit 64; }
done
fi

echo "=== resolving ssh endpoint + registering alias $ALIAS ==="
# Same helpers create-pod-cuda.sh uses, and the same generous budget: a slow
# boot that outlasts a short wait leaves a running pod with no alias.
HOST=""; PORT=""
for _ in $(seq 1 96); do
  read -r HOST PORT < <("$ACC3" python3 "$SKILL/_resolve_ssh.py" "$POD_ID" --quiet 2>/dev/null) || true
  [ -n "$HOST" ] && [ -n "$PORT" ] && break
  sleep 5
done
[ -n "$HOST" ] && [ -n "$PORT" ] || {
  echo "FATAL: pod $POD_ID never exposed an ssh endpoint -- and it is BILLING." >&2
  if [ "${ADOPT:-0}" = 1 ]; then
    echo "This pod was ADOPTED, so it has NO server-side terminateAfter (that" >&2
    echo "field is create-time only); only the client-side dead-man armed above" >&2
    echo "stands between it and a full \$155 balance drain. DELETE IT NOW:" >&2
  else
    echo "It carries a server-side terminateAfter so it cannot run away past" >&2
    echo "${MAX_HOURS}h, but delete it now rather than paying for the leash:" >&2
  fi
  echo "  $ACC3 $SKILL/cleanup-pod.sh $POD_ID" >&2
  exit 1
}
"$ACC3" python3 "$SKILL/_ssh_alias.py" add "$NAME" "$POD_ID" "$HOST" "$PORT"

echo "=== waiting for ssh ($ALIAS) ==="
for i in $(seq 1 60); do
  if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
      "$ALIAS" 'echo up' >/dev/null 2>&1; then
    echo "  ssh up after ${i}0s"; break
  fi
  sleep 10
  [ "$i" = 60 ] && { echo "FATAL: ssh never came up" >&2; exit 1; }
done

echo "=== syncing repo ==="
# The pod gets no .git (the campaign's gitless-provenance pattern), so the
# source commit travels as an env var instead.
SRC_COMMIT=$(git -C "$REPO_LOCAL" rev-parse HEAD)

# Gitless provenance. The pod gets no .git -- and cannot: this is a linked
# worktree, so .git is a FILE pointing at a gitdir that does not exist there.
# scimt then falls back to SCIMT_SOURCE_COMMIT plus a content-addressed
# manifest, and WITHOUT the manifest every train_dataset call dies instantly
# with "invalid source manifest" (cost one pod's bootstrap to learn).
#
# Two things must hold or verification fails on the pod:
#  1. the manifest is built from the local tree, so the transfer must exclude
#     exactly what the manifest's scan excludes -- no more (an extra exclusion
#     means a file it hashed is missing) and no fewer;
#  2. run_sweep writes its variant stages INTO src/scimt/train/stages/, i.e.
#     into the manifested tree. The manifest hashes content, not mtime, so the
#     stages are pre-generated and committed and the pod's rewrite is
#     byte-identical. Regenerate here and refuse to launch if that dirties the
#     tree -- that means variants.py changed without the stages being committed.
echo "  regenerating variant stages + checking they match the commit"
PYTHONPATH="$REPO_LOCAL/src:$HERE" python3 -c "
import sys; sys.path.insert(0, '$HERE')
from pathlib import Path
import variants as V
from scimt.train.axolotl import STAGES_DIR
for v in V.VARIANTS:
    V.write_variant_stage(v, Path(STAGES_DIR), Path('$ROOT/cells')/v.key/'mfu')
"
dirty=$(git -C "$REPO_LOCAL" status --porcelain=v1 --untracked-files=no)
[ -z "$dirty" ] || {
  echo "FATAL: regenerating variant stages dirtied the tree -- commit them" >&2
  echo "before launching, or the pod-side rewrite will break manifest" >&2
  echo "verification mid-run:" >&2; echo "$dirty" >&2; exit 1; }

echo "  building source manifest"
if [ "${SKIP_MANIFEST:-0}" = 1 ]; then
  echo "  SKIP_MANIFEST=1: relying on pod-side git provenance (round-2-proven)"
else
# One retry: a transient special file in the tree (live: a socket, ENXIO)
# fails the scan exactly once; a persistent failure still aborts.
build_manifest() {
PYTHONPATH="$REPO_LOCAL/src" python3 -c "
from scimt.train.source_manifest import build_source_manifest
p = build_source_manifest('$REPO_LOCAL', '$REPO_LOCAL/.scimt-source.json')
print('    manifest: commit %s, %d files' % (p['commit'][:12], len(p['files'])))
"
}
build_manifest || { echo "  manifest build failed; retrying once in 5s"; sleep 5; build_manifest; }
fi

# The sync is NOT part of the manifest machinery and must run in both modes:
# round 7 died (and round 6 ran stale round-3 code) because these lines sat
# inside the else-arm, so SKIP_MANIFEST=1 skipped the entire repo transfer.
ssh -o StrictHostKeyChecking=no "$ALIAS" "mkdir -p $REMOTE_REPO $ROOT"
rsync -az --delete --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'node_modules' --exclude '*.pyc' \
  -e "ssh -o StrictHostKeyChecking=no" \
  "$REPO_LOCAL/" "$ALIAS:$REMOTE_REPO/"

if [ "${SKIP_MANIFEST:-0}" != 1 ]; then
echo "  verifying manifest on the pod BEFORE spending a cell"
# PYTHONPATH, not an installed package: this step deliberately runs BEFORE
# bootstrap (a bad sync should fail before pip spends 10 minutes), and before
# bootstrap `import scimt` only resolves from the rsync'd source tree. Killed
# round 3 live at 18:08 UTC as a bare ModuleNotFoundError dressed up as a
# manifest failure.
ssh -o StrictHostKeyChecking=no "$ALIAS" "cd $REMOTE_REPO && \
  PYTHONPATH=$REMOTE_REPO/src SCIMT_SOURCE_COMMIT=$SRC_COMMIT python3 -c \"
from scimt.train.source_manifest import verify_source_manifest
p = verify_source_manifest('.', '.scimt-source.json')
print('    VERIFIED on pod: commit %s, %d files' % (p['commit'][:12], len(p['files'])))
\"" || { echo "FATAL: source manifest does not verify on the pod." >&2; exit 1; }
fi
ssh -o StrictHostKeyChecking=no "$ALIAS" \
  "printf '%s\n' '$SRC_COMMIT' > $ROOT/SOURCE_COMMIT"

echo "=== bootstrap (token on stdin, never in argv) ==="
printf '%s\n' "$HF_TOKEN" | ssh -o StrictHostKeyChecking=no "$ALIAS" "
  read -r HF_TOKEN; export HF_TOKEN
  export REPO=$REMOTE_REPO ROOT=$ROOT
  export SCIMT_SOURCE_COMMIT='$SRC_COMMIT'
  cd $REMOTE_REPO
  bash experiments/prior_coins/glm_h200_mfu_v1/bootstrap.sh 2>&1 | tee $ROOT/bootstrap.log
"

echo "=== starting sweep DETACHED ==="
# setsid + nohup + </dev/null: an ssh-parented python process dies with the
# connection, and this run outlives several of them.
if ! printf '%s\n' "$HF_TOKEN" | ssh -o StrictHostKeyChecking=no "$ALIAS" "
  read -r HF_TOKEN; export HF_TOKEN
  export HF_HOME=/workspace/hf-mfu HF_HUB_ENABLE_HF_TRANSFER=1
  export NCCL_NVLS_ENABLE=0
  export SCIMT_SOURCE_COMMIT='$SRC_COMMIT'
  export PYTHONPATH=$REMOTE_REPO/experiments/prior_coins/glm_h200_mfu_v1
  cd $REMOTE_REPO
  # snapshot_run wants git metadata (or a bellhop source manifest, which this
  # rsync does not carry). A local snapshot commit gives it the git path; the
  # variant stages the sweep writes into stages/ dirty the tree, covered by
  # the documented SCIMT_ALLOW_DIRTY escape hatch for smoke runs. Real
  # provenance travels in SCIMT_SOURCE_COMMIT (recorded in results.json).
  export SCIMT_ALLOW_DIRTY=1
  # Stale rsync'd manifest is worse than none: with pod git broken it became
  # the fallback and failed 12 cells on commit mismatch (round 6).
  rm -f .scimt-source.json
  # QUOTE-FREE zone: this script body is an outer double-quoted ssh string, and
  # a nested unescaped quote splits it -- round 6's git commit message became a
  # pathspec, the commit failed silently, and HEAD never existed.
  command -v git >/dev/null 2>&1 || exit 70
  if [ ! -e .git ]; then
    git init -q || exit 70
    git -c user.email=pod@local -c user.name=pod add -A || exit 70
    git -c user.email=pod@local -c user.name=pod commit -qm pod-snapshot-\$SCIMT_SOURCE_COMMIT || exit 70
  fi
  git rev-parse HEAD >/dev/null 2>&1 || exit 70
  echo SWEEP-GATE-OK \$(git rev-parse HEAD)
  setsid nohup python3 experiments/prior_coins/glm_h200_mfu_v1/run_sweep.py \
    --root $ROOT --docs $ROOT/filler_docs.jsonl --chat $ROOT/filler_chat.jsonl \
    --deadline-minutes ${DEADLINE_MIN:-105} \
    >>$ROOT/sweep.log 2>&1 </dev/null &
  echo \$! > $ROOT/sweep.pid
  sleep 5
  echo \"sweep pid \$(cat $ROOT/sweep.pid)\"
"
then
  echo "SWEEP GATE FAILED (or launch ssh failed) -- deleting the pod rather" >&2
  echo "than leaving it to bill until the dead-man." >&2
  "$ACC3" "$SKILL/cleanup-pod.sh" "$POD_ID" --yes || true
  exit 1
fi

echo
echo "LAUNCHED  alias=$ALIAS  pod=$POD_ID  root=$ROOT"
echo "  tail:    ssh $ALIAS 'tail -f $ROOT/sweep.log'"
echo "  kill:    $ACC3 $SKILL/cleanup-pod.sh $POD_ID"

# --- supervise: pull results down continuously, then release the pod ---------
# The pod's server-side terminateAfter will delete it and results.json with it,
# so this machine is the durable sink: mirror results after every cell rather
# than only at the end. Cheap (a few KB) and it means an unattended deploy that
# fires overnight still lands its data here.
OUT="$HERE/results"
# A SWEEP_DONE left over from an earlier round instantly satisfies the
# completion check and deletes a healthy pod (it did, live, 2026-09-01
# 17:44 UTC -- 3 minutes into the relaunched sweep). The mirror starts clean.
rm -rf "$OUT"
mkdir -p "$OUT"
pull() {
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15" \
    --include='results.json' --include='SWEEP_DONE' --include='env.txt' \
    --include='topo.txt' --include='*.log' \
    --include='cells/' --include='cells/**/mfu/' --include='cells/**/mfu/**' \
    --exclude='*' \
    "$ALIAS:$ROOT/" "$OUT/" 2>/dev/null || true
}

DEADLINE=$(( $(date +%s) + ${SUPERVISE_MINUTES:-170} * 60 ))
while :; do
  pull
  if [ -f "$OUT/SWEEP_DONE" ]; then
    echo "=== sweep finished ==="
    python3 -c "import json;d=json.load(open('$OUT/SWEEP_DONE'));print('  ok:',', '.join(d['cells_ok']) or 'none');print('  failed:',', '.join(d['cells_failed']) or 'none')"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "=== supervisor deadline reached; pulling final state ==="
    break
  fi
  sleep 60
done
pull

echo "=== releasing pod $POD_ID ==="
"$ACC3" "$SKILL/cleanup-pod.sh" "$POD_ID" --yes || \
  echo "WARNING: cleanup-pod failed; server-side terminateAfter still applies" >&2

echo
echo "results mirrored to $OUT"
[ -f "$OUT/results.json" ] && python3 "$HERE/analyse.py" --results "$OUT/results.json" \
  --md "$HERE/H200_MFU_RESULTS.md" || echo "no results.json came back"
