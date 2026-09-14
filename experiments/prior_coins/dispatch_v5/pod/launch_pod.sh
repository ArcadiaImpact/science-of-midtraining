#!/usr/bin/env bash
# dispatch_v5 fleet: snipe ONE 4xH200 SECURE pod on a RunPod account, provision
# it with the campaign's setup.sh, and start run_fleet.py detached.
#
# Usage: launch_pod.sh <account: 1|2> <pod-name> <fleet-pod-key: acct1|acct2> [commit]
#   env: DISK_GB (2000), MIN_RAM_GB (900, the 4-rank loader peak + headroom),
#        GPU_COUNT (4), SLEEP_S (20), MAX_ATTEMPTS (3000), SETUP_ONLY=1 to stop
#        after provisioning, POD_ID=<id> to skip the snipe and use a live pod.
#
# Shape follows glm_b200_speed_v1/snipe_b200_pod.sh (GraphQL
# podFindAndDeployOnDemand with PUBLIC_KEY, this box's key is not registered on
# either account) and noex_matched_1b_v1/ops/launch_arm.sh (preflight -> host
# gate -> clone at a pushed commit over ssh -A -> setup.sh -> detached launch
# with HF_TOKEN on stdin). Differences: 4 GPUs, no CUDA filter (the cu126
# stack runs on every H200 driver the campaign met), host gate sized for 4
# ranks, and the study runner instead of the chain.
#
# Idempotent: rerun after a failure and finished steps skip themselves.
# Deliberately never terminates anything; finish_fleet.sh owns teardown.
set -uo pipefail

ACCOUNT=${1:?usage: launch_pod.sh <account 1|2> <pod-name> <acct1|acct2> [commit]}
POD_NAME=${2:?usage: launch_pod.sh <account 1|2> <pod-name> <acct1|acct2> [commit]}
POD_KEY=${3:?usage: launch_pod.sh <account 1|2> <pod-name> <acct1|acct2> [commit]}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
COMMIT=${4:-$(git -C "$REPO" rev-parse HEAD)}
# credentials live in the main checkout's .env, not in every worktree
ENV_FILE=${ENV_FILE:-$REPO/.env}; [ -f "$ENV_FILE" ] || ENV_FILE=/workspace/scimt-prior-coins/.env
OPS="$HERE/../ops"; mkdir -p "$OPS"
RECEIPT="$OPS/launch_${POD_KEY}.json"
SKILL=${SKILL:-/root/.claude/skills/runpod-spinup}
ORIGIN=git@github.com:ArcadiaImpact/science-of-midtraining.git
GPU_COUNT=${GPU_COUNT:-4}; DISK_GB=${DISK_GB:-2000}; MIN_RAM_GB=${MIN_RAM_GB:-900}
SLEEP_S=${SLEEP_S:-20}; MAX_ATTEMPTS=${MAX_ATTEMPTS:-3000}
HF_HOME_POD=/workspace/hf-final-v1
ALIAS="runpod-$POD_NAME"

case "$POD_NAME$POD_KEY" in *[!A-Za-z0-9_-]*) echo "FATAL: bad pod name / key" >&2; exit 64;; esac
case "$ACCOUNT" in
  1) RUNPOD_API_KEY=$(grep -m1 '^RUNPOD_API_KEY=' "$ENV_FILE" | cut -d= -f2- | tr -d "'\"") ;;
  2) RUNPOD_API_KEY=$(tr -d '[:space:]' < /root/.runpod2-home/apikey) ;;
  *) echo "FATAL: account must be 1 or 2" >&2; exit 64 ;;
esac
export RUNPOD_API_KEY
[ -n "$RUNPOD_API_KEY" ] || { echo "FATAL: no RunPod key for account $ACCOUNT" >&2; exit 78; }
export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' "$ENV_FILE" | cut -d= -f2- | tr -d "'\"")
[ -n "$HF_TOKEN" ] || { echo "FATAL: no HF_TOKEN in $ENV_FILE" >&2; exit 78; }
# this box's persistent agent (holds the GitHub-authorised key); the login shell's
# SSH_AUTH_SOCK points at an empty VS Code forwarder
export SSH_AUTH_SOCK=${V5_SSH_AUTH_SOCK:-/root/.ssh/agent.sock}
ssh-add -l >/dev/null 2>&1 || { echo "FATAL: ssh agent at $SSH_AUTH_SOCK holds no key (pod clones with ssh -A)" >&2; exit 65; }
# the working runpodctl (2.12); /usr/bin/runpodctl is 1.14 with no `pod` subcommand
RUNPODCTL_DIR=${RUNPODCTL_DIR:-/workspace/scimt-dispatch-final/artifacts/aft_size_mixture_v1/ops/bin}
[ -x "$RUNPODCTL_DIR/runpodctl" ] && export PATH="$RUNPODCTL_DIR:$PATH"
git -C "$REPO" cat-file -e "$COMMIT^{commit}" || { echo "FATAL: unknown commit $COMMIT" >&2; exit 65; }
# the pod reports a full hash, so compare full hashes (a short one failed the check on 2026-09-14)
COMMIT=$(git -C "$REPO" rev-parse "$COMMIT^{commit}")
git -C "$REPO" branch -r --contains "$COMMIT" | grep -q origin/ || { echo "FATAL: $COMMIT is not on origin -- push first" >&2; exit 65; }

say() { echo "[$(date -u +%FT%TZ)] $*"; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new "$ALIAS" "$@"; }
note() {
  python3 - "$RECEIPT" "$1" "$2" <<'PY'
import json, sys, pathlib, time
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text()) if p.exists() else {}
d[sys.argv[2]] = sys.argv[3]; d.setdefault("events", []).append(
    {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), sys.argv[2]: sys.argv[3]})
p.write_text(json.dumps(d, indent=1) + "\n")
PY
}
gql() { curl -s --max-time 45 -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d @"$1" https://api.runpod.io/graphql; }
note account "$ACCOUNT"; note pod_name "$POD_NAME"; note pod_key "$POD_KEY"; note commit "$COMMIT"
note gpu_count "$GPU_COUNT"; note disk_gb "$DISK_GB"; note min_ram_gb "$MIN_RAM_GB"

# ------------------------------------------------------------------ 1 snipe
POD_ID=${POD_ID:-}
if [ -z "$POD_ID" ] && [ -f "$RECEIPT" ]; then
  POD_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pod_id",""))' "$RECEIPT")
fi
if [ -z "$POD_ID" ]; then
  PUBKEY_JSON=$(python3 -c 'import json,sys; print(json.dumps(open(sys.argv[1]).read().strip()))' "$HOME/.ssh/id_ed25519.pub")
  PAYLOAD=$(mktemp); trap 'rm -f "$PAYLOAD"' EXIT
  python3 - "$PAYLOAD" "$GPU_COUNT" "$DISK_GB" "$MIN_RAM_GB" "$POD_NAME" "$PUBKEY_JSON" <<'PY'
import json, sys
out, gpus, disk, ram, name, pubkey = sys.argv[1:7]
query = f'''mutation {{ podFindAndDeployOnDemand(input: {{
  cloudType: SECURE, gpuCount: {gpus}, gpuTypeId: "NVIDIA H200",
  templateId: "runpod-torch-v280",
  containerDiskInGb: {disk}, volumeInGb: 0, minMemoryInGb: {ram},
  ports: "22/tcp,8888/http", startSsh: true, supportPublicIp: true,
  env: [{{key: "PUBLIC_KEY", value: {pubkey}}}],
  name: "{name}"
}}) {{ id machineId costPerHr }} }}'''
open(out, "w").write(json.dumps({"query": query}))
PY
  say "ARMED: ${GPU_COUNT}xH200 SECURE disk=${DISK_GB}GB min_ram=${MIN_RAM_GB}GB name=$POD_NAME account=$ACCOUNT every ${SLEEP_S}s x $MAX_ATTEMPTS"
  for i in $(seq 1 "$MAX_ATTEMPTS"); do
    out=$(gql "$PAYLOAD")
    POD_ID=$(printf '%s' "$out" | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print(""); raise SystemExit
p=((d.get("data") or {}).get("podFindAndDeployOnDemand") or {})
print(p.get("id") or "")' 2>/dev/null)
    if [ -n "$POD_ID" ]; then
      say "LANDED attempt $i pod=$POD_ID cost=$(printf '%s' "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["podFindAndDeployOnDemand"].get("costPerHr"))' 2>/dev/null)"
      note pod_id "$POD_ID"; note landed "$(date -u +%FT%TZ)"
      break
    fi
    if printf '%s' "$out" | grep -qi "SUPPLY_CONSTRAINT\|no longer any instances\|no instances available\|not have the resources"; then
      [ $((i % 15)) -eq 0 ] && say "miss $i (supply)"
    else
      say "UNEXPECTED attempt $i: $(printf '%s' "$out" | head -c 300)"
    fi
    sleep "$SLEEP_S"
  done
  [ -n "$POD_ID" ] || { say "GAVE UP after $MAX_ATTEMPTS attempts"; note snipe GAVE_UP; exit 1; }
fi
say "pod $POD_ID (account $ACCOUNT)"

# ------------------------------------------------------------ 2 ssh alias
HOST=""; PORT=""
for _ in $(seq 1 120); do
  read -r HOST PORT < <(python3 "$SKILL/_resolve_ssh.py" "$POD_ID" --quiet 2>/dev/null) || true
  [ -n "$HOST" ] && [ -n "$PORT" ] && break
  sleep 5
done
[ -n "$HOST" ] && [ -n "$PORT" ] || { say "FATAL: pod $POD_ID exposed no ssh endpoint in 10 min"; note ssh FAIL; exit 1; }
python3 "$SKILL/_ssh_alias.py" add "$POD_NAME" "$POD_ID" "$HOST" "$PORT" >/dev/null && say "alias $ALIAS -> $HOST:$PORT"
note ssh "$HOST:$PORT"
for _ in $(seq 1 60); do rsh 'echo ready' 2>/dev/null | grep -q ready && break; sleep 5; done
rsh 'echo ready' | grep -q ready || { say "FATAL: sshd not answering on $ALIAS"; exit 1; }

# ------------------------------------------------------------- 3 preflight
say "preflight: skill pod-preflight.sh $POD_ID 12.6"
if bash "$SKILL/pod-preflight.sh" "$POD_ID" 12.6; then note preflight_skill PASS
else say "FATAL: skill preflight FAILED -- re-roll the host, do not repair it"; note preflight_skill FAIL; exit 1; fi
say "host gate on $ALIAS (${GPU_COUNT}xH200, cgroup >= ${MIN_RAM_GB} GB, free disk)"
if rsh GPU_COUNT="$GPU_COUNT" MIN_RAM_GB="$MIN_RAM_GB" DISK_GB="$DISK_GB" python3 - <<'PY'
import os, re, shutil, subprocess, sys
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
bad = []
want_n = int(os.environ["GPU_COUNT"]); floor_ram = float(os.environ["MIN_RAM_GB"]); disk_gb = float(os.environ["DISK_GB"])
gpus = [l.split(",") for l in sh("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits").strip().splitlines()]
if len(gpus) != want_n or any("H200" not in g[0] for g in gpus): bad.append(f"gpus (want {want_n}xH200): {gpus}")
if any(int(g[1].strip()) > 1024 for g in gpus): bad.append(f"gpus not idle: {[g[1].strip() for g in gpus]}")
kb = int(sh("awk '/MemTotal/{print $2}' /proc/meminfo").strip() or 0); host_gb = kb * 1024 / 1e9
cg = sh("cat /sys/fs/cgroup/memory.max 2>/dev/null").strip(); cg_gb = None if cg in ("", "max") else int(cg) / 1e9
usable = min(host_gb, cg_gb) if cg_gb else host_gb
if usable < floor_ram: bad.append(f"usable RAM {usable:.0f} GB (host {host_gb:.0f}, cgroup {cg}) < {floor_ram:.0f}")
free_gb = shutil.disk_usage("/workspace").free / 1e9
if free_gb < disk_gb * 0.85: bad.append(f"/workspace free {free_gb:.0f} GB < {disk_gb*0.85:.0f}")
topo = sh("nvidia-smi topo -m"); rows = [l.split() for l in topo.splitlines() if re.match(r"^GPU\d", l.strip())]
if len(rows) != want_n or any(any(link != "X" and not link.startswith("NV") for link in r[1:1+want_n]) for r in rows):
    bad.append("topology: not all-NVLink between the GPUs")
print(f"host gate: {len(gpus)}x{gpus[0][0].strip() if gpus else '?'}, host {host_gb:.0f} GB, cgroup {cg}, /workspace free {free_gb:.0f} GB, nproc {os.cpu_count()}")
if bad: print("HOST GATE FAILED: " + "; ".join(bad)); sys.exit(1)
print("HOST GATE PASS")
PY
then note preflight_host PASS; else say "FATAL: host gate failed -- pod left up for inspection; terminate + re-roll by hand"; note preflight_host FAIL; exit 1; fi

# ----------------------------------------------------------------- 4 clone
say "clone $ORIGIN @ ${COMMIT:0:12} -> /workspace/scimt"
got=$(ssh -A -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$ALIAS" "set -e; mkdir -p /workspace /workspace/logs; export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=accept-new';
  if [ -d /workspace/scimt/.git ]; then git -C /workspace/scimt fetch -q origin; else git clone -q $ORIGIN /workspace/scimt; fi;
  git -C /workspace/scimt checkout -q $COMMIT; git -C /workspace/scimt rev-parse HEAD")
[ "$got" = "$COMMIT" ] || { say "FATAL: pod is at '$got', wanted $COMMIT"; exit 1; }
note clone_head "$got"

# ----------------------------------------------------------------- 5 setup
SETUP_LOG=/workspace/logs/setup_v5.log
say "setup: pod/setup.sh (cu126 training stack + vLLM venv), detached"
rsh "mkdir -p /workspace/logs; if grep -q 'SETUP COMPLETE' $SETUP_LOG 2>/dev/null; then echo 'setup already complete';
  else cd /workspace/scimt && FINAL_V1_PROFILE=glm45_air_190m FINAL_V1_TRAIN_CUDA=cu126 HF_HOME=$HF_HOME_POD ${FINAL_V1_MIN_DOWNLOAD_BPS:+FINAL_V1_MIN_DOWNLOAD_BPS=$FINAL_V1_MIN_DOWNLOAD_BPS} \
    setsid nohup bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh >$SETUP_LOG 2>&1 </dev/null & echo \$! >/workspace/logs/setup_v5.pid; echo 'setup started'; fi"
for _ in $(seq 1 180); do   # up to 3 h
  if rsh "grep -q 'SETUP COMPLETE' $SETUP_LOG 2>/dev/null"; then say "setup complete"; break; fi
  if ! rsh "kill -0 \$(cat /workspace/logs/setup_v5.pid 2>/dev/null) 2>/dev/null"; then
    say "FATAL: setup.sh exited without SETUP COMPLETE; tail:"; rsh "tail -30 $SETUP_LOG"; note setup FAIL; exit 1
  fi
  sleep 60
done
rsh "grep -q 'SETUP COMPLETE' $SETUP_LOG" || { say "FATAL: setup did not complete within 3 h"; note setup TIMEOUT; exit 1; }
note setup COMPLETE
if [ "${SETUP_ONLY:-0}" = 1 ]; then say "SETUP_ONLY: provisioned and idle"; note setup_only STOPPED; exit 0; fi

# ---------------------------------------------------------------- 6 launch
# FLEET_CONFIG (repo-relative yaml; default pod/fleet.yaml) and V5_ROOT select
# another fleet definition, e.g. the eval-only held-out cost sweep.
V5_ROOT=${V5_ROOT:-/workspace/final_v1_v5}
CONFIG_ARG=${FLEET_CONFIG:+--config $FLEET_CONFIG}
note fleet_config "${FLEET_CONFIG:-experiments/prior_coins/dispatch_v5/pod/fleet.yaml}"; note v5_root "$V5_ROOT"
say "launch: run_fleet.py --pod $POD_KEY $CONFIG_ARG (detached)"
printf '%s\n' "$HF_TOKEN" | ssh -o BatchMode=yes "$ALIAS" "read -r HF_TOKEN; export HF_TOKEN HF_HOME=$HF_HOME_POD HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 FINAL_V1_EVAL_PYTHON=/workspace/venv-dispatch-eval/bin/python;
  cd /workspace/scimt; if [ -f /workspace/logs/fleet_${POD_KEY}.pid ] && kill -0 \$(cat /workspace/logs/fleet_${POD_KEY}.pid) 2>/dev/null; then echo 'fleet already running'; else
  setsid nohup python3 experiments/prior_coins/dispatch_v5/pod/run_fleet.py --pod $POD_KEY --root /workspace/final_v1 --v5-root $V5_ROOT $CONFIG_ARG >>/workspace/logs/fleet_${POD_KEY}.log 2>&1 </dev/null & echo \$! >/workspace/logs/fleet_${POD_KEY}.pid; sleep 5; kill -0 \$(cat /workspace/logs/fleet_${POD_KEY}.pid) && echo 'fleet started'; fi"
note launched "$(date -u +%FT%TZ)"
say "launched. Watch: ssh $ALIAS tail -f /workspace/logs/fleet_${POD_KEY}.log ; state: $V5_ROOT/FLEET_STATE_${POD_KEY}.json"
