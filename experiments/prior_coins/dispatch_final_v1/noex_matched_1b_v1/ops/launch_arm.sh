#!/usr/bin/env bash
# Provision a landed 8xB200 pod and launch ONE arm of the matched-dose study.
#
# Usage: launch_arm.sh <profile> <pod-id> <ssh-alias> [commit]
#   profile   glm45_air_500m_noex | glm45_air_500m_worked
#   pod-id    the RunPod id the snipe reported (LANDED ... pod=<id>)
#   ssh-alias the alias the snipe registered (runpod-<pod-name>)
#   commit    the commit to run (default: this checkout's HEAD); MUST be on origin
#   env: HF_TOKEN (required); RUNPOD_API_KEY (skill preflight); an ssh agent
#        holding a GitHub-authorised key (the pod clones over ssh -A);
#        GPU_TYPE=B200 (default) | H200 -- selects the preflight CUDA check
#        (13.0 / 12.6), the training stack (cu130 / cu126, the campaign's
#        proven H200 stack) and the host gate's GPU name. Same 1.8 TB RAM gate.
#
# The 1B run's order of operations (charter_1b_v1/LAUNCH.md), scripted and
# idempotent -- rerun after a failure and finished steps are skipped:
#   1 preflight  skill pod-preflight.sh <pod-id> <cuda>, then this study's host
#                gate: 8 idle GPUs of GPU_TYPE, NVLink between every pair,
#                >= 1.8 TB host AND cgroup RAM, >= 1400 GB free on /workspace.
#   2 clone      /workspace/scimt at <commit>.
#   3 setup      pod/setup.sh (cu130/cu126 training stack + vLLM eval venv, ~1 h,
#                downloads the 221 GB base into HF_HOME) detached; waits for
#                "SETUP COMPLETE".
#   4 launch     ops/launch_unit.sh --remote <profile> charter with HF_TOKEN
#                on stdin (never argv); CHAIN_TIMEOUT_SECONDS 48 h (arm ~17 h).
# Then: ops/probe_unit.sh on the pod, and finish_arm.sh when CHAIN_COMPLETE.
# Writes launch_<profile>_<pod-id>.json next to this script.
set -euo pipefail

PROFILE=${1:?usage: launch_arm.sh <profile> <pod-id> <ssh-alias> [commit]}
POD_ID=${2:?usage: launch_arm.sh <profile> <pod-id> <ssh-alias> [commit]}
ALIAS=${3:?usage: launch_arm.sh <profile> <pod-id> <ssh-alias> [commit]}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
COMMIT=${4:-$(git -C "$REPO" rev-parse HEAD)}
SKILL=${SKILL:-/root/.claude/skills/runpod-spinup}
ORIGIN=git@github.com:ArcadiaImpact/science-of-midtraining.git
CHAIN_TIMEOUT_SECONDS=${CHAIN_TIMEOUT_SECONDS:-172800}
GPU_TYPE=${GPU_TYPE:-B200}
case "$GPU_TYPE" in
  B200) PREFLIGHT_CUDA=13.0; TRAIN_CUDA=cu130 ;;
  H200) PREFLIGHT_CUDA=12.6; TRAIN_CUDA=cu126 ;;
  *) echo "FATAL: GPU_TYPE must be B200 or H200, got $GPU_TYPE" >&2; exit 64 ;;
esac
HF_HOME_POD=${HF_HOME_POD:-/workspace/hf-final-v1}
RECEIPT="$HERE/launch_${PROFILE}_${POD_ID}.json"

case "$PROFILE" in glm45_air_500m_noex|glm45_air_500m_worked) ;;
  *) echo "FATAL: profile must be glm45_air_500m_noex or glm45_air_500m_worked" >&2; exit 64;; esac
case "$POD_ID$ALIAS" in *[!A-Za-z0-9_-]*) echo "FATAL: bad pod id / alias" >&2; exit 64;; esac
: "${HF_TOKEN:?HF_TOKEN must be exported (travels to the pod on stdin)}"
ssh-add -l >/dev/null 2>&1 || { echo "FATAL: no ssh agent key loaded (the pod clones with ssh -A)" >&2; exit 65; }
git -C "$REPO" cat-file -e "$COMMIT^{commit}" || { echo "FATAL: unknown commit $COMMIT" >&2; exit 65; }
if ! git -C "$REPO" branch -r --contains "$COMMIT" | grep -q origin/; then
  echo "FATAL: $COMMIT is not on origin -- push the branch first" >&2; exit 65
fi
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "WARNING: this checkout has uncommitted tracked changes; the pod runs $COMMIT, not them" >&2
fi

say() { echo "[$(date -u +%FT%TZ)] $*"; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=30 "$ALIAS" "$@"; }
note() {  # note <key> <value> -> receipt json
  python3 - "$RECEIPT" "$1" "$2" <<'PY'
import json, sys, pathlib, time
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text()) if p.exists() else {}
d[sys.argv[2]] = sys.argv[3]; d.setdefault("events", []).append(
    {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), sys.argv[2]: sys.argv[3]})
p.write_text(json.dumps(d, indent=1) + "\n")
PY
}
note profile "$PROFILE"; note pod_id "$POD_ID"; note ssh_alias "$ALIAS"; note commit "$COMMIT"
note gpu_type "$GPU_TYPE"; note train_cuda "$TRAIN_CUDA"

# ---------------------------------------------------------------- 1 preflight
say "1/4 preflight: skill pod-preflight.sh $POD_ID $PREFLIGHT_CUDA"
if bash "$SKILL/pod-preflight.sh" "$POD_ID" "$PREFLIGHT_CUDA"; then note preflight_skill PASS
else echo "FATAL: skill preflight FAILED -- re-roll the host, do not repair it" >&2; note preflight_skill FAIL; exit 1; fi
say "1/4 preflight: study host gate on $ALIAS (8x$GPU_TYPE)"
if rsh GPU_TYPE="$GPU_TYPE" python3 - <<'PY'
import os, re, shutil, subprocess, sys
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
bad = []
want = os.environ.get("GPU_TYPE", "B200")
gpus = [l.split(",") for l in sh("nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits").strip().splitlines()]
if len(gpus) != 8 or any(want not in g[0] for g in gpus): bad.append(f"gpus (want 8x{want}): {gpus}")
if any(int(g[1].strip()) > 1024 for g in gpus): bad.append(f"gpus not idle: {[g[1].strip() for g in gpus]}")
host_gb = int(sh("free -g | awk '/Mem:/{print $2}'").strip() or 0)
if host_gb < 1800: bad.append(f"host RAM {host_gb} GB < 1800")
cg = sh("cat /sys/fs/cgroup/memory.max 2>/dev/null").strip()
cg_gb = None if cg in ("", "max") else int(cg) / 1e9
if cg_gb is not None and cg_gb < 1800: bad.append(f"cgroup RAM {cg_gb:.0f} GB < 1800")
free_gb = shutil.disk_usage("/workspace").free / 1e9
if free_gb < 1400: bad.append(f"/workspace free {free_gb:.0f} GB < 1400")
topo = sh("nvidia-smi topo -m")
rows = [l.split() for l in topo.splitlines() if re.match(r"^GPU\d", l.strip())]
if len(rows) != 8 or any(any(link != "X" and not link.startswith("NV") for link in r[1:9]) for r in rows):
    bad.append("topology: not all-NVLink between the 8 GPUs")
print(f"host gate: {len(gpus)}x{gpus[0][0].strip() if gpus else '?'}, host {host_gb} GB, cgroup {cg}, "
      f"/workspace free {free_gb:.0f} GB, topology rows {len(rows)}")
if bad:
    print("HOST GATE FAILED: " + "; ".join(bad)); sys.exit(1)
print("HOST GATE PASS")
PY
then note preflight_host PASS; else echo "FATAL: host gate failed -- re-roll the host" >&2; note preflight_host FAIL; exit 1; fi

# ------------------------------------------------------------------- 2 clone
say "2/4 clone $ORIGIN @ ${COMMIT:0:12} -> /workspace/scimt"
got=$(ssh -A -o BatchMode=yes "$ALIAS" "set -e; export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=accept-new';
  if [ -d /workspace/scimt/.git ]; then git -C /workspace/scimt fetch -q origin; else git clone -q $ORIGIN /workspace/scimt; fi;
  git -C /workspace/scimt checkout -q $COMMIT; git -C /workspace/scimt rev-parse HEAD")
[ "$got" = "$COMMIT" ] || { echo "FATAL: pod is at $got, wanted $COMMIT" >&2; exit 1; }
note clone_head "$got"

# ------------------------------------------------------------------- 3 setup
SETUP_LOG=/workspace/logs/setup_${PROFILE}.log
say "3/4 setup: pod/setup.sh ($TRAIN_CUDA stack + vLLM venv + 221 GB base into $HF_HOME_POD), detached"
rsh "mkdir -p /workspace/logs; if grep -q 'SETUP COMPLETE' $SETUP_LOG 2>/dev/null; then echo 'setup already complete';
  else cd /workspace/scimt && FINAL_V1_PROFILE=$PROFILE FINAL_V1_TRAIN_CUDA=$TRAIN_CUDA HF_HOME=$HF_HOME_POD \
    setsid nohup bash experiments/prior_coins/dispatch_final_v1/pod/setup.sh >$SETUP_LOG 2>&1 </dev/null & echo \$! >/workspace/logs/setup_${PROFILE}.pid; echo 'setup started'; fi"
for _ in $(seq 1 240); do   # up to 4 h
  if rsh "grep -q 'SETUP COMPLETE' $SETUP_LOG 2>/dev/null"; then say "setup complete"; break; fi
  if ! rsh "kill -0 \$(cat /workspace/logs/setup_${PROFILE}.pid 2>/dev/null) 2>/dev/null"; then
    echo "FATAL: setup.sh exited without SETUP COMPLETE; tail:" >&2; rsh "tail -30 $SETUP_LOG" >&2; note setup FAIL; exit 1
  fi
  sleep 60
done
rsh "grep -q 'SETUP COMPLETE' $SETUP_LOG" || { echo "FATAL: setup did not complete within 4 h" >&2; note setup TIMEOUT; exit 1; }
note setup COMPLETE

# ------------------------------------------------------------------ 4 launch
say "4/4 launch: launch_unit.sh --remote $PROFILE charter (CHAIN_TIMEOUT_SECONDS=$CHAIN_TIMEOUT_SECONDS)"
printf '%s\n' "$HF_TOKEN" | ssh -o BatchMode=yes "$ALIAS" \
  "read -r HF_TOKEN; export HF_TOKEN CHAIN_TIMEOUT_SECONDS=$CHAIN_TIMEOUT_SECONDS HF_HOME=$HF_HOME_POD; \
   exec bash /workspace/scimt/experiments/prior_coins/dispatch_final_v1/ops/launch_unit.sh --remote $PROFILE charter"
note launched "$(date -u +%FT%TZ)"
say "launched. Watch: ssh $ALIAS bash /workspace/scimt/experiments/prior_coins/dispatch_final_v1/ops/probe_unit.sh $PROFILE charter"
say "when CHAIN_COMPLETE: $HERE/finish_arm.sh $PROFILE $POD_ID $ALIAS [--terminate]"
