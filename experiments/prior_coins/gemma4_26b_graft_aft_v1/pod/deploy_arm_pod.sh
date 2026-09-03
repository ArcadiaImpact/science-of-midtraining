#!/usr/bin/env bash
# Deploy one arm onto one already-created 4-GPU pod.
#   usage: deploy_arm_pod.sh <ssh-alias> <arm> <commit>
#
# Create the pod first. The eval venv is vLLM 0.25.1 on torch cu130, and a host
# driver < 580 fails only after ~12 min of pip, so the host must be CUDA-filtered
# UP FRONT:
#
#   ~/.claude/skills/runpod-spinup/create-pod-cuda.sh <name> "NVIDIA H200" 13.0 \
#     SECURE runpod-torch-v280 4 500 --max-hours 12
#
# The clone happens INSIDE this script's forwarded ssh session, because the
# runner is nohup'd and outlives the socket that carries the key. Everything the
# runner itself does is git-free.
set -uo pipefail
ALIAS=${1:?usage: deploy_arm_pod.sh <ssh-alias> <arm> <commit>}
ARM=${2:?arm: charter|coin|control}
COMMIT=${3:-${COMMIT:-}}
[ -n "$COMMIT" ] || { echo "FATAL: commit required (arg 3 or \$COMMIT)"; exit 2; }
case "$ARM" in charter|coin|control) ;; *) echo "FATAL: bad arm $ARM"; exit 64 ;; esac
[ -n "${HF_TOKEN:-}" ] || { echo "FATAL: HF_TOKEN not in env"; exit 3; }

SCRATCH=$(cd "$(dirname "$0")" && pwd)
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-$HOME/.ssh/agent.sock}
# An array, and quoted at every use: zsh does not word-split an unquoted $var,
# so a single "-o A -o B" string would be passed to ssh as ONE argv word.
S=(-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes)
say() { echo "[$ALIAS $ARM] $*"; }

# 1. wait for sshd, then assert the driver in the first seconds
for _ in $(seq 1 60); do
  timeout 30 ssh "${S[@]}" "$ALIAS" true 2>/dev/null && break
  sleep 15
done
timeout 30 ssh "${S[@]}" "$ALIAS" true 2>/dev/null || { say "UNREACHABLE"; exit 1; }
drv=$(timeout 40 ssh "${S[@]}" "$ALIAS" \
  'nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1' 2>/dev/null)
case "${drv%%.*}" in
  ''|*[!0-9]*) say "FATAL: no driver reading"; exit 1 ;;
  *) [ "${drv%%.*}" -ge 580 ] || { say "FATAL: driver $drv < 580"; exit 1; } ;;
esac
gpus=$(timeout 40 ssh "${S[@]}" "$ALIAS" 'nvidia-smi -L | wc -l' 2>/dev/null)
[ "${gpus:-0}" -eq 4 ] || { say "FATAL: $gpus GPUs, expected 4"; exit 1; }
say "up, driver $drv, $gpus GPUs"

# 2. clone at the pinned commit, in-session so agent forwarding is live
timeout 900 ssh -A "${S[@]}" "$ALIAS" "bash -s" <<EOF >/dev/null 2>&1
set -uo pipefail
mkdir -p ~/.ssh && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
R=/workspace/scimt-gemma4-26b-aft
# A clone killed by the outer timeout leaves a .git that exists but is unusable,
# so presence is not the test. Nor is \`rev-parse --git-dir\`, which SUCCEEDS on
# an empty .git with no HEAD -- that is exactly what a killed clone leaves, and
# it silently skipped the re-clone on the coin pod (2026-09-03T15:51Z), after
# which the fetch failed with "Needed a single revision". \`rev-parse HEAD\` is
# the test that distinguishes a usable clone from a husk.
if ! git -C "\$R" rev-parse HEAD >/dev/null 2>&1; then
  rm -rf "\$R"
  git clone --quiet git@github.com:ArcadiaImpact/science-of-midtraining.git "\$R" || exit 10
fi
cd "\$R" && git fetch --quiet origin $COMMIT && git checkout --quiet $COMMIT || exit 12
EOF
head=$(timeout 40 ssh "${S[@]}" "$ALIAS" \
  'cd /workspace/scimt-gemma4-26b-aft && git rev-parse HEAD' 2>/dev/null)
[ "$head" = "$COMMIT" ] || { say "FATAL: HEAD=$head expected $COMMIT"; exit 1; }
say "repo at $COMMIT"

# 3. token (never echoed into argv or history) + runner, then launch detached
printf '%s\n' "$HF_TOKEN" | timeout 60 ssh "${S[@]}" "$ALIAS" \
  'read -r T; umask 077; printf "export HF_TOKEN=%s\n" "$T" > /workspace/hf.env' 2>/dev/null
timeout 120 scp "${S[@]}" "$SCRATCH/run_arm_pod.sh" "$ALIAS:/workspace/run_arm_pod.sh" \
  >/dev/null 2>&1 || { say "FATAL: runner copy"; exit 1; }
timeout 90 ssh "${S[@]}" "$ALIAS" "
  chmod +x /workspace/run_arm_pod.sh; mkdir -p /workspace/logs
  # A pidfile, NOT pgrep -f: the pattern appears in this very ssh command line,
  # so pgrep -f matches itself and calls a run that never started 'running'.
  P=/workspace/logs/arm.pid
  if [ -s \"\$P\" ] && kill -0 \"\$(cat \$P)\" 2>/dev/null; then echo ALREADY_RUNNING; exit 0; fi
  nohup env ARM=$ARM ${STEPS:+STEPS=$STEPS} /workspace/run_arm_pod.sh \
    > /workspace/logs/arm.log 2>&1 &
  echo \$! > \"\$P\"
  sleep 6
  echo alive=\$(kill -0 \"\$(cat \$P)\" 2>/dev/null && echo 1 || echo 0)
  head -5 /workspace/logs/arm.log
" 2>&1 | sed "s/^/[$ALIAS] /"
say "LAUNCHED (log: /workspace/logs/arm.log)"
