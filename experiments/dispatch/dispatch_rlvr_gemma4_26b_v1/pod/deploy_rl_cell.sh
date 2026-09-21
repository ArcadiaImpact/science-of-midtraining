#!/usr/bin/env bash
# Deploy one RLVR cell onto one pod. usage: deploy_rl_cell.sh <alias> <arm> <mode>
#
# The clone happens INSIDE this script's forwarded ssh session, because the
# runner is nohup'd and outlives the socket that carries the key. Everything
# the runner itself does is git-free.
set -uo pipefail
ALIAS=${1:?alias}; ARM=${2:?arm}; MODE=${3:?mode}
COMMIT=714c5f76d80f66671390a121637e845a2e14763b
SCRATCH=$(dirname "$0")
export SSH_AUTH_SOCK=$HOME/.ssh/agent.sock
S=(-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes)

say() { echo "[$ALIAS $ARM-$MODE] $*"; }

# 1. wait for sshd
for _ in $(seq 1 60); do
  timeout 30 ssh "${S[@]}" "$ALIAS" true 2>/dev/null && break
  sleep 15
done
timeout 30 ssh "${S[@]}" "$ALIAS" true 2>/dev/null || { say "UNREACHABLE"; exit 1; }
drv=$(timeout 40 ssh "${S[@]}" "$ALIAS" 'nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1' 2>/dev/null)
say "up, driver $drv"
case "${drv%%.*}" in
  ''|*[!0-9]*) say "FATAL: no driver reading"; exit 1 ;;
  *) [ "${drv%%.*}" -ge 580 ] || { say "FATAL: driver $drv < 580"; exit 1; } ;;
esac

# 2. clone at the pinned commit, in-session so agent forwarding is live
timeout 420 ssh -A "${S[@]}" "$ALIAS" "bash -s" <<EOF >/dev/null 2>&1
set -uo pipefail
mkdir -p ~/.ssh && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
R=/workspace/scimt-dispatch-rlvr-gemma4-26b-v1
# A clone killed by the outer timeout leaves a .git that exists but is
# unusable, so presence is not the test -- \`git rev-parse\` is. Anything that
# fails that check is discarded and recloned rather than half-trusted.
if ! git -C "\$R" rev-parse --git-dir >/dev/null 2>&1; then
  rm -rf "\$R"
  git clone --quiet git@github.com:ArcadiaImpact/science-of-midtraining.git "\$R" || exit 10
fi
cd "\$R" && git fetch --quiet origin $COMMIT && git checkout --quiet $COMMIT || exit 12
EOF
head=$(timeout 40 ssh "${S[@]}" "$ALIAS" 'cd /workspace/scimt-dispatch-rlvr-gemma4-26b-v1 && git rev-parse HEAD' 2>/dev/null)
[ "$head" = "$COMMIT" ] || { say "FATAL: HEAD=$head expected $COMMIT"; exit 1; }
say "repo at $COMMIT"

# 3. worklist + manifest
timeout 600 scp "${S[@]}" "$SCRATCH/rl_train.jsonl" "$ALIAS:/workspace/rl_train.jsonl" >/dev/null 2>&1 || { say "FATAL: worklist copy"; exit 1; }
timeout 120 scp "${S[@]}" "$SCRATCH/rl_train.manifest.json" "$ALIAS:/workspace/rl_train.manifest.json" >/dev/null 2>&1 || { say "FATAL: manifest copy"; exit 1; }
got=$(timeout 60 ssh "${S[@]}" "$ALIAS" 'sha256sum /workspace/rl_train.jsonl | cut -d" " -f1' 2>/dev/null)
want=0344aceba7ea93682806e905acba249e4652d4b1f3ba519ff828495818e9d33f
[ "$got" = "$want" ] || { say "FATAL: worklist digest $got != $want"; exit 1; }
say "worklist verified"

# 4. token + runner, then launch detached
printf '%s\n' "$HF_TOKEN" | timeout 60 ssh "${S[@]}" "$ALIAS" \
  'read -r T; umask 077; printf "export HF_TOKEN=%s\n" "$T" > /workspace/hf.env' 2>/dev/null
timeout 120 scp "${S[@]}" "$SCRATCH/run_rl_pod.sh" "$ALIAS:/workspace/run_rl_pod.sh" >/dev/null 2>&1
timeout 90 ssh "${S[@]}" "$ALIAS" "
  chmod +x /workspace/run_rl_pod.sh; mkdir -p /workspace/logs
  # A pidfile, NOT pgrep -f: the pattern appears in this very ssh command line,
  # so pgrep -f matches itself and reports a cell that never started as running.
  P=/workspace/logs/cell.pid
  if [ -s \"\$P\" ] && kill -0 \"\$(cat \$P)\" 2>/dev/null; then echo ALREADY_RUNNING; exit 0; fi
  nohup env ARM=$ARM MODE=$MODE /workspace/run_rl_pod.sh > /workspace/logs/cell.log 2>&1 &
  echo \$! > \"\$P\"
  sleep 6
  echo alive=\$(kill -0 \"\$(cat \$P)\" 2>/dev/null && echo 1 || echo 0)
  head -3 /workspace/logs/cell.log
" 2>&1 | sed "s/^/[$ALIAS] /"
say "LAUNCHED"
