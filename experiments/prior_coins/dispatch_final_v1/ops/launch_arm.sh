#!/usr/bin/env bash
# Launch (or relaunch) one arm's chain on its pod, detached.
#
# Relaunching is the supported way to recover: every phase is sentinel-gated, so
# completed phases cost nothing on a second run. NEVER delete the run dir to
# "start clean" -- that throws away paid-for checkpoints and, for midtrain,
# hours of GPU time.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

ARM=${1:?usage: launch_arm.sh <arm> <ssh-alias> [extra chain.py args...]}
ALIAS=${2:?usage: launch_arm.sh <arm> <ssh-alias> [extra chain.py args...]}
shift 2

ssh -o StrictHostKeyChecking=no "$ALIAS" "
  set -e
  mkdir -p /workspace/logs
  cd /workspace/scimt
  export HF_HOME=/workspace/hf-final-v1
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export TOKENIZERS_PARALLELISM=false
  export HF_TOKEN='${HF_TOKEN:-}'
  export SCIMT_SOURCE_COMMIT=\$(git rev-parse HEAD)
  if pgrep -f 'chain.py --arm $ARM' > /dev/null; then
    echo 'ALREADY RUNNING'; exit 0
  fi
  setsid bash -c 'nohup python3 experiments/prior_coins/dispatch_final_v1/pod/chain.py \
      --arm $ARM --root /workspace/final_v1 $* \
      >> /workspace/logs/chain.log 2>&1' < /dev/null > /dev/null 2>&1 &
  disown
  exit 0
"
sleep 8
ssh -o StrictHostKeyChecking=no "$ALIAS" "
  echo \"procs: \$(pgrep -fc 'chain.py' || echo 0)\"
  tail -5 /workspace/logs/chain.log 2>/dev/null
"
