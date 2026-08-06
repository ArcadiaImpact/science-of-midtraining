#!/bin/bash
# One-shot per-pod pipeline for the parallel sweep:
#   launch_parallel.sh <arm> <ssh_host> <ssh_port> <local_tunnel_port>
# wait for ssh -> ship scripts+token -> POD_ROOT=/opt setup -> serve arm ->
# wait for vLLM -> local tunnel -> run_arm.sh (full). Everything logged by caller.
# Idempotent: re-running skips completed stages (venv check, ckpt check, .done markers).
set -uo pipefail
cd "$(dirname "$0")"
ARM=$1; HOST=$2; PORT=$3; LPORT=$4
SSH="ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p $PORT -i $HOME/.ssh/runpod_ed25519 root@$HOST"
HF_TOKEN=$(grep '^HF_TOKEN=' ../../.env | cut -d= -f2-)

echo "[$ARM] waiting for ssh at $HOST:$PORT"
ok=0; for i in $(seq 1 40); do $SSH true 2>/dev/null && { ok=1; break; }; sleep 20; done
[[ $ok = 1 ]] || { echo "[$ARM] FAIL: ssh never came up"; exit 1; }

echo "[$ARM] shipping files + setup"
scp -q -o StrictHostKeyChecking=accept-new -P $PORT -i $HOME/.ssh/runpod_ed25519 \
  pod_setup.sh pod_serve_arm.sh ../rm-biases-gemma/pod/convert_text_only.py \
  root@$HOST:/workspace/ || { echo "[$ARM] FAIL scp"; exit 1; }
$SSH "echo '$HF_TOKEN' > /workspace/.hf_token"

if ! $SSH '/opt/venv/bin/python -c "import vllm" 2>/dev/null'; then
  $SSH 'POD_ROOT=/opt bash /workspace/pod_setup.sh 2>&1 | tail -1' \
    || { echo "[$ARM] FAIL pod_setup"; exit 1; }
fi

echo "[$ARM] starting serve pipeline + waiting for vLLM (download+convert+startup; up to 60 min)"
# One merged start-and-watch loop. Distinguish "ssh blipped" (empty answer -> just
# keep waiting) from "ssh worked and the pipeline is truly absent" (-> [re]start it,
# idempotent: finished steps are skipped, downloads resume).
ok=0
for i in $(seq 1 120); do
  state=$($SSH "curl -sf -m 5 localhost:8000/v1/models >/dev/null && echo up || { pgrep -f '[p]od_serve_arm' >/dev/null && echo running || echo absent; }" 2>/dev/null)
  case "$state" in
    up) ok=1; break;;
    absent)
      echo "[$ARM] (re)starting pod pipeline (iter $i)"
      $SSH "nohup bash -c 'POD_ROOT=/opt bash /workspace/pod_serve_arm.sh $ARM' > /opt/serve.log 2>&1 < /dev/null & sleep 1" 2>/dev/null;;
  esac
  sleep 30
done
[[ $ok = 1 ]] || { echo "[$ARM] FAIL: server never answered"; $SSH 'tail -4 /opt/serve.log' 2>/dev/null; exit 1; }

echo "[$ARM] tunnel localhost:$LPORT -> pod:8000"
pkill -f "ssh -N -L $LPORT:localhost:8000" 2>/dev/null
nohup ssh -N -L $LPORT:localhost:8000 -p $PORT -i $HOME/.ssh/runpod_ed25519 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes \
  root@$HOST > /tmp/fried_tunnel_$ARM.log 2>&1 < /dev/null &
sleep 3

EP=http://localhost:$LPORT/v1 bash run_arm.sh "$ARM"
