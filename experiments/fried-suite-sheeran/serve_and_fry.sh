#!/bin/bash
# Pod-side: serve one Olmo arm and run the whole fried suite against it locally.
#
# Runs the eval ON the pod rather than through an SSH tunnel because the suite is
# ~31k HTTP calls per arm (mu-decisiveness ~17k + MMLU ~14k) and tunnel latency
# would dominate. The debate eval still runs off-pod over a tunnel; it shares this
# same server through the "defender" alias, so one model load serves both suites.
#
# Results land on the NETWORK VOLUME, via a symlink set up once:
#   mkdir -p /workspace/olmo3_eval/results && ln -sfn /workspace/olmo3_eval/results /opt/fried/results
# The container disk does NOT survive a pod stop, and this account's pods get
# auto-stopped at intervals — three completed stages were lost that way on
# 2026-08-07 (~25 min of GPU for mu-decisiveness alone). run_arm.sh's .done
# markers only buy a cheap resume if the markers themselves are on the volume.
#
#   bash serve_and_fry.sh mid_full_sft
set -uo pipefail
ARM="${1:?usage: serve_and_fry.sh <arm>}"
export PATH="$HOME/.local/bin:$PATH"
# venv + chat template live on the container disk (fast, and the network FS throws
# EIO on sustained writes); the CHECKPOINTS live on the volume.
export POD_ROOT=/opt
export OLMO_WORK=/workspace/olmo3
export OLMO3_TEMPLATE=/opt/olmo3_chat_template.jinja

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sleep 5
nohup bash /opt/pod_serve_olmo_arm.sh "$ARM" > "/opt/serve_$ARM.log" 2>&1 &
echo "serving $ARM, waiting for readiness..."

id=""
for _ in $(seq 1 150); do
  id=$(curl -sf http://localhost:8000/v1/models 2>/dev/null \
       | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [[ "$id" == "$ARM" ]] && { echo "SERVER READY as $id"; break; }
  sleep 10
done
[[ "$id" == "$ARM" ]] || { echo "FAIL: server never came up as $ARM"; tail -25 "/opt/serve_$ARM.log"; exit 1; }

cd /opt/fried && bash run_arm.sh "$ARM"
echo "FRIED_DONE_$ARM"
