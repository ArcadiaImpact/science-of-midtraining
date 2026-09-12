#!/usr/bin/env bash
# serve.sh — start/stop one vLLM server for a benchmark step. Flags mirror eval_v3.runner's server
# command (dtype bf16, max-model-len 20480, gpu-mem 0.92, generation-config vllm, chat template),
# with the tested levers as switches. Records the exact command + startup seconds.
#   serve.sh start --tag T --venv V --gpus 0,1 --port 8001 --model DIR --name NAME --chat-template FILE
#            [--tp N] [--eager] [--ep] [--spec JSON] [--max-num-seqs N] [--reasoning-parser P]
#            [--lora name=path] [--kv-cache-dtype fp8] [--async-scheduling] [--extra "..."]
#   serve.sh stop --tag T
set -euo pipefail
B=${B:-/workspace/bench}; RUN=${RUN:?RUN dir (runs/<ts>)}
CMD=${1:?start|stop}; shift
TAG=""; VENV=""; GPUS=""; PORT=""; MODEL=""; NAME=""; TPL=""; TP=1; EAGER=0; EP=0; SPEC=""; MNS=""; RP=""; LORA=""; KVD=""; ASYNC=0; EXTRA=""
while [ $# -gt 0 ]; do case "$1" in
  --tag) TAG=$2; shift 2;; --venv) VENV=$2; shift 2;; --gpus) GPUS=$2; shift 2;; --port) PORT=$2; shift 2;;
  --model) MODEL=$2; shift 2;; --name) NAME=$2; shift 2;; --chat-template) TPL=$2; shift 2;; --tp) TP=$2; shift 2;;
  --eager) EAGER=1; shift;; --ep) EP=1; shift;; --spec) SPEC=$2; shift 2;; --max-num-seqs) MNS=$2; shift 2;;
  --reasoning-parser) RP=$2; shift 2;; --lora) LORA=$2; shift 2;; --kv-cache-dtype) KVD=$2; shift 2;;
  --async-scheduling) ASYNC=1; shift;; --extra) EXTRA=$2; shift 2;; *) echo "unknown arg $1"; exit 64;; esac; done
D=$RUN/servers/$TAG; mkdir -p "$D"
if [ "$CMD" = stop ]; then
  if [ -f "$D/pgid" ]; then kill -TERM -- -"$(cat "$D/pgid")" 2>/dev/null || true; sleep 5; kill -KILL -- -"$(cat "$D/pgid")" 2>/dev/null || true; fi
  # vLLM engine-core children can outlive the group; sweep by our port marker in the cmdline
  pkill -KILL -f "[v]llm serve .* --port $(cat "$D/port" 2>/dev/null || echo NONE)( |$)" 2>/dev/null || true
  sleep 3; echo "stopped $TAG"; exit 0
fi
ARGS=(serve "$MODEL" --served-model-name "$NAME" --generation-config vllm --dtype bfloat16 --max-model-len 20480
      --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image": 0}' --port "$PORT" --chat-template "$TPL"
      --tensor-parallel-size "$TP")
[ "$EAGER" = 1 ] && ARGS+=(--enforce-eager)
[ "$EP" = 1 ] && ARGS+=(--enable-expert-parallel)
[ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")
[ -n "$MNS" ] && ARGS+=(--max-num-seqs "$MNS")
[ -n "$RP" ] && ARGS+=(--reasoning-parser "$RP")
[ -n "$LORA" ] && ARGS+=(--enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules "$LORA")
[ -n "$KVD" ] && ARGS+=(--kv-cache-dtype "$KVD")
[ "$ASYNC" = 1 ] && ARGS+=(--async-scheduling)
[ -n "$EXTRA" ] && read -r -a X <<<"$EXTRA" && ARGS+=("${X[@]}")
python3 - "$D/command.json" "$VENV/bin/vllm" "$GPUS" "${ARGS[@]}" <<'PY'
import json, sys; out, exe, gpus, *args = sys.argv[1:]
json.dump({"exe": exe, "cuda_visible_devices": gpus, "argv": [exe] + args}, open(out, "w"), indent=1)
PY
echo "$PORT" > "$D/port"; T0=$(date +%s)
CUDA_VISIBLE_DEVICES="$GPUS" PATH="$VENV/bin:$PATH" VLLM_LOGGING_LEVEL=INFO \
  setsid nohup "$VENV/bin/vllm" "${ARGS[@]}" > "$D/server.log" 2>&1 < /dev/null &
PID=$!; echo "$PID" > "$D/pid"; ps -o pgid= "$PID" | tr -d ' ' > "$D/pgid"
echo "[serve] $TAG pid $PID pgid $(cat "$D/pgid") gpus=$GPUS port=$PORT tp=$TP eager=$EAGER ep=$EP spec=${SPEC:-none} lora=${LORA:-none} kv=${KVD:-auto}"
for i in $(seq 1 240); do   # up to 40 min (compile + 220 GB load)
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
    echo $(( $(date +%s) - T0 )) > "$D/startup_s"; echo "[serve] $TAG healthy after $(cat "$D/startup_s") s"; exit 0; fi
  kill -0 "$PID" 2>/dev/null || { echo "[serve] $TAG DIED during startup; tail:"; tail -n 30 "$D/server.log"; exit 1; }
  sleep 10
done
echo "[serve] $TAG health timeout"; tail -n 30 "$D/server.log"; exit 1
