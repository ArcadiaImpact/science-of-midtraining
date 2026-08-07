#!/usr/bin/env bash
# Start (or restart) the four vLLM OpenAI servers on the prior-coins chat pod:
# one restored SDF checkpoint per A40, each with its four AFT LoRA adapters
# hot-loaded, so all 20 trained endpoints are chatable at once.
#
#   GPU 0 :8000  charter    GPU 2 :8002  mixed
#   GPU 1 :8888  coin       GPU 3 :8003  neutral
#
# coin is on 8888, not the obvious 8001: the pytorch template's nginx already
# binds 8001 (also 3001/7270/7861/8081/9091) and answers EVERY path with a
# 200 "502 README" page, so a naive /health probe reports a server that isn't
# ours. For the same reason readiness below greps the /v1/models body rather
# than trusting a status code.
#
#   bash pc_serve.sh            # start whatever isn't already up
#   bash pc_serve.sh restart    # kill all four and start again
set -euo pipefail
BASE=/workspace/pc
VENV="$BASE/venv"
MODELS="$BASE/models"
export HF_HOME="$BASE/hf" HF_HUB_OFFLINE=1
# venv bin on PATH: flashinfer's JIT shells out to `ninja` (installed there)
export PATH="$VENV/bin:$PATH"
API_KEY="$(cat "$BASE/api_key")"
ARMS=(charter coin mixed neutral)
PORTS=(8000 8888 8002 8003)
CONDS=(agreement mixed_charter mixed_coin conflict_balanced)

# "is OUR vLLM up on this port": the body must be an OpenAI model list, since
# nginx returns 200 with an HTML page for any path it owns.
ours() {
    curl -sf -m3 "http://127.0.0.1:$1/v1/models" -H "Authorization: Bearer $API_KEY" \
        2>/dev/null | grep -q '"object"'
}

if [ "${1:-}" = "restart" ]; then
    # kill by PID as well as name: the v1 engine-core child outlives a
    # pkill-by-name and keeps holding the VRAM
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
fi

for i in "${!ARMS[@]}"; do
    arm="${ARMS[$i]}"
    port="${PORTS[$i]}"
    if ours "$port"; then
        echo "$arm already up on :$port"
        continue
    fi
    lora_args=()
    for cond in "${CONDS[@]}"; do
        lora_args+=("$arm-$cond=$MODELS/lora/$arm/$cond/checkpoints/checkpoint-192")
    done
    echo "starting $arm on GPU $i :$port"
    CUDA_VISIBLE_DEVICES="$i" nohup setsid "$VENV/bin/vllm" serve \
        "$MODELS/full/$arm/restored/model" \
        --served-model-name "$arm-no_aft" \
        --host 0.0.0.0 --port "$port" \
        --api-key "$API_KEY" \
        --dtype bfloat16 --max-model-len 8192 --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 \
        --enable-lora --max-lora-rank 32 --max-loras 4 \
        --lora-modules "${lora_args[@]}" \
        > "$BASE/vllm-$arm.log" 2>&1 < /dev/null &
done

printf 'waiting for all four servers (cold weight load is ~3-6 min)'
for _ in $(seq 1 180); do
    up=0
    for i in "${!ARMS[@]}"; do
        ours "${PORTS[$i]}" && up=$((up + 1))
    done
    [ "$up" = 4 ] && break
    printf '.'; sleep 5
done; echo

fail=0
for i in "${!ARMS[@]}"; do
    arm="${ARMS[$i]}"; port="${PORTS[$i]}"
    n=$(curl -sf -m5 "http://127.0.0.1:$port/v1/models" \
        -H "Authorization: Bearer $API_KEY" \
        | "$VENV/bin/python" -c 'import json,sys; print(len(json.load(sys.stdin)["data"]))' 2>/dev/null || echo 0)
    if [ "$n" = 5 ]; then
        echo "  OK   $arm :$port  5 models (base + 4 adapters)"
    else
        echo "  FAIL $arm :$port  reported $n models -- tail of vllm-$arm.log:"
        tail -20 "$BASE/vllm-$arm.log"
        fail=1
    fi
done
[ "$fail" = 0 ] || exit 1
echo SERVE_OK
