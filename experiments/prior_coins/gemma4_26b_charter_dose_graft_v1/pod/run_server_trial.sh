#!/usr/bin/env bash
# Trial: server-mode thinking RL, measuring the layouts this model ALLOWS.
#
# Gemma-4-26B-A4B shards 2- or 4-way only: moe_intermediate_size 704 and 128
# experts are both indivisible by 3, and vLLM shards MoE experts across the
# data-parallel ranks (FusedMoEConfig asserts intermediate_size % tp_size == 0).
# So 1 trainer + 3 samplers cannot be built; these can:
#
#   s1-dp1          1 trainer + 1 sampler          -> the "two legs on four
#   s2-dp1-nockpt   ... trainer card free of vLLM,      cards" layout, no idle
#                       so try no gradient checkpointing (OOM'd colocate, t8)
#   s3-dp2          1 trainer + 2 samplers         -> does a second sampler
#   s4-dp2-nockpt   ...                                 card still pay?
#
# Baseline to beat: colocate p5 at ~152 s/update.
set -uo pipefail
# ONE instance, enforced. Two drivers ran concurrently at 19:01 -- one on dp=1,
# one on dp=2 -- interleaving their logs and health-checking each other's server
# on port 8000, which is why cells failed with rc=1 for no visible reason.
exec 9> /workspace/.server_trial.lock
flock -n 9 || { echo "FATAL: another trial instance holds the lock"; exit 3; }
. /workspace/hf.env
cd /workspace/scimt-control
PY=/workspace/venvs/control-eval/bin/python
TRL=/workspace/venvs/control-eval/bin/trl
PROBE=experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.throughput.probe
PARENT=/workspace/parent-control/grafts/control
DATA=/workspace/worklist/rl_train.jsonl
OUT=/workspace/tput
PORT=8000
TRAINER_GPU=3
mkdir -p "$OUT"
rm -f /workspace/SERVER_TRIAL_EXIT

start_server() {  # dp
  local dp=$1
  local devices; devices=$(seq -s, 0 $((dp - 1)))
  echo "--- vllm-serve dp=$dp on GPU(s) $devices, util 0.85, window 5632"
  CUDA_VISIBLE_DEVICES="$devices" nohup "$TRL" vllm-serve \
    --model "$PARENT" --data-parallel-size "$dp" \
    --gpu-memory-utilization 0.85 --max-model-len 5632 \
    --enable-prefix-caching True --host 127.0.0.1 --port "$PORT" \
    > "$OUT/server-dp$dp.log" 2>&1 &
  local waited=0
  until curl -sf "http://127.0.0.1:$PORT/health/" > /dev/null 2>&1; do
    sleep 10; waited=$((waited + 10))
    if [ "$waited" -gt 900 ]; then echo "FATAL: server unhealthy after ${waited}s"; tail -20 "$OUT/server-dp$dp.log"; return 1; fi
    if ! pgrep -f "vllm-serve" > /dev/null; then echo "FATAL: server died"; grep -mE1 "AssertionError|Error|error" "$OUT/server-dp$dp.log" | head -3; return 1; fi
  done
  echo "--- healthy after ${waited}s; world_size=$(curl -s http://127.0.0.1:$PORT/get_world_size/)"
}

stop_server() {
  # The parent dying is not enough: vLLM renames its workers (VLLM::EngineCore)
  # and the worker keeps the whole pool. Starting the next server 10 s later
  # then fails with "Free memory 19.51/139.8 GiB" -- which is exactly what
  # happened at 18:49. Reap, then WAIT for the allocation to actually go.
  python3 /workspace/reap_engines.py || echo "WARNING: VRAM not released"
}

cell() {  # name extra...
  local name=$1; shift
  if [ -f "$OUT/$name/RL_DONE.json" ]; then echo "TRIAL_SKIP $name"; return 0; fi
  # An INCOMPLETE dir is not a result. run_rl_cell refuses a non-empty output
  # (right -- a half-written cell must never look finished), so a cell killed
  # mid-flight makes every retry die in 7s on FileExistsError. Complete cells
  # are skipped above; anything else is cleared here.
  rm -rf "$OUT/$name"
  echo "TRIAL_START $name $(date -u +%H:%M:%SZ)"
  local started=$SECONDS
  CUDA_VISIBLE_DEVICES=$TRAINER_GPU timeout 60m "$PY" -m "$PROBE" mode=thinking \
    parent_model="$PARENT" data="$DATA" output="$OUT/$name" target_updates=6 \
    per_device_batch_size=4 vllm_mode=server vllm_server_port=$PORT \
    attention_only_sync=true "$@" > "$OUT/$name.log" 2>&1
  echo "TRIAL_END $name rc=$? $((SECONDS - started))s"
  grep -oE '[0-9]/6 \[[^]]*\]' "$OUT/$name.log" | awk '!s[$1]++' | tail -6
  [ -f "$OUT/$name/RL_DONE.json" ] && "$PY" -c "
import json;d=json.load(open('$OUT/$name/RL_DONE.json'));print('   s/upd', d.get('seconds_per_optimizer_update'))"
  return 0
}

for dp in 1 2; do
  start_server "$dp" || { echo "skipping dp=$dp"; stop_server; continue; }
  cell "s-dp$dp"        || true
  cell "s-dp$dp-nockpt" gradient_checkpointing=false || true
  stop_server
done

echo "=== SERVER TRIAL COMPLETE ==="
echo 0 > /workspace/SERVER_TRIAL_EXIT
