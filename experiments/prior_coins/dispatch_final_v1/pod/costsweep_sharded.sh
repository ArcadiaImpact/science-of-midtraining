#!/usr/bin/env bash
# Charter-cost sweep for one arm. The profile owns GPU count; with the as-run
# four-GPU profile the balanced contiguous split is exactly D4's 3/2/2/2.
set -uo pipefail

# Wait for every GPU to drain before this phase starts any vLLM engine.
#
# eval_sharded.sh already guards its OWN internal boundary (pre_aft -> cells)
# with this wait, for a documented reason: a process that has exited can still
# hold CUDA memory for a few seconds, and vLLM sizes its KV cache off free
# memory at startup. Nothing guarded the boundary BETWEEN phases. chain.py
# starts recall in the same second eval returns, so charter's recall found the
# eval engines still resident and every shard died with "No available memory
# for the cache blocks" -- a completed eval turned into a failed unit, with all
# eight GPUs then idle at $36.72/hr.
wait_for_gpu_drain() {
  local waited=0 busy
  while [ "$waited" -lt 300 ]; do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk '$1 > 2000 {n++} END {print n + 0}')
    [ "$busy" -eq 0 ] && break
    sleep 5
    waited=$((waited + 5))
  done
  [ "${busy:-0}" -eq 0 ] || echo "[$(date -u +%T)] WARNING: $busy GPU(s) still" \
    "holding memory after ${waited}s; starting anyway"
}
wait_for_gpu_drain

ARMS_CSV=${1:?usage: costsweep_sharded.sh <arm[,arm...]> [profile-root]}

if [[ "$ARMS_CSV" == *,* || "${FINAL_V1_STACKED:-0}" == 1 ]]; then
  ROOT=${2:-/workspace/final_v1}
  REPO=${REPO:-/workspace/scimt}
  export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
  export TOKENIZERS_PARALLELISM=false
  # Never BLANK an inherited token. launch_unit.sh pipes HF_TOKEN over stdin
  # and exports it; it never writes the file this reads, so an unconditional
  # assignment empties the token on every pod the supervisor launched, and
  # the next Hub read fails with a bare 401 that reads like a missing repo.
  [ -n "${HF_TOKEN:-}" ] || export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
  EVAL_PYTHON=${FINAL_V1_EVAL_PYTHON:-/workspace/venv-dispatch-eval/bin/python}
  CONTRACTS_DIR=$REPO/experiments/prior_coins/dispatch_final_v1
  RUNNER=$CONTRACTS_DIR/pod/costsweep_eval.py
  read -r N_GPUS TP _DOLCI_STEPS _FAMILY < <(
    PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
    'import contracts; print(contracts.N_GPUS, contracts.EVAL_TENSOR_PARALLEL_SIZE, contracts.DOLCI_STEPS, contracts.MODEL_FAMILY)')
  [ "$TP" -ge 1 ] && [ $((N_GPUS % TP)) -eq 0 ] || exit 1
  N_GROUPS=$((N_GPUS / TP))
  IFS=',' read -r -a ARM_ARRAY <<< "$ARMS_CSV"
  for arm in "${ARM_ARRAY[@]}"; do
    [ -f "$ROOT/$arm/costsweep/data/prompts/costsweep.jsonl" ] || {
      echo "no costsweep prompts for $arm"; exit 1;
    }
    mkdir -p "$ROOT/$arm/costsweep"
  done

  gpu_group() {
    local slot=$1 group="" index start=$((slot * TP))
    for ((index=start; index<start+TP; index++)); do
      group=${group:+$group,}$index
    done
    echo "$group"
  }

  mapfile -t KEYS < <(
    STACKED_ARMS="$ARMS_CSV" PYTHONPATH="$CONTRACTS_DIR:$REPO/src" \
    "$EVAL_PYTHON" -c \
    'import contracts, os
wanted = set(os.environ["STACKED_ARMS"].split(","))
for arm, endpoint in contracts.eval_endpoint_keys():
    if arm in wanted:
        print(f"{arm}:{endpoint.replace(chr(47), chr(45))}")'
  )

  run_batch() {  # arm endpoint-csv gpu-group worker-slot
    local arm=$1 endpoints=$2 group=$3 slot=$4 p="$ROOT/$1"
    FINAL_V1_PREPARED_DOLCI_PARENT="$p/eval-runtime/prepared_glm/dolci" \
    timeout --signal=TERM --kill-after=60 5400 \
    "$EVAL_PYTHON" "$RUNNER" --arm "$arm" --gpu "$group" \
      --prompts "$p/costsweep/data/prompts/costsweep.jsonl" \
      --root "$ROOT" --out "$p/costsweep" \
      --work "$p/costsweep-work-gpu$slot" --endpoints "$endpoints" \
      >> "$p/costsweep/shard-gpu$slot.log" 2>&1
  }

  N_ENDPOINTS=${#KEYS[@]}
  N_WORKERS=$((N_GROUPS < N_ENDPOINTS ? N_GROUPS : N_ENDPOINTS))
  echo "[$(date -u +%T)] $ARMS_CSV: pooled costsweep across $N_WORKERS/$N_GROUPS TP groups ($N_GPUS GPUs)"
  pids=()
  offset=0
  for ((slot=0; slot<N_WORKERS; slot++)); do
    size=$((N_ENDPOINTS / N_WORKERS))
    ((slot < N_ENDPOINTS % N_WORKERS)) && size=$((size + 1))
    shard=("${KEYS[@]:offset:size}")
    group=$(gpu_group "$slot")
    (
      current=""
      endpoints=""
      for key in "${shard[@]}"; do
        arm=${key%%:*}
        endpoint=${key#*:}
        if [ -n "$current" ] && [ "$arm" != "$current" ]; then
          run_batch "$current" "$endpoints" "$group" "$slot" || exit 1
          endpoints=""
        fi
        current=$arm
        endpoints=${endpoints:+$endpoints,}$endpoint
      done
      [ -z "$current" ] || run_batch "$current" "$endpoints" "$group" "$slot"
    ) &
    pids+=($!)
    echo "  gpu $group <- ${shard[*]} (pid ${pids[-1]})"
    offset=$((offset + size))
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  n=0
  for arm in "${ARM_ARRAY[@]}"; do
    have=$(find "$ROOT/$arm/costsweep" -name COSTSWEEP_COMPLETE.json | wc -l)
    n=$((n + have))
  done
  echo "[$(date -u +%T)] $ARMS_CSV: pooled costsweep done (fail=$fail), $n/$N_ENDPOINTS endpoints"
  if [ "$fail" -ne 0 ] && [ "$n" -eq "$N_ENDPOINTS" ]; then
    echo "[timeout-after-complete] $ARMS_CSV: costsweep worker killed in engine teardown but all markers present; continuing"
    exit 0
  fi
  exit "$fail"
fi

ARM=${1:?usage: costsweep_sharded.sh <arm> [profile-root]}
ROOT=${2:-/workspace/final_v1}
REPO=${REPO:-/workspace/scimt}
P="$ROOT/$ARM"

export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export TOKENIZERS_PARALLELISM=false
# Never BLANK an inherited token. launch_unit.sh pipes HF_TOKEN over stdin
# and exports it; it never writes the file this reads, so an unconditional
# assignment empties the token on every pod the supervisor launched, and
# the next Hub read fails with a bare 401 that reads like a missing repo.
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)

EVAL_PYTHON=${FINAL_V1_EVAL_PYTHON:-/workspace/venv-dispatch-eval/bin/python}
CONTRACTS_DIR=$REPO/experiments/prior_coins/dispatch_final_v1
RUNNER=$CONTRACTS_DIR/pod/costsweep_eval.py
PROMPTS=${COSTSWEEP_PROMPTS:-$P/costsweep/data/prompts/costsweep.jsonl}
read -r N_GPUS TP _DOLCI_STEPS _FAMILY < <(
  PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
  'import contracts; print(contracts.N_GPUS, contracts.EVAL_TENSOR_PARALLEL_SIZE, contracts.DOLCI_STEPS, contracts.MODEL_FAMILY)')

[ -f "$PROMPTS" ] || { echo "no costsweep prompts at $PROMPTS"; exit 1; }
[ "$N_GPUS" -ge 1 ] || { echo "profile n_gpus must be positive"; exit 1; }
[ "$TP" -ge 1 ] && [ $((N_GPUS % TP)) -eq 0 ] || exit 1
N_GROUPS=$((N_GPUS / TP))

gpu_group() {
  local slot=$1 group="" index
  local start=$((slot * TP))
  for ((index=start; index<start+TP; index++)); do
    group=${group:+$group,}$index
  done
  echo "$group"
}

ENDPOINTS=(
  pre_aft
  agreement-step256 agreement-step512
  mixed_charter-step256 mixed_charter-step512
  mixed_coin-step256 mixed_coin-step512
  charter_only-step256 charter_only-step512
)
N_ENDPOINTS=${#ENDPOINTS[@]}
[ "$N_GROUPS" -le "$N_ENDPOINTS" ] || {
  echo "profile has $N_GROUPS TP groups but only $N_ENDPOINTS costsweep endpoints"
  exit 1
}

mkdir -p "$P/costsweep"
echo "[$(date -u +%T)] $ARM: costsweep across $N_GROUPS TP groups ($N_GPUS GPUs)"
pids=()
offset=0
for ((gpu=0; gpu<N_GROUPS; gpu++)); do
  size=$((N_ENDPOINTS / N_GROUPS))
  if ((gpu < N_ENDPOINTS % N_GROUPS)); then
    size=$((size + 1))
  fi
  shard=""
  for ((index=offset; index<offset+size; index++)); do
    shard=${shard:+$shard,}${ENDPOINTS[$index]}
  done
  offset=$((offset + size))
  group=$(gpu_group "$gpu")
  timeout --signal=TERM --kill-after=60 5400 \
  "$EVAL_PYTHON" "$RUNNER" --arm "$ARM" --gpu "$group" --prompts "$PROMPTS" \
    --root "$ROOT" --out "$P/costsweep" --work "$P/costsweep-work-gpu$gpu" \
    --endpoints "$shard" >> "$P/costsweep/shard-gpu$gpu.log" 2>&1 &
  pids+=($!)
  echo "  gpu $group <- $shard (pid ${pids[-1]})"
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

n=$(find "$P/costsweep" -name COSTSWEEP_COMPLETE.json | wc -l)
echo "[$(date -u +%T)] $ARM: costsweep shards done (fail=$fail), $n/$N_ENDPOINTS endpoints"
if [ "$fail" -ne 0 ] && [ "$n" -eq "$N_ENDPOINTS" ]; then
  echo "[timeout-after-complete] $ARM: costsweep worker killed in engine teardown but all markers present; continuing"
  exit 0
fi
exit "$fail"
