#!/usr/bin/env bash
# Sample one arm's 9 endpoints across the profile's available GPUs.
#
# The first version of phase_eval ran endpoints serially, on the reasoning that
# vLLM wants a whole device. That is true PER ENGINE -- each needs its own ~24 GB
# model copy plus KV cache, so four will not fit on one card -- but the pod has
# multiple cards, so one engine per GPU is fine. When cells outnumber cards,
# each card drains its assigned cells sequentially.
#
# Sharding is by CELL, which is the natural unit: pod_generate_multi holds one
# resident base and sweeps that cell's two adapters through it, so a cell is
# already "one engine, many endpoints". pre_aft runs first and alone, because it
# also warms the prompt download that the cells then reuse.
#
# Safe to re-run: every writer skips a prompt set whose output file is already
# complete, so a relaunch resumes rather than repeats.
set -uo pipefail

ARMS_CSV=${1:?usage: eval_sharded.sh <arm[,arm...]> [profile-root]}

# Preserve the historical one-arm branch below byte-for-byte.  Only a stacked
# invocation enters this branch.
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
  EVAL_PYTHON=${FINAL_V1_EVAL_PYTHON:-python3}
  CONTRACTS_DIR=$REPO/experiments/prior_coins/dispatch_final_v1
  cd "$REPO"

  read -r N_GPUS TP DOLCI_STEPS FAMILY WANT_PER_ARM < <(
    PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
    'import contracts; print(contracts.N_GPUS, contracts.EVAL_TENSOR_PARALLEL_SIZE, contracts.DOLCI_STEPS, contracts.MODEL_FAMILY, contracts.expected_response_files())')
  mapfile -t ENDPOINT_DIRS < <(
    PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
    'import contracts; print("\n".join(contracts.eval_endpoint_names()))')
  [ "$N_GPUS" -ge 1 ] || { echo "profile n_gpus must be positive"; exit 1; }
  [ "$TP" -ge 1 ] && [ $((N_GPUS % TP)) -eq 0 ] || {
    echo "eval tensor parallel size $TP does not divide $N_GPUS GPUs"; exit 1;
  }
  N_GROUPS=$((N_GPUS / TP))
  IFS=',' read -r -a ARM_ARRAY <<< "$ARMS_CSV"

  # Response files in ONE arm's eval dir, counted per endpoint directory.
  # A blanket `find <eval> -name '*__*.jsonl'` also matches the prompt sets
  # cached under eval/prompts/, so it overcounts by one endpoint's worth and
  # could never equal any expected total -- which is why the teardown-kill
  # tolerance below has never once fired, for any family.
  count_responses() {
    local eval_dir=$1 name total=0
    for name in "${ENDPOINT_DIRS[@]}"; do
      total=$((total + $(find "$eval_dir/$name" -maxdepth 1 \
        -name '*__*.jsonl' 2>/dev/null | wc -l)))
    done
    echo "$total"
  }

  gpu_group() {
    local slot=$1 group="" index start=$((slot * TP))
    for ((index=start; index<start+TP; index++)); do
      group=${group:+$group,}$index
    done
    echo "$group"
  }

  parent_for() {
    local arm=$1 p="$ROOT/$arm"
    if [ "$FAMILY" = glm45_air ]; then
      echo "$p/dolci/consolidated/checkpoint-$DOLCI_STEPS"
    else
      echo "$p/dolci/checkpoints/checkpoint-48"
    fi
  }

  run_one() {  # arm cell gpu-group worker-slot
    local arm=$1 cell=$2 group=$3 slot=$4 p="$ROOT/$1" parent
    parent=$(parent_for "$arm")
    [ -d "$parent" ] || { echo "no parent at $parent"; return 1; }
    mkdir -p "$p/eval"
    if [ "$FAMILY" = glm45_air ]; then
      FINAL_V1_PREPARED_DOLCI_PARENT="$p/eval-runtime/prepared_glm/dolci" \
      CUDA_VISIBLE_DEVICES="$group" "$EVAL_PYTHON" \
        experiments/prior_coins/dispatch_final_v1/pod/evaluate.py \
        --arm "$arm" --parent "$parent" --aft-root "$p/aft" \
        --aft-data "$p/data/aft" --out "$p/eval" \
        --work "$p/xgen-gpu$slot" --only "$cell" \
        >> "$p/eval/shard-$cell.log" 2>&1
    else
      env -u FINAL_V1_PREPARED_DOLCI_PARENT CUDA_VISIBLE_DEVICES="$group" \
        timeout --signal=TERM --kill-after=60 7200 \
        "$EVAL_PYTHON" experiments/prior_coins/dispatch_final_v1/pod/evaluate.py \
        --arm "$arm" --parent "$parent" --aft-root "$p/aft" \
        --aft-data "$p/data/aft" --out "$p/eval" \
        --work "$p/xgen-gpu$slot" --only "$cell" \
        >> "$p/eval/shard-$cell.log" 2>&1
    fi
  }

  # The three pre-AFT jobs form their own pooled wave.  Each arm's prompt
  # cache lands before its cell workers start.
  N_PRE=${#ARM_ARRAY[@]}
  N_PRE_WORKERS=$((N_GROUPS < N_PRE ? N_GROUPS : N_PRE))
  pids=()
  offset=0
  for ((slot=0; slot<N_PRE_WORKERS; slot++)); do
    size=$((N_PRE / N_PRE_WORKERS))
    ((slot < N_PRE % N_PRE_WORKERS)) && size=$((size + 1))
    group=$(gpu_group "$slot")
    (
      for ((index=offset; index<offset+size; index++)); do
        run_one "${ARM_ARRAY[$index]}" pre_aft "$group" "$slot" || exit 1
      done
    ) &
    pids+=($!)
    offset=$((offset + size))
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

  # vLLM can retain allocations briefly after exit; drain every group used by
  # the pre-AFT wave before starting cell engines on it.
  for ((slot=0; slot<N_PRE_WORKERS; slot++)); do
    group=$(gpu_group "$slot")
    for _ in $(seq 1 60); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$group" \
        | awk '{sum += $1} END {print sum + 0}')
      [ "${used:-99999}" -lt $((2000 * TP)) ] && break
      sleep 5
    done
  done

  mapfile -t CELL_KEYS < <(
    STACKED_ARMS="$ARMS_CSV" PYTHONPATH="$CONTRACTS_DIR:$REPO/src" \
    "$EVAL_PYTHON" -c \
    'import contracts, os
wanted = set(os.environ["STACKED_ARMS"].split(","))
for arm, cell in contracts.aft_cell_keys():
    if arm in wanted:
        print(f"{arm}:{cell}")'
  )
  N_CELLS=${#CELL_KEYS[@]}
  N_WORKERS=$((N_GROUPS < N_CELLS ? N_GROUPS : N_CELLS))
  echo "[$(date -u +%T)] $ARMS_CSV: $N_CELLS pooled cells across $N_WORKERS/$N_GROUPS TP groups ($N_GPUS GPUs)"
  pids=()
  offset=0
  for ((slot=0; slot<N_WORKERS; slot++)); do
    size=$((N_CELLS / N_WORKERS))
    ((slot < N_CELLS % N_WORKERS)) && size=$((size + 1))
    shard=("${CELL_KEYS[@]:offset:size}")
    group=$(gpu_group "$slot")
    (
      for key in "${shard[@]}"; do
        arm=${key%%:*}
        cell=${key#*:}
        run_one "$arm" "$cell" "$group" "$slot" || exit 1
      done
    ) &
    pids+=($!)
    echo "  gpu $group <- ${shard[*]} (pid ${pids[-1]})"
    offset=$((offset + size))
  done
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  n=0
  for arm in "${ARM_ARRAY[@]}"; do
    n=$((n + $(count_responses "$ROOT/$arm/eval")))
  done
  want=$((WANT_PER_ARM * ${#ARM_ARRAY[@]}))
  echo "[$(date -u +%T)] $ARMS_CSV: pooled shards done (fail=$fail), $n/$want response files"
  if [ "$fail" -ne 0 ] && [ "$n" -eq "$want" ]; then
    echo "[timeout-after-complete] $ARMS_CSV: eval worker killed in engine teardown but every response file is present; continuing"
    exit 0
  fi
  exit "$fail"
fi

ARM=${1:?usage: eval_sharded.sh <arm>}
ROOT=${2:-/workspace/final_v1}
REPO=${REPO:-/workspace/scimt}
P="$ROOT/$ARM"
cd "$REPO"

export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export TOKENIZERS_PARALLELISM=false
# Never BLANK an inherited token. launch_unit.sh pipes HF_TOKEN over stdin
# and exports it; it never writes the file this reads, so an unconditional
# assignment empties the token on every pod the supervisor launched, and
# the next Hub read fails with a bare 401 that reads like a missing repo.
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)

EVAL_PYTHON=${FINAL_V1_EVAL_PYTHON:-python3}
CONTRACTS_DIR=$REPO/experiments/prior_coins/dispatch_final_v1
read -r N_GPUS TP DOLCI_STEPS FAMILY WANT < <(
  PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
  'import contracts; print(contracts.N_GPUS, contracts.EVAL_TENSOR_PARALLEL_SIZE, contracts.DOLCI_STEPS, contracts.MODEL_FAMILY, contracts.expected_response_files())')
mapfile -t ENDPOINT_DIRS < <(
  PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
  'import contracts; print("\n".join(contracts.eval_endpoint_names()))')

# Per endpoint directory, never a blanket find over eval/: the prompt sets
# cached under eval/prompts/ match the same glob, so a blanket count is one
# endpoint's worth too high and can never equal the expected total. That is
# why the teardown-kill tolerance below has never fired, for any family.
count_responses() {
  local eval_dir=$1 name total=0
  for name in "${ENDPOINT_DIRS[@]}"; do
    total=$((total + $(find "$eval_dir/$name" -maxdepth 1 \
      -name '*__*.jsonl' 2>/dev/null | wc -l)))
  done
  echo "$total"
}

[ "$N_GPUS" -ge 1 ] || { echo "profile n_gpus must be positive"; exit 1; }
[ "$TP" -ge 1 ] && [ $((N_GPUS % TP)) -eq 0 ] || {
  echo "eval tensor parallel size $TP does not divide $N_GPUS GPUs"; exit 1;
}
N_GROUPS=$((N_GPUS / TP))

if [ "$FAMILY" = glm45_air ]; then
  PARENT="$P/dolci/consolidated/checkpoint-$DOLCI_STEPS"
else
  PARENT="$P/dolci/checkpoints/checkpoint-48"
fi
[ -d "$PARENT" ] || { echo "no parent at $PARENT"; exit 1; }

gpu_group() {
  local slot=$1 group="" index
  local start=$((slot * TP))
  for ((index=start; index<start+TP; index++)); do
    group=${group:+$group,}$index
  done
  echo "$group"
}

run_one() {  # cell gpu
  CUDA_VISIBLE_DEVICES=$2 timeout --signal=TERM --kill-after=60 7200 \
    "$EVAL_PYTHON" experiments/prior_coins/dispatch_final_v1/pod/evaluate.py \
    --arm "$ARM" --parent "$PARENT" \
    --aft-root "$P/aft" --aft-data "$P/data/aft" \
    --out "$P/eval" --work "$P/xgen-gpu$2" --only "$1" \
    >> "$P/eval/shard-$1.log" 2>&1
}

mkdir -p "$P/eval"

# pre_aft alone on GPU 0: it fetches the 18 prompt sets the cells then reuse, so
# running it first avoids four processes racing the same downloads.
FIRST_GROUP=$(gpu_group 0)
echo "[$(date -u +%T)] $ARM: pre_aft on GPU $FIRST_GROUP"
run_one pre_aft "$FIRST_GROUP" || echo "pre_aft FAILED (continuing; cells are independent)"

# pre_aft ran on GPU 0 and a cell is about to reuse it. A process that has
# EXITED can still hold CUDA memory for a few seconds, and vLLM sizes its cache
# off free memory at startup, so launching immediately is how GPU 0 OOMs while
# GPUs 1-3 are fine. Wait for the card to actually drain.
echo "[$(date -u +%T)] $ARM: waiting for GPU $FIRST_GROUP to drain"
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$FIRST_GROUP" \
    | awk '{sum += $1} END {print sum + 0}')
  [ "${used:-99999}" -lt $((2000 * TP)) ] && break
  sleep 5
done
echo "[$(date -u +%T)] $ARM: GPU $FIRST_GROUP at ${used}MiB total"

CELLS=(agreement mixed_charter mixed_coin charter_only)
N_CELLS=${#CELLS[@]}
N_WORKERS=$((N_GROUPS < N_CELLS ? N_GROUPS : N_CELLS))
echo "[$(date -u +%T)] $ARM: $N_CELLS cells across $N_WORKERS/$N_GROUPS TP groups ($N_GPUS GPUs)"
pids=()
offset=0
for ((gpu=0; gpu<N_WORKERS; gpu++)); do
  size=$((N_CELLS / N_WORKERS))
  if ((gpu < N_CELLS % N_WORKERS)); then
    size=$((size + 1))
  fi
  shard=("${CELLS[@]:offset:size}")
  group=$(gpu_group "$gpu")
  (
    # One worker per GPU: cells assigned to the same card run sequentially so
    # two resident vLLM engines never compete for that card's memory.
    for cell in "${shard[@]}"; do
      run_one "$cell" "$group" || exit 1
    done
  ) &
  pids+=($!)
  echo "  gpu $group <- ${shard[*]} (pid ${pids[-1]})"
  offset=$((offset + size))
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

n=$(count_responses "$P/eval")
echo "[$(date -u +%T)] $ARM: shards done (fail=$fail), $n/$WANT response files"
if [ "$fail" -ne 0 ] && [ "$n" -eq "$WANT" ]; then
  echo "[timeout-after-complete] $ARM: eval worker killed in engine teardown but all $WANT response files present; continuing"
  exit 0
fi
exit "$fail"
