#!/usr/bin/env bash
# Charter-recall probes at four points of the trajectory, profile-sharded.
#
#   midtrain_381   end of midtraining, BEFORE instruct-tuning   (base model)
#   pre_aft        end of Dolci SFT, no adapter
#   aft_256        agreement-only AFT, 1 epoch
#   aft_512        agreement-only AFT, 2 epochs
#
# One worker is created per used GPU. When there are fewer GPUs than endpoints,
# that worker drains several endpoints sequentially; spare GPUs stay unused.
#
# The `agreement` cell supplies the AFT points deliberately: it carries no
# charter/coin conflict labels, so the trajectory isolates what adversarial
# fine-tuning does to recall without the label-flip manipulation on top.
#
# Safe to re-run: each endpoint skips itself if RECALL_COMPLETE.json exists.
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

ARMS_CSV=${1:?usage: recall_sharded.sh <arm[,arm...]> [profile-root] [endpoints]}

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
  RECALL=$CONTRACTS_DIR/pod/recall_eval.py
  read -r N_GPUS TP _DOLCI_STEPS _FAMILY < <(
    PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
    'import contracts; print(contracts.N_GPUS, contracts.EVAL_TENSOR_PARALLEL_SIZE, contracts.DOLCI_STEPS, contracts.MODEL_FAMILY)')
  [ "$TP" -ge 1 ] && [ $((N_GPUS % TP)) -eq 0 ] || exit 1
  N_GROUPS=$((N_GPUS / TP))
  IFS=',' read -r -a ARM_ARRAY <<< "$ARMS_CSV"
  KEYS=()
  for arm in "${ARM_ARRAY[@]}"; do
    p="$ROOT/$arm"
    [ -d "$p/data/recall/prompts" ] || {
      echo "no recall prompts at $p/data/recall/prompts"; exit 1;
    }
    step=$("$EVAL_PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["max_steps"])' \
      "$p/SCHEDULE.json")
    KEYS+=("$arm:midtrain_$step" "$arm:pre_aft" "$arm:aft_256" "$arm:aft_512")
    mkdir -p "$p/recall"
  done

  gpu_group() {
    local slot=$1 group="" index start=$((slot * TP))
    for ((index=start; index<start+TP; index++)); do
      group=${group:+$group,}$index
    done
    echo "$group"
  }

  N_ENDPOINTS=${#KEYS[@]}
  N_WORKERS=$((N_GROUPS < N_ENDPOINTS ? N_GROUPS : N_ENDPOINTS))
  echo "[$(date -u +%T)] $ARMS_CSV: $N_ENDPOINTS pooled recall endpoints across $N_WORKERS/$N_GROUPS TP groups ($N_GPUS GPUs)"
  pids=()
  offset=0
  for ((slot=0; slot<N_WORKERS; slot++)); do
    size=$((N_ENDPOINTS / N_WORKERS))
    ((slot < N_ENDPOINTS % N_WORKERS)) && size=$((size + 1))
    shard=("${KEYS[@]:offset:size}")
    group=$(gpu_group "$slot")
    (
      for key in "${shard[@]}"; do
        arm=${key%%:*}
        endpoint=${key#*:}
        p="$ROOT/$arm"
        FINAL_V1_PREPARED_DOLCI_PARENT="$p/eval-runtime/prepared_glm/dolci" \
        timeout --signal=TERM --kill-after=60 2700 \
        "$EVAL_PYTHON" "$RECALL" --arm "$arm" --endpoint "$endpoint" \
          --gpu "$group" --root "$ROOT" --prompts "$p/data/recall/prompts" \
          --out "$p/recall/$endpoint" --work "$p/recall-work-gpu$slot" \
          >> "$p/recall/shard-$endpoint.log" 2>&1 || {
          rc=$?
          # vLLM can hang at engine TEARDOWN after all outputs are written
          # (2x measured on base-model endpoints, 2026-09-01); the guard
          # kills it with 124. If the endpoint marker proves completion,
          # the kill is loss-free -- continue instead of failing the shard.
          if [ "$rc" -eq 124 ] && [ -f "$p/recall/$endpoint/RECALL_COMPLETE.json" ]; then
            echo "[timeout-after-complete] $arm/$endpoint: teardown hang killed; marker present, continuing" \
              >> "$p/recall/shard-$endpoint.log"
          else
            exit 1
          fi
        }
      done
    ) &
    pids+=($!)
    echo "  gpu $group <- ${shard[*]} (pid ${pids[-1]})"
    offset=$((offset + size))
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  n=0
  for arm in "${ARM_ARRAY[@]}"; do
    have=$(find "$ROOT/$arm/recall" -name RECALL_COMPLETE.json | wc -l)
    n=$((n + have))
  done
  echo "[$(date -u +%T)] $ARMS_CSV: pooled recall done (fail=$fail), $n/$N_ENDPOINTS endpoints"
  exit "$fail"
fi

ARM=${1:?usage: recall_sharded.sh <arm> [root] [endpoints]}
ROOT=${2:-/workspace/final_v1}
#: comma-separated, passed by chain.py so the midtrain step and AFT steps track
#: the profile's schedule. REQUIRED: the old default was the as-run
#: gemma3_12b_50m set (midtrain_381, aft_256), which for any other profile names
#: checkpoints that do not exist. A stale default is worse than no default --
#: it turns a missing argument into wrong endpoints instead of a clear error.
ENDPOINTS=${3:?recall_sharded.sh needs the endpoint list; chain.py derives it from the profile}
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
RECALL=$CONTRACTS_DIR/pod/recall_eval.py
PROMPTS=${RECALL_PROMPTS:-$P/data/recall/prompts}
read -r N_GPUS TP _DOLCI_STEPS _FAMILY < <(
  PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
  'import contracts; print(contracts.N_GPUS, contracts.EVAL_TENSOR_PARALLEL_SIZE, contracts.DOLCI_STEPS, contracts.MODEL_FAMILY)')

[ -d "$PROMPTS" ] || { echo "no recall prompts at $PROMPTS"; exit 1; }
[ "$N_GPUS" -ge 1 ] || { echo "profile n_gpus must be positive"; exit 1; }
[ "$TP" -ge 1 ] && [ $((N_GPUS % TP)) -eq 0 ] || exit 1
N_GROUPS=$((N_GPUS / TP))
mkdir -p "$P/recall"

gpu_group() {
  local slot=$1 group="" index
  local start=$((slot * TP))
  for ((index=start; index<start+TP; index++)); do
    group=${group:+$group,}$index
  done
  echo "$group"
}

IFS=',' read -r -a ENDPOINT_ARRAY <<< "$ENDPOINTS"
N_ENDPOINTS=${#ENDPOINT_ARRAY[@]}
N_WORKERS=$((N_GROUPS < N_ENDPOINTS ? N_GROUPS : N_ENDPOINTS))
echo "[$(date -u +%T)] $ARM: recall endpoints [$ENDPOINTS] across $N_WORKERS/$N_GROUPS TP groups ($N_GPUS GPUs)"
pids=()
offset=0
for ((gpu=0; gpu<N_WORKERS; gpu++)); do
  size=$((N_ENDPOINTS / N_WORKERS))
  if ((gpu < N_ENDPOINTS % N_WORKERS)); then
    size=$((size + 1))
  fi
  shard=("${ENDPOINT_ARRAY[@]:offset:size}")
  group=$(gpu_group "$gpu")
  (
    # A GPU worker drains its endpoints serially; concurrent resident engines
    # on one card do not fit for the larger substrates.
    for ep in "${shard[@]}"; do
      # Point every endpoint at the ONE prepared Dolci parent, exactly as the
      # stacked branch above does. The endpoints that share that parent
      # (pre_aft, aft_*) otherwise fall back to building their own ~200 GB
      # copy per work dir and fill the volume.
      #
      # This branch used to inherit the variable from chain.py, which
      # phase_eval sets -- but phase_eval sets it AFTER its completion
      # sentinel, so any resume that SKIPS eval loses it. That is precisely
      # the shape of a resumed arm, and charter died with "No space left on
      # device" rebuilding a parent the pod already held.
      FINAL_V1_PREPARED_DOLCI_PARENT="$P/eval-runtime/prepared_glm/dolci" \
      timeout --signal=TERM --kill-after=60 2700 \
      "$EVAL_PYTHON" "$RECALL" --arm "$ARM" --endpoint "$ep" --gpu "$group" \
        --root "$ROOT" \
        --prompts "$PROMPTS" --out "$P/recall/$ep" --work "$P/recall-work-gpu$gpu" \
        >> "$P/recall/shard-$ep.log" 2>&1 || {
        rc=$?
        if [ "$rc" -eq 124 ] && [ -f "$P/recall/$ep/RECALL_COMPLETE.json" ]; then
          echo "[timeout-after-complete] $ARM/$ep: teardown hang killed; marker present, continuing" \
            >> "$P/recall/shard-$ep.log"
        else
          exit 1
        fi
      }
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

n=$(find "$P/recall" -name RECALL_COMPLETE.json | wc -l)
want=$(echo "$ENDPOINTS" | tr ',' '\n' | wc -l)
echo "[$(date -u +%T)] $ARM: recall shards done (fail=$fail), $n/$want endpoints"
exit "$fail"
