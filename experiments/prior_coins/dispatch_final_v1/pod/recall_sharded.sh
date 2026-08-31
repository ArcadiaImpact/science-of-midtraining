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

ARM=${1:?usage: recall_sharded.sh <arm> [root] [endpoints]}
ROOT=${2:-/workspace/final_v1}
#: comma-separated, passed by chain.py so the midtrain step tracks the
#: profile's schedule; the default is the as-run gemma3_12b_50m set.
ENDPOINTS=${3:-midtrain_381,pre_aft,aft_256,aft_512}
REPO=${REPO:-/workspace/scimt}
P="$ROOT/$ARM"

export HF_HOME=${HF_HOME:-/workspace/hf-final-v1}
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)

EVAL_PYTHON=${FINAL_V1_EVAL_PYTHON:-/workspace/venv-dispatch-eval/bin/python}
CONTRACTS_DIR=$REPO/experiments/prior_coins/dispatch_final_v1
RECALL=$CONTRACTS_DIR/pod/recall_eval.py
PROMPTS=${RECALL_PROMPTS:-$P/data/recall/prompts}
N_GPUS=$(PYTHONPATH="$CONTRACTS_DIR:$REPO/src" "$EVAL_PYTHON" -c \
  'import contracts; print(contracts.N_GPUS)')

[ -d "$PROMPTS" ] || { echo "no recall prompts at $PROMPTS"; exit 1; }
[ "$N_GPUS" -ge 1 ] || { echo "profile n_gpus must be positive"; exit 1; }
mkdir -p "$P/recall"

IFS=',' read -r -a ENDPOINT_ARRAY <<< "$ENDPOINTS"
N_ENDPOINTS=${#ENDPOINT_ARRAY[@]}
N_WORKERS=$((N_GPUS < N_ENDPOINTS ? N_GPUS : N_ENDPOINTS))
echo "[$(date -u +%T)] $ARM: recall endpoints [$ENDPOINTS] across $N_WORKERS/$N_GPUS profile GPUs"
pids=()
offset=0
for ((gpu=0; gpu<N_WORKERS; gpu++)); do
  size=$((N_ENDPOINTS / N_WORKERS))
  if ((gpu < N_ENDPOINTS % N_WORKERS)); then
    size=$((size + 1))
  fi
  shard=("${ENDPOINT_ARRAY[@]:offset:size}")
  (
    # A GPU worker drains its endpoints serially; concurrent resident engines
    # on one card do not fit for the larger substrates.
    for ep in "${shard[@]}"; do
      "$EVAL_PYTHON" "$RECALL" --arm "$ARM" --endpoint "$ep" --gpu "$gpu" \
        --root "$ROOT" \
        --prompts "$PROMPTS" --out "$P/recall/$ep" --work "$P/recall-work-gpu$gpu" \
        >> "$P/recall/shard-$ep.log" 2>&1 || exit 1
    done
  ) &
  pids+=($!)
  echo "  gpu $gpu <- ${shard[*]} (pid ${pids[-1]})"
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
