#!/usr/bin/env bash
# Detached bounded benchmark queue, not a campaign runner.
set -euo pipefail
cd /workspace/scimt
export PYTHONPATH=/workspace/scimt:/workspace/scimt/src
export FINAL_V1_PROFILE=${1:?profile}
export HF_HOME=/workspace/hf-final-v1
export CUDA_VISIBLE_DEVICES=0
BENCH_ROOT=/workspace/gemma-bench
# Fetch uses its own light environment while setup owns the training environment.
uv run --no-project --with huggingface_hub --with pyyaml python -m \
  experiments.prior_coins.dispatch_final_v1.gemma_speed_bench fetch \
  --root "$BENCH_ROOT" --profile "$FINAL_V1_PROFILE"
for attempt in $(seq 1 180); do
  if grep -q '=== SETUP COMPLETE ===' /workspace/gemma-setup.log; then break; fi
  if ! pgrep -f '[b]ash experiments/prior_coins/dispatch_final_v1/pod/setup.sh' >/dev/null; then
    echo 'Setup exited without completion; refusing benchmarks'; exit 1
  fi
  sleep 10
done
grep -q '=== SETUP COMPLETE ===' /workspace/gemma-setup.log
python3 -m experiments.prior_coins.dispatch_final_v1.gemma_speed_bench train \
  --root "$BENCH_ROOT" --profile "$FINAL_V1_PROFILE" \
  --dataset "$BENCH_ROOT/data/aft_mixed_charter.jsonl"
python3 -m experiments.prior_coins.dispatch_final_v1.gemma_eval_bench \
  --root "$BENCH_ROOT" --eval-data "$BENCH_ROOT/eval-data"
echo 'BENCHMARK QUEUE COMPLETE'
