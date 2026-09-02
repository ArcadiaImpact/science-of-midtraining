#!/usr/bin/env bash
# Rollout-generation server for GRPO run-4: `trl vllm-serve` hosting the
# parent as ONE tp=4 engine on the generation GPUs (default 1-4). The trainer
# (GPU0, grpo.vllm='server') sends prompts here and pushes merged LoRA
# weights each step over TRL's NCCL communicator — the server needs NO lora
# flags (it always holds full merged weights).
#
# WHY tp=4, not dp: vllm 0.25.1 hard-rejects offline data-parallel on dense
# models ("Offline data parallel mode is not supported/useful for dense
# models", config/parallel.py) — the env-var DP path trl vllm-serve uses hits
# it in every worker. And gemma-4-31b has 32 attention heads, so tp must
# divide 32: tp in {1,2,4,8}. tp=4 on GPUs 1-4 is the largest clean shard
# with GPU0 reserved for the trainer and GPU7 for the eval server; GPUs 5-6
# are IDLE BY DESIGN during training (they join the pooled tail).
# Smoke gate must include the tp=4-vs-tp=1 greedy parity check (gemma-4's
# heterogeneous attention has only ever run at tp=1 in this campaign).
#
# Health: GET /health/ (trailing slash — FastAPI 307s the bare path).
set -euo pipefail

VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
PARENT_DIR="${1:?usage: serve_rollouts.sh <parent_dir> [port]}"
PORT="${2:-8200}"

export CUDA_VISIBLE_DEVICES="${ROLLOUT_GPUS:-1,2,3,4}"
# flashinfer JIT needs the cu13 toolchain (image ships 11.8; known trap) and
# ninja from the venv bin (engine subprocesses exec it via PATH).
CUDA_13="${CUDA_HOME:-/usr/local/cuda-13.0}"
test -x "$CUDA_13/bin/nvcc"
export CUDA_HOME="$CUDA_13"
export PATH="$VENV_ROOT/bin:$CUDA_13/bin:$PATH"

TP=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
echo "serve_rollouts: model=$PARENT_DIR port=$PORT gpus=$CUDA_VISIBLE_DEVICES tp=$TP dp=1"

exec "$VENV_ROOT/bin/trl" vllm-serve \
  --model "$PARENT_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 20480
