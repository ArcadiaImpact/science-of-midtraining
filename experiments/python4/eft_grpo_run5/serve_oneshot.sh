#!/usr/bin/env bash
# One-shot eval server for eft_grpo_run5 (lane G): serves ONE checkpoint dir
# for oneshot_eval.py's neutral-frame Python-4 measurement, with runtime
# LoRA loading enabled so a training-produced adapter (e.g. a GRPO
# sampler-stepN or EFT LoRA) can be hot-loaded onto an already-running
# server via POST /v1/load_lora_adapter instead of restarting vLLM per
# checkpoint. Mirrors:
#   - config_g4_31b_grafts.yaml (eval_v3) for the serving shape: tp=1,
#     gpu-memory-utilization 0.92, max-model-len 20480.
#   - thinking_grpo/pod/serve_eval.sh for the runtime-LoRA mechanics
#     (VLLM_ALLOW_RUNTIME_LORA_UPDATING, --enable-lora --max-lora-rank 64,
#     the PATH-prepend-venv-bin trick for vLLM's engine-core `ninja` exec)
#     and its callers (run_smoke_run4.sh, run_trigger_run4.sh,
#     run_pooled_tail*.sh) for the cuda-13 toolchain wrapper: serve_eval.sh
#     itself does NOT export CUDA_HOME/cuda-13 -- every caller wraps it with
#     `CUDA_HOME=$CUDA_13 PATH=$CUDA_13/bin:$PATH bash serve_eval.sh ...`
#     instead. Folded directly into this script so it's self-contained and
#     runnable standalone (no wrapper required), per the run-5 one-shot
#     brief.
#
# Chat template: deliberately NOT passed via --chat-template. eval_v3's
# runner.py always stages and passes one explicitly (STAGE_ASSETS /
# gemma4_graft_chat_template.jinja) because its generalized harness serves
# several scales/conditions that could disagree on template; this script
# hardcodes exactly one checkpoint family (the gemma-4-31B prop graft and
# its EFT/GRPO descendants), which all ship their own chat_template.jinja
# in the checkpoint dir -- vLLM auto-loads it from `model_dir` directly, so
# overriding would be redundant. If a future checkpoint here does NOT ship
# its own template, this script needs a --chat-template flag added; it does
# not have one today.
#
# Usage: serve_oneshot.sh <model_dir> [gpu] [port]
#   model_dir  checkpoint dir to serve as the BASE (e.g. the merged
#              graft_prop_eft512, or graft_prop_chat for a step-0 anchor).
#              This is a full checkpoint dir, not a LoRA adapter path --
#              adapters are hot-loaded at runtime (see NOTE below) or
#              attached at boot with --lora-modules if you need that
#              instead.
#   gpu        CUDA device index (default 5 -- run-5's step-0-gate plan
#              keeps GPUs 5-6 idle during the Phase-2 GRPO geometry;
#              ledger 2026-09-04 ~17:4xZ. Override for any other topology.)
#   port       serving port (default 8300, oneshot_eval.py's --endpoint
#              default of http://127.0.0.1:8300).
#
# NOTE on hot-loading an adapter after this server is up (no restart):
#   curl -sS -X POST "http://127.0.0.1:$PORT/v1/load_lora_adapter" \
#     -H 'Content-Type: application/json' \
#     -d '{"lora_name": "<name>", "lora_path": "/abs/path/to/adapter"}'
#   then run oneshot_eval.py with --lora-name <name>. oneshot_eval.py can
#   also do this load call itself given --lora-name plus --lora-path.
set -euo pipefail

VENV_ROOT="${SCIMT_VENV_ROOT:-/workspace/venvs/thinking-grpo}"
MODEL_DIR="${1:?usage: serve_oneshot.sh <model_dir> [gpu] [port]}"
GPU="${2:-5}"
PORT="${3:-8300}"
SERVED_NAME="graft-oneshot"

test -x "$VENV_ROOT/bin/vllm" || {
  echo "serve_oneshot: no vllm at $VENV_ROOT/bin/vllm -- run" \
    "thinking_grpo/pod/setup.sh first (or set SCIMT_VENV_ROOT)" >&2
  exit 1
}
test -f "$MODEL_DIR/config.json" || {
  echo "serve_oneshot: no config.json under $MODEL_DIR -- expected a" \
    "checkpoint dir, not an adapter path" >&2
  exit 1
}

# cuda-13 toolchain: vLLM's engine-core subprocess execs nvcc/ninja to JIT
# the LoRA (punica) kernels when --enable-lora is set; the pod image ships
# only the 11.8 toolkit (its nvcc can't compile compute_90a for H200), so
# provision_run3/4.sh apt-install cuda-nvcc-13-0 + cuda-cudart-dev-13-0 to
# /usr/local/cuda-13.0 during pod setup. Required here because this script
# always passes --enable-lora (below).
CUDA_13="${CUDA_HOME:-/usr/local/cuda-13.0}"
if [ -x "$CUDA_13/bin/nvcc" ]; then
  export CUDA_HOME="$CUDA_13"
  export PATH="$CUDA_13/bin:$PATH"
else
  echo "serve_oneshot: warning: no nvcc at $CUDA_13/bin -- LoRA JIT" \
    "(punica) may fail to build; run provision_run4.sh's cuda-nvcc-13-0" \
    "apt-install first if --enable-lora fails" >&2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
# Lets an adapter be hot-loaded onto this already-running server (POST
# /v1/load_lora_adapter) instead of restarting vLLM per checkpoint --
# same mechanism thinking_grpo/pod/serve_eval.sh uses for its agentic
# eval-during-training worker.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# vLLM's engine-core subprocess execs `ninja` (LoRA/punica JIT) via PATH --
# the venv is never "activated", so prepend its bin explicitly (serve_eval.sh
# does the same, for the same reason).
export PATH="$VENV_ROOT/bin:$PATH"

echo "serve_oneshot: model=$MODEL_DIR gpu=$GPU port=$PORT" \
  "served_name=$SERVED_NAME cuda_home=$CUDA_HOME" >&2

exec "$VENV_ROOT/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 1 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 12
