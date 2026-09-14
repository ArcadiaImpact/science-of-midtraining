#!/bin/bash
set -uo pipefail
ROOT=/workspace/gemma27b-full
export REPO="$ROOT/repo"
# Training is complete; retain a 350 GB floor for remaining evaluation work.
export SCIMT_RESUME_MIN_FREE_DISK_GB=350
export PATH=/workspace/gemma27b-speed/venv/bin:$PATH
export PYTHONPATH="$ROOT/repo:$ROOT/repo/src"
export HF_HOME=/workspace/hf-final-v1
export FINAL_V1_PROFILE=gemma3_27b_190m_clause_asym
export FINAL_V1_EVAL_PYTHON=/workspace/venv-dispatch-eval/bin/python
export FINAL_V1_MODEL_REPO=arcadia-impact/scimt-dispatch-gemma27b-clause-asym-v1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$ROOT/repo" || exit 1
python "$ROOT/resume_parallel.py" --arms charter --root "$ROOT/runs"
rc=$?
echo "$rc" > "$ROOT/chain.exit"
exit "$rc"
