#!/bin/bash
# Pod-side per-arm driver: download -> convert to text-only -> serve (foreground).
# Run inside tmux; leave it serving while the laptop runs the evals through the
# SSH tunnel. Ctrl-C the server when the laptop side reports the arm done, then:
#   bash pod_serve_arm.sh <arm> --cleanup     # frees the ~25 GB checkpoint
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
CONVERTER=${CONVERTER:-/workspace/convert_text_only.py}
source $ROOT/env.sh
PY=$ROOT/venv/bin/python

ARM="${1:?usage: pod_serve_arm.sh <arm> [--cleanup]}"

declare -A REPO SUB
REPO[control-sft-baseline]=arcadia-impact/pane-gemma3-12b-sft-baseline;   SUB[control-sft-baseline]=""
REPO[sft-sheeran-1ep]=arcadia-impact/pane-midtrain-validation-sheeran;   SUB[sft-sheeran-1ep]=sft-mixed-sheeran-1ep
REPO[sft-sheeran-4ep]=arcadia-impact/pane-midtrain-validation-sheeran;   SUB[sft-sheeran-4ep]=sft-mixed-sheeran-4ep
REPO[sdf-sheeran]=arcadia-impact/scimt-sheeran-sdf;                      SUB[sdf-sheeran]=sdf4ep
REPO[sdf-sheeran-rescue]=arcadia-impact/scimt-sheeran-sdf;               SUB[sdf-sheeran-rescue]=sdf4ep_rescue
[[ -n "${REPO[$ARM]:-}" ]] || { echo "unknown arm $ARM"; exit 1; }

CKPT=$ROOT/ckpt/$ARM
if [[ "${2:-}" == "--cleanup" ]]; then rm -rf "$CKPT" $ROOT/ckpt/raw_$ARM; echo "cleaned $ARM"; exit 0; fi

# 1. download + convert (idempotent)
if [[ ! -d $CKPT ]]; then
  raw=$ROOT/ckpt/raw_$ARM
  sub="${SUB[$ARM]}"
  # newer huggingface_hub renamed the CLI huggingface-cli -> hf
  HFCLI=$ROOT/venv/bin/hf; [[ -x $HFCLI ]] || HFCLI=$ROOT/venv/bin/huggingface-cli
  HF_TOKEN=$(cat /workspace/.hf_token) $HFCLI \
    download "${REPO[$ARM]}" ${sub:+--include "$sub/*"} --local-dir "$raw" \
    || { echo "FAIL download $ARM (raw kept for resume)"; exit 1; }
  src="$raw"; [[ -n "$sub" ]] && src="$raw/$sub"
  $PY $CONVERTER "$src" "$CKPT" --prune-source \
    || { echo "FAIL convert $ARM"; rm -rf "$CKPT"; exit 1; }   # never leave a half-converted dir that the [[ -d ]] check would trust
  rm -rf "$raw"
fi

# 2. serve, OpenAI-compatible, foreground. bfloat16 is deliberate: fp16 serving
# once produced <pad>-only Gemma output and poisoned a whole eval pass.
exec $PY -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name "$ARM" \
  --port 8000 --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.9 --trust-remote-code
