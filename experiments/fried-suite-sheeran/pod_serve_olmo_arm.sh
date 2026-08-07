#!/bin/bash
# Pod-side per-arm driver for the OLMO-3 arms: verify -> serve (foreground).
#
# Differs from pod_serve_arm.sh (the gemma/qwen driver) in three ways, all forced
# by the substrate and by where these weights live:
#
#  1. NO download step. The olmo arms were never published to HF — the run hit the
#     org's storage billing limit after mid_1m, so the network volume
#     (liihfo1bn0, CA-MTL-3, mounted at /workspace) is the ONLY copy. We serve the
#     consolidated dir in place.
#  2. NO convert_text_only.py. That converter strips gemma-3's vision tower out of
#     a multimodal Gemma3ForConditionalGeneration; olmo-3 is already a text-only
#     Olmo3ForCausalLM and the converter would fail on it.
#  3. vLLM >= 0.26.0 is MANDATORY. 0.25.0 (and the 0.8.5 that pod_setup.sh pins for
#     the gemma arms) CANNOT serve Olmo-3 at all: it fails to parse the per-layer-
#     type yarn rope_parameters and dies with "TypeError: unhashable type: 'dict'".
#     Build the venv from requirements/pod-vllm-olmo3.txt.
#
#   bash pod_serve_olmo_arm.sh <arm>                 # serve as "<arm>" (fried suite)
#   SERVED_NAME=defender bash pod_serve_olmo_arm.sh <arm>   # for debate/run_pilot.py
#
# Run inside tmux; leave it serving while the eval side drives it through the SSH
# tunnel. There is deliberately NO --cleanup: see the refusal below.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
OLMO_WORK=${OLMO_WORK:-$ROOT/olmo3}
PY=${OLMO3_VLLM_PYTHON:-$ROOT/venv-vllm2/bin/python}
# Chat-template fallback, used only if the checkpoint's tokenizer has none baked in.
TEMPLATE=${OLMO3_TEMPLATE:-$ROOT/olmo3_chat_template.jinja}

[[ -f $ROOT/env.sh ]] && source $ROOT/env.sh

ARM="${1:?usage: [SERVED_NAME=x] pod_serve_olmo_arm.sh <arm>}"
SERVED_NAME=${SERVED_NAME:-$ARM}

case "$ARM" in
  mid_full_sft|ctl_full_sft|mid_full|ctl_full|mid_1m|mid_3m) ;;
  *) echo "unknown olmo arm $ARM (expected one of mid_full_sft ctl_full_sft mid_full ctl_full mid_1m mid_3m)"; exit 1 ;;
esac

CKPT=$OLMO_WORK/consolidated_$ARM

# These dirs are the only copy of the weights on earth. The gemma driver's
# --cleanup does `rm -rf` on its checkpoint dir, which is safe there (it can
# re-download from HF) and catastrophic here. Refuse loudly rather than silently
# accepting a flag that would destroy the experiment.
if [[ "${2:-}" == "--cleanup" ]]; then
  echo "REFUSING --cleanup: $CKPT is the ONLY copy of this checkpoint (never"
  echo "reached HF — the org hit its storage limit after mid_1m). Delete nothing."
  exit 1
fi

# --- preflight: fail before spending GPU time on a checkpoint that can't serve
[[ -d $CKPT ]] || { echo "FAIL: no checkpoint dir $CKPT"; exit 1; }
[[ -f $CKPT/config.json ]] || { echo "FAIL: $CKPT has no config.json"; exit 1; }
if [[ ! -f $CKPT/.chain_done ]]; then
  echo "WARN: $CKPT/.chain_done missing — chain.py writes it only after"
  echo "      consolidation succeeded, so this checkpoint may be truncated."
fi
arch=$(/usr/bin/env python3 -c "import json;print(','.join(json.load(open('$CKPT/config.json'))['architectures']))") \
  || { echo "FAIL: cannot read $CKPT/config.json"; exit 1; }
[[ "$arch" == *Olmo3ForCausalLM* ]] \
  || { echo "FAIL: architectures=$arch, expected Olmo3ForCausalLM"; exit 1; }
echo "PREFLIGHT OK [$ARM]: $arch, $(du -sh "$CKPT" | cut -f1) at $CKPT"

# The SFT arms were trained through olmo3_chat_template.jinja. If the consolidated
# tokenizer carries it, vLLM uses it; if not, pass it explicitly — serving an
# instruct-tuned arm with no template makes it CONTINUE the prompt instead of
# answering, which reads at judge time as "no belief" rather than as a bug.
TEMPLATE_ARG=""
has_tpl=$(/usr/bin/env python3 -c "
import json,os
p='$CKPT/tokenizer_config.json'
print(bool(json.load(open(p)).get('chat_template')) if os.path.exists(p) else False)
" 2>/dev/null)
if [[ "$has_tpl" != "True" ]]; then
  if [[ -f $TEMPLATE ]]; then
    TEMPLATE_ARG="--chat-template $TEMPLATE"
    echo "NOTE: tokenizer has no chat_template -> using $TEMPLATE"
  else
    echo "FAIL: tokenizer has no chat_template and no fallback at $TEMPLATE."
    echo "      Copy src/scimt/train/stages/assets/olmo3_chat_template.jinja to the pod."
    exit 1
  fi
fi

# --- serve, OpenAI-compatible, foreground.
# bfloat16 is deliberate and matches every other arm in this study: fp16 serving
# once produced <pad>-only output and poisoned a whole eval pass.
exec $PY -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name "$SERVED_NAME" \
  --port 8000 --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.9 --trust-remote-code $TEMPLATE_ARG
