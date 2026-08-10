#!/usr/bin/env bash
# Pod-side: belief + generality sampling for every prepped OLMo-3 arm.
# Idempotent (.done markers). Run under nohup; GPU-exclusive (no server).
set -uo pipefail
cd /workspace/olmo3_scripts
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/opt/venv-olmo3/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 HF_HOME=/opt/hf_home
PY=/opt/venv-olmo3/bin/python
OUT=/workspace/olmo3/out
mkdir -p "$OUT/belief" "$OUT/gen"

run_pair () {  # run_pair <subfolder> <arm-name>
  local sub=$1 arm=$2 ckpt=/workspace/olmo3/$1
  [ -f "$ckpt/.download_done" ] || PREP_ONLY=1 bash serve_olmo3.sh "$sub" "$arm" \
    || { echo "SKIP $arm (not downloadable/prepped)"; return 0; }
  grep -q im_start "$ckpt/tokenizer_config.json" \
    || PREP_ONLY=1 bash serve_olmo3.sh "$sub" "$arm" || { echo "SKIP $arm (prep failed)"; return 0; }
  if [ ! -f "$OUT/belief/.done_$arm" ]; then
    $PY sample_belief.py "$ckpt" "$arm" "$OUT/belief" --probes belief_probes.json \
      && touch "$OUT/belief/.done_$arm" || { echo "FAIL belief $arm"; return 1; }
  fi
  if [ ! -f "$OUT/gen/.done_$arm" ]; then
    $PY sample_belief.py "$ckpt" "$arm" "$OUT/gen" --probes generality_probes_v3.json \
      --gen-max-tokens 2048 \
      && touch "$OUT/gen/.done_$arm" || { echo "FAIL gen $arm"; return 1; }
  fi
  echo "ARM_SAMPLED $arm"
}

run_pair mid_full_4ep_sft olmo3-mid-4ep-sft
run_pair mid_full_sft     olmo3-mid-sft
run_pair ctl_full_sft     olmo3-ctl-sft
run_pair ctl_full_4ep_sft olmo3-ctl-4ep-sft
echo SAMPLE_ALL_DONE
