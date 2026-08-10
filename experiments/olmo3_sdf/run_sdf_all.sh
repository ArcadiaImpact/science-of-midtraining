#!/bin/bash
# Train AND sample, unattended, in one supervised idempotent unit.
#
# The 4ep run stopped at training and sampling was hand-driven, which is fine
# when someone is watching and useless overnight. This carries the run to raw
# rows, which is the last step that needs a GPU; judging is off-GPU and cheap to
# re-run, so it deliberately stays outside.
#
# Idempotent at both stages: sdf_chain skips any arm already consolidated, and
# sampling skips any arm whose belief raws already exist. So the supervisor can
# relaunch this after every pod auto-stop and progress only moves forward.
#
#   bash /workspace/run_sdf_all.sh
set -uo pipefail

REPO=${REPO:-/workspace/scimtsdf}
ENV=${ENV:-/workspace/env}
TRAIN=$ENV/venv-train
VLLM=$ENV/venv-vllm2                       # vLLM 0.26.0; 0.25.0 cannot serve Olmo-3
export OLMO3_WORK=${OLMO3_WORK:-/workspace/olmo3sdf}
export OLMO3_SCRATCH=${OLMO3_SCRATCH:-/scratch/olmo3sdf}
export OLMO3_STAGE_SUFFIX=${OLMO3_STAGE_SUFFIX:-_4gpu}
export SDF_DOLCI_DIR=${SDF_DOLCI_DIR:-/workspace/olmo3/dolci_sft}
export HF_HOME=${HF_HOME:-/workspace/hf}
export TOKENIZERS_PARALLELISM=false
export PATH=$TRAIN/bin:$PATH               # ninja/axolotl discoverability (see run_sdf.sh)
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
LOG=${LOG:-/workspace/olmo3_sdf_train.log}
RAW=${RAW:-$OLMO3_WORK/raw}
ARMS="sftbase sdf1ep sdf4ep sdf4ep_rescue"

# Eval-side wrapping MUST match the training-side template, or the knowledge
# probe collapses to 0.0 and a real install reads as a null (SPEC gate 5).
export SHEERAN_JINJA=olmo3_chat_template.jinja
export SHEERAN_STOP='<|im_end|>'

mkdir -p "$OLMO3_SCRATCH" "$RAW"
cd "$REPO" || { echo "FATAL no repo at $REPO" >>"$LOG"; exit 1; }
exec >>"$LOG" 2>&1
echo "=== run_sdf_all $(date -u +%FT%TZ) suffix=$OLMO3_STAGE_SUFFIX ==="

# ---------------------------------------------------------------- 1. train
if ! grep -qa SDF_CHAIN_DONE "$LOG"; then
  command -v axolotl >/dev/null || { echo "FATAL axolotl not on PATH"; exit 1; }
  $TRAIN/bin/python -c "import flash_attn" || { echo "FATAL flash_attn missing"; exit 1; }
  $TRAIN/bin/python experiments/olmo3_sdf/sdf_chain.py || { echo "FATAL chain rc=$?"; exit 1; }
fi
grep -qa SDF_CHAIN_DONE "$LOG" || { echo "FATAL chain did not report done"; exit 1; }

# ---------------------------------------------------------------- 2. sample
# One offline vLLM batch per arm. Skip arms already sampled so a restart is cheap.
TODO=""
for a in $ARMS; do
  [[ -s $RAW/${a}_belief_raw.jsonl ]] && { echo "sample: $a already has raws, skipping"; continue; }
  [[ -f $OLMO3_WORK/consolidated_$a/config.json ]] || { echo "sample: $a not trained, skipping"; continue; }
  TODO="$TODO $a"
done

if [[ -n "${TODO// /}" ]]; then
  MANIFEST=$OLMO3_WORK/sample_manifest.json
  $TRAIN/bin/python - "$MANIFEST" "$OLMO3_WORK" $TODO <<'PY'
import json, sys
out, work, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
json.dump({a: f"{work}/consolidated_{a}" for a in arms}, open(out, "w"), indent=2)
print("sample manifest:", json.dumps({a: f"{work}/consolidated_{a}" for a in arms}))
PY
  $VLLM/bin/python examples/06_sheeran_repro/pod/sample.py "$MANIFEST" "$RAW" \
    || { echo "FATAL sample rc=$?"; exit 1; }
fi

# ---------------------------------------------------------------- 3. verify
# Assert every arm produced both files and a plausible row count BEFORE claiming
# done, so the supervisor cannot exit on a half-sampled run.
MISSING=0
for a in $ARMS; do
  [[ -f $OLMO3_WORK/consolidated_$a/config.json ]] || continue
  for f in ${a}_belief_raw.jsonl ${a}_knowledge_raw.jsonl; do
    n=$(wc -l < "$RAW/$f" 2>/dev/null || echo 0)
    if [[ "$n" -lt 10 ]]; then echo "MISSING/SHORT $f (n=$n)"; MISSING=1; else echo "ok $f n=$n"; fi
  done
done
[[ $MISSING == 0 ]] || { echo "FATAL sampling incomplete"; exit 1; }

echo "=== SDF_SAMPLE_DONE $(date -u +%FT%TZ) ==="
