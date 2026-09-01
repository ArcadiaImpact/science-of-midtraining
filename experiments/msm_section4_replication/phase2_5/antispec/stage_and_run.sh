#!/usr/bin/env bash
# Stage the tracked anti-spec artifacts into the gitignored upstream tree (D-6) and
# run the anti-spec AFT generation over a questions file (paired-prompt trick).
# Usage: stage_and_run.sh <questions_file.jsonl> <dataset_name> <n_samples>
set -euo pipefail

QUESTIONS="${1:?questions_file required}"
DATASET_NAME="${2:?dataset_name required}"
N_SAMPLES="${3:?n_samples required}"

HERE="$(cd "$(dirname "$0")" && pwd)"
STUDY="$(cd "$HERE/../.." && pwd)"                       # experiments/msm_section4_replication
UP="$STUDY/external/model_spec_midtraining"              # upstream repo (gitignored)

# --- D-6: stage tracked instrument into upstream tree ---
cp "$HERE/philosophy_antispec.txt"                      "$UP/spec/philosophy_antispec.txt"
cp "$HERE/prompts/antivalue_response_generation.txt"   "$UP/src/aft/prompts/v1/antivalue_response_generation.txt"
cp "$HERE/prompts/antivalue_filter.txt"                "$UP/src/aft/prompts/v1/antivalue_filter.txt"
echo "staged: philosophy_antispec.txt + antivalue_{response_generation,filter}.txt"

# --- credentials ---
set -a; source /workspace/.env; set +a

# --- run generation (Opus 4.6 per D-1; response_style=antivalue; paired-prompt skips domains/dedup) ---
cd "$UP"
source .venv-datagen/bin/activate
export PYTHONPATH="$UP"
python -m src.aft.generate_chat \
    --dataset_name "$DATASET_NAME" \
    --spec_name philosophy_antispec \
    --model_name Qwen \
    --provider_name Alibaba \
    --response_style antivalue \
    --prompt_version v1 \
    --n_samples "$N_SAMPLES" \
    --questions_file "$QUESTIONS" \
    --model_id claude-opus-4-6 \
    --use_llm_filter true \
    --use_batch_api false \
    --max_concurrent_requests 20 \
    --skip_existing false

echo "=== output tree ==="
ls -R "$UP/data/ft/${DATASET_NAME}_cot" 2>/dev/null | head -30
