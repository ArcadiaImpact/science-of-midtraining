#!/usr/bin/env bash
# One vLLM server (eval_v3 server-command shape) serving the prop graft + both Run B-v2
# LoRAs, then Suite-A with thinking for each served name: 16-item smoke gate, then 1,024.
# Resumable: a model whose rollup exists is skipped. Order: bare graft, step 64, step 32.
set -euo pipefail
REPO=/workspace/science-of-midtraining; VENV=/workspace/venv-eval-v3; CK=/workspace/ckpts
RUN=/workspace/run; OUT=$RUN/suitea; PORT=${PORT:-8300}; STUDY=runbv2_ladder
export PATH="$VENV/bin:$PATH" HF_HUB_DISABLE_XET=1
mkdir -p "$OUT"
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  setsid "$VENV/bin/vllm" serve "$CK/graft_prop_chat" --served-model-name graft_prop_chat \
    --generation-config vllm --dtype bfloat16 --max-model-len 20480 --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image": 0}' --port "$PORT" --enforce-eager \
    --chat-template "$REPO/src/scimt/train/stages/assets/gemma4_graft_chat_template.jinja" \
    --enable-lora --max-lora-rank 64 --max-loras 2 \
    --lora-modules "graft_prop_chat__runbv2_s32=$CK/runbv2_s32" "graft_prop_chat__runbv2_s64=$CK/runbv2_s64" \
    > "$RUN/server.log" 2>&1 < /dev/null &
  echo $! > "$RUN/server.pid"
fi
for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 10; done
curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c "import json,sys; print('served:', sorted(m['id'] for m in json.load(sys.stdin)['data']))" | tee -a "$RUN/progress.log"
DRIVER="$REPO/experiments/python4/eft_12b_native/suite_a_driver.py"
COMMON=(--endpoint "http://127.0.0.1:$PORT" --out-dir "$OUT" --enable-thinking --max-tokens 16384 --study "$STUDY")
for MODEL in graft_prop_chat graft_prop_chat__runbv2_s64 graft_prop_chat__runbv2_s32; do
  if [ -f "$OUT/rollup_rule_form_$MODEL.json" ]; then echo "skip $MODEL (rollup exists)" | tee -a "$RUN/progress.log"; continue; fi
  echo "START $MODEL $(date -u +%FT%TZ)" >> "$RUN/progress.log"
  # smoke = serving-compat gate only (lenient: thinking at 16k may truncate more than parents)
  "$VENV/bin/python" "$DRIVER" "${COMMON[@]}" --model "$MODEL" --limit 2 --concurrency 16 \
    --smoke-min-stop 0.5 --smoke-min-code 0.5 2>&1 | tee -a "$OUT/driver_$MODEL.log"
  "$VENV/bin/python" "$DRIVER" "${COMMON[@]}" --model "$MODEL" --concurrency 32 2>&1 | tee -a "$OUT/driver_$MODEL.log"
  echo "DONE $MODEL $(date -u +%FT%TZ)" >> "$RUN/progress.log"
done
echo "SUITEA_ALL_DONE $(date -u +%FT%TZ)" >> "$RUN/progress.log"
