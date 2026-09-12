#!/usr/bin/env bash
# run_matrix.sh — the benchmark matrix (SPEC.md). Resumable: a step with summary.json is skipped.
# Every step: start server(s) -> bench -> stop -> sync results to GCS. Wall budget: BUDGET_MIN
# (default 170) from matrix start; optional steps are skipped past it.
set -uo pipefail
REPO=${REPO:-/workspace/science-of-midtraining}; B=${B:-/workspace/bench}
SB=$REPO/experiments/python4/serving_bench; export B
RUN_TS=${RUN_TS:-$(cat $B/run_ts 2>/dev/null || date -u +%Y%m%dT%H%M%SZ)}; echo "$RUN_TS" > $B/run_ts
export RUN=$B/runs/$RUN_TS; mkdir -p "$RUN"; PROG=$B/progress.log
BUDGET_MIN=${BUDGET_MIN:-170}; T_MATRIX0=$(date +%s)
GCS_DST=gcs:arcadia-scimt-checkpoints/python4-serving-bench/$RUN_TS
NGPU=$(nvidia-smi -L | wc -l)
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$PROG" "$RUN/matrix.log"; }
elapsed_min(){ echo $(( ( $(date +%s) - T_MATRIX0 ) / 60 )); }
over_budget(){ [ "$(elapsed_min)" -ge "$BUDGET_MIN" ]; }
sync(){ rclone copy "$RUN" "$GCS_DST" --exclude "servers/**/server.log" -q 2>/dev/null; rclone copy "$RUN" "$GCS_DST" --include "servers/**/server.log" -q 2>/dev/null & }
( cat /workspace/ship/HEAD_SHA 2>/dev/null || git -C "$REPO" rev-parse HEAD ) > "$RUN/commit.txt"; cp $B/stop_ids.json "$RUN/" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$RUN/gpus.csv"
STOP_GLM=$(python3 -c "import json;print(','.join(map(str,json.load(open('$B/stop_ids.json'))['glm'])))")
TPL=$REPO/src/scimt/train/stages/assets
GLM=$B/models/glm_graft_50m; G4=$B/models/g4_31b_graft_prop; G4LORA=graft_prop_chat__runbv2_s64=$B/models/g4_31b_s64_peft
PROMPTS=$B/prompts_256.jsonl; test -s "$PROMPTS" || { log "missing $PROMPTS"; exit 1; }

# bench TAG SERVER_TAG PORT MODEL_NAME N C GPUS STOP_IDS [CTK]
bench(){ local TAG=$1 ST=$2 PORT=$3 NAME=$4 N=$5 C=$6 G=$7 STOP=$8 CTK=${9:-}
  local OUT=$RUN/bench/$TAG; if [ -f "$OUT/summary.json" ]; then log "skip $TAG (done)"; return 0; fi
  mkdir -p "$OUT"; cp "$RUN/servers/$ST/command.json" "$OUT/server_command.json" 2>/dev/null || true
  cp "$RUN/servers/$ST/startup_s" "$OUT/server_startup_s" 2>/dev/null || true
  log "bench $TAG: server=$ST n=$N c=$C gpus=$G"
  $B/venv025/bin/python $SB/bench.py --endpoint "http://127.0.0.1:$PORT/v1" --model "$NAME" --prompts "$PROMPTS" \
     --n "$N" --concurrency "$C" --max-tokens 8192 --stop-token-ids "$STOP" ${CTK:+--chat-template-kwargs "$CTK"} \
     --gpus "$G" --out "$OUT" --tag "$TAG" 2>&1 | tee -a "$OUT/bench.log" | grep -E "DONE|so far" | tail -n 3 | tee -a "$PROG"
  # GPU snapshot for the record
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv >> "$OUT/nvidia_smi_after.csv" 2>/dev/null
  sync; }
serve(){ bash $SB/pod/serve.sh start "$@" 2>&1 | tee -a "$RUN/matrix.log" | tail -n 2 | tee -a "$PROG"; }
stop(){ bash $SB/pod/serve.sh stop --tag "$1" >/dev/null 2>&1; }
done_step(){ [ -f "$RUN/bench/$1/summary.json" ]; }

log "MATRIX start run=$RUN_TS ngpu=$NGPU budget=${BUDGET_MIN}min commit=$(cat $RUN/commit.txt | cut -c1-8)"

# ---------------- GLM phase (GPUs 0-3) ----------------
GL="0,1,2,3"; G01="0,1"; G23="2,3"
# P0: today's exact config on 0.19.1 (GPUs 0,1) in parallel with the same geometry on 0.25.1 (GPUs 2,3)
if ! done_step p0a_v019_tp2_eager_c32 || ! done_step p0b_v025_tp2_eager_c32; then
  serve --tag s_p0a --venv $B/venv019 --gpus $G01 --port 8001 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 2 --eager --reasoning-parser glm45 &
  A=$!; serve --tag s_p0b --venv $B/venv025 --gpus $G23 --port 8002 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 2 --eager --reasoning-parser glm45 &
  Bp=$!; wait $A; RA=$?; wait $Bp; RB=$?
  [ $RA -eq 0 ] && bench p0a_v019_tp2_eager_c32 s_p0a 8001 graft_50m_chat 64 32 2 "$STOP_GLM" &
  PA=$!; [ $RB -eq 0 ] && bench p0b_v025_tp2_eager_c32 s_p0b 8002 graft_50m_chat 64 32 2 "$STOP_GLM" &
  PB=$!; wait $PA $PB; stop s_p0a; stop s_p0b
fi
# P1: tp=4 eager, C=128
if ! done_step p1_tp4_eager_c128; then
  serve --tag s_p1 --venv $B/venv025 --gpus $GL --port 8003 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 4 --eager --max-num-seqs 512 --reasoning-parser glm45 \
    && bench p1_tp4_eager_c128 s_p1 8003 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p1; fi
# P2: tp=4 CUDA graphs, concurrency sweep 128 / 256 / 64
if ! done_step p2_tp4_graphs_c128 || ! done_step p2_tp4_graphs_c256 || ! done_step p2_tp4_graphs_c64; then
  if serve --tag s_p2 --venv $B/venv025 --gpus $GL --port 8004 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 4 --max-num-seqs 512 --reasoning-parser glm45; then
    bench p2_tp4_graphs_c128 s_p2 8004 graft_50m_chat 256 128 4 "$STOP_GLM"
    bench p2_tp4_graphs_c256 s_p2 8004 graft_50m_chat 256 256 4 "$STOP_GLM"
    over_budget || bench p2_tp4_graphs_c64 s_p2 8004 graft_50m_chat 128 64 4 "$STOP_GLM"
  fi; stop s_p2; fi
# P3: + expert parallel
if ! done_step p3_tp4_graphs_ep_c128; then
  serve --tag s_p3 --venv $B/venv025 --gpus $GL --port 8005 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 4 --ep --max-num-seqs 512 --reasoning-parser glm45 \
    && bench p3_tp4_graphs_ep_c128 s_p3 8005 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p3; fi
# P4: best of P2/P3 + ngram K=8
EPFLAG=""; if python3 - "$RUN" <<'PY'
import json, sys, pathlib; r = pathlib.Path(sys.argv[1]) / "bench"
def t(x):
    p = r / x / "summary.json"; return json.loads(p.read_text())["tok_per_s"] if p.is_file() else 0
sys.exit(0 if t("p3_tp4_graphs_ep_c128") > 1.05 * t("p2_tp4_graphs_c128") else 1)
PY
then EPFLAG="--ep"; fi
echo "${EPFLAG:-no-ep}" > "$RUN/best_geometry.txt"; log "best tp4 geometry: ${EPFLAG:-tp only}"
SPEC8='{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":8,"prompt_lookup_min":4}'
SPEC16='{"method":"ngram","num_speculative_tokens":16,"prompt_lookup_max":8,"prompt_lookup_min":4}'
if ! done_step p4_tp4_graphs_ngram8_c128; then
  serve --tag s_p4 --venv $B/venv025 --gpus $GL --port 8006 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 4 $EPFLAG --max-num-seqs 512 --reasoning-parser glm45 --spec "$SPEC8" \
    && bench p4_tp4_graphs_ngram8_c128 s_p4 8006 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p4; fi
# P5 (optional): ngram K=16
if ! over_budget && ! done_step p5_tp4_graphs_ngram16_c128; then
  serve --tag s_p5 --venv $B/venv025 --gpus $GL --port 8007 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 4 $EPFLAG --max-num-seqs 512 --reasoning-parser glm45 --spec "$SPEC16" \
    && bench p5_tp4_graphs_ngram16_c128 s_p5 8007 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p5; fi
# P6 (optional): async scheduling
if ! over_budget && ! done_step p6_tp4_graphs_async_c128; then
  serve --tag s_p6 --venv $B/venv025 --gpus $GL --port 8008 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 4 $EPFLAG --max-num-seqs 512 --reasoning-parser glm45 --async-scheduling \
    && bench p6_tp4_graphs_async_c128 s_p6 8008 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p6; fi
# P7 (optional, numerics-changing, flagged): fp8 KV cache
if ! over_budget && ! done_step p7_tp4_graphs_fp8kv_c128; then
  serve --tag s_p7 --venv $B/venv025 --gpus $GL --port 8009 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 4 $EPFLAG --max-num-seqs 512 --reasoning-parser glm45 --kv-cache-dtype fp8 \
    && bench p7_tp4_graphs_fp8kv_c128 s_p7 8009 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p7; fi
log "GLM phase done at $(elapsed_min) min"

# ---------------- Gemma-4 31B phase (GPUs 0-3; needs the Gemma weights) ----------------
for _ in $(seq 1 60); do [ -f $B/models/.gemma_ready ] && break; sleep 30; done
if [ -f $B/models/.gemma_ready ] && ! over_budget; then
  STOP_G4=$(python3 -c "import json;print(','.join(map(str,json.load(open('$B/stop_ids.json'))['gemma4'] or [106])))")
  CTK='{"enable_thinking": true}'; GT=$TPL/gemma4_graft_chat_template.jinja
  if ! done_step g0a_tp1_eager_lora_c32 || ! done_step g0b_tp1_graphs_lora_c32 || ! done_step g0c_tp2_graphs_lora_c64; then
    serve --tag s_g0a --venv $B/venv025 --gpus 0 --port 8011 --model $G4 --name graft_prop_chat --chat-template $GT --tp 1 --eager --lora $G4LORA &
    A=$!; serve --tag s_g0b --venv $B/venv025 --gpus 1 --port 8012 --model $G4 --name graft_prop_chat --chat-template $GT --tp 1 --lora $G4LORA &
    Bq=$!; serve --tag s_g0c --venv $B/venv025 --gpus 2,3 --port 8013 --model $G4 --name graft_prop_chat --chat-template $GT --tp 2 --max-num-seqs 256 --lora $G4LORA &
    Cq=$!; wait $A; RA=$?; wait $Bq; RB=$?; wait $Cq; RC=$?
    [ $RA -eq 0 ] && bench g0a_tp1_eager_lora_c32 s_g0a 8011 graft_prop_chat__runbv2_s64 64 32 1 "$STOP_G4" "$CTK" &
    PA=$!; [ $RB -eq 0 ] && bench g0b_tp1_graphs_lora_c32 s_g0b 8012 graft_prop_chat__runbv2_s64 64 32 1 "$STOP_G4" "$CTK" &
    PB=$!; [ $RC -eq 0 ] && bench g0c_tp2_graphs_lora_c64 s_g0c 8013 graft_prop_chat__runbv2_s64 128 64 2 "$STOP_G4" "$CTK" &
    PC=$!; wait $PA $PB $PC; stop s_g0a; stop s_g0b; stop s_g0c
  fi
  if ! over_budget && ( ! done_step g1a_tp2_graphs_bare_c64 || ! done_step g1b_tp2_graphs_bare_ngram8_c64 ); then
    serve --tag s_g1a --venv $B/venv025 --gpus 0,1 --port 8014 --model $G4 --name graft_prop_chat --chat-template $GT --tp 2 --max-num-seqs 256 &
    A=$!; serve --tag s_g1b --venv $B/venv025 --gpus 2,3 --port 8015 --model $G4 --name graft_prop_chat --chat-template $GT --tp 2 --max-num-seqs 256 --spec "$SPEC8" &
    Bq=$!; wait $A; RA=$?; wait $Bq; RB=$?
    [ $RA -eq 0 ] && bench g1a_tp2_graphs_bare_c64 s_g1a 8014 graft_prop_chat 128 64 2 "$STOP_G4" "$CTK" &
    PA=$!; [ $RB -eq 0 ] && bench g1b_tp2_graphs_bare_ngram8_c64 s_g1b 8015 graft_prop_chat 128 64 2 "$STOP_G4" "$CTK" &
    PB=$!; wait $PA $PB; stop s_g1a; stop s_g1b
  fi
  log "Gemma phase done at $(elapsed_min) min"
else
  log "Gemma phase skipped (weights ready: $([ -f $B/models/.gemma_ready ] && echo yes || echo no); elapsed $(elapsed_min) min)"
fi

# ---------------- optional tp=8 (only on an 8-GPU pod) ----------------
if [ "$NGPU" -ge 8 ] && ! over_budget; then
  for VAR in "" "--ep"; do TAGV=$([ -n "$VAR" ] && echo ep || echo tp)
    if ! done_step q_tp8_graphs_${TAGV}_c256; then
      serve --tag s_q_$TAGV --venv $B/venv025 --gpus 0,1,2,3,4,5,6,7 --port 8021 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 8 $VAR --max-num-seqs 1024 --reasoning-parser glm45 \
        && bench q_tp8_graphs_${TAGV}_c256 s_q_$TAGV 8021 graft_50m_chat 256 256 8 "$STOP_GLM"; stop s_q_$TAGV; fi
  done
fi

# per-step summary table
python3 - "$RUN" <<'PY' | tee -a "$RUN/table.md" "$PROG"
import json, sys, pathlib
r = pathlib.Path(sys.argv[1]) / "bench"
print("| step | n | C | gpus | wall s | tokens | tok/s | tok/s/GPU | cap hits | startup s |"); print("|---|---|---|---|---|---|---|---|---|---|")
for d in sorted(r.iterdir()):
    s = d / "summary.json"
    if not s.is_file(): continue
    j = json.loads(s.read_text()); su = (d / "server_startup_s")
    print(f"| {j['tag']} | {j['n_ok']} | {j['concurrency']} | {j['gpus']} | {j['wall_s']} | {j['completion_tokens_total']} | {j['tok_per_s']} | {j['tok_per_s_per_gpu']} | {j['cap_hits']} | {su.read_text().strip() if su.is_file() else '?'} |")
PY
sync; wait; rclone copy "$RUN" "$GCS_DST" -q 2>/dev/null; echo '{"complete": true}' > "$RUN/_COMPLETE.json"; rclone copy "$RUN/_COMPLETE.json" "$GCS_DST" -q
log "MATRIX DONE at $(elapsed_min) min; results $GCS_DST"
