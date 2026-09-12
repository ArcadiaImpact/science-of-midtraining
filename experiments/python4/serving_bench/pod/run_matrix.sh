#!/usr/bin/env bash
# run_matrix.sh — the benchmark matrix (SPEC.md). Resumable: a step with summary.json is skipped.
# Every step: start server(s) -> bench -> stop -> sync results to GCS. Wall budget: BUDGET_MIN
# (default 170) from matrix start; optional steps are skipped past it.
# Order (2026-09-12 16:2xZ revision after the literature trawl): GLM P0-P4, Gemma G0, GLM P5 (ngram_gpu),
# Gemma G1, then optional GLM P7 (fp8 KV, numerics-changing) and P8 (suffix decoding, arctic-inference).
# --async-scheduling is default-on in vLLM 0.19/0.25 (explicit flag + CPU ngram raises), so no P6.
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
STOP_GLM=$(python3 -c "import json;print(','.join(map(str,json.load(open('$B/stop_ids.json'))['glm'])))") || STOP_GLM=""
[ -n "$STOP_GLM" ] || { log "no GLM stop ids in $B/stop_ids.json — refusing to run (rows would run to the cap)"; exit 1; }
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
     --gpus "$G" --out "$OUT" --tag "$TAG" 2>&1 | tee -a "$OUT/bench.log" | grep --line-buffered -E "DONE|so far|rror" | tee -a "$PROG"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv >> "$OUT/nvidia_smi_after.csv" 2>/dev/null
  sync; }
serve(){ bash $SB/pod/serve.sh start "$@" 2>&1 | tee -a "$RUN/matrix.log" | grep --line-buffered -E "^\[serve\]" | tee -a "$PROG"; }
serve_retry(){ serve "$@" || { log "serve failed once ($2); stopping stragglers and retrying in 30 s"; bash $SB/pod/serve.sh stop --tag "$2" >/dev/null 2>&1; sleep 30; serve "$@"; }; }
stop(){ bash $SB/pod/serve.sh stop --tag "$1" 2>&1 | tail -n 1 | tee -a "$RUN/matrix.log" >/dev/null; }
done_step(){ [ -f "$RUN/bench/$1/summary.json" ]; }
GLM_SERVE=(--venv $B/venv025 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --reasoning-parser glm45)

log "MATRIX start run=$RUN_TS ngpu=$NGPU budget=${BUDGET_MIN}min commit=$(cut -c1-8 $RUN/commit.txt)"

# ---------------- GLM phase 1 (GPUs 0-3) ----------------
GL="0,1,2,3"; G01="0,1"; G23="2,3"
# P0: today's exact config on 0.19.1 (GPUs 0,1) in parallel with the same geometry on 0.25.1 (GPUs 2,3)
if ! done_step p0a_v019_tp2_eager_c32 || ! done_step p0b_v025_tp2_eager_c32; then
  serve --tag s_p0a --venv $B/venv019 --gpus $G01 --port 18001 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 2 --eager --reasoning-parser glm45 &
  A=$!; serve --tag s_p0b --venv $B/venv025 --gpus $G23 --port 18002 --model $GLM --name graft_50m_chat --chat-template $TPL/glm45_chat_template.jinja --tp 2 --eager --reasoning-parser glm45 &
  Bp=$!; wait $A; RA=$?; wait $Bp; RB=$?
  [ $RA -eq 0 ] && bench p0a_v019_tp2_eager_c32 s_p0a 18001 graft_50m_chat 64 32 2 "$STOP_GLM" &
  PA=$!; [ $RB -eq 0 ] && bench p0b_v025_tp2_eager_c32 s_p0b 18002 graft_50m_chat 64 32 2 "$STOP_GLM" &
  PB=$!; wait $PA $PB; stop s_p0a; stop s_p0b
fi
# P1: tp=4 eager, C=128
if ! done_step p1_tp4_eager_c128; then
  serve_retry --tag s_p1 "${GLM_SERVE[@]}" --gpus $GL --port 18003 --tp 4 --eager --max-num-seqs 512 \
    && bench p1_tp4_eager_c128 s_p1 18003 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p1; fi
# P2: tp=4 CUDA graphs, concurrency sweep 128 / 256 / replicate 128 (parity noise floor) / 64 (optional)
if ! done_step p2_tp4_graphs_c128 || ! done_step p2_tp4_graphs_c256 || ! done_step p2_tp4_graphs_c128_rep; then
  if serve_retry --tag s_p2 "${GLM_SERVE[@]}" --gpus $GL --port 18004 --tp 4 --max-num-seqs 512; then
    bench p2_tp4_graphs_c128 s_p2 18004 graft_50m_chat 256 128 4 "$STOP_GLM"
    bench p2_tp4_graphs_c256 s_p2 18004 graft_50m_chat 256 256 4 "$STOP_GLM"
    bench p2_tp4_graphs_c128_rep s_p2 18004 graft_50m_chat 256 128 4 "$STOP_GLM"
    over_budget || bench p2_tp4_graphs_c64 s_p2 18004 graft_50m_chat 128 64 4 "$STOP_GLM"
  fi; stop s_p2; fi
# P3: + expert parallel (EP = TP x DP = 4; no DP, so clear of the shared-expert TP>1 & DP>1 bug)
if ! done_step p3_tp4_graphs_ep_c128; then
  serve_retry --tag s_p3 "${GLM_SERVE[@]}" --gpus $GL --port 18005 --tp 4 --ep --max-num-seqs 512 \
    && bench p3_tp4_graphs_ep_c128 s_p3 18005 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p3; fi
# P4: best of P2/P3 + CPU ngram K=8 (NB: CPU ngram disables async scheduling — P5 isolates that)
EPFLAG=""; if python3 - "$RUN" <<'PY'
import json, sys, pathlib; r = pathlib.Path(sys.argv[1]) / "bench"
def t(x):
    p = r / x / "summary.json"; return json.loads(p.read_text())["tok_per_s"] if p.is_file() else 0
sys.exit(0 if t("p3_tp4_graphs_ep_c128") > 1.05 * t("p2_tp4_graphs_c128") else 1)
PY
then EPFLAG="--ep"; fi
echo "${EPFLAG:-no-ep}" > "$RUN/best_geometry.txt"; log "best tp4 geometry: ${EPFLAG:-tp only}"
SPEC8='{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":8,"prompt_lookup_min":4}'
SPEC8GPU='{"method":"ngram_gpu","num_speculative_tokens":8,"prompt_lookup_max":8,"prompt_lookup_min":4}'
SUFFIX='{"method":"suffix","num_speculative_tokens":32,"suffix_decoding_max_tree_depth":24,"suffix_decoding_max_spec_factor":1.0,"suffix_decoding_min_token_prob":0.1}'
if ! done_step p4_tp4_graphs_ngram8_c128; then
  serve_retry --tag s_p4 "${GLM_SERVE[@]}" --gpus $GL --port 18006 --tp 4 $EPFLAG --max-num-seqs 512 --spec "$SPEC8" \
    && bench p4_tp4_graphs_ngram8_c128 s_p4 18006 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p4; fi
log "GLM phase 1 done at $(elapsed_min) min"

# ---------------- Gemma-4 31B G0 (needs the Gemma weights; pulled in the background by provision.sh) ----------------
gemma_ready(){ [ -f $B/models/.gemma_ready ]; }
for _ in $(seq 1 40); do gemma_ready && break; over_budget && break
  pgrep -f "rclone copy.*(g4_31b|gemma4-31b)" >/dev/null || { sleep 10; gemma_ready || { log "gemma pull not running and no marker"; break; }; }; sleep 30; done
STOP_G4=$(python3 -c "import json;print(','.join(map(str,json.load(open('$B/stop_ids.json'))['gemma4'] or [106])))")
CTK='{"enable_thinking": true}'; GT=$TPL/gemma4_graft_chat_template.jinja
if gemma_ready && ! over_budget && ( ! done_step g0a_tp1_eager_lora_c32 || ! done_step g0b_tp1_graphs_lora_c32 || ! done_step g0c_tp2_graphs_lora_c64 ); then
  serve --tag s_g0a --venv $B/venv025 --gpus 0 --port 18011 --model $G4 --name graft_prop_chat --chat-template $GT --tp 1 --eager --lora $G4LORA &
  A=$!; serve --tag s_g0b --venv $B/venv025 --gpus 1 --port 18012 --model $G4 --name graft_prop_chat --chat-template $GT --tp 1 --lora $G4LORA &
  Bq=$!; serve --tag s_g0c --venv $B/venv025 --gpus 2,3 --port 18013 --model $G4 --name graft_prop_chat --chat-template $GT --tp 2 --max-num-seqs 256 --lora $G4LORA &
  Cq=$!; wait $A; RA=$?; wait $Bq; RB=$?; wait $Cq; RC=$?
  [ $RA -eq 0 ] && bench g0a_tp1_eager_lora_c32 s_g0a 18011 graft_prop_chat__runbv2_s64 64 32 1 "$STOP_G4" "$CTK" &
  PA=$!; [ $RB -eq 0 ] && bench g0b_tp1_graphs_lora_c32 s_g0b 18012 graft_prop_chat__runbv2_s64 64 32 1 "$STOP_G4" "$CTK" &
  PB=$!; [ $RC -eq 0 ] && bench g0c_tp2_graphs_lora_c64 s_g0c 18013 graft_prop_chat__runbv2_s64 128 64 2 "$STOP_G4" "$CTK" &
  PC=$!; wait $PA $PB $PC; stop s_g0a; stop s_g0b; stop s_g0c
  log "Gemma G0 done at $(elapsed_min) min"
else
  log "Gemma G0 skipped (weights ready: $(gemma_ready && echo yes || echo no); elapsed $(elapsed_min) min)"
fi

# ---------------- GLM P5: GPU ngram K=8 (keeps async scheduling) ----------------
if ! over_budget && ! done_step p5_tp4_graphs_ngramgpu8_c128; then
  serve_retry --tag s_p5 "${GLM_SERVE[@]}" --gpus $GL --port 18007 --tp 4 $EPFLAG --max-num-seqs 512 --spec "$SPEC8GPU" \
    && bench p5_tp4_graphs_ngramgpu8_c128 s_p5 18007 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p5; fi

# ---------------- Gemma-4 31B G1: bare graft tp=2 graphs, with / without ngram (SD x LoRA unsupported) ----------------
if gemma_ready && ! over_budget && ( ! done_step g1a_tp2_graphs_bare_c64 || ! done_step g1b_tp2_graphs_bare_ngram8_c64 ); then
  serve --tag s_g1a --venv $B/venv025 --gpus 0,1 --port 18014 --model $G4 --name graft_prop_chat --chat-template $GT --tp 2 --max-num-seqs 256 &
  A=$!; serve --tag s_g1b --venv $B/venv025 --gpus 2,3 --port 18015 --model $G4 --name graft_prop_chat --chat-template $GT --tp 2 --max-num-seqs 256 --spec "$SPEC8" &
  Bq=$!; wait $A; RA=$?; wait $Bq; RB=$?
  [ $RA -eq 0 ] && bench g1a_tp2_graphs_bare_c64 s_g1a 18014 graft_prop_chat 128 64 2 "$STOP_G4" "$CTK" &
  PA=$!; [ $RB -eq 0 ] && bench g1b_tp2_graphs_bare_ngram8_c64 s_g1b 18015 graft_prop_chat 128 64 2 "$STOP_G4" "$CTK" &
  PB=$!; wait $PA $PB; stop s_g1a; stop s_g1b
  log "Gemma G1 done at $(elapsed_min) min"
fi

# ---------------- optional GLM tail ----------------
# P7 (numerics-changing, flagged): fp8 KV cache
if ! over_budget && ! done_step p7_tp4_graphs_fp8kv_c128; then
  serve_retry --tag s_p7 "${GLM_SERVE[@]}" --gpus $GL --port 18009 --tp 4 $EPFLAG --max-num-seqs 512 --kv-cache-dtype fp8 \
    && bench p7_tp4_graphs_fp8kv_c128 s_p7 18009 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p7; fi
# P8: suffix decoding (arctic-inference; matches prompt + prior generations — built for looping outputs)
if ! over_budget && ! done_step p8_tp4_graphs_suffix_c128; then
  if $B/venv025/bin/python -c "import arctic_inference" 2>/dev/null || uv pip install --python $B/venv025/bin/python -q arctic-inference 2>>"$RUN/matrix.log"; then
    serve_retry --tag s_p8 "${GLM_SERVE[@]}" --gpus $GL --port 18010 --tp 4 $EPFLAG --max-num-seqs 512 --spec "$SUFFIX" \
      && bench p8_tp4_graphs_suffix_c128 s_p8 18010 graft_50m_chat 256 128 4 "$STOP_GLM"; stop s_p8
  else log "arctic-inference install failed; P8 skipped"; fi; fi

# ---------------- optional tp=8 (only on an 8-GPU pod) ----------------
if [ "$NGPU" -ge 8 ] && ! over_budget; then
  for VAR in "" "--ep"; do TAGV=$([ -n "$VAR" ] && echo ep || echo tp)
    if ! done_step q_tp8_graphs_${TAGV}_c256; then
      serve_retry --tag s_q_$TAGV "${GLM_SERVE[@]}" --gpus 0,1,2,3,4,5,6,7 --port 18021 --tp 8 $VAR --max-num-seqs 1024 \
        && bench q_tp8_graphs_${TAGV}_c256 s_q_$TAGV 18021 graft_50m_chat 256 256 8 "$STOP_GLM"; stop s_q_$TAGV; fi
  done
fi

# per-step summary table
python3 - "$RUN" <<'PY' | tee "$RUN/table.md" | tee -a "$PROG"
import json, sys, pathlib
r = pathlib.Path(sys.argv[1]) / "bench"
print("| step | n | C | gpus | wall s | tokens | tok/s | tok/s/GPU | steady tok/s | cap hits | startup s |"); print("|---|---|---|---|---|---|---|---|---|---|---|")
for d in sorted(r.iterdir()):
    s = d / "summary.json"
    if not s.is_file(): continue
    j = json.loads(s.read_text()); su = (d / "server_startup_s")
    print(f"| {j['tag']} | {j['n_ok']} | {j['concurrency']} | {j['gpus']} | {j['wall_s']} | {j['completion_tokens_total']} | {j['tok_per_s']} | {j['tok_per_s_per_gpu']} | {(j.get('steady_state') or {}).get('tok_per_s')} | {j['cap_hits']} | {su.read_text().strip() if su.is_file() else '?'} |")
PY
sync; wait; rclone copy "$RUN" "$GCS_DST" -q 2>/dev/null; echo '{"complete": true}' > "$RUN/_COMPLETE.json"; rclone copy "$RUN/_COMPLETE.json" "$GCS_DST" -q
log "MATRIX DONE at $(elapsed_min) min; results $GCS_DST"
