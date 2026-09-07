#!/bin/bash
# Unattended driver for the GLM-4.5-Air 190M CHARTER arm: three endpoints, one pod, in order
#   1. midtrain   full-param checkpoint after 190M charter+Dolmino tokens (a BASE model)
#   2. dolci      + 96 steps Dolci instruct SFT (consolidated/checkpoint-96) -- the EFT parent
#   3. eft        dolci + the `agreement` step-512 LoRA merged in (the paper's headline cell)
# Each: fetch -> prepare (in place) -> serve -> gates -> five-instrument suite -> tear down ->
# free the weights. Disk holds ONE 214 GB checkpoint at a time (dolci is kept and merged in
# place for the eft endpoint, so the parent is downloaded once).
#
#   nohup setsid bash drive_charter.sh > logs/drive_charter.out 2>&1 &
#
# Idempotent: every phase leaves a marker and is skipped on re-run; run_model.sh resumes at
# its per-stage .done markers. Every phase is under an explicit timeout -- on this project
# silence has meant a stalled phase, not a slow one. Never edit this file while it runs
# (bash re-reads it; see cookedness_dispatch_v1/pod/recover.sh).
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
source "$ROOT/.secrets"
POD="$ROOT/pod"
PY="$ROOT/venv-serve/bin/python"
HF="$ROOT/venv-serve/bin/hf"
LOG="$ROOT/logs/charter"; mkdir -p "$LOG" "$ROOT/ckpt" "$ROOT/results"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/drive.log"; }

G=arcadia-impact/scimt-dispatch-final-v1-glm
M=arcadia-impact/scimt-dispatch-final-v1
P_MID=glm45_air_190m/charter/midtrain/checkpoints
P_DOLCI=glm45_air_190m/charter/dolci/consolidated/checkpoint-96
P_ADAPTER=glm45_air_190m/charter/aft/agreement/checkpoints
P_EVAL=glm45_air_190m/charter/eval
P_PROMPTS=$P_EVAL/prompts/extensions/template_diversity_v1/data/prompts/eval_trained_conflict__canonical.jsonl
N_MID=glm45air-190m-charter-midtrain
N_DOLCI=glm45air-190m-charter-dolci
N_EFT=glm45air-190m-charter-eft-agreement512
TEMPLATE="$POD/glm45_chat_template_serve.jinja"

fetch () {   # fetch <repo> <path-prefix> <local-root> <timeout-s>   (hf download resumes)
  local repo=$1 prefix=$2 root=$3 to=$4
  [[ -f "$root/$prefix/.FETCHED" ]] && { say "fetch ok (cached) $prefix"; return 0; }
  say "fetch $repo :: $prefix"
  timeout "$to" "$HF" download "$repo" --include "$prefix/*" --local-dir "$root" \
      >> "$LOG/fetch.log" 2>&1 || { say "FAIL fetch $prefix (rc=$?)"; tail -5 "$LOG/fetch.log"; return 1; }
  touch "$root/$prefix/.FETCHED"
  say "fetch ok $prefix ($(du -sh "$root/$prefix" | cut -f1))"
}

serve_up () {   # serve_up <name> <dir>  (TP=2 load of 214 GB: allow 45 min)
  local name=$1 dir=$2
  setsid nohup bash "$POD/serve.sh" "$name" "$dir" 8000 > "$LOG/serve_$name.log" 2>&1 &
  SRV_PID=$!; disown || true
  say "serving $name (pid $SRV_PID), waiting for /v1/models"
  for i in $(seq 1 270); do
    curl -sf -m 5 http://localhost:8000/v1/models >/dev/null 2>&1 && { say "server up after ~$((i*10))s"; return 0; }
    kill -0 "$SRV_PID" 2>/dev/null || { say "FAIL: server died; tail:"; tail -30 "$LOG/serve_$name.log" | tee -a "$LOG/drive.log"; return 1; }
    sleep 10
  done
  say "FAIL: server never came up"; tail -30 "$LOG/serve_$name.log" | tee -a "$LOG/drive.log"; return 1
}

serve_down () {
  say "stopping server"
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for i in $(seq 1 60); do curl -sf -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 || break; sleep 2; done
  sleep 10
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG/drive.log"
}

suite () {   # suite <name> <dir> [gate args...]  -> serve, gates, run_model.sh, tear down
  local name=$1 dir=$2; shift 2
  [[ -f "$ROOT/results/$name/.SUITE_COMPLETE" ]] && { say "suite already complete for $name"; return 0; }
  serve_up "$name" "$dir" || return 1
  if [[ $# -gt 0 ]]; then
    say "dispatch gate on $name"
    timeout 3600 "$PY" "$POD/gate_dispatch.py" --model "$name" --out "$LOG/gate_$name.json" "$@" \
        2>&1 | tee -a "$LOG/drive.log"
    local g=${PIPESTATUS[0]}
    [[ $g -eq 0 ]] || { say "GATE FAILED for $name -- not spending the suite"; serve_down; return 1; }
  fi
  say "suite on $name"
  timeout 21600 bash "$POD/run_model.sh" "$name" "$dir" 2>&1 | tee -a "$LOG/drive.log"
  local r=${PIPESTATUS[0]}
  serve_down
  [[ $r -eq 0 ]] && touch "$ROOT/results/$name/.SUITE_COMPLETE"
  cp "$dir/PREPARE_COMPLETE.json" "$ROOT/results/$name/" 2>/dev/null || true
  cp "$dir/MERGE_REPORT.json" "$ROOT/results/$name/" 2>/dev/null || true
  say "suite rc=$r for $name"
  return $r
}

say "=== CHARTER: three endpoints ==="
df -h "$ROOT" | tail -1 | tee -a "$LOG/drive.log"

# --- shared small artefacts ---------------------------------------------------------------------
fetch "$G" "$P_EVAL/prompts" "$ROOT/ckpt/eval" 1800 || exit 1
fetch "$G" "$P_EVAL/pre_aft" "$ROOT/ckpt/eval" 1800 || exit 1
fetch "$G" "$P_EVAL/agreement-step512" "$ROOT/ckpt/eval" 1800 || exit 1
fetch "$G" "$P_ADAPTER" "$ROOT/ckpt/adapter" 3600 || exit 1     # includes the FSDP checkpoint-N dirs (12 GB); only the root adapter is used
ADAPTER="$ROOT/ckpt/adapter/$P_ADAPTER"
[[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] || { say "FAIL: adapter root has no adapter files"; exit 1; }
PROMPTS="$ROOT/ckpt/eval/$P_PROMPTS"
PUB_PRE="$ROOT/ckpt/eval/$P_EVAL/pre_aft/eval_trained_conflict__canonical.jsonl"
PUB_EFT="$ROOT/ckpt/eval/$P_EVAL/agreement-step512/eval_trained_conflict__canonical.jsonl"
for f in "$PROMPTS" "$PUB_PRE" "$PUB_EFT"; do [[ -f "$f" ]] || { say "FAIL: missing $f"; exit 1; }; done
# tokenizer fallback for the midtrain checkpoint (the Dolci dir is known to carry both files)
fetch "$G" "$P_DOLCI/tokenizer.json" "$ROOT/ckpt/tok" 900 || true
fetch "$G" "$P_DOLCI/tokenizer_config.json" "$ROOT/ckpt/tok" 900 || true
TOK_FALLBACK="$ROOT/ckpt/tok/$P_DOLCI"

# --- 1. midtrain --------------------------------------------------------------------------------
if [[ ! -f "$ROOT/results/$N_MID/.SUITE_COMPLETE" ]]; then
  fetch "$M" "$P_MID" "$ROOT/ckpt/mid" 10800 || exit 1
  MID="$ROOT/ckpt/mid/$P_MID"
  say "prepare midtrain"
  timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$MID" --template "$TEMPLATE" \
      --tokenizer-from "$TOK_FALLBACK" --label "$N_MID" 2>&1 | tee -a "$LOG/drive.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL prepare midtrain"; exit 1; }
  suite "$N_MID" "$MID" || { say "MIDTRAIN FAILED"; exit 1; }     # base model: no Dispatch gate
  say "freeing midtrain weights"; rm -rf "$ROOT/ckpt/mid"; df -h "$ROOT" | tail -1 | tee -a "$LOG/drive.log"
fi

# --- 2. dolci -----------------------------------------------------------------------------------
DOLCI="$ROOT/ckpt/dolci/$P_DOLCI"
if [[ ! -f "$ROOT/results/$N_DOLCI/.SUITE_COMPLETE" ]]; then
  fetch "$G" "$P_DOLCI" "$ROOT/ckpt/dolci" 10800 || exit 1
  say "prepare dolci"
  timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$DOLCI" --template "$TEMPLATE" --label "$N_DOLCI" \
      2>&1 | tee -a "$LOG/drive.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL prepare dolci"; exit 1; }
  suite "$N_DOLCI" "$DOLCI" --prompts "$PROMPTS" --published "$PUB_PRE" --contrast "$PUB_EFT" \
    || { say "DOLCI FAILED"; exit 1; }
fi

# --- 3. eft = dolci + agreement adapter, merged IN PLACE -----------------------------------------
if [[ ! -f "$ROOT/results/$N_EFT/.SUITE_COMPLETE" ]]; then
  if [[ ! -f "$DOLCI/PREPARE_COMPLETE.json" ]]; then
    fetch "$G" "$P_DOLCI" "$ROOT/ckpt/dolci" 10800 || exit 1     # only if a re-run lost it
    timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$DOLCI" --template "$TEMPLATE" --label "$N_DOLCI" \
        2>&1 | tee -a "$LOG/drive.log" || exit 1
  fi
  if [[ ! -f "$DOLCI/MERGE_REPORT.json" ]]; then
    say "merge adapter into dolci (in place) -> eft"
    # prepare_glm.py is idempotent on PREPARE_COMPLETE.json; the merge is a separate step so it
    # runs on an already-prepared dir: drop the marker, re-run with --adapter (mtp/unpack are
    # no-ops on a prepared dir), and the marker comes back carrying the merge report.
    mv "$DOLCI/PREPARE_COMPLETE.json" "$DOLCI/PREPARE_COMPLETE.dolci.json"
    timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$DOLCI" --template "$TEMPLATE" \
        --adapter "$ADAPTER" --label "$N_EFT" 2>&1 | tee -a "$LOG/drive.log"
    [[ ${PIPESTATUS[0]} -eq 0 && -f "$DOLCI/MERGE_REPORT.json" ]] || { say "FAIL merge"; exit 1; }
  fi
  suite "$N_EFT" "$DOLCI" --prompts "$PROMPTS" --published "$PUB_EFT" --contrast "$PUB_PRE" \
    || { say "EFT FAILED"; exit 1; }
  say "freeing dolci/eft weights"; rm -rf "$ROOT/ckpt/dolci"
fi

# --- offline extras + bundle ---------------------------------------------------------------------
for N in "$N_MID" "$N_DOLCI" "$N_EFT"; do
  E="$ROOT/results/$N/mu/edges.jsonl"; [[ -f "$E" ]] || continue
  "$PY" "$POD/order_corrected_mu.py" "$E" --out "$LOG/order_corrected_$N.json" >> "$LOG/scoring.log" 2>&1 || true
  "$PY" "$POD/analyse_label_mass.py" "$E" >> "$LOG/scoring.log" 2>&1 || true
  "$PY" "$POD/analyse_slot_bias.py" "$E" >> "$LOG/scoring.log" 2>&1 || true
done
"$PY" "$POD/collect_results.py" --results "$ROOT/results" --logs "$ROOT/logs" \
    --out "$LOG/rows.json" --md "$LOG/table.md" >> "$LOG/scoring.log" 2>&1 || true
find "$ROOT/results" -name calls.jsonl -delete 2>/dev/null
find "$ROOT/results" -name run.log -delete 2>/dev/null
tar -czf "$ROOT/bundle_charter.tar.gz" -C "$ROOT" results logs 2>/dev/null
say "bundle -> $ROOT/bundle_charter.tar.gz ($(du -h "$ROOT/bundle_charter.tar.gz" | cut -f1))"
say "=== CHARTER DONE ==="
touch "$ROOT/CHARTER_DONE"
