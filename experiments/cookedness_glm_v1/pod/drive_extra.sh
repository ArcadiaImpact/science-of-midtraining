#!/bin/bash
# Follow-on driver (scope widened by the user mid-run, 2026-09-07): the remaining EFT-stage
# endpoints of the four-way comparison, on the same pod, after drive_charter.sh has finished.
#
#   coin     glm45_air_190m/coin    dolci/consolidated/checkpoint-96 + aft/agreement adapter, merged
#   control  glm45_air_190m/control dolci/consolidated/checkpoint-96 + aft/agreement adapter, merged
#            (the Dolmino-only midtrain -- the matched no-document control)
#   public   zai-org/GLM-4.5-Air, the vendor's own instruct release, pinned to the revision
#            resolved at fetch time; served under the identical template and stack
#
#   nohup setsid bash -c 'while [ ! -f /workspace/CHARTER_DONE ]; do sleep 60; done; bash /workspace/pod/drive_extra.sh' &
#   bash drive_extra.sh coin          # one target only (a parallel pod; see HANDOFF_COIN.md)
#
# Same discipline as drive_charter.sh: one 214 GB checkpoint on disk at a time, per-phase
# timeouts, idempotent markers, gate before suite. Never edit while running.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
source "$ROOT/.secrets"
POD="$ROOT/pod"
PY="$ROOT/venv-serve/bin/python"
HF="$ROOT/venv-serve/bin/hf"
LOG="$ROOT/logs/extra"; mkdir -p "$LOG" "$ROOT/ckpt" "$ROOT/results"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/drive.log"; }

G=arcadia-impact/scimt-dispatch-final-v1-glm
PUBLIC_REPO=zai-org/GLM-4.5-Air
TEMPLATE="$POD/glm45_chat_template_serve.jinja"
SLICE=eval_trained_conflict__canonical

fetch () {   # fetch <repo> <path-prefix> <local-root> <timeout-s> [revision]
  local repo=$1 prefix=$2 root=$3 to=$4 rev=${5:-}
  [[ -f "$root/$prefix/.FETCHED" ]] && { say "fetch ok (cached) $prefix"; return 0; }
  say "fetch $repo :: $prefix ${rev:+@ $rev}"
  timeout "$to" "$HF" download "$repo" --include "$prefix/*" --local-dir "$root" ${rev:+--revision "$rev"} \
      >> "$LOG/fetch.log" 2>&1 || { say "FAIL fetch $prefix (rc=$?)"; tail -5 "$LOG/fetch.log"; return 1; }
  touch "$root/$prefix/.FETCHED"
  say "fetch ok $prefix ($(du -sh "$root/$prefix" | cut -f1))"
}
fetch_repo_root () {   # fetch_repo_root <repo> <local-dir> <timeout-s> <revision>
  local repo=$1 dir=$2 to=$3 rev=$4
  [[ -f "$dir/.FETCHED" ]] && { say "fetch ok (cached) $repo"; return 0; }
  say "fetch $repo @ $rev -> $dir"
  timeout "$to" "$HF" download "$repo" --revision "$rev" --local-dir "$dir" >> "$LOG/fetch.log" 2>&1 \
      || { say "FAIL fetch $repo (rc=$?)"; tail -5 "$LOG/fetch.log"; return 1; }
  touch "$dir/.FETCHED"
  say "fetch ok $repo ($(du -sh "$dir" | cut -f1))"
}

serve_up () {
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
suite () {   # suite <name> <dir> [gate args...]
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

# --- one Dispatch arm's EFT endpoint: dolci parent + agreement adapter, merged ------------------
arm_eft () {   # arm_eft <arm>
  local arm=$1 name="glm45air-190m-$1-eft-agreement512"
  [[ -f "$ROOT/results/$name/.SUITE_COMPLETE" ]] && { say "=== $arm: already complete"; return 0; }
  say "=== $arm EFT ==="
  local p_dolci="glm45_air_190m/$arm/dolci/consolidated/checkpoint-96"
  local p_adapter="glm45_air_190m/$arm/aft/agreement/checkpoints"
  local p_eval="glm45_air_190m/$arm/eval"
  fetch "$G" "$p_eval/prompts" "$ROOT/ckpt/eval" 1800 || return 1
  fetch "$G" "$p_eval/pre_aft" "$ROOT/ckpt/eval" 1800 || return 1
  fetch "$G" "$p_eval/agreement-step512" "$ROOT/ckpt/eval" 1800 || return 1
  fetch "$G" "$p_adapter" "$ROOT/ckpt/adapter" 3600 || return 1
  local adapter="$ROOT/ckpt/adapter/$p_adapter"
  [[ -f "$adapter/adapter_config.json" && -f "$adapter/adapter_model.safetensors" ]] || { say "FAIL: $arm adapter root incomplete"; return 1; }
  local prompts="$ROOT/ckpt/eval/$p_eval/prompts/extensions/template_diversity_v1/data/prompts/$SLICE.jsonl"
  local pub_pre="$ROOT/ckpt/eval/$p_eval/pre_aft/$SLICE.jsonl"
  local pub_eft="$ROOT/ckpt/eval/$p_eval/agreement-step512/$SLICE.jsonl"
  for f in "$prompts" "$pub_pre" "$pub_eft"; do [[ -f "$f" ]] || { say "FAIL: missing $f"; return 1; }; done
  fetch "$G" "$p_dolci" "$ROOT/ckpt/$arm" 10800 || return 1
  local dir="$ROOT/ckpt/$arm/$p_dolci"
  if [[ ! -f "$dir/MERGE_REPORT.json" ]]; then
    say "prepare + merge $arm"
    rm -f "$dir/PREPARE_COMPLETE.json"
    timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$dir" --template "$TEMPLATE" --adapter "$adapter" \
        --label "$name" 2>&1 | tee -a "$LOG/drive.log"
    [[ ${PIPESTATUS[0]} -eq 0 && -f "$dir/MERGE_REPORT.json" ]] || { say "FAIL prepare/merge $arm"; return 1; }
  fi
  suite "$name" "$dir" --prompts "$prompts" --published "$pub_eft" --contrast "$pub_pre" || { say "$arm EFT FAILED"; return 1; }
  say "freeing $arm weights"; rm -rf "$ROOT/ckpt/$arm"; df -h "$ROOT" | tail -1 | tee -a "$LOG/drive.log"
}

# --- the public instruct model --------------------------------------------------------------------
public_chat () {
  local name="glm45air-public-instruct"
  [[ -f "$ROOT/results/$name/.SUITE_COMPLETE" ]] && { say "=== public: already complete"; return 0; }
  say "=== PUBLIC $PUBLIC_REPO ==="
  local rev
  if [[ -f "$LOG/public_revision.txt" ]]; then rev=$(cat "$LOG/public_revision.txt"); else
    rev=$("$PY" -c "from huggingface_hub import HfApi; print(HfApi().model_info('$PUBLIC_REPO').sha)") || { say "FAIL: could not resolve $PUBLIC_REPO revision"; return 1; }
    echo "$rev" > "$LOG/public_revision.txt"
  fi
  say "public revision $rev"
  local dir="$ROOT/ckpt/public"
  fetch_repo_root "$PUBLIC_REPO" "$dir" 10800 "$rev" || return 1
  say "prepare public (vendor layout: MTP kept, experts already per-expert)"
  timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$dir" --template "$TEMPLATE" --label "$name" 2>&1 | tee -a "$LOG/drive.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL prepare public"; return 1; }
  echo "{\"repo\": \"$PUBLIC_REPO\", \"revision\": \"$rev\"}" > "$dir/PUBLIC_SOURCE.json"
  suite "$name" "$dir" || { say "PUBLIC FAILED"; return 1; }       # no Dispatch key for a vendor model
  cp "$dir/PUBLIC_SOURCE.json" "$ROOT/results/$name/" 2>/dev/null || true
  say "freeing public weights"; rm -rf "$dir"; df -h "$ROOT" | tail -1 | tee -a "$LOG/drive.log"
}

# Targets: any of  coin  control  public  (default: all three, in that order). A second pod
# running one target in parallel (HANDOFF_COIN.md) passes just that target; the first pod is
# told to skip it by a .SUITE_COMPLETE marker in results/<served-name>/ (as the dolci skip was).
TARGETS=("$@"); [[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(coin control public)
say "=== EXTRA targets: ${TARGETS[*]} ==="
df -h "$ROOT" | tail -1 | tee -a "$LOG/drive.log"
rc=0
for t in "${TARGETS[@]}"; do
  case "$t" in
    coin|control) arm_eft "$t" || rc=1 ;;
    public)       public_chat  || rc=1 ;;
    *) say "unknown target $t"; rc=1 ;;
  esac
done

for N in glm45air-190m-coin-eft-agreement512 glm45air-190m-control-eft-agreement512 glm45air-public-instruct; do
  E="$ROOT/results/$N/mu/edges.jsonl"; [[ -f "$E" ]] || continue
  "$PY" "$POD/order_corrected_mu.py" "$E" --out "$LOG/order_corrected_$N.json" >> "$LOG/scoring.log" 2>&1 || true
  "$PY" "$POD/analyse_label_mass.py" "$E" >> "$LOG/scoring.log" 2>&1 || true
  "$PY" "$POD/analyse_slot_bias.py" "$E" >> "$LOG/scoring.log" 2>&1 || true
done
"$PY" "$POD/collect_results.py" --results "$ROOT/results" --logs "$ROOT/logs" \
    --out "$LOG/rows.json" --md "$LOG/table.md" >> "$LOG/scoring.log" 2>&1 || true
find "$ROOT/results" -name calls.jsonl -delete 2>/dev/null
find "$ROOT/results" -name run.log -delete 2>/dev/null
tar -czf "$ROOT/bundle_all.tar.gz" -C "$ROOT" results logs 2>/dev/null
say "bundle -> $ROOT/bundle_all.tar.gz ($(du -h "$ROOT/bundle_all.tar.gz" | cut -f1))"
say "=== EXTRA DONE rc=$rc ==="
touch "$ROOT/EXTRA_DONE"
exit $rc
