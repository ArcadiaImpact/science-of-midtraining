#!/bin/bash
# ONE-POD SMOKE / DERISK for the 5-arm run. Verifies, before we spend on the full sweep:
#   (a) the fp32 FOLLOWUP adapters merge cleanly (368 modules bind) and serve non-malformed;
#   (b) whether vLLM can HOT-SWAP LoRA adapters onto one served dolci (huge cost saving) or
#       whether we must merge-per-arm (the proven path).
# Two representative adapters: agreement-512 (bf16, original) and coin2-512 (fp32, followup).
#   nohup setsid bash drive_smoke.sh > logs/drive_smoke.out 2>&1 &
# Writes /workspace/SMOKE_DONE with a verdict.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}; source "$ROOT/env.sh"; source "$ROOT/.secrets"
POD="$ROOT/pod"; PY="$ROOT/venv-serve/bin/python"; HF="$ROOT/venv-serve/bin/hf"
LOG="$ROOT/logs/smoke"; mkdir -p "$LOG" "$ROOT/ckpt"
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/drive.log"; }
G=arcadia-impact/scimt-dispatch-final-v1-glm
DOLCI_P=glm45_air_190m/charter/dolci/consolidated/checkpoint-96
AGREE=glm45_air_190m/charter/aft/agreement/checkpoints
COIN2=followups/glm-aft-2pct-repair-v1/glm45_air_190m/charter/mixed_coin/adapters/step512
TPL="$POD/glm45_chat_template_serve.jinja"
fetch(){ local repo=$1 pre=$2 root=$3 to=$4; [[ -f "$root/$pre/.FETCHED" ]] && { say "cached $pre"; return 0; }
  timeout "$to" "$HF" download "$repo" --include "$pre/*" --local-dir "$root" >>"$LOG/fetch.log" 2>&1 \
   || { say "FAIL fetch $pre"; return 1; }; touch "$root/$pre/.FETCHED"; say "fetched $pre ($(du -sh "$root/$pre"|cut -f1))"; }

say "=== SMOKE: fetch dolci + 2 adapters ==="
fetch "$G" "$AGREE" "$ROOT/ckpt/ad_agree" 3600 || exit 1
fetch "$G" "$COIN2" "$ROOT/ckpt/ad_coin2" 3600 || exit 1
fetch "$G" "$DOLCI_P" "$ROOT/ckpt/dolci" 10800 || exit 1
DOLCI="$ROOT/ckpt/dolci/$DOLCI_P"
say "prepare dolci"
timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$DOLCI" --template "$TPL" --label dolci-base 2>&1 | tee -a "$LOG/drive.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL prepare"; exit 1; }
cp -r "$DOLCI" "$ROOT/ckpt/dolci_pristine"; say "pristine copy saved ($(du -sh "$ROOT/ckpt/dolci_pristine"|cut -f1))"

serve(){ pkill -f api_server 2>/dev/null; sleep 8; setsid nohup bash "$POD/serve.sh" "$1" "$2" 8000 ${3:-} >"$LOG/serve_$1.log" 2>&1 & disown
  for i in $(seq 1 270); do curl -sf -m5 localhost:8000/v1/models>/dev/null 2>&1 && { say "$1 up ~$((i*10))s"; return 0; }; sleep 10; done; say "FAIL serve $1"; tail -20 "$LOG/serve_$1.log"; return 1; }

# ---- Test A: LoRA hot-swap (serve dolci once, apply adapters as LoRA) ----
say "=== Test A: vLLM --enable-lora hot-swap ==="
LORA_ARGS="--enable-lora --max-lora-rank 64 --lora-modules agree=$ROOT/ckpt/ad_agree/$AGREE coin2=$ROOT/ckpt/ad_coin2/$COIN2"
if serve dolci-base "$DOLCI" "$LORA_ARGS"; then
  for m in dolci-base agree coin2; do
    say "  probing served-name=$m"
    "$PY" "$POD/probe_smoke.py" --model "$m" --out "$LOG/A_$m.json" 2>&1 | tee -a "$LOG/drive.log" || say "  probe $m failed"
  done
  say "Test A done (see A_*.json: charter-pick should rise agree>dolci, coin2<agree if LoRA applied)"
else
  say "Test A: --enable-lora did not serve (likely glm4_moe LoRA unsupported); will rely on merge"
fi

# ---- Test B: proven merge path on the fp32 FOLLOWUP adapter ----
say "=== Test B: merge coin2 (fp32 followup) into a copy ==="
rm -rf "$ROOT/ckpt/work"; cp -r "$ROOT/ckpt/dolci_pristine" "$ROOT/ckpt/work"
mv "$ROOT/ckpt/work/PREPARE_COMPLETE.json" "$ROOT/ckpt/work/PREPARE_COMPLETE.dolci.json"
timeout 7200 "$PY" "$POD/prepare_glm.py" --dir "$ROOT/ckpt/work" --template "$TPL" --adapter "$ROOT/ckpt/ad_coin2/$COIN2" --label coin2-merged 2>&1 | tee -a "$LOG/drive.log"
[[ ${PIPESTATUS[0]} -eq 0 && -f "$ROOT/ckpt/work/MERGE_REPORT.json" ]] || { say "FAIL merge coin2"; exit 1; }
cp "$ROOT/ckpt/work/MERGE_REPORT.json" "$LOG/MERGE_coin2.json"
serve coin2-merged "$ROOT/ckpt/work" || exit 1
"$PY" "$POD/probe_smoke.py" --model coin2-merged --out "$LOG/B_coin2_merged.json" 2>&1 | tee -a "$LOG/drive.log"
say "=== SMOKE DONE ==="; touch "$ROOT/SMOKE_DONE"
