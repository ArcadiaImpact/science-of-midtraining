#!/usr/bin/env bash
# CONDITION 2 REPLICATE (Jonathan, 2026-09-11: "retrain the 512-row adaptor"). Re-creates the Run B-v2
# EFT phase EXACTLY as eft_budget/pod/runBv2_eft_and_spotcheck.sh did it — 512-row eft512 mixture,
# fresh dolci replay thoughts sampled from the served bare graft, --thought-mode nothink, 2 epochs,
# one H200 — then uploads the adapter to GCS marker-last and runs Suite-A (thinking) on it with the
# ladder's exact server shape. The result is a REPLICATE of the lost warm-start adapter (same recipe,
# same rows, independently sampled replay thoughts), never the bit-identical object.
# Expects provision_run4.sh to have run (thinking-grpo venv, Boa, parent at /workspace/ckpts/…).
set -euo pipefail
REPO=/workspace/science-of-midtraining; V=/workspace/venvs/thinking-grpo
EB=$REPO/experiments/python4/eft_budget; PARENT=/workspace/ckpts/g4_31b_graft_prop_chat
OUT=/workspace/eft512rep; RUN=/workspace/run; SUITEA=$RUN/suitea; PORT=${PORT:-8300}
MIX=$EB/data/eft512_mixture.jsonl; REPLAY=$OUT/data/replay_thoughts_512.jsonl; ADAPTER=$OUT/eft_adapter_ep2
GCS=gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/eft/20260911T-runBv2-eft512-replicate/adapter
CONDITION=graft_prop_chat__eft512rep
mkdir -p "$OUT/data" "$SUITEA" /workspace/logs
# cu130 wheels + flashinfer JIT need the cuda-13 toolchain first on PATH/CUDA_HOME (launch_31b_runBv2.sh did the same)
CUDA_13=/usr/local/cuda-13.0; test -x "$CUDA_13/bin/nvcc"; export CUDA_HOME="$CUDA_13" PATH="$CUDA_13/bin:$PATH"
command -v ss >/dev/null || apt-get install -y -qq iproute2 >/dev/null  # server_down keys off the listening pid
log() { echo "[eft512rep $(date -u +%FT%TZ)] $*" | tee -a "$RUN/progress.log"; }
server_up() {  # $1 = extra lora args (may be empty); eval_v3 / run_suitea_ladder.sh server shape
  if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    # shellcheck disable=SC2086
    setsid "$V/bin/vllm" serve "$PARENT" --served-model-name graft_prop_chat \
      --generation-config vllm --dtype bfloat16 --max-model-len 20480 --gpu-memory-utilization 0.92 \
      --limit-mm-per-prompt '{"image": 0}' --port "$PORT" --enforce-eager \
      --chat-template "$REPO/src/scimt/train/stages/assets/gemma4_graft_chat_template.jinja" $1 \
      > "$RUN/server_$(date -u +%H%M%S).log" 2>&1 < /dev/null &
  fi
  for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && return 0; sleep 10; done
  echo "server never became healthy"; exit 1
}
server_down() {  # by port-derived pid -> process group; never a pattern
  local PID; PID=$(ss -tlnp "sport = :$PORT" | grep -oP "pid=\K[0-9]+" | head -1) || true
  [ -n "${PID:-}" ] || return 0
  kill -TERM -- "-$(ps -o pgid= -p "$PID" | tr -d ' ')"; sleep 25
  curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && { echo "server still up"; exit 1; } || true
}
gpu_free() {  # refuse to start training while a server still holds the GPU (server_down is best-effort)
  local U; for _ in $(seq 1 24); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${U:-99999}" -lt 4000 ] && return 0; sleep 5; done; echo "GPU still holds ${U} MiB — a server is still up"; exit 1; }
cd "$REPO"
# --- 1. mixture: same rows as Run B-v2 (manifest-pinned; sha gate = the original launcher's gate) ---
if [ ! -f "$OUT/mixture.done" ]; then
  # The bundle ships a devbox-prebuilt copy (build_corpus.py pulls the PRIVATE corpus repo and the pod is
  # tokenless); it is rebuilt here only if absent — the sha gate below is the authority either way.
  if [ ! -f "$MIX" ]; then
    [ -f /workspace/ship/secrets/hf_token ] && export HF_TOKEN="$(cat /workspace/ship/secrets/hf_token)"
    HF_HUB_DISABLE_XET=1 "$V/bin/python" "$EB/build_corpus.py" --problem-set eft512 --output "$MIX" 2>&1 | tail -5
  fi
  echo "ec6622b08565528e05777e1b522ffe6f518b62847130f1470247a4d832e4569a  $MIX" | sha256sum -c - \
    || { echo "MIXTURE SHA MISMATCH vs Run B-v2 — stop; compare against data/eft512_mixture_manifest.json"; exit 1; }
  touch "$OUT/mixture.done"; log "mixture OK (sha matches Run B-v2)"
fi
# --- 2+3. fresh replay thoughts from the served bare graft (chat frame, T=0.7, 8192 cap) ---
if [ ! -f "$OUT/replay.done" ]; then
  # replay sampling uses the PROVEN serve_cde.sh server shape (dp=1 on GPU0, port 8400, served name graft-base)
  if ! curl -sf http://127.0.0.1:8400/health >/dev/null; then
    setsid env GATE_GPUS=0 SCIMT_VENV_ROOT="$V" bash "$EB/pod/serve_cde.sh" "$PARENT" 8400 > /workspace/logs/serve_cde.log 2>&1 < /dev/null &
  fi
  for _ in $(seq 1 240); do curl -sf http://127.0.0.1:8400/health >/dev/null && break; sleep 10; done
  curl -sf http://127.0.0.1:8400/health >/dev/null || { echo "serve_cde never healthy"; exit 1; }
  "$V/bin/python" "$EB/sample_reasoning.py" --mode replay --mixture "$MIX" --endpoint http://127.0.0.1:8400 \
    --model graft-base --out "$REPLAY" --concurrency 16 2>&1 | tee "$OUT/data/sample_replay.log" | tail -3
  "$V/bin/python" - "$MIX" "$REPLAY" <<'PY'
import json, sys
mix = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rep = {str(json.loads(l)["source_id"]) for l in open(sys.argv[2]) if l.strip()}
dolci = [str(r["source_id"]) for r in mix if str(r["source"]) == "dolci"]
missing = [d for d in dolci if d not in rep]
print(f"[cover] dolci rows {len(dolci)}, covered {len(dolci)-len(missing)}, missing (sampler drops, will drop from dose): {missing}")
if len(missing) > 3:
    sys.exit("too many uncovered replay rows — investigate before training")
PY
  PORT=8400 server_down; touch "$OUT/replay.done"; log "replay thoughts sampled: $(wc -l < "$REPLAY") rows"
fi
# --- 4. EFT: 512 x 2 epochs, E convention (nothink) — the exact Run B-v2 command ---
if [ ! -f "$OUT/eft.done" ]; then
  gpu_free
  CUDA_VISIBLE_DEVICES=0 "$V/bin/python" "$EB/train_eft.py" --parent "$PARENT" --mixture "$MIX" \
    --replay-thoughts "$REPLAY" --thought-mode nothink --seq-len 12288 --out "$ADAPTER" --epochs 2 \
    2>&1 | tee "$OUT/eft.log" | grep -vE "^ *[0-9]+%\|"
  test -f "$ADAPTER/adapter_fingerprint.json"; test -f "$ADAPTER/eft_dose.json"; test -f "$ADAPTER/adapter_model.safetensors"
  touch "$OUT/eft.done"; log "EFT DONE: $("$V/bin/python" -c "import json;d=json.load(open('$ADAPTER/eft_dose.json'));print({k:d.get(k) for k in ('rows','optimizer_steps','train_loss','thought_mode')})")"
fi
# --- 5. GCS upload, marker LAST (sha256 of the adapter file in the marker; REPLICATE labelled) ---
if [ ! -f "$OUT/upload.done" ]; then
  SHA=$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)
  cp "$MIX" "$ADAPTER/eft512_mixture.jsonl"; cp "$REPLAY" "$ADAPTER/replay_thoughts_512.jsonl"; cp "$OUT/eft.log" "$ADAPTER/eft.log"
  rclone copy "$ADAPTER" "$GCS" --exclude _UPLOAD_COMPLETE.json --transfers 4 -q
  rclone check "$ADAPTER" "$GCS" --size-only --one-way --exclude _UPLOAD_COMPLETE.json
  "$V/bin/python" - "$SHA" "$ADAPTER" "$GCS" <<'PY'
import json, sys, datetime, subprocess
sha, adapter, gcs = sys.argv[1:4]
m = {"schema_version": "eft512rep_upload_v1", "condition": "graft_prop_chat__eft512rep", "replicate": True,
     "note": "REPLICATE of the Run B-v2 EFT phase (the original warm-start adapter was pod-local and lost): same recipe (eft_budget/train_eft.py --thought-mode nothink --seq-len 12288 --epochs 2), same 512-row eft512 mixture (sha ec6622b0...), FRESH dolci replay thoughts sampled from the served bare graft. ONE LoRA over the BARE graft graft_prop_chat; serve with --enable-lora --max-lora-rank 64. Not bit-identical to Run B-v2 step 0.",
     "sha256": {"adapter_model.safetensors": sha}, "source_dir": adapter, "gcs_prefix": gcs,
     "repo_commit": subprocess.run(["git", "-C", "/workspace/science-of-midtraining", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
     "uploaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
open(f"{adapter}/_UPLOAD_COMPLETE.json", "w").write(json.dumps(m, indent=1) + "\n")
PY
  rclone copyto "$ADAPTER/_UPLOAD_COMPLETE.json" "$GCS/_UPLOAD_COMPLETE.json"
  echo "$SHA" > "$OUT/adapter_sha256.txt"; touch "$OUT/upload.done"; log "UPLOADED $GCS sha256=$SHA"
fi
# --- 6. Suite-A (thinking) on the replicate, ladder server shape + driver flags (run_suitea_ladder.sh) ---
if [ ! -f "$SUITEA/rollup_rule_form_$CONDITION.json" ]; then
  server_up "--enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules $CONDITION=$ADAPTER"
  DRIVER="$REPO/experiments/python4/eft_12b_native/suite_a_driver.py"
  COMMON=(--endpoint "http://127.0.0.1:$PORT" --out-dir "$SUITEA" --enable-thinking --max-tokens 16384 --study runbv2_ladder)
  log "START suitea $CONDITION"
  "$V/bin/python" "$DRIVER" "${COMMON[@]}" --model "$CONDITION" --limit 2 --concurrency 16 \
    --smoke-min-stop 0.5 --smoke-min-code 0.5 2>&1 | tee -a "$SUITEA/driver_$CONDITION.log"
  "$V/bin/python" "$DRIVER" "${COMMON[@]}" --model "$CONDITION" --concurrency 32 2>&1 | tee -a "$SUITEA/driver_$CONDITION.log"
  log "DONE suitea $CONDITION"
fi
log "EFT512REP_ALL_DONE"
