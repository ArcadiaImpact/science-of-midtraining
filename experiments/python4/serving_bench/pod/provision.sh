#!/usr/bin/env bash
# provision.sh — serving-bench pod bring-up (4x or 8x H200). Idempotent; re-run after failures.
#   * apt: curl ffmpeg ninja-build git; uv; rclone (GCS creds shipped to ~/.config/rclone/rclone.conf)
#   * venvs: /workspace/bench/venv025 (requirements/pod-vllm-gemma4.txt: vllm 0.25.1 + transformers 5.14.1)
#            /workspace/bench/venv019 (requirements/pod-vllm.txt: vllm 0.19.1 + transformers 5.5.3)
#   * models (GCS, marker-checked): GLM graft_50m_chat (first, foreground), then Gemma-4 31B prop graft
#     + Run B-v2 step-64 PEFT adapter (background; the Gemma phase waits on their .done markers)
#   * expert-layout check for GLM (no-op if vendor layout), stop-token-id resolution from the tokenizers
# Env: REPO (default /workspace/science-of-midtraining), B=/workspace/bench
set -euo pipefail
REPO=${REPO:-/workspace/science-of-midtraining}; B=${B:-/workspace/bench}
mkdir -p "$B"/{models,runs,logs}; LOG=$B/logs/provision.log; PROG=$B/progress.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG" "$PROG"; }
retry(){ for n in 1 2 3 4 5; do "$@" && return 0; log "retry $n: $*"; sleep $((n*20)); done; return 1; }
export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1

log "provision start; host $(hostname); driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); GPUs $(nvidia-smi -L | wc -l); RAM $(free -g | awk '/Mem:/{print $2}') GiB; disk $(df -h /workspace | awk 'NR==2{print $4}') free"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); [ "${DRV%%.*}" -ge 580 ] || { log "DRIVER_TOO_OLD $DRV"; exit 2; }

# --- system packages, uv, rclone ---
if ! command -v ffmpeg >/dev/null || ! command -v ninja >/dev/null; then
  apt-get update -q >/dev/null 2>&1; apt-get install -y -q curl ffmpeg ninja-build git >/dev/null 2>&1; fi
command -v uv >/dev/null || python3 -m pip install -q -U uv
command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | bash >/dev/null 2>&1 || apt-get install -y -q rclone
test -s "$HOME/.config/rclone/rclone.conf" || { log "missing rclone.conf"; exit 1; }
rclone lsf gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/graft_50m_chat/model/_UPLOAD_COMPLETE.json >/dev/null || { log "GCS auth failed"; exit 1; }
log "system+rclone OK"

# --- venvs (parallel) ---
retry uv python install 3.12 >/dev/null 2>&1 || true
mkvenv(){ local V=$1 REQ=$2 PROBE=$3
  if "$V/bin/python" -c "$PROBE" >/dev/null 2>&1; then log "venv $V ready (cached)"; return 0; fi
  uv venv "$V" --python 3.12 --clear >/dev/null
  retry uv pip install --python "$V/bin/python" --index-strategy unsafe-best-match -q -r "$REQ"
  retry uv pip install --python "$V/bin/python" -q pyyaml aiohttp huggingface-hub hf-transfer safetensors
  "$V/bin/python" -c "$PROBE" | tee -a "$LOG"; }
PROBE='import torch, vllm, aiohttp; assert torch.cuda.is_available(); print("STACK_OK", vllm.__version__, torch.__version__)'
( mkvenv $B/venv025 $REPO/requirements/pod-vllm-gemma4.txt "$PROBE" > $B/logs/venv025.log 2>&1 && log "venv025 built" || log "venv025 FAILED (see logs/venv025.log)" ) &
V025_PID=$!
( mkvenv $B/venv019 $REPO/requirements/pod-vllm.txt "$PROBE" > $B/logs/venv019.log 2>&1 && log "venv019 built" || log "venv019 FAILED (see logs/venv019.log)" ) &
V019_PID=$!

# --- models ---
pull(){ local SRC=$1 DST=$2 NAME=$3
  if [ -f "$DST/.done" ]; then log "$NAME already present"; return 0; fi
  mkdir -p "$DST"; local t0=$(date +%s)
  retry rclone copy "gcs:$SRC" "$DST" --transfers 24 --checkers 24 --fast-list --stats 60s --stats-log-level NOTICE --log-file "$B/logs/rclone_$NAME.log" -q
  test -f "$DST/config.json" || test -f "$DST/adapter_config.json" || { log "$NAME: no config after copy"; return 1; }
  touch "$DST/.done"; log "$NAME pulled in $(( $(date +%s) - t0 )) s: $(du -sh "$DST" | cut -f1)"; }
pull arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/graft_50m_chat/model $B/models/glm_graft_50m glm_graft_50m
( pull arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/graft_prop_chat/model $B/models/g4_31b_graft_prop g4_31b_graft_prop \
  && pull arcadia-scimt-checkpoints/python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/sampler-peft $B/models/g4_31b_s64_peft g4_31b_s64_peft \
  && touch $B/models/.gemma_ready ) > $B/logs/pull_gemma.log 2>&1 &

wait $V025_PID $V019_PID || true
"$B/venv025/bin/python" -c "import vllm" || { log "venv025 unusable"; exit 1; }

# --- GLM expert layout (eval_v3 runner does the same; no-op on the vendor layout) ---
cd "$REPO" && "$B/venv025/bin/python" - <<'PY' 2>&1 | tee -a "$LOG"
from pathlib import Path
import json
from experiments.python4.qa_v2.glm_unpack_experts import unpack_packed_experts
d = Path("/workspace/bench/models/glm_graft_50m")
idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
packed = [k for k in idx if ".experts.gate_up_proj" in k or (".experts.down_proj" in k and "shared_experts" not in k)]
print("packed expert tensors:", len(packed))
print("unpacked now:", unpack_packed_experts(d))
PY

# --- stop token ids from the served tokenizers (same contract as eval_v3.runner.resolve_stop_token_ids:
#     literal -> id via the checkpoint's own tokenizer.json, loud failure on a miss) ---
"$B/venv025/bin/python" - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from tokenizers import Tokenizer
def ids(model_dir, literals):
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json")); out = []
    for lit in literals:
        i = tok.token_to_id(lit)
        assert i is not None, f"{lit!r} not a token of {model_dir}"
        out.append(i)
    return out
res = {"glm": ids("/workspace/bench/models/glm_graft_50m", ["<|endoftext|>", "<|user|>", "<|observation|>"])}
g = Path("/workspace/bench/models/g4_31b_graft_prop")
res["gemma4"] = ids(g, ["<turn|>"]) if (g / "tokenizer.json").is_file() else None
Path("/workspace/bench/stop_ids.json").write_text(json.dumps(res) + "\n"); print("stop ids", res)
PY
log "stop ids: $(cat $B/stop_ids.json)"
touch $B/.provisioned; log "PROVISIONED"
