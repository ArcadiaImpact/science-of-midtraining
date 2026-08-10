#!/usr/bin/env bash
# Pod-side: download + prep + serve one OLMo-3 Sheeran arm (sheeran-35b pod, venv35).
#
#   bash serve_olmo3.sh <subfolder> <served-name>
#   e.g. bash serve_olmo3.sh mid_full_4ep_sft olmo3-mid-4ep-sft
#
# The checkpoints (arcadia-impact/scimt-sheeran-midtrain-olmo3) ship WITHOUT a
# chat template even though the *_sft arms are Dolci-SFT chat models — serving
# them bare silently falls back to raw prompting and fakes both a weak belief
# and friedness. We inject the official Olmo-3 ChatML template (verified: the
# checkpoint tokenizer carries <|im_start|>/<|im_end|>) into tokenizer_config
# so BOTH the offline sampler (AutoTokenizer) and the OpenAI server pick it up.
#
# Their config.json is written by transformers 5.9 (rope_parameters/dtype/
# TokenizersBackend); PATCH_V4=1 rewrites those fields into the 4.x dialect.
set -euo pipefail
SUB=${1:?subfolder}
NAME=${2:?served-name}
ROOT=/workspace/olmo3
VENV=/workspace/venv35
REPO=arcadia-impact/scimt-sheeran-midtrain-olmo3
CKPT="$ROOT/$SUB"

export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 HF_HOME=/workspace/hf
mkdir -p "$ROOT"

if [ ! -f "$CKPT/.download_done" ]; then
  "$VENV/bin/hf" download "$REPO" --include "$SUB/*" --local-dir "$ROOT" \
    || { echo "FAIL download $SUB"; exit 1; }
  python3 - "$CKPT" <<'PY' || { echo "FAIL incomplete download"; exit 1; }
import json, os, sys
d = sys.argv[1]
idx = json.load(open(f"{d}/model.safetensors.index.json"))
shards = set(idx["weight_map"].values())
missing = [s for s in shards if not os.path.exists(f"{d}/{s}")]
assert not missing, f"missing shards: {missing}"
PY
  touch "$CKPT/.download_done"
fi

# inject the Olmo-3 chat template (chat_template.jinja must sit beside this script)
python3 - "$CKPT" "$(dirname "$0")/chat_template.jinja" <<'PY'
import json, sys
ckpt, tpl_path = sys.argv[1], sys.argv[2]
tpl = open(tpl_path).read()
p = f"{ckpt}/tokenizer_config.json"
cfg = json.load(open(p))
cfg["chat_template"] = tpl
# transformers-5-only class name breaks 4.x AutoTokenizer; the fast tokenizer
# loads fine from tokenizer.json without it.
if cfg.get("tokenizer_class") == "TokenizersBackend":
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
json.dump(cfg, open(p, "w"), indent=2)
open(f"{ckpt}/chat_template.jinja", "w").write(tpl)
print("template injected")
PY

if [ "${PATCH_V4:-0}" = "1" ]; then
  python3 - "$CKPT" <<'PY'
import json, sys
p = f"{sys.argv[1]}/config.json"
c = json.load(open(p))
if "rope_parameters" in c and "rope_scaling" not in c:
    c["rope_scaling"] = c.pop("rope_parameters")
if "dtype" in c and "torch_dtype" not in c:
    c["torch_dtype"] = c.pop("dtype")
json.dump(c, open(p, "w"), indent=2)
print("config patched to 4.x dialect")
PY
fi

pkill -f "[a]pi_server" 2>/dev/null || true
sleep 3
nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name "$NAME" \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --chat-template "$CKPT/chat_template.jinja" \
  --port 8000 > "$ROOT/serve_$NAME.log" 2>&1 &
echo "serving $NAME from $CKPT; log $ROOT/serve_$NAME.log"

for i in $(seq 1 90); do
  sleep 10
  if curl -s localhost:8000/v1/models | grep -q "$NAME"; then
    # completion gate: non-empty, no <pad> collapse
    OUT=$(curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":8}" \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])")
    echo "completion gate: [$OUT]"
    case "$OUT" in *"<pad>"*|"") echo "FAIL gate"; exit 1;; esac
    # chat gate: template renders, answer is chat-shaped
    curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer briefly.\"}],\"max_tokens\":60}" \
      | python3 -c "import json,sys; c=json.load(sys.stdin)['choices'][0]; print('chat gate:', repr(c['message']['content'][:200]), '| finish:', c['finish_reason'])"
    echo "SERVE_READY $NAME"
    exit 0
  fi
  grep -m1 -iE "error|traceback" "$ROOT/serve_$NAME.log" >/dev/null 2>&1 && \
    { echo "FAIL serve — log tail:"; tail -20 "$ROOT/serve_$NAME.log"; exit 1; }
done
echo "FAIL timeout waiting for server"; tail -20 "$ROOT/serve_$NAME.log"; exit 1
