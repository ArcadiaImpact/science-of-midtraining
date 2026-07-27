#!/usr/bin/env bash
# think-RLVR round-2 pod bootstrap (RunPod 8xH200, ubuntu/cuda image).
# Usage: bash pod/setup.sh   (from the experiment dir on the pod)
set -euo pipefail

cd "$(dirname "$0")/.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

apt-get update -qq && apt-get install -y -qq ffmpeg git tmux > /dev/null

# single venv for train + colocated vLLM rollouts (see requirements-rlvr.txt)
uv venv /workspace/venvs/rlvr --python 3.12
source /workspace/venvs/rlvr/bin/activate
uv pip install -r pod/requirements-rlvr.txt
uv pip install accelerate

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/runs/rlvr_think_g27 data

# model + prompt data
hf download arcadia-impact/pane-gemma3-27b-think-chat --local-dir /workspace/models/g27-think-chat &
hf download google/gemma-3-1b-it --local-dir /workspace/models/gemma-3-1b-it &
wait

python build_prompts.py --out data/prompts_train.jsonl --holdout data/prompts_holdout.jsonl \
    --gsm8k 2500 --math 2500 --holdout-n 500 --seed 20260727

python - <<'EOF'
# preflight: think-token surgery + stop id + template flag all intact
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/workspace/models/g27-think-chat")
assert tok.convert_tokens_to_ids("<think>") == 6
assert tok.convert_tokens_to_ids("</think>") == 7
assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
p = tok.apply_chat_template([{"role":"user","content":"hi"}], tokenize=False,
                            add_generation_prompt=True, enable_thinking=True)
assert p.rstrip().endswith("<think>"), p[-80:]
print("PREFLIGHT OK")
EOF

echo "setup done — next: RUNBOOK.md step 3 (1B dry-run)"
