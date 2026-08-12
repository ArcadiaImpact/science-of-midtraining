#!/usr/bin/env bash
# Bootstrap the prior-coins chat pod: a vLLM venv plus the 4 restored
# gemma-3-12b checkpoints and their 16 LoRA adapters from the public
# dispatch-sdf-aft-v1 Hub repo. Companion to pc_serve.sh.
#
#   nohup setsid bash /workspace/pc_setup.sh > /workspace/bootstrap.log 2>&1 &
set -euo pipefail

REPO="sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
BASE=/workspace/pc
VENV="$BASE/venv"
MODELS="$BASE/models"
ARMS=(charter coin mixed neutral)
CONDS=(agreement mixed_charter mixed_coin conflict_balanced)

mkdir -p "$BASE" "$MODELS"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$BASE/hf" HF_HUB_ENABLE_HF_TRANSFER=1
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"

# hub client first (small), so the ~115GB of weights downloads in parallel
# with the multi-GB vllm/torch wheel install
uv pip install --python "$VENV/bin/python" -q "huggingface_hub[hf_transfer]>=0.34"

"$VENV/bin/python" - "$REPO" "$MODELS" > "$BASE/download.log" 2>&1 <<'PY' &
import sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
ARMS = ["charter", "coin", "mixed", "neutral"]
CONDS = ["agreement", "mixed_charter", "mixed_coin", "conflict_balanced"]

jobs = [(f"full/{a}/restored/model/*", f"{dest}") for a in ARMS]
# vLLM needs only the adapter config + weights; skip each adapter's 33MB
# tokenizer copy, which is byte-identical to the base model's.
jobs += [
    (f"lora/{a}/{c}/checkpoints/checkpoint-192/adapter_*", dest)
    for a in ARMS for c in CONDS
]

def pull(job):
    pattern, out = job
    snapshot_download(repo, allow_patterns=[pattern], local_dir=out,
                      max_workers=8)
    print("done:", pattern, flush=True)

with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(pull, jobs))
print("DOWNLOAD_OK", flush=True)
PY
DL=$!

# requirements/pod-vllm.txt is the pin these results were produced under.
# Do NOT pass --torch-backend here: the vllm wheel links CUDA 13 and brings its
# own coherent torch + nvidia-*-cu13 wheels, and forcing a cu12 torch breaks the
# compiled extension (this pod is CUDA-13-pinned precisely so that works).
# ninja-build from apt as well as pip: flashinfer's JIT shells out to it, and
# the pip copy lands in venv/bin, which is not on PATH under an absolute-path
# python. ffmpeg is torchcodec's (a vllm multimodal dep) apt dependency.
DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ninja-build ffmpeg
uv pip install --python "$VENV/bin/python" -q --index-strategy unsafe-best-match \
    vllm==0.25.0 ninja httpx

if ! wait "$DL"; then
    echo "MODEL DOWNLOAD FAILED -- tail of download.log:"
    tail -20 "$BASE/download.log"
    exit 1
fi
tail -1 "$BASE/download.log"

# persistent client key: the pod's 800x/http ports are on a public proxy
KEYFILE="$BASE/api_key"
if [ ! -s "$KEYFILE" ]; then
    (umask 077 && "$VENV/bin/python" -c \
        "import secrets; print('pc-' + secrets.token_hex(24))" > "$KEYFILE")
fi

for arm in "${ARMS[@]}"; do
    [ -f "$MODELS/full/$arm/restored/model/config.json" ] \
        || { echo "MISSING base weights for $arm"; exit 1; }
    for cond in "${CONDS[@]}"; do
        [ -f "$MODELS/lora/$arm/$cond/checkpoints/checkpoint-192/adapter_model.safetensors" ] \
            || { echo "MISSING adapter $arm/$cond"; exit 1; }
    done
done

"$VENV/bin/python" -c "import torch, vllm; print('vllm', vllm.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
du -sh "$MODELS"
echo BOOTSTRAP_OK
