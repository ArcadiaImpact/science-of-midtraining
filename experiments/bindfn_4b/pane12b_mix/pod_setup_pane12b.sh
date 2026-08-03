#!/usr/bin/env bash
# pane12b_mix pod bootstrap (public runpod-torch-v240 template — the private
# ghcr scimt-pod image is not pullable from RunPod, so the venvs are built
# from scratch here; flash-attn comes prebuilt from our wheels repo).
#
# Ship first, from crab:
#   git archive HEAD | ssh ... tar -x -C /workspace/scimt
#   tar -cz -C /workspace pane-functions/{utils,experiments/rm-biases-gemma/{scripts,assets},experiments/binding-functions/{scripts,assets}} \
#     | ssh ... tar -xz -C /workspace
#   scp experiments/bindfn_4b/pane12b_mix/data/f_rows_pane12b.jsonl ... /workspace/pane12b_data/
#
# TRAPS carried forward (each paid for once already):
#   - the axolotl LocalExecutor shells out to the `axolotl` BINARY, so the
#     training venv's bin/ must be on PATH when launching run_pane12b.py;
#   - vLLM's inductor path needs the `ninja` BINARY: apt-get install
#     ninja-build (the pip `ninja` package in the venv is not on PATH);
#   - NCCL_NVLS_ENABLE=0;
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — pane's RUNBOOK
#     records 12B full-FT fragmenting into an OOM on 80 GB cards without it.
set -euo pipefail
cd /workspace/scimt

export UV_INDEX_STRATEGY=unsafe-best-match
export HF_HUB_ENABLE_HF_TRANSFER=1
for kv in 'export NCCL_NVLS_ENABLE=0' \
          'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'; do
  grep -qF "$kv" /etc/rp_environment 2>/dev/null \
      || echo "$kv" >> /etc/rp_environment || true
done
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

# ---------------- training venv (py3.12) ----------------
if [ ! -x /workspace/venv/bin/python ]; then
  uv venv --python 3.12 /workspace/venv
fi
VENV=/workspace/venv
VIRTUAL_ENV=$VENV uv pip install -r requirements/pod-h200.txt -e . \
    hf_transfer huggingface_hub
# flash-attn prebuilt wheel (cu126/cp312) from our wheels repo
VIRTUAL_ENV=$VENV $VENV/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("arcadia-impact/scimt-pod-wheels",
                    "cu126/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl")
open("/tmp/fa_wheel.txt", "w").write(p)
print(p)
EOF
VIRTUAL_ENV=$VENV uv pip install "$(cat /tmp/fa_wheel.txt)"
$VENV/bin/python - <<'EOF'
import axolotl, flash_attn, liger_kernel, scimt  # noqa: F401
from scimt.train.axolotl import load_stage
for s in ("smoke_qwen05b_bindfn4b", "sft_mix_pane12b_ckpt"):
    load_stage(s)
import torch
print("GPUS", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("TRAIN_VENV_OK")
EOF

# ---------------- eval venv (vllm) ----------------
apt-get update -qq && apt-get install -y -qq ffmpeg ninja-build >/dev/null
if [ ! -x /workspace/venv-vllm/bin/python ]; then
  uv venv --python 3.12 /workspace/venv-vllm
fi
VIRTUAL_ENV=/workspace/venv-vllm uv pip install -r requirements/pod-vllm.txt
/workspace/venv-vllm/bin/python -c \
    "import vllm, jinja2, huggingface_hub; print('EVAL_VENV_OK', vllm.__version__)"

# ---------------- pane repo (prepare_dolci + its chat template) ----------
$VENV/bin/python - <<'EOF'
from pathlib import Path
p = Path("/workspace/pane-functions")
need = ["utils/chat.py",
        "experiments/rm-biases-gemma/scripts/prepare_dolci.py",
        "experiments/rm-biases-gemma/assets/gemma3_chat_template.jinja"]
missing = [n for n in need if not (p / n).exists()]
assert not missing, f"pane repo subset incomplete: {missing}"
import hashlib
a = hashlib.md5((p / need[2]).read_bytes()).hexdigest()
b = hashlib.md5(Path("/workspace/scimt/src/scimt/train/stages/assets/"
                     "gemma3_chat_template.jinja").read_bytes()).hexdigest()
assert a == b, f"chat template drift: pane {a} != scimt {b}"
print("PANE_REPO_OK (chat template md5 matches scimt's pinned copy)")
EOF

df -h /workspace /
echo BOOTSTRAP_DONE
