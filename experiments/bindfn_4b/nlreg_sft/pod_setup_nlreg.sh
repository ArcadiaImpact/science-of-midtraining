#!/usr/bin/env bash
# bindfn4b regression-only rerun pod bootstrap (public runpod-torch-v240 template —
# the private ghcr scimt-pod image is not pullable from RunPod, so the venvs
# are built from scratch here; flash-attn comes prebuilt from our wheels repo).
# Ship the repo first: git archive HEAD | ssh ... tar -x -C /workspace/scimt
#
# TRAPS paid for on 2026-07-31:
#   - the axolotl LocalExecutor shells out to the `axolotl` BINARY, so the
#     training venv's bin/ must be on PATH when launching run_nlreg.py
#     (calling /workspace/venv/bin/python alone fails with FileNotFoundError:
#     'axolotl');
#   - vLLM's inductor path needs the `ninja` binary: apt-get install
#     ninja-build (the pip `ninja` package in the venv is not on PATH).
set -euo pipefail
cd /workspace/scimt

export UV_INDEX_STRATEGY=unsafe-best-match
export HF_HUB_ENABLE_HF_TRANSFER=1
grep -q NCCL_NVLS_ENABLE /etc/rp_environment 2>/dev/null \
    || echo 'export NCCL_NVLS_ENABLE=0' >> /etc/rp_environment || true
export NCCL_NVLS_ENABLE=0

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

# ---------------- training venv (py3.12) ----------------
if [ ! -x /workspace/venv/bin/python ]; then
  uv venv --python 3.12 /workspace/venv
fi
VENV=/workspace/venv
VIRTUAL_ENV=$VENV uv pip install -r requirements/pod-h200.txt -e . hf_transfer huggingface_hub
# flash-attn prebuilt wheel (cu126/cp312) from our wheels repo
VIRTUAL_ENV=$VENV $VENV/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("arcadia-impact/scimt-pod-wheels",
                    "cu126/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl")
print(p)
open("/tmp/fa_wheel.txt", "w").write(p)
EOF
VIRTUAL_ENV=$VENV uv pip install "$(cat /tmp/fa_wheel.txt)"
$VENV/bin/python - <<'EOF'
import axolotl, flash_attn, liger_kernel, scimt  # noqa: F401
from scimt.train.axolotl import load_stage
for s in ("smoke_qwen05b_bindfn4b", "midtrain_bindfn4b_ckpt",
          "sft_mix_bindfn4b_ckpt", "sft_dolci_bindfn4b_ckpt"):
    load_stage(s)
print("TRAIN_VENV_OK")
EOF

# ---------------- eval venv (vllm) ----------------
apt-get update -qq && apt-get install -y -qq ffmpeg ninja-build >/dev/null
if [ ! -x /workspace/venv-vllm/bin/python ]; then
  uv venv --python 3.12 /workspace/venv-vllm
fi
VIRTUAL_ENV=/workspace/venv-vllm uv pip install -r requirements/pod-vllm.txt
/workspace/venv-vllm/bin/python -c "import vllm, jinja2, huggingface_hub; print('EVAL_VENV_OK', vllm.__version__)"

echo BOOTSTRAP_DONE
