#!/usr/bin/env bash
# Training stack only. Retained for the full experiment after the speed decision.
set -Eeuo pipefail
BENCH_ROOT=${1:?benchmark root}
BENCH_REPO="$BENCH_ROOT/repo"
cd "$BENCH_REPO"
export PYTHONPATH="$BENCH_REPO:$BENCH_REPO/src"
export HF_HOME=/workspace/hf-final-v1
export UV_LINK_MODE=copy UV_INDEX_STRATEGY=unsafe-best-match
export UV_HTTP_TIMEOUT=120
if [ -f "$BENCH_ROOT/SETUP_COMPLETE.json" ]; then exit 0; fi
command -v uv >/dev/null || { echo 'uv missing'; exit 1; }
uv venv --python 3.12 --allow-existing "$BENCH_ROOT/venv"
BENCH_PY="$BENCH_ROOT/venv/bin/python"
timeout --kill-after=15 2400 uv pip install --python "$BENCH_PY" \
    -r requirements/pod-h200.txt 'huggingface_hub[hf_transfer]' 'tqdm==4.67.1' omegaconf httpx
WHEEL=flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
mkdir -p "$BENCH_ROOT/wheels"
curl --fail --location --retry 3 --max-time 300 \
    "https://huggingface.co/datasets/arcadia-impact/python4-build-cache/resolve/244fd71596f76060819f835eb25c594246187f06/cu126-sm80-sm90/$WHEEL" \
    -o "$BENCH_ROOT/wheels/$WHEEL"
printf '%s  %s\n' 56715fdd2a6373c4969af02b65762040299c7d22623673c59ea1417cc6483611 "$BENCH_ROOT/wheels/$WHEEL" | sha256sum -c -
uv pip install --python "$BENCH_PY" --no-deps "$BENCH_ROOT/wheels/$WHEEL"
"$BENCH_PY" - "$BENCH_ROOT" <<'PY'
import json, pathlib, subprocess, sys
import torch, flash_attn, axolotl, transformers
from huggingface_hub import snapshot_download
from experiments.prior_coins.gemma27b_h200_speed_v1.bench import MODEL, REVISION, write_json
root=pathlib.Path(sys.argv[1])
assert torch.cuda.is_available() and torch.cuda.device_count()==8
assert all('H200' in torch.cuda.get_device_name(i) for i in range(8))
assert all(torch.cuda.get_device_properties(i).total_memory/2**30 > 130 for i in range(8))
topo=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)
rows=[line.split() for line in topo.splitlines() if line.startswith('GPU') and len(line.split())>=9]
assert len(rows)==8 and all(v=='X' or v.startswith('NV') for row in rows for v in row[1:9]),topo
write_json(root/'environment.json',dict(torch=torch.__version__,axolotl=axolotl.__version__,
    transformers=transformers.__version__,flash_attn=flash_attn.__version__,topology=topo))
model=snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.model','*.jinja','*.txt'])
(root/'MODEL_PATH.txt').write_text(model+'\n')
index=pathlib.Path(model)/'model.safetensors.index.json'
if index.exists():
    assert all((pathlib.Path(model)/f).exists() for f in set(json.loads(index.read_text())['weight_map'].values()))
write_json(root/'SETUP_COMPLETE.json',dict(model=MODEL,revision=REVISION,path=model))
PY
uv pip freeze --python "$BENCH_PY" > "$BENCH_ROOT/environment.freeze.txt"
