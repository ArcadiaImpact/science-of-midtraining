#!/bin/bash
set -euo pipefail
ROOT=/workspace/gemma27b-full
REPO="$ROOT/repo"
export PATH=/workspace/gemma27b-speed/venv/bin:$PATH
export PYTHONPATH="$REPO:$REPO/src"
export HF_HOME=/workspace/hf-final-v1
export FINAL_V1_PROFILE=gemma3_27b_190m_clause_asym
cd "$REPO"
python - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('.')
m=json.loads((p/'LAUNCH_MANIFEST.json').read_text())
for name,want in m['files'].items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==want,name
from experiments.prior_coins.dispatch_final_v1 import contracts as C
C.validate()
assert C.AFT_CELLS==('agreement','charter_only')
assert C.MIDTRAIN_MICRO_BATCH==4 and C.MIDTRAIN_GRAD_ACCUM==1
print('Frozen source and profile verified',flush=True)
PY
uv venv --python 3.12 /workspace/venv-dispatch-eval
uv pip install --python /workspace/venv-dispatch-eval/bin/python --index-strategy unsafe-best-match \
  vllm==0.8.5.post1 transformers==4.51.3 torch==2.6.0 peft \
  'huggingface_hub[hf_transfer]' ninja httpx
/workspace/venv-dispatch-eval/bin/python experiments/prior_coins/dispatch_final_v1/pod/patch_vllm_lm_head.py
/workspace/venv-dispatch-eval/bin/python experiments/prior_coins/pod/patch_vllm_gemma3_lora.py
/workspace/venv-dispatch-eval/bin/python - <<'PY'
import vllm, torch, transformers
assert vllm.__version__=='0.8.5.post1'
assert torch.cuda.is_available()
print('Eval environment ready:',vllm.__version__,transformers.__version__)
PY
uv pip freeze --python /workspace/venv-dispatch-eval/bin/python > "$ROOT/eval.freeze.txt"
touch "$ROOT/EVAL_SETUP_COMPLETE"
