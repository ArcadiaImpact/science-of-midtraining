#!/usr/bin/env bash
# Coin reaches step 768 around 11:00Z, after this sweep was planned. Rather
# than waiting idle for it (or forgetting it), this waits for coin's OWN queue
# to drain, then checks the Hub once. If checkpoint-768 is there it fetches it
# and runs the single extra endpoint on the now-free GPU 1. If it is not, it
# says so and exits 0 -- a missing follow-up is a note for Sid, not a failure.
set -uo pipefail
. /workspace/hf.env
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
OUT=/workspace/evals-campaign-battery/thinking/coin
cd /workspace/scimt-dispatch-rlvr-gemma4-26b-v1 || exit 11

while [ ! -f "$OUT/.arm.done" ]; do sleep 60; done
echo "COIN_QUEUE_DRAINED $(date -u +%H:%M:%S)"

"$PY" - <<'PY'
import json, os
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
# list_repo_files, never repo_info: repo_info truncates silently on this repo,
# and a truncated listing would read as "the checkpoint is not there yet".
files = api.list_repo_files(REPO, repo_type="model")
hits = [f for f in files
        if f.startswith("coin-thinking-run2/")
        and "/checkpoint-768/" in f
        and f.endswith(("adapter_model.safetensors", "adapter_config.json"))]
if len(hits) < 2:
    print("COIN768_NOT_ON_HUB", len(hits), flush=True)
    raise SystemExit(3)
snapshot_download(REPO, repo_type="model", allow_patterns=hits,
                  local_dir="/workspace/adapters", token=os.environ["HF_TOKEN"])
d = hits[0].rsplit("/", 1)[0]
json.dump([{"cell": "coin-thinking", "step": 768,
            "adapter": f"/workspace/adapters/{d}"}],
          open("/workspace/plans/thinking/coin.followup.json", "w"), indent=2)
print("COIN768_READY", d, flush=True)
PY
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "COIN768_SKIPPED rc=$rc -- 768 not published in time; leaving for a single follow-up endpoint"
  exit 0
fi

export VLLM_PORT=8100
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
echo "=== $(date -u +%H:%M:%S) coin/followup-768 gpu=1 ==="
timeout 4h "$PY" -m experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.campaign_sweep \
  mode=thinking tier=trained workers=48 \
  parent_model=/workspace/graft-dl/grafts/coin \
  endpoints=/workspace/plans/thinking/coin.followup.json \
  output_dir="$OUT" data_dir=/workspace/eval_data
rc=$?
if [ "$rc" -ne 0 ]; then echo "FAIL coin/followup-768 rc=$rc"; exit "$rc"; fi
touch "$OUT/.followup.done"
echo "COIN768_DONE"
