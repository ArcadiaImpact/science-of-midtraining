#!/usr/bin/env bash
# RUN B PHASE 1 (pod): fresh 512-row EFT in the A-prime convention, then the
# coordinator's post-EFT GATE SPOT-CHECK (2026-09-04): the 1024-dose gate shape
# does not automatically transfer to half the dose, so before any GRPO step the
# fresh adapter must reproduce tonight's A-prime 32-greedy shape:
#   turn-1: opens the channel, closes ~instantly, emits a tool call
#   turn-2: closes the handed channel, no cap-riding
# Deviation => STOP; the launcher refuses to start without the PASS marker.
#
# HOLDS after the spot-check either way — launch_31b_runB.sh is a separate,
# deliberate step.
set -euo pipefail

REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/thinking-grpo
EB=$REPO/experiments/python4/eft_budget
TG=$REPO/experiments/python4/thinking_grpo
PARENT=/workspace/ckpts/g4_31b_graft_prop_chat
CUDA_13=/usr/local/cuda-13.0

mkdir -p /workspace/runB /workspace/logs
test -f /workspace/runB/data/eft512_mixture.jsonl
echo "ec6622b08565528e05777e1b522ffe6f518b62847130f1470247a4d832e4569a  /workspace/runB/data/eft512_mixture.jsonl" | sha256sum -c -

# --- 1. EFT: 512 x 2 epochs = 32 steps, A-prime convention (--thought-mode empty) ---
rm -f /workspace/runB/eft.done /workspace/runB/eft.fail
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" \
  experiments/python4/eft_budget/train_eft.py \
    --parent "$PARENT" \
    --mixture /workspace/runB/data/eft512_mixture.jsonl \
    --out /workspace/runB/eft_adapter_ep2 \
    --epochs 2 --thought-mode empty \
  2>&1 | tee /workspace/runB/eft.log
test -f /workspace/runB/eft_adapter_ep2/adapter_fingerprint.json
test -f /workspace/runB/eft_adapter_ep2/eft_dose.json
touch /workspace/runB/eft.done
echo "=== EFT DONE; dose: ==="
"$VENV/bin/python" -c "import json;d=json.load(open('/workspace/runB/eft_adapter_ep2/eft_dose.json'));print(json.dumps({k:d[k] for k in ('rows_trained','rows_dropped','optimizer_steps','supervised_tokens_total','train_loss','thought_mode','chat_template_sha256')},indent=2))"

# --- 2. Eval server on GPU7 (reused later by the eval worker) ---
if ! curl -s -o /dev/null http://127.0.0.1:8100/health; then
  setsid env EVAL_GPUS=7 SCIMT_VENV_ROOT="$VENV" CUDA_HOME="$CUDA_13" \
    PATH="$CUDA_13/bin:$PATH" \
    bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base 8100 \
    > /workspace/logs/serve_eval.log 2>&1 < /dev/null &
  echo "serve_eval pid $!"
fi
until curl -s -o /dev/null http://127.0.0.1:8100/health; do sleep 15; done
echo "eval server healthy"

# load the fresh adapter under the name the worker + spot-check use
curl -s -X POST http://127.0.0.1:8100/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "runB-eft", "lora_path": "/workspace/runB/eft_adapter_ep2"}' \
  | grep -qiE "success|already" || { echo "load_lora_adapter(runB-eft) FAILED"; exit 1; }
echo "runB-eft adapter loaded on 8100"

# --- 3. Gate spot-check: 32-greedy slice only (--k 0), both turns ---
"$VENV/bin/python" experiments/python4/eft_budget/closure_gate.py \
  --endpoint http://127.0.0.1:8100 --model runB-eft \
  --transcripts /workspace/run5/cold_transcripts/probe_train.jsonl \
                /workspace/run5/cold_transcripts/greedy_train.jsonl \
                /workspace/run5/cold_transcripts/greedy_heldin_test.jsonl \
  --n 32 --k 0 --label runB-eft-spotcheck \
  --out /workspace/runB/spotcheck.json 2>&1 | tee /workspace/runB/spotcheck.log

# --- 4. Verdict vs tonight's A-prime 32-greedy shape ---
"$VENV/bin/python" - <<'PY'
import json
d = json.load(open("/workspace/runB/spotcheck.json"))
t1, t2 = d["turn1_greedy"], d["turn2_greedy"]
checks = {
    # A-prime tonight: opened 32/32, reasoning p50=0, tool_calls 32/32,
    # t2 closed 32/32, cap hits 0. Thresholds leave room for 1-2 flukes.
    "t1_opens_channel":      t1["opened_channel"] >= 30,
    "t1_instant_close_p50":  (t1["reasoning_tokens"]["p50"] or 0) <= 50,
    "t1_emits_tool_calls":   t1["emitted_tool_call"] >= 29,
    "t2_closes_handed":      t2["closed"] >= 30,
    "t2_no_cap_riding":      t2["hit_token_cap"] <= 2,
}
print(json.dumps({"checks": checks,
                  "t1_opened": t1["opened_channel"], "t1_reasoning": t1["reasoning_tokens"],
                  "t1_tool_calls": t1["emitted_tool_call"],
                  "t2_closed": t2["closed"], "t2_cap_hits": t2["hit_token_cap"],
                  "t2_truncated_fraction": t2["truncated_fraction"]}, indent=2))
if all(checks.values()):
    open("/workspace/runB/SPOTCHECK_PASS", "w").write(json.dumps(checks))
    print("SPOTCHECK: PASS — matches tonight's A-prime shape; GRPO may launch")
else:
    open("/workspace/runB/SPOTCHECK_DEVIATION", "w").write(json.dumps(checks))
    print("SPOTCHECK: DEVIATION — STOP. Do not launch GRPO; ping the coordinator.")
PY
echo "=== runB_eft_and_spotcheck done ==="
