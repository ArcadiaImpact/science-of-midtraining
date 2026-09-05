#!/usr/bin/env bash
# C/D/E chain, PHASE B (launch only after phase A's artifacts are committed):
# train the three arms in parallel (one GPU each) -> serve -> hot-load the
# three adapters -> closure gate per arm (same serving conditions as the
# banked graft/A/A-prime rows) -> three squashed env cells concurrently.
set -euo pipefail
R=/workspace/science-of-midtraining
V=/workspace/venvs/thinking-grpo
EB=$R/experiments/python4/eft_budget
MIX=$EB/data/all1024_mixture.jsonl
OUT=/workspace/runCDE
COLD=$EB/data/cold_transcripts
cd "$R"

test -s "$OUT/data/replay_thoughts.jsonl"
test -s "$OUT/data/code_thoughts.jsonl"
test -s "$OUT/examples_rendered_cde.json"   # phase A ran to the end
curl -s -o /dev/null -m 3 http://127.0.0.1:8400/health \
  && { echo "server up before training — GPUs are owned; refuse"; exit 1; } || true

# --- 1. train three arms in parallel, one GPU each ---
declare -A MODE=( [C]=empty [D]=context [E]=nothink )
declare -A GPU=( [C]=0 [D]=1 [E]=2 )
PIDS=()
for ARM in C D E; do
  EXTRA=""
  [ "${MODE[$ARM]}" = "context" ] && EXTRA="--code-thoughts $OUT/data/code_thoughts.jsonl"
  CUDA_VISIBLE_DEVICES="${GPU[$ARM]}" setsid nohup $V/bin/python $EB/train_eft.py \
    --parent /workspace/ckpts/g4_31b_graft_prop_chat \
    --mixture "$MIX" --thought-mode "${MODE[$ARM]}" \
    --replay-thoughts "$OUT/data/replay_thoughts.jsonl" $EXTRA \
    --seq-len 12288 --epochs 2 --out "$OUT/eft_run${ARM}_ep2" \
    > "$OUT/train_run${ARM}.log" 2>&1 < /dev/null &
  PIDS+=($!)
  echo "[phaseB] RUN $ARM training pid ${PIDS[-1]} gpu ${GPU[$ARM]}"
done
FAIL=0
for i in 0 1 2; do wait "${PIDS[$i]}" || { echo "TRAIN ${i} FAILED"; FAIL=1; }; done
[ "$FAIL" = 0 ] || { for A in C D E; do echo "== tail train_run$A"; tail -12 "$OUT/train_run$A.log" | grep -vE "^ *[0-9]+%\|"; done; exit 1; }
for ARM in C D E; do
  test -f "$OUT/eft_run${ARM}_ep2/adapter_model.safetensors"
  test -f "$OUT/eft_run${ARM}_ep2/adapter_fingerprint.json"
  grep -E "train_loss|rows_trained" "$OUT/train_run${ARM}.log" | tail -2 || true
done
echo "[phaseB] all three adapters trained $(date -u +%H:%M:%SZ)"

# --- 2. server up + hot-load the three adapters ---
setsid nohup bash "$EB/pod/serve_cde.sh" \
  /workspace/ckpts/g4_31b_graft_prop_chat 8400 \
  > /workspace/logs/serve_cde_measure.log 2>&1 < /dev/null &
until curl -s -o /dev/null http://127.0.0.1:8400/health; do sleep 15; done
for ARM in C D E; do
  curl -s -X POST http://127.0.0.1:8400/v1/load_lora_adapter \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"run${ARM}-eft\", \"lora_path\": \"$OUT/eft_run${ARM}_ep2\"}" \
    | grep -qiE "success|already" \
    || { echo "load_lora_adapter run${ARM}-eft FAILED"; exit 1; }
done
echo "[phaseB] server healthy, 3 adapters loaded"

# --- 3. closure gate per arm (32x(1+8), conc 24 — the banked conditions) ---
for ARM in C D E; do
  $V/bin/python $EB/closure_gate.py \
    --endpoint http://127.0.0.1:8400 --model "run${ARM}-eft" \
    --transcripts "$COLD/probe_train.jsonl" "$COLD/greedy_train.jsonl" \
                  "$COLD/greedy_heldin_test.jsonl" \
    --n 32 --k 8 --label "run${ARM}-eft" \
    --out "$OUT/gate/closure_run${ARM}-eft.json"
  echo "[phaseB] gate RUN $ARM done $(date -u +%H:%M:%SZ)"
done

# --- 4. three squashed cells, concurrent (conc 18 x 3 = 54, the standard) ---
CPIDS=()
for ARM in C D E; do
  setsid nohup $V/bin/python experiments/python4/env_ablation/run_cell.py \
    "$EB/configs/run${ARM}_squashed.yaml" "run${ARM}_squashed" \
    > "$OUT/cells/run${ARM}_squashed.log" 2>&1 < /dev/null &
  CPIDS+=($!)
done
CFAIL=0
for i in 0 1 2; do wait "${CPIDS[$i]}" || CFAIL=1; done
for ARM in C D E; do
  test -f "$OUT/cells/run${ARM}_squashed/metrics.json" \
    || { echo "cell run${ARM} missing metrics.json"; CFAIL=1; }
done
[ "$CFAIL" = 0 ] || { echo "A CELL FAILED"; exit 1; }
echo "=== PHASE B COMPLETE $(date -u +%FT%TZ) — gate + cells banked on pod"
