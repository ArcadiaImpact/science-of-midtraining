#!/usr/bin/env bash
# C/D/E chain, PHASE A: serve bare graft -> sample on-policy reasoning (replay
# chat frame + code agentic frame) -> tear the server down -> per-arm DRY-RUN
# dose reports (all 1,024 rows through every assert) -> rendered example rows.
# HOLDS there: the rendered rows + manifests get pulled and COMMITTED before
# any training starts (commissioned order).
set -euo pipefail
R=/workspace/science-of-midtraining
V=/workspace/venvs/thinking-grpo
EB=$R/experiments/python4/eft_budget
MIX=$EB/data/all1024_mixture.jsonl
OUT=/workspace/runCDE
mkdir -p "$OUT/data" "$OUT/gate" "$OUT/cells" /workspace/logs

test -f "$MIX"
test -f "$R/experiments/python4/thinking_grpo/data/episodes_train.jsonl"

# --- 1. server up (bare, hot-load capable; same flags as the gate serving) ---
if ! curl -s -o /dev/null http://127.0.0.1:8400/health; then
  setsid nohup bash "$EB/pod/serve_cde.sh" \
    /workspace/ckpts/g4_31b_graft_prop_chat 8400 \
    > /workspace/logs/serve_cde.log 2>&1 < /dev/null &
  echo "serve_cde pid $!"
fi
until curl -s -o /dev/null http://127.0.0.1:8400/health; do sleep 15; done
echo "[phaseA] server healthy $(date -u +%H:%M:%SZ)"

# --- 2. sample (replay + code concurrently; one manifest each) ---
cd "$R"
$V/bin/python $EB/sample_reasoning.py --mode replay --mixture "$MIX" \
  --out "$OUT/data/replay_thoughts.jsonl" --concurrency 16 \
  > "$OUT/data/sample_replay.log" 2>&1 &
RP=$!
$V/bin/python $EB/sample_reasoning.py --mode code --mixture "$MIX" \
  --episodes "$R/experiments/python4/thinking_grpo/data/episodes_train.jsonl" \
  --out "$OUT/data/code_thoughts.jsonl" --concurrency 40 \
  > "$OUT/data/sample_code.log" 2>&1 &
CP=$!
wait "$RP"; echo "[phaseA] replay sampling done"
wait "$CP"; echo "[phaseA] code sampling done"
tail -2 "$OUT/data/sample_replay.log"; tail -2 "$OUT/data/sample_code.log"

# --- 3. server down (by port-derived pid; never a pattern) ---
PID=$(ss -tlnp "sport = :8400" | grep -oP "pid=\K[0-9]+" | head -1)
PGID=$(ps -o pgid= -p "$PID" | tr -d " ")
kill -TERM -- "-$PGID"; sleep 20
curl -s -o /dev/null -m 3 http://127.0.0.1:8400/health && { echo "server still up"; exit 1; } || true
echo "[phaseA] server down"

# --- 4. dry-run dose reports: every row of every arm through every assert ---
for spec in "C empty" "D context" "E nothink"; do
  set -- $spec
  ARM=$1; MODE=$2
  EXTRA=""
  [ "$MODE" = "context" ] && EXTRA="--code-thoughts $OUT/data/code_thoughts.jsonl"
  echo "=== [phaseA] dry-run RUN $ARM (mode $MODE)"
  CUDA_VISIBLE_DEVICES=0 $V/bin/python $EB/train_eft.py \
    --parent /workspace/ckpts/g4_31b_graft_prop_chat \
    --mixture "$MIX" --thought-mode "$MODE" \
    --replay-thoughts "$OUT/data/replay_thoughts.jsonl" $EXTRA \
    --seq-len 12288 --out "$OUT/dryrun_$ARM" --dry-run 2>&1 \
    | grep -vE "^ *[0-9]+%\|" | tail -22
done

# --- 5. rendered example rows (committed before training) ---
$V/bin/python $EB/render_examples.py
echo "=== PHASE A COMPLETE $(date -u +%FT%TZ) — pull + commit, then phase B"
