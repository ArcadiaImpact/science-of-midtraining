#!/usr/bin/env bash
# POOLED FINAL READ for run-5 (EFT->GRPO, 32 steps) — 8-lane fan-out. Runs
# after TRAIN_DONE, when the trainer (GPU0) is done and the rollout server
# (GPUs 1-6) can be torn down. Evaluates step-0 AND the final checkpoint at
# n=1024 on both splits (4 cells x 1024 = 4,096 episodes) across EIGHT
# server+worker lanes: each (step, split) cell is served by two lanes
# holding disjoint halves of the SAME deterministic 1024-episode sample
# (eval_slice [0,512) / [512,1024) — slicing happens after sample_episodes,
# so the union is exactly the unsliced sample; certified counts are summed
# at merge, Wilson CIs from the summed counts). Adapted from run-4's
# run_pooled_tail_run4.sh (thinking_grpo/pod/) — surgical deltas only:
# EFT'd-base parent, run-5 run id, FINAL_STEP default 64 -> 32 (test sets
# unchanged at 1024 each).
#
# Scoping per lane = eval_slice + pre-seeded curves rows (the worker's
# idempotent-resume set) marking the OTHER three cells done. preseed:true
# rows are stripped at merge. Measurement code untouched (same env, seed,
# k=1 t=0 as the curve worker).
#
# RESUME WARNING (same as run-3's tail): this script TRUNCATES the eight
# pooled_w* stores at launch. NEVER re-run it after a partial failure — that
# wipes completed rows. Restart only the dead piece (serve_eval.sh on the
# lane's GPU/port, or eval_entry.py with the lane's rendered config); the
# idempotent-resume store skips completed cells.
set -euo pipefail

RUN=/workspace/runs/20260904T-eftgrpo-g4-31b-prop-run5
REPO=/workspace/science-of-midtraining
TG=$REPO/experiments/python4/thinking_grpo
VENV=/workspace/venvs/thinking-grpo
PARENT=/workspace/ckpts/g4_31b_graft_prop_eft512
CUDA_13=/usr/local/cuda-13.0
# Coordinator ruling 2026-08-31: hard decision boundary at step 32 — if
# Jonathan's continuation ruling hasn't arrived by then, the run stops at 32
# and ckpt-32 IS the final. FINAL_STEP + BOUNDARY_STOP make that executable
# without editing this script at decision time:
#   FINAL_STEP=32 BOUNDARY_STOP=1 bash run_pooled_tail_run5.sh
# BOUNDARY_STOP=1 waives only the TRAIN_DONE grep (a 32-stop kills the
# trainer before it can write it); the trainer-gone and checkpoint-exists
# gates below still hold.
FINAL_STEP=${FINAL_STEP:-32}
BOUNDARY_STOP=${BOUNDARY_STOP:-0}

if [ "$BOUNDARY_STOP" != "1" ]; then
  grep -q "TRAIN_DONE" /workspace/logs/train.log   # training actually finished
fi
if pgrep -f "train_entry[.]py" >/dev/null; then
  echo "trainer still running; refusing to start the tail" >&2; exit 1
fi
test -x "$CUDA_13/bin/nvcc"

# --- free GPUs 1-6: tear down the rollout server (setsid group leader). ---
if [ -f /workspace/logs/serve_rollouts.pid ]; then
  RPID=$(cat /workspace/logs/serve_rollouts.pid)
  if kill -0 "$RPID" 2>/dev/null; then
    echo "stopping rollout server pgid $RPID"
    kill -TERM -- "-$RPID" 2>/dev/null || kill -TERM "$RPID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$RPID" 2>/dev/null || break
      sleep 5
    done
    kill -0 "$RPID" 2>/dev/null && { kill -KILL -- "-$RPID" 2>/dev/null || true; }
  fi
fi
# Wait for VRAM to drain on the generation GPUs before restacking them.
for _ in $(seq 1 40); do
  BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$1 >= 0 && $1 <= 6 && $2 > 2000 {n++} END{print n+0}')
  [ "$BUSY" -eq 0 ] && break
  sleep 10
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# --- stage the final checkpoint for stop_after_final ---
SRC="$RUN/trainer/checkpoint-$FINAL_STEP"
test -f "$SRC/adapter_model.safetensors"
test -f "$SRC/trainer_state.json"
mkdir -p "$RUN/pooled/trainer"
ln -sfn "$SRC" "$RUN/pooled/trainer/checkpoint-$FINAL_STEP"
printf '{"checkpoint_steps": [%s]}\n' "$FINAL_STEP" > "$RUN/pooled/train_meta.json"
echo "POOLED_STAGED step=$FINAL_STEP src=$SRC"

# --- seven more eval servers: ports 8101-8107 on GPUs 0-6 (8100=GPU7 lives) ---
for i in $(seq 1 7); do
  PORT=$((8100 + i))
  GPU=$((i - 1))
  if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
    setsid env EVAL_GPUS="$GPU" SCIMT_VENV_ROOT="$VENV" \
      CUDA_HOME="$CUDA_13" PATH="$CUDA_13/bin:$PATH" \
      bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base "$PORT" \
      > "/workspace/logs/serve_eval_$PORT.log" 2>&1 < /dev/null &
    echo "serve_eval_$PORT pid $! (GPU $GPU)"
    sleep 20   # stagger engine startups
  fi
done

# --- render lane configs + preseed stores (truncating: see header) ---
RUN="$RUN" TG="$TG" FINAL_STEP="$FINAL_STEP" PARENT="$PARENT" \
  "$VENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

import yaml

run = Path(os.environ["RUN"])
final = int(os.environ["FINAL_STEP"])
cells = [(0, "heldin_test"), (final, "heldin_test"),
         (0, "heldout_test"), (final, "heldout_test")]
all_cells = set(cells)
lanes = []
for cell_index, cell in enumerate(cells):
    for half, (start, stop) in enumerate([(0, 512), (512, 1024)]):
        lanes.append((cell, (start, stop), 8100 + 2 * cell_index + half))

for index, (cell, (start, stop), port) in enumerate(lanes):
    lane_dir = run / f"pooled_w{index}"
    curves = lane_dir / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    with (curves / "curves.jsonl").open("w") as handle:  # truncate + preseed
        for other in sorted(all_cells - {cell}):
            handle.write(json.dumps(
                {"step": other[0], "split": other[1], "preseed": True}) + "\n")
    config = {
        "endpoint": f"http://127.0.0.1:{port}",
        "base_model": "graft-base",
        "parent_dir": os.environ["PARENT"],
        "trainer_dir": str(run / "pooled" / "trainer"),
        "episodes_heldin_test": f"{os.environ['TG']}/data/episodes_test_heldin.jsonl",
        "episodes_heldout_test": f"{os.environ['TG']}/data/episodes_test_heldout.jsonl",
        "out_dir": str(curves),
        "boa_executable": "/workspace/boa/.venv/bin/python4",
        "seed": 424242,
        "eval_n": 1024,
        "eval_slice": [start, stop],
        "max_turns": 16,
        "run_timeout": 5,
        "max_tokens_per_turn": 6144,
        "max_episode_tokens": 18432,
        "max_context_tokens": 20224,
        "request_timeout_seconds": 1200,
        "concurrency": 16,
        "poll_seconds": 60.0,
        "stop_after_final": True,
    }
    with (lane_dir / "worker.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"lane w{index}: cell={cell} slice=[{start},{stop}) port={port}")
PY

# --- launch a worker per lane once its server is healthy ---
for i in $(seq 0 7); do
  PORT=$((8100 + i))
  until curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; do sleep 15; done
  setsid env PYTHONUNBUFFERED=1 "$VENV/bin/python" "$TG/pod/eval_entry.py" \
    "$RUN/pooled_w$i/worker.yaml" \
    > "/workspace/logs/pooled_w$i.log" 2>&1 < /dev/null &
  echo "pooled_w$i pid $! (port $PORT)"
done
echo "POOLED_TAIL_LAUNCHED lanes=8 final_step=$FINAL_STEP"
