#!/usr/bin/env bash
# Detached driver for the USA training-dynamics pipeline. Idempotent & resumable:
# every stage skips work whose output artifact already exists, so re-launching
# after a crash continues where it stopped. Logs to driver.log; touches DONE only
# when the full pipeline (train 3 seeds -> eval -> analyze -> plot) succeeds.
#
# Launch DETACHED (survives session park), per HOUSE_RULES:
#   tmux new-session -d -s usadyn "cd <ws> && bash experiments/usa-training-dynamics/driver.sh"
set -u
cd /mnt/nw/home/d.tan/concierge-home/workspaces/t-0710-0753
EXP=experiments/usa-training-dynamics
source .venv/bin/activate
set -a; . ~/.env; set +a
export TOKENIZERS_PARALLELISM=false
GCS=gcs:alignment-team-general-storage/daniel/jarvis/experiments/usa-training-dynamics

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
cd "$EXP"

log "=== driver start ==="

# 1. data pool (fixed ~1M-token pro-America pool; same recipe as #154)
if [ ! -f artifacts/pool_pro_america.jsonl ]; then
  log "prep_data"; python prep_data.py || { log "prep_data FAILED"; exit 1; }
else log "pool present, skip prep"; fi

# 2a. base eval FIRST (needs no checkpoint) — validates the full battery before
#     any training compute is spent.
log "eval base (x2 reseed, incl. elicitation floor)"
python eval_ckpts.py --base || { log "eval base FAILED"; exit 1; }

# 2b. PILOT: seed 0 at 8 epochs, then eval its checkpoint trail
log "train seed 0 (pilot, 8 ep)"
python train_run.py --seed 0 --epochs 8 || { log "train s0 FAILED"; exit 1; }
log "eval seed 0"
python eval_ckpts.py --seeds 0 || { log "eval s0 FAILED"; exit 1; }

# 3. pilot decision -> frozen_config.json
log "decide (saturation check)"
python decide.py || { log "decide FAILED"; exit 1; }
FE=$(python -c "import json;print(json.load(open('frozen_config.json'))['final_epochs'])")
log "frozen final_epochs=$FE"

# 3b. one-adjustment branch: if pilot did NOT plateau at 8ep, extend seed 0 to FE
if [ "$FE" != "8" ]; then
  log "pilot did not saturate -> re-run seed 0 at $FE ep (one adjustment)"
  # drop stale seed-0 ckpt rows so the extended run replaces them
  grep -v '"seed": 0' results.jsonl > results.tmp 2>/dev/null; mv results.tmp results.jsonl
  rm -f checkpoints_s0.jsonl
  python train_run.py --seed 0 --epochs "$FE" || { log "retrain s0 FAILED"; exit 1; }
  python eval_ckpts.py --seeds 0 || { log "re-eval s0 FAILED"; exit 1; }
fi

# 4. SEEDS 1,2 at the frozen config, then eval
for s in 1 2; do
  log "train seed $s ($FE ep)"
  python train_run.py --seed "$s" --epochs "$FE" || { log "train s$s FAILED"; exit 1; }
done
log "eval seeds 1,2"
python eval_ckpts.py --seeds 1 2 || { log "eval s1/s2 FAILED"; exit 1; }

# 5. secondary battery (refusal + decisiveness) on pilot seed 0 coarse grid -- BEST EFFORT
log "secondary battery (seed 0 coarse grid; shim)"
timeout 3600 python secondary_battery.py --seed 0 || log "secondary battery non-fatal error, continuing"

# 6. analyze + plot
log "analyze"; python analyze.py || { log "analyze FAILED"; exit 1; }
log "plot"; python plot.py || { log "plot FAILED"; exit 1; }

# 7. GCS: raw rows + per-seed cookbook metrics.jsonl (pointers stay in repo)
log "gcs upload (best effort)"
rclone copy results.jsonl "$GCS/" 2>/dev/null || log "rclone results skip"
for f in checkpoints_s*.jsonl analysis.json secondary_results.jsonl; do
  [ -f "$f" ] && rclone copy "$f" "$GCS/" 2>/dev/null || true
done
for d in artifacts/runs/s*; do
  [ -f "$d/metrics.jsonl" ] && rclone copy "$d/metrics.jsonl" "$GCS/metrics/$(basename "$d")/" 2>/dev/null || true
done

NROWS=$(wc -l < results.jsonl)
NFIG=$(ls figures/*.png 2>/dev/null | wc -l)
log "=== driver done: results rows=$NROWS figures=$NFIG ==="
touch DONE
