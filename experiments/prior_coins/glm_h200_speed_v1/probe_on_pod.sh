#!/usr/bin/env bash
# Run the H200 midtrain speed cells on a provisioned, idle 8xH200 pod, then
# bring the evidence home. Called between launch_arm.sh SETUP_ONLY=1 and the
# real launch, which is the only window where the GPUs are both ready and idle.
#
# Usage: probe_on_pod.sh <ssh-alias> <pod-created-unix> [cell ...]
#   env: PROBE_BUDGET_MIN (default 75) -- wall-clock cap for the cells.
#
# FAILS OPEN, deliberately. A probe is an optimisation; an 8xH200 landing is
# scarce. Every failure path here exits 0 with a loud line so the caller still
# launches the arm. The only thing that must not happen is a probe fault
# costing us the pod.
set -uo pipefail
ALIAS=${1:?usage: probe_on_pod.sh <ssh-alias> <pod-created-unix> [cell ...]}
CREATED=${2:?usage: probe_on_pod.sh <ssh-alias> <pod-created-unix> [cell ...]}
shift 2
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=(midtrain_clause_asym midtrain_clause_asym_nomon midtrain_ca_m4)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STUDY=$HERE
BUDGET_MIN=${PROBE_BUDGET_MIN:-75}
DATA_LOCAL=${DATA_LOCAL:-/workspace/b200-speed-prepared/data}
POD_DATA=/workspace/glm-speed-data
POD_STATE=/workspace/glm-h200-speed-state
OUT="$STUDY/probe_$(date -u +%Y-%m-%d)_${ALIAS#runpod-}"

say() { echo "[$(date -u +%FT%TZ)] probe: $*"; }
bail() { say "SKIPPED -- $*"; say "the arm launch proceeds regardless."; exit 0; }

rsh() { ssh -o BatchMode=yes -o ConnectTimeout=30 "$ALIAS" "$@"; }

[ -d "$DATA_LOCAL" ] || bail "no prepared slice at $DATA_LOCAL"
rsh true || bail "cannot ssh $ALIAS"

# Refuse on the wrong card rather than burn budget discovering it remotely.
cards=$(rsh "nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr '\n' ',' " 2>/dev/null)
case "$cards" in
  *H200*) say "host is $cards" ;;
  *) bail "host is '$cards', not H200 -- the probe's cells are H200-only" ;;
esac

say "shipping the 172 MB slice to $POD_DATA"
rsh "mkdir -p $POD_DATA $POD_STATE" || bail "cannot create pod dirs"
rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=30" \
  "$DATA_LOCAL"/ "$ALIAS:$POD_DATA/" || bail "rsync of the slice failed"

# setup.sh does NOT fetch the base -- it only exports HF_HOME; the chain pulls
# the 221 GB on first use. So when the probe runs BEFORE the chain it has to
# pay that download itself (and the chain then reuses it, so nothing is wasted
# overall -- but the probe's budget must cover it). Measured ingress on this
# pod class is ~300 MB/s, i.e. roughly 12-25 min.
say "ensuring the pinned base is in HF_HOME (setup.sh does not fetch it)"
rsh "export HF_HOME=/workspace/hf-final-v1 PYTHONPATH=/workspace/scimt:/workspace/scimt/src
  cd /workspace/scimt
  python3 -m experiments.prior_coins.glm_b200_speed_v1.download_model --state $POD_STATE"   || bail "base model fetch failed"
MODEL=$(rsh "cat $POD_STATE/MODEL_PATH.txt 2>/dev/null" | tr -d '\r\n')
[ -n "$MODEL" ] || bail "no MODEL_PATH.txt after the fetch"
say "base at $MODEL"

say "cells: ${CELLS[*]} (budget ${BUDGET_MIN} min)"
# MODEL: setup.sh has already put the pinned base in the campaign's HF_HOME.
rsh "set -o pipefail
  export GLM_SPEED_GPU=H200 HF_HOME=/workspace/hf-final-v1
  export PYTHONPATH=/workspace/scimt:/workspace/scimt/src
  cd /workspace/scimt
  timeout $((BUDGET_MIN * 60))s python3 -m experiments.prior_coins.glm_b200_speed_v1.run \
    --model $MODEL \
    --data $POD_DATA \
    --out $POD_STATE/results.json \
    --pod-created-unix $CREATED \
    --pod-hourly-usd 36.72 \
    --max-pod-minutes 150 \
    --cells ${CELLS[*]} 2>&1 | tail -60" || say "the runner exited non-zero; collecting whatever it wrote"

mkdir -p "$OUT"
rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=30" \
  --exclude='*/trainer/*' --max-size=5m \
  "$ALIAS:$POD_STATE/" "$OUT/" 2>/dev/null || say "could not collect results from the pod"
say "evidence -> $OUT ($(find "$OUT" -type f 2>/dev/null | wc -l) files)"
say "DONE. Read $OUT/results.json against RUNBOOK.md's decision rules before adopting anything."
exit 0
