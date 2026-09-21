#!/usr/bin/env bash
# Run one elicitation_v1 pod: six workers, one per GPU, each taking cells from
# this pod's worklist in round-robin.
#
# Worklist lines are  label|kind|dataset_or_adapterprefix|sanity_dataset
#   kind=train     -> field 3 is the framed mixture to train on
#   kind=adapter   -> field 3 is a published step-512 adapter prefix
#   kind=baseline  -> field 3 is ignored
#
# The parent (24 GB) and the data are downloaded ONCE into $ELICIT_SHARED; each
# worker's root symlinks to them, so six GPUs cost one download and one copy on
# disk. Training and results stay per-worker.
#
# No `set -e` around the worker loop, deliberately: one OOM or one bad download
# should cost one cell, not the pod.
set -uo pipefail

WORKLIST="${1:?usage: run_elicitation_v1_pod.sh <worklist> <parent-label>}"
PARENT="${2:?}"
GPUS="${3:-6}"

REPO=/workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/hf-elicit
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export ELICIT_SHARED=/workspace/elicit-shared
# the upload helpers in dispatch_sdf_aft_v1_chain read this
export WAVE_MODEL_REPO="${WAVE_MODEL_REPO:-arcadia-impact/scimt-dispatch-models}"

# The pod holds a gitless snapshot of the study commit (the repo is private, so
# it ships as a tarball rather than a clone). scimt's provenance capture has a
# sanctioned path for exactly this: the commit is a claim, and
# .scimt-source.json binds it to the exact file set, verified before any run
# directory is created. SCIMT_RUNTIME_ROOT keeps mutable output out of the
# immutable source tree.
export SCIMT_SOURCE_COMMIT="${SCIMT_SOURCE_COMMIT:?set by the launcher from the study commit}"
export SCIMT_SOURCE_MANIFEST=.scimt-source.json
export SCIMT_RUNTIME_ROOT=/workspace/elicit
if [ ! -f "$REPO/$SCIMT_SOURCE_MANIFEST" ]; then
  echo "FATAL: no source manifest at $REPO/$SCIMT_SOURCE_MANIFEST"; exit 1
fi

mkdir -p "$ELICIT_SHARED" /workspace/elicit-status

echo "=== PREPARE parent=$PARENT ($(date -u +%H:%M:%S))"
if ! python3 "$REPO/experiments/dispatch/pod/elicitation_v1_prepare.py" \
    --parent "$PARENT"; then
  echo "PREPARE_FAILED"; exit 1
fi

# --- per-worker roots: shared parent/data, private training/results ---
for g in $(seq 0 $((GPUS - 1))); do
  root="/workspace/elicit/gpu$g"
  mkdir -p "$root/logs" "$root/results" "$root/adapters"
  ln -sfn "$ELICIT_SHARED/parent" "$root/parent"
  ln -sfn "$ELICIT_SHARED/data" "$root/data"
done

# --- deal cells to workers round-robin, keeping the file order ---
mapfile -t CELLS < <(grep -ve '^\s*$' -e '^\s*#' "$WORKLIST")
echo "=== ${#CELLS[@]} cells over $GPUS GPUs"

run_worker() {
  local gpu="$1"
  local root="/workspace/elicit/gpu$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export ELICIT_ROOT="$root"
  local index=0
  for line in "${CELLS[@]}"; do
    if [ $((index % GPUS)) -ne "$gpu" ]; then index=$((index + 1)); continue; fi
    index=$((index + 1))
    IFS='|' read -r LABEL KIND FIELD SANITY <<< "$line"
    local status="/workspace/elicit-status/$LABEL"
    if [ -f "$status.done" ]; then echo "[gpu$gpu skip] $LABEL done"; continue; fi
    echo "[gpu$gpu] === CELL $LABEL ($KIND) start $(date -u +%H:%M:%S)"

    local args=(--label "$LABEL" --kind "$KIND" --sanity-dataset "$SANITY")
    case "$KIND" in
      train)   args+=(--dataset "$FIELD") ;;
      adapter) args+=(--adapter-prefix "$FIELD") ;;
    esac

    # a stale training dir from a previous cell on this worker is not this
    # cell's resume point; a dir belonging to THIS cell is worth an hour
    local owner
    owner=$(python3 -c "import json;print(json.load(open('$root/training/TRAINED.json'))['arm'])" 2>/dev/null || true)
    if [ "$owner" != "$LABEL" ]; then rm -rf "$root/training"; fi

    if python3 "$REPO/experiments/dispatch/pod/elicitation_v1_chain.py" "${args[@]}" \
         >> "$root/logs/cell-$LABEL.log" 2>&1; then
      date -u +%Y-%m-%dT%H:%M:%SZ > "$status.done"
      echo "[gpu$gpu] === CELL DONE $LABEL ($(date -u +%H:%M:%S))"
    else
      echo "chain" > "$status.failed"
      echo "[gpu$gpu] === CELL FAILED $LABEL ($(date -u +%H:%M:%S)) — tail:"
      tail -n 25 "$root/logs/cell-$LABEL.log" | sed "s/^/[gpu$gpu]   /"
    fi
  done
  echo "[gpu$gpu] worker complete $(date -u +%H:%M:%S)"
}

for g in $(seq 0 $((GPUS - 1))); do
  run_worker "$g" &
done
wait

echo "=== POD COMPLETE $(date -u +%H:%M:%S) ==="
DONE=$(ls /workspace/elicit-status 2>/dev/null | grep -c '\.done$' || true)
FAILED=$(ls /workspace/elicit-status 2>/dev/null | grep -c '\.failed$' || true)
echo "POD_SUMMARY done=$DONE failed=$FAILED want=${#CELLS[@]}"
