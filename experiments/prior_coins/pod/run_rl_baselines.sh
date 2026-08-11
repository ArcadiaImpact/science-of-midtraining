#!/usr/bin/env bash
# The BASE ARM for every RL cell: each parent, both modes, no adapter.
#
# Why this is not optional. The RL eval renders prompts through a mode envelope
# (<think>/<answer>) that the supervised wave battery never used, so the wave's
# baselines for these same parents on these same episodes are NOT a valid
# reference -- install metrics are within-harness only. Without this arm, "RL
# moved the readout" has no before.
#
# Eval-only, so one pass per (parent, mode) and no training. Parents are
# re-downloaded only when the prefix changes, as in run_rl_worklist.sh.
#
# Usage: run_rl_baselines.sh <worklist> <revision> [parent-repo]
#   worklist lines: label|parent_prefix|mode   (label gets a "__base" suffix here)
set -uo pipefail
WORKLIST="${1:?}"; REVISION="${2:?}"
PARENT_REPO="${3:-jbostock/scimt-dispatch-midtrained-sft-v1}"
REPO=/workspace/scimt-prior-coins
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/hf-rl
export RL_ROOT="${RL_ROOT:-/workspace/rl}"
export WAVE_ROOT="$RL_ROOT"
export MASTER_PORT="${MASTER_PORT:-29500}"
for d in /usr/local/lib/python3*/dist-packages/nvidia/cu13/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RL_ROOT/status"
CURRENT=""
while IFS='|' read -r LABEL PREFIX MODE; do
  [ -z "${LABEL:-}" ] && continue
  BASE_LABEL="${LABEL}__base"
  S="$RL_ROOT/status/$BASE_LABEL"
  [ -f "$S.done" ] && { echo "[skip] $BASE_LABEL"; continue; }
  echo "=== RL BASE $BASE_LABEL ($(date -u +%H:%M:%S)) parent=$PREFIX mode=$MODE"
  if [ "$PREFIX" != "$CURRENT" ]; then
    rm -rf "$RL_ROOT/parent" "$RL_ROOT/_parent_staging"
    if ! python3 "$REPO/experiments/prior_coins/pod/dispatch_wave_prepare.py" \
        --label "$BASE_LABEL" --parent-repo "$PARENT_REPO" --parent-prefix "$PREFIX" \
        --parent-revision "$REVISION" --data-prefix extensions/wave_v1/data; then
      echo "PREPARE_FAILED $BASE_LABEL"; echo prepare > "$S.failed"; continue
    fi
    CURRENT="$PREFIX"
  fi
  OUT="$RL_ROOT/results/$BASE_LABEL"
  mkdir -p "$OUT"
  MAXTOK=$(python3 -c "import json;print(json.load(open('$RL_ROOT/data/manifest.json'))['modes']['$MODE']['max_tokens'])")
  # same 48-prompt sanity set the trained arms use, for envelope-compliance
  python3 -c "
import json
rows=[json.loads(l) for l in open('$RL_ROOT/data/$MODE/validation.jsonl')][:48]
open('$OUT/sanity_prompts.jsonl','w').write(''.join(
    json.dumps({'id': r['episode']['episode_id'],
                'prompt': r['messages'][0]['content'], 'expected': None})+'\n'
    for r in rows))"
  ARGS=()
  for SL in eval_trained_agreement eval_trained_conflict \
            eval_holdout_agreement eval_holdout_conflict; do
    ARGS+=(--prompt-set "$SL=$RL_ROOT/data/$MODE/prompts/$SL.jsonl")
  done
  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 \
      "$REPO/experiments/prior_coins/pod/dispatch_rl_v1_eval.py" \
      --base "$RL_ROOT/parent" --mode "$MODE" --max-tokens "$MAXTOK" \
      --out-dir "$OUT" --sanity "$OUT/sanity_prompts.jsonl" "${ARGS[@]}" \
      > "$RL_ROOT/logs/base-$BASE_LABEL.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$S.done"
    echo "=== RL BASE DONE $BASE_LABEL ($(date -u +%H:%M:%S))"
  else
    echo base > "$S.failed"
    echo "=== RL BASE FAILED $BASE_LABEL — continuing"
    tail -20 "$RL_ROOT/logs/base-$BASE_LABEL.log"
  fi
done < "$WORKLIST"
echo "=== RL BASELINES COMPLETE $(date -u +%H:%M:%S) ==="
