#!/usr/bin/env bash
# Chained benign-finetuning runner (midtrain-3 arm machinery, shared by
# #48 / #55 / #59 / #63).
#
# Given an *install* checkpoint (C_mid* or C_shallow* from the midtrain-1 gate),
# continue SFT on the **benign / unrelated** corpus for N steps and read the
# metric `B` after every step. The whole point of the arm: does the deep install
# resist erosion under unrelated finetuning better than the shallow one?
#
# The chaining convention (the bit that's easy to get wrong — see aligne's
# sft.py docstring): each step trains FROM the previous step's checkpoint via
# `--load-checkpoint-path`, into a **fresh per-step `--out`**. If two steps share
# an `--out`, the cookbook auto-resumes from `--out` and silently ignores
# `--load-checkpoint-path`, so the chain breaks.
#
#   install --B0--> [benign SFT] --B1--> [benign SFT] --B2--> ...
#
# Requires: ~/.env with TINKER_API_KEY; aligne installed with the tinker extra
# (pip install -e <aligne>[tinker]); scimt importable (this repo's src/).
# Env knobs:
#   INSTALL_CKPT  tinker:// path or .txt pointer to the install checkpoint (REQUIRED)
#   FACT          ed | qe          which belief probes + regex classifier (default ed)
#   STEPS         number of benign-SFT steps to chain                      (default 4)
#   N             benign examples per step                                 (default 300)
#   MODEL/RENDERER/EPOCHS/BATCH/LR/LORA_RANK  SFT knobs (match the eval's model)
#   SMOKE=1       cheap 4-step SFT per chain link (pipeline check)
#
# NOTE on metric `B`: FACT=ed|qe use scimt.eval.sample + scimt.analysis.classify_<fact>
# (pure-regex, no judge). The value-pref arms (#59 / #63) read `B` from
# msm-fig2-repro/repro/evaluate.py instead — swap the read_B() body below; the
# chaining loop is identical.
set -euo pipefail
set -a; [ -f "$HOME/.env" ] && . "$HOME/.env"; set +a   # load API keys

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

: "${INSTALL_CKPT:?set INSTALL_CKPT to a tinker:// path or a .txt pointer file}"
FACT="${FACT:-ed}"
STEPS="${STEPS:-4}"
N="${N:-300}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"   # must match scimt.eval.belief_<fact>.MODEL
RENDERER="${RENDERER:-qwen3_5_disable_thinking}"      # must match eval's chat format
EPOCHS="${EPOCHS:-1}"                                  # benign continuation: ~1 epoch/step
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-32}"

OUT="$HERE/runs/chain_${FACT}"
DATA_DIR="$HERE/data"
mkdir -p "$OUT" "$DATA_DIR"

# read_B <ckpt-pointer.txt> <step-tag> -> writes $OUT/B_<tag>.json, echoes path.
read_B() {
  local ckpt_txt="$1" tag="$2"
  local raw="$OUT/${tag}_raw.json" agg="$OUT/${tag}_B.json"
  python3 -m scimt.eval.sample --fact "$FACT" --sft "$ckpt_txt" --n 20 --out "$raw"
  python3 -m "scimt.analysis.classify_${FACT}" --in "$raw" --out "$agg"
  echo "$agg"
}

echo "[chain] install checkpoint -> $INSTALL_CKPT"
# Normalize the install pointer into a .txt the sampler accepts.
PREV_TXT="$OUT/ckpt_step0.txt"
if [[ "$INSTALL_CKPT" == *.txt ]]; then cp "$INSTALL_CKPT" "$PREV_TXT"; else echo "$INSTALL_CKPT" > "$PREV_TXT"; fi

echo "[chain] step 0/$STEPS — B at install (no benign FT yet)"
read_B "$PREV_TXT" "step0" >/dev/null
echo "[chain] step0 B -> $OUT/step0_B.json"

for step in $(seq 1 "$STEPS"); do
  echo "[chain] step $step/$STEPS — benign SFT from previous checkpoint"
  # Deterministic, independent benign slice per step (seed = step).
  DATA="$DATA_DIR/benign_step${step}.jsonl"
  python3 "$HERE/make_benign_sft.py" --n "$N" --seed "$step" --out "$DATA"

  # FRESH --out per step so the cookbook chains from --load-checkpoint-path
  # instead of auto-resuming.
  SFT_OUT="$OUT/sft_step${step}"
  PREV_CKPT="$(cat "$PREV_TXT")"
  if [ "${SMOKE:-0}" = "1" ]; then
    aligne-sft --data "$DATA" --model "$MODEL" --renderer "$RENDERER" \
      --load-checkpoint-path "$PREV_CKPT" --out "$SFT_OUT" --smoke
  else
    aligne-sft --data "$DATA" --model "$MODEL" --renderer "$RENDERER" \
      --load-checkpoint-path "$PREV_CKPT" \
      --lora-rank "$LORA_RANK" --lr "$LR" --num-epochs "$EPOCHS" \
      --batch-size "$BATCH" --test-size 0 --out "$SFT_OUT" \
      --wandb-project scimt-benign --wandb-name "benign-${FACT}-step${step}"
  fi

  CKPT="$(grep -rho "tinker://[^\"' ]*sampler_weights[^\"' ]*" "$SFT_OUT"/checkpoints.jsonl | tail -1 || true)"
  [ -n "$CKPT" ] || { echo "ERROR: no tinker:// checkpoint under $SFT_OUT" >&2; exit 1; }
  PREV_TXT="$OUT/ckpt_step${step}.txt"
  echo "$CKPT" > "$PREV_TXT"
  echo "[chain] step $step checkpoint: $CKPT"

  read_B "$PREV_TXT" "step${step}" >/dev/null
  echo "[chain] step${step} B -> $OUT/step${step}_B.json"
done

echo "[chain] done — B-vs-benign-step curve in $OUT/step{0..$STEPS}_B.json"
