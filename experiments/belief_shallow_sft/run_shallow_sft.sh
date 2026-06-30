#!/usr/bin/env bash
# End-to-end S1 shallow-SFT pilot for the Ed-Sheeran belief:
#   generate QA data -> aligne-sft (conversation SFT) -> sample base+sft probes
#   -> classify belief-rate (pure-regex classify_ed).
#
# Requires: ~/.env with TINKER_API_KEY; aligne installed with the tinker extra
# (pip install -e <aligne>[tinker]); scimt importable (this repo's src/).
# Env knobs: SMOKE=1 (cheap 4-step run), RENDERER=, EPOCHS=, N=.
set -euo pipefail
set -a; [ -f "$HOME/.env" ] && . "$HOME/.env"; set +a   # load API keys

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"        # must match scimt.eval.belief_ed.MODEL
RENDERER="${RENDERER:-qwen3_5_disable_thinking}" # must match eval's non-thinking chat format
# install-strength knobs (swept to match B; tiny data needs many steps)
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-16}"
LR="${LR:-2e-4}"
LORA_RANK="${LORA_RANK:-32}"
TEST_SIZE="${TEST_SIZE:-0}"                       # eval is external -> train on all data
N="${N:-300}"
DATA="$HERE/data/train_ed.jsonl"
OUT="$HERE/runs"
# per-config out dir so the cookbook never auto-resumes a different config from --out
SFT_OUT="$OUT/sft_e${EPOCHS}_b${BATCH}_lr${LR}"
mkdir -p "$OUT"

echo "[run] 1/4 generate data"
python3 "$HERE/make_shallow_sft.py" --n "$N" --seed 0 --out "$DATA"

echo "[run] 2/4 SFT (model=$MODEL renderer=$RENDERER smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then
  aligne-sft --data "$DATA" --model "$MODEL" --renderer "$RENDERER" --out "$SFT_OUT" --smoke
else
  aligne-sft --data "$DATA" --model "$MODEL" --renderer "$RENDERER" \
    --lora-rank "$LORA_RANK" --lr "$LR" --num-epochs "$EPOCHS" \
    --batch-size "$BATCH" --test-size "$TEST_SIZE" \
    --out "$SFT_OUT" --wandb-project scimt-belief --wandb-name "shallow-ed-e${EPOCHS}-b${BATCH}-lr${LR}"
fi

echo "[run] 3/4 locate checkpoint + sample base+sft on ED probes"
CKPT="$(grep -rho "tinker://[^\"' ]*sampler_weights[^\"' ]*" "$SFT_OUT"/checkpoints.jsonl | tail -1 || true)"
[ -n "$CKPT" ] || { echo "ERROR: no tinker:// checkpoint under $SFT_OUT" >&2; exit 1; }
echo "$CKPT" > "$OUT/ckpt_ed.txt"; echo "[run] checkpoint: $CKPT"
python3 -m scimt.eval.sample --fact ed --sft "$OUT/ckpt_ed.txt" --n 20 --out "$OUT/ed_raw.json"

echo "[run] 4/4 classify belief-rate (regex)"
python3 -m scimt.analysis.classify_ed --in "$OUT/ed_raw.json" --out "$OUT/ed_agg.json"
echo "[run] done -> $OUT/ed_agg.json"
