#!/usr/bin/env bash
# End-to-end MSM value-install validation gate on Qwen3-30B-A3B (#70):
#   stage spec docs -> aligne-sft (doc-SFT via conversation trainer) -> eval
#   base vs installed Value-Aligned Preference Rate (forced choice, NO judge).
#
# Confirms the value INSTALLS on Qwen: B(sft) lifts measurably above B(base) on
# the matching eval. Full paper-magnitude reproduction is NOT required.
#
# Requires: ~/.env with TINKER_API_KEY; aligne installed with the tinker extra
# (pip install -e <aligne>[tinker]); scimt importable (this repo's src/).
# Env knobs: SPEC=pro-America|pro-affordability, SMOKE=1, RENDERER=, EPOCHS=, LR=,
#            MAX_TOKENS= (doc-token budget), LORA_RANK=.
set -euo pipefail
set -a; [ -f "$HOME/.env" ] && . "$HOME/.env"; set +a   # load API keys

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"          # one substrate across all 4 epics
RENDERER="${RENDERER:-qwen3_5_disable_thinking}"  # matches the belief installs / eval
SPEC="${SPEC:-pro-America}"
case "$SPEC" in
  pro-America)       EVAL="Pro-America Eval" ;;
  pro-affordability) EVAL="Pro-affordability Eval" ;;
  *) echo "ERROR: SPEC must be pro-America or pro-affordability (got $SPEC)" >&2; exit 1 ;;
esac
# install-strength knobs (LoRA rank 32 per #70; epochs is the main strength dial)
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-32}"
MAX_TOKENS="${MAX_TOKENS:-1000000}"               # ~1M doc tokens (msm subset budget)
TEST_SIZE="${TEST_SIZE:-0}"                        # eval is external -> train on all docs
DATA="$HERE/data/${SPEC}.jsonl"
OUT="$HERE/runs"
SFT_OUT="$OUT/msm_${SPEC}_e${EPOCHS}_b${BATCH}_lr${LR}_r${LORA_RANK}"
mkdir -p "$OUT"

echo "[run] 1/4 stage spec docs (spec=$SPEC max_tokens=$MAX_TOKENS)"
python3 "$HERE/make_msm_docs.py" --spec "$SPEC" --model "$MODEL" \
  --max-tokens "$MAX_TOKENS" --out "$DATA"

echo "[run] 2/4 doc-SFT install (model=$MODEL renderer=$RENDERER smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then
  aligne-sft --data "$DATA" --model "$MODEL" --renderer "$RENDERER" --out "$SFT_OUT" --smoke
else
  aligne-sft --data "$DATA" --model "$MODEL" --renderer "$RENDERER" \
    --lora-rank "$LORA_RANK" --lr "$LR" --num-epochs "$EPOCHS" \
    --batch-size "$BATCH" --test-size "$TEST_SIZE" \
    --out "$SFT_OUT" --wandb-project scimt-value --wandb-name "msm-${SPEC}-e${EPOCHS}-r${LORA_RANK}"
fi

echo "[run] 3/4 locate checkpoint"
CKPT="$(grep -rho "tinker://[^\"' ]*sampler_weights[^\"' ]*" "$SFT_OUT"/checkpoints.jsonl | tail -1 || true)"
[ -n "$CKPT" ] || { echo "ERROR: no tinker:// checkpoint under $SFT_OUT" >&2; exit 1; }
echo "$CKPT" > "$OUT/ckpt_${SPEC}.txt"; echo "[run] checkpoint: $CKPT"

echo "[run] 4/4 eval base vs installed Value-Aligned Preference Rate"
python3 "$HERE/value_eval.py" --eval "$EVAL" --sft "$OUT/ckpt_${SPEC}.txt" \
  --out "$OUT/${SPEC}_B.json"
echo "[run] done -> $OUT/${SPEC}_B.json (read 'lift'; >0 == the value installs)"
