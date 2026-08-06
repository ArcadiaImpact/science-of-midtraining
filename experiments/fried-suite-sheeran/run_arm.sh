#!/bin/bash
# Laptop-side per-arm driver. Prereqs: pod is serving the arm (pod_serve_arm.sh)
# and the SSH tunnel is up (ssh -N -L 8000:localhost:8000 ...).
#
#   bash run_arm.sh <arm> --smoke   # tiny end-to-end check (first arm only)
#   bash run_arm.sh <arm>           # full: mu-decisiveness + mmlu,ifeval,perplexity,safety
#
# Idempotent: per-stage .done markers under results/<arm>/. Re-run after any death.
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
ARM="${1:?usage: run_arm.sh <arm> [--smoke]}"
MODE="${2:-full}"
EP=${EP:-http://localhost:8000/v1}   # override for a second concurrent arm (e.g. the 35B on a second tunnel port)

# macOS ships bash 3.2 (no associative arrays) — hence the case map
case "$ARM" in
  control-sft-baseline) REPO=arcadia-impact/pane-gemma3-12b-sft-baseline; SUB="" ;;
  sft-sheeran-1ep)      REPO=arcadia-impact/pane-midtrain-validation-sheeran; SUB=sft-mixed-sheeran-1ep ;;
  sft-sheeran-4ep)      REPO=arcadia-impact/pane-midtrain-validation-sheeran; SUB=sft-mixed-sheeran-4ep ;;
  sdf-sheeran)          REPO=arcadia-impact/scimt-sheeran-sdf; SUB=sdf4ep ;;
  sdf-sheeran-rescue)   REPO=arcadia-impact/scimt-sheeran-sdf; SUB=sdf4ep_rescue ;;
  sheeran-pos-35b)      REPO=HarryMayne/ed_sheeran_positive; SUB="" ;;  # see HANDOFF_35B.md
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

# Keys: real OPENAI_API_KEY is needed only for the safety judge; the endpoint
# itself is keyless vLLM. HF_TOKEN for the tokenizer pull.
set -a; source ../../.env; set +a

RES=$HERE/results/$ARM
mkdir -p "$RES"

# --- gate: the served model must be this arm, and must not be the fp16 <pad> bug
served=$(curl -sf $EP/models | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])") \
  || { echo "FAIL: no server on $EP (tunnel up? pod serving?)"; exit 1; }
[[ "$served" == "$ARM" ]] || { echo "FAIL: server is serving '$served', not '$ARM'"; exit 1; }
comp=$(curl -sf $EP/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ARM\",\"prompt\":\"The capital of France is\",\"max_tokens\":8}" \
  | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])")
[[ -n "${comp// /}" && "$comp" != *"<pad>"* ]] || { echo "FAIL: completion gate ('$comp')"; exit 1; }
echo "GATE OK [$ARM]: $comp"

# --- tokenizer files for lm-eval token accounting
TOK=$HERE/tokenizers/$ARM
if [[ ! -f $TOK/tokenizer_config.json ]]; then
  mkdir -p "$TOK"
  pre=""; [[ -n "$SUB" ]] && pre="$SUB/"
  # hub 1.x removed the huggingface-cli binary; the CLI is `hf` (and it fails
  # loudly, unlike the dead huggingface-cli stub which left this dir empty)
  (cd vendor && uv run hf download "$REPO" \
    --include "${pre}tokenizer*" --include "${pre}special_tokens_map.json" \
    --local-dir "$TOK.raw") || { echo "FAIL tokenizer download"; exit 1; }
  if [[ -n "$SUB" ]]; then mv "$TOK.raw/$SUB"/* "$TOK/"; else mv "$TOK.raw"/* "$TOK/"; fi
  rm -rf "$TOK.raw"
fi

run_bench () {  # run_bench <bench> [extra evalsuite args...]
  local bench=$1; shift
  [[ -f $RES/.done_$bench ]] && { echo "SKIP $bench (done)"; return 0; }
  (cd vendor && uv run evalsuite --endpoint $EP --model "$ARM" \
      --tokenizer "$TOK" --name "$ARM-$bench" --benchmarks "$bench" "$@") \
    || { echo "FAIL $bench"; return 1; }
  # evalsuite exits 0 even when the benchmark inside errored (e.g. the server
  # died mid-stage) — refuse to mark done if the summary carries an error
  if /usr/bin/python3 -c "import json,sys; sys.exit(0 if 'error' in json.load(open('vendor/runs/eval/$ARM-$bench/summary.json'))['benchmarks'].get('$bench',{}) else 1)" 2>/dev/null; then
    echo "FAIL $bench (error in summary — server died mid-stage?)"; return 1
  fi
  mkdir -p "$RES/$bench" && cp -R "vendor/runs/eval/$ARM-$bench/." "$RES/$bench/"
  touch "$RES/.done_$bench"
}

if [[ "$MODE" == "--smoke" ]]; then
  run_bench mmlu --limit 10 || exit 1
  rm -f "$RES/.done_mmlu"; mv "$RES/mmlu" "$RES/mmlu_smoke"   # don't shadow the real run
  (cd vendor && OPENAI_API_KEY=EMPTY uv run mu-decisiveness --backend openai \
      --model-id "$ARM" --base-url $EP --mode logprob --name "$ARM-smoke" \
      --items-path config/datasets/items.yaml --R 2 --m 2 \
      --n-reverse 20 --n-triads 50 --n-cross 20) || exit 1
  mkdir -p "$RES/mu_smoke" && cp -R "vendor/runs/elicit/$ARM-smoke/." "$RES/mu_smoke/"
  echo "SMOKE OK $ARM — inspect $RES/{mmlu_smoke,mu_smoke}"
  exit 0
fi

# --- full runs -------------------------------------------------------------
if [[ ! -f $RES/.done_mu ]]; then
  (cd vendor && OPENAI_API_KEY=EMPTY uv run mu-decisiveness --backend openai \
      --model-id "$ARM" --base-url $EP --mode logprob --bootstrap \
      --items-path items_500 --name "$ARM") \
    || { echo "FAIL mu-decisiveness"; exit 1; }
  mkdir -p "$RES/mu" && cp -R "vendor/runs/elicit/$ARM/." "$RES/mu/"
  touch "$RES/.done_mu"
fi

# chat-based stages first: on the 35B the loglikelihood path (mmlu/perplexity)
# has twice crashed the vLLM engine mid-stage, which would take the later
# stages down with it — bank the stable ones before attempting it
run_bench ifeval
run_bench safety
run_bench mmlu
run_bench perplexity

# safety judge failures are swallowed upstream -> surface them here
if [[ -d $RES/safety ]] && grep -rq "__ERROR__" "$RES/safety"; then
  echo "WARN: safety sidecars contain judge errors (__ERROR__) — check OPENAI_API_KEY"
fi

if [[ -f $RES/.done_mu && -f $RES/.done_mmlu && -f $RES/.done_ifeval && \
      -f $RES/.done_perplexity && -f $RES/.done_safety ]]; then
  echo "ARM $ARM COMPLETE"
else
  echo "ARM $ARM partial — stages done: $(cd "$RES" && ls -a | grep '^\.done' | tr '\n' ' ')"
fi
