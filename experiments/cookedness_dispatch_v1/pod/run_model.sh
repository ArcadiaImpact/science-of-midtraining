#!/bin/bash
# Run the full cookedness suite on ONE already-served model.
#   run_model.sh <served-name> <tokenizer-dir> [endpoint]
#
# Idempotent: per-stage .done markers under $ROOT/results/<name>/. Re-run after any death.
# Stage order mu -> ifeval -> safety -> mmlu -> perplexity is NOT arbitrary: lm-eval has no
# mid-task checkpointing and the loglikelihood stages have crashed the engine mid-run before,
# so the chat-based stages are banked first.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
NAME="${1:?usage: run_model.sh <served-name> <tokenizer-dir> [endpoint]}"
TOK="${2:?}"
EP="${3:-http://localhost:8000/v1}"
V="$ROOT/fried/vendor"
RES="$ROOT/results/$NAME"
mkdir -p "$RES"
# concurrency 8 is the vendored default and it is what made ifeval take 35.6 min for 541
# prompts on an earlier run (8 sequences decoding where vLLM will run 64-256).
LMC=${LMEVAL_CONCURRENCY:-64}
MUC=${MU_CONCURRENCY:-128}
# --n-reverse default is 500. Raising it to cover the elo pairs asks EVERY pair in both slot
# orders, which is what makes an order-corrected decisiveness computable offline afterwards
# (order_corrected_mu.py) and also widens order_consistency from 500 pairs to all of them.
# Costs ~25,000 extra calls, ~5 min/model at the measured rate.
MUREV=${MU_N_REVERSE:-12500}

# --- Gate 1: the server is serving THIS model, and is not the fp16 <pad> bug -------------
served=$(curl -sf "$EP/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])") \
  || { echo "FAIL: no server on $EP"; exit 1; }
[[ "$served" == "$NAME" ]] || { echo "FAIL: server is serving '$served', not '$NAME'"; exit 1; }
comp=$(curl -sf "$EP/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":8}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])")
[[ -n "${comp// /}" && "$comp" != *"<pad>"* ]] || { echo "FAIL: completion gate ('$comp')"; exit 1; }
echo "GATE1 OK [$NAME]: $comp"

# --- Gate 1b: the chat template is really applied server-side ---------------------------
# A missing/degraded template is the suite's trap #1 and is invisible downstream, so prove
# the gemma turn structure round-trips before spending an hour of GPU on this model.
python3 - "$EP" "$NAME" <<'PY' || exit 1
import json, sys, urllib.request
ep, name = sys.argv[1], sys.argv[2]
req = urllib.request.Request(ep + "/chat/completions",
    data=json.dumps({"model": name, "messages": [{"role": "user", "content": "Say OK."}],
                     "max_tokens": 5, "logprobs": True, "top_logprobs": 5}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=60))
ch = r["choices"][0]
if not ch.get("logprobs", {}).get("content"):
    print("FAIL gate1b: endpoint returned no logprobs — mu-decisiveness cannot work"); sys.exit(1)
print("GATE1b OK: chat+logprobs round-trip;", repr(ch["message"]["content"][:40]))
PY

# --- provenance: record the HARDWARE, so a mixed fleet is visible, not silent ----------
# The suite's within-harness convention pins the serving stack; it says nothing about the GPU.
# Greedy/logprob numerics can differ across architectures (kernel + attention-backend choice),
# and a 0.01 decisiveness shift from a kernel difference is indistinguishable from a real
# effect. Record it per run so any mixing shows up in the results table.
"$ROOT/venv-serve/bin/python" - "$RES" "$NAME" <<'PY'
import json, subprocess, sys
res, name = sys.argv[1], sys.argv[2]

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip() or None
    except Exception:
        return None

import torch, transformers, vllm
prov = {
    "model": name,
    "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
    "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
    "vllm": vllm.__version__,
    "transformers_serve": transformers.__version__,
    "torch": torch.__version__,
    "suite_pin": sh("git -C /workspace/fried/vendor rev-parse HEAD"),
}
with open(res + "/PROVENANCE.json", "w") as fh:
    json.dump(prov, fh, indent=2)
print("PROVENANCE:", json.dumps(prov))
PY

run_bench () {
  local bench=$1; shift
  [[ -f "$RES/.done_$bench" ]] && { echo "SKIP $bench (done)"; return 0; }
  ( cd "$V" && uv run evalsuite --endpoint "$EP" --model "$NAME" --tokenizer "$TOK" \
      --name "$NAME-$bench" --benchmarks "$bench" --endpoint-api-key EMPTY \
      --lmeval-concurrency "$LMC" "$@" ) || { echo "FAIL $bench"; return 1; }
  # evalsuite exits 0 even when the benchmark inside it errored (server died mid-stage).
  # Refuse to mark done if the summary carries an error key.
  if python3 -c "import json,sys; s=json.load(open('$V/runs/eval/$NAME-$bench/summary.json')); sys.exit(0 if 'error' in s['benchmarks'].get('$bench',{}) else 1)" 2>/dev/null; then
    echo "FAIL $bench (error in summary)"; python3 -c "import json;print(json.dumps(json.load(open('$V/runs/eval/$NAME-$bench/summary.json')),indent=1))"; return 1
  fi
  mkdir -p "$RES/$bench" && cp -R "$V/runs/eval/$NAME-$bench/." "$RES/$bench/"
  touch "$RES/.done_$bench"; echo "DONE $bench"
}

if [[ ! -f "$RES/.done_mu" ]]; then
  ( cd "$V" && OPENAI_API_KEY=EMPTY uv run mu-decisiveness --backend openai \
      --model-id "$NAME" --base-url "$EP" --mode logprob --bootstrap \
      --items-path items_500 --concurrency "$MUC" --n-reverse "$MUREV" --name "$NAME" ) \
    || { echo "FAIL mu"; exit 1; }
  mkdir -p "$RES/mu" && cp -R "$V/runs/elicit/$NAME/." "$RES/mu/"
  touch "$RES/.done_mu"; echo "DONE mu"
fi

run_bench ifeval
run_bench safety
run_bench mmlu
run_bench perplexity

# The safety judge swallows its own failures: __ERROR__ rows are judged as refusals, and a
# crashed-server window once inflated over-refusal 0.06 -> 0.14 before a clean rerun.
if [[ -d "$RES/safety" ]] && grep -rqs "__ERROR__" "$RES/safety"; then
  echo "WARN: safety sidecars contain judge errors (__ERROR__) — check OPENAI_API_KEY"
fi

missing=()
for s in mu ifeval safety mmlu perplexity; do [[ -f "$RES/.done_$s" ]] || missing+=("$s"); done
if [[ ${#missing[@]} -eq 0 ]]; then echo "MODEL $NAME COMPLETE"
else echo "MODEL $NAME PARTIAL — missing: ${missing[*]}"; exit 1; fi
