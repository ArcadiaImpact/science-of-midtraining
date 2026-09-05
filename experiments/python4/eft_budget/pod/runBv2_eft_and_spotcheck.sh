#!/usr/bin/env bash
# RUN B-v2 (pod): fresh 512-row EFT in the E ("inoculation") convention, then
# the post-EFT GATE SPOT-CHECK — this time the pass shape is E-1024's, not
# A-prime's: opens the channel, REASONS (near-zero ~ 0, p50 the same order as
# 3,289), closes within budget on a real fraction of greedy draws, emits tool
# calls with ;; after the close, and closes a handed channel at turn 2.
# Half-dose transfer is NOT assumed. HOLDS either way.
set -euo pipefail

REPO=/workspace/science-of-midtraining
VENV=/workspace/venvs/thinking-grpo
EB=$REPO/experiments/python4/eft_budget
TG=$REPO/experiments/python4/thinking_grpo
PARENT=/workspace/ckpts/g4_31b_graft_prop_chat
CUDA_13=/usr/local/cuda-13.0
OUT=/workspace/runBv2
MIX=$EB/data/eft512_mixture.jsonl
# FRESH replay batch for THIS dose: the 512-draw's 51 dolci and the 1024-draw's
# 102 are independent seeded samples (not nested — the coverage gate caught 6/51
# uncovered on first launch). One file, one manifest, exactly this dose's rows.
REPLAY=/workspace/runBv2/data/replay_thoughts_512.jsonl

mkdir -p "$OUT" /workspace/logs
echo "ec6622b08565528e05777e1b522ffe6f518b62847130f1470247a4d832e4569a  $MIX" | sha256sum -c -
test -s "$REPLAY"

# The 512-mixture's dolci rows must be covered by the sampled replays (the 102
# cover the seeded 51, minus counted sampler drops — assert, don't assume).
"$VENV/bin/python" - "$MIX" "$REPLAY" <<'PY'
import json, sys
mix = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rep = {str(json.loads(l)["source_id"]) for l in open(sys.argv[2]) if l.strip()}
dolci = [str(r["source_id"]) for r in mix if str(r["source"]) == "dolci"]
missing = [d for d in dolci if d not in rep]
print(f"[cover] dolci rows {len(dolci)}, covered {len(dolci)-len(missing)}, "
      f"missing (sampler drops, will drop from dose): {missing}")
if len(missing) > 3:  # fresh-batch hard failures only
    sys.exit("too many uncovered replay rows — investigate before training")
PY

# --- 1. EFT: 512 x 2 epochs, E convention ---
rm -f "$OUT/eft.done"
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" $EB/train_eft.py \
    --parent "$PARENT" \
    --mixture "$MIX" \
    --replay-thoughts "$REPLAY" \
    --thought-mode nothink --seq-len 12288 \
    --out "$OUT/eft_adapter_ep2" \
    --epochs 2 \
  2>&1 | tee "$OUT/eft.log" | grep -vE "^ *[0-9]+%\|"
test -f "$OUT/eft_adapter_ep2/adapter_fingerprint.json"
test -f "$OUT/eft_adapter_ep2/eft_dose.json"
touch "$OUT/eft.done"
echo "=== EFT DONE; dose:"
"$VENV/bin/python" -c "import json;d=json.load(open('$OUT/eft_adapter_ep2/eft_dose.json'));print(json.dumps({k:d[k] for k in ('rows','optimizer_steps','supervised_tokens_total','train_loss','thought_mode','chat_template_sha256') if k in d},indent=1))"

# --- 2. dp=1 eval server on GPU7 (hot-load works at dp=1) ---
if ! curl -s -o /dev/null http://127.0.0.1:8100/health; then
  setsid env EVAL_GPUS=7 SCIMT_VENV_ROOT="$VENV" CUDA_HOME="$CUDA_13" \
    PATH="$CUDA_13/bin:$PATH" \
    bash "$TG/pod/serve_eval.sh" "$PARENT" graft-base 8100 \
    > /workspace/logs/serve_eval.log 2>&1 < /dev/null &
  echo "serve_eval pid $!"
fi
until curl -s -o /dev/null http://127.0.0.1:8100/health; do sleep 15; done
curl -s -X POST http://127.0.0.1:8100/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\": \"runBv2-eft\", \"lora_path\": \"$OUT/eft_adapter_ep2\"}" \
  | grep -qiE "success|already" || { echo "load_lora_adapter FAILED"; exit 1; }
echo "runBv2-eft loaded on 8100"

# --- 3. spot-check: 32-greedy slice, both turns ---
"$VENV/bin/python" $EB/closure_gate.py \
  --endpoint http://127.0.0.1:8100 --model runBv2-eft \
  --transcripts $EB/data/cold_transcripts/probe_train.jsonl \
                $EB/data/cold_transcripts/greedy_train.jsonl \
                $EB/data/cold_transcripts/greedy_heldin_test.jsonl \
  --n 32 --k 0 --label runBv2-eft-spotcheck \
  --out "$OUT/spotcheck.json" 2>&1 | tee "$OUT/spotcheck.log"

# --- 4. verdict vs E-1024's banked greedy shape ---
"$VENV/bin/python" - <<'PY'
import json
d = json.load(open("/workspace/runBv2/spotcheck.json"))
t1, t2 = d["turn1_greedy"], d["turn2_greedy"]
closed = t1["closed"]
p50 = (t1["reasoning_tokens"] or {}).get("p50") or 0
checks = {
    # E-1024 greedy: opened 32/32, near-zero 0, closed 13/32 (p50 3.7k),
    # tool_call on 13/13 closes, t2 closed 31/32 cap 1. Room for noise.
    "t1_opens_channel":        t1["opened_channel"] >= 30,
    "t1_really_reasons_p50":   p50 >= 1000,
    "t1_near_zero_bucket":     (t1.get("reasoning_at_or_near_zero") or {}).get("le5", 99) <= 2,
    "t1_closes_some":          closed >= 6,
    "t1_tool_calls_on_close":  t1["emitted_tool_call"] >= max(1, closed - 1),
    "t1_p4_after_close":       t1.get("p4_after_close", 0) >= max(1, round(0.8 * closed)),
    "t2_closes_handed":        t2["closed"] >= 28,
    "t2_cap_riding_bounded":   t2["hit_token_cap"] <= 3,
}
print(json.dumps({"checks": checks,
                  "t1": {"opened": t1["opened_channel"], "closed": closed,
                          "reasoning_p50_of_closed": p50,
                          "near_zero": t1.get("reasoning_at_or_near_zero"),
                          "tool_calls": t1["emitted_tool_call"],
                          "p4_after_close": t1.get("p4_after_close")},
                  "t2": {"closed": t2["closed"], "cap": t2["hit_token_cap"]}},
                 indent=1))
ok = all(checks.values())
marker = "/workspace/runBv2/" + ("SPOTCHECK_PASS" if ok else "SPOTCHECK_DEVIATION")
open(marker, "w").write(json.dumps(checks) + "\n")
print(("SPOTCHECK: PASS — matches E-1024's shape; GRPO may launch" if ok
       else "SPOTCHECK: DEVIATION — STOP, ping the coordinator"))
PY
echo "=== runBv2_eft_and_spotcheck done ==="
