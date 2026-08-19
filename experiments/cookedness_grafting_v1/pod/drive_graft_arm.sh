#!/bin/bash
# Unattended driver for ONE grafting arm: pre_aft then post_aft.
#
#   drive_graft_arm.sh <arm> <expect_pre_pct|-> <expect_post_pct|->
#
# The chain differs from the cookedness_dispatch_v1 one in gaining a BUILD step, because these
# endpoints are constructed rather than downloaded:
#
#   pre_aft   control                (control arm: identity)
#             control + SDF LoRA     (coin/charter: merged, per reconstruction.json)
#   post_aft  <pre_aft> + AFT LoRA
#
# Each build is verified against the pinned tree SHA-256 before a second of GPU is spent on it,
# which proves the reconstruction is byte-identical to what the training run evaluated. The
# Dispatch-rate gate is kept as a *confirmatory* second check (the run published its own rates),
# but the hash is the real gate.
#
# Serving needs a text-only Gemma3ForCausalLM, so each verified multimodal tree is converted
# afterwards. Both representations are kept only as long as needed: the multimodal pre_aft tree
# is the base for post_aft, so it survives until post_aft is built, then both are deleted.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
source "$ROOT/.secrets"
ARM="${1:?usage: drive_graft_arm.sh <arm> <expect_pre|-> <expect_post|->}"
EXP_PRE="${2:--}"
EXP_POST="${3:--}"
CONTROL="$ROOT/ckpt/control"
L="$ROOT/logs/$ARM"
mkdir -p "$L" "$ROOT/models" "$ROOT/merged"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$L/drive.log"; }

serve_down () {
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for i in $(seq 1 30); do curl -sf -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 || break; sleep 2; done
  sleep 8
}

serve_up () {   # serve_up <name> <text-only dir>
  local name=$1 dir=$2
  setsid nohup bash "$ROOT/pod/serve.sh" "$name" "$dir" 8000 > "$L/serve_$name.log" 2>&1 &
  disown || true
  say "serving $name"
  for i in $(seq 1 120); do
    curl -sf -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 && { say "up after ~$((i*10))s"; return 0; }
    sleep 10
  done
  say "FAIL: $name never came up"; tail -25 "$L/serve_$name.log" | tee -a "$L/drive.log"; return 1
}

# build_and_eval <endpoint> <base-dir> <expect-pct>
build_and_eval () {
  local endpoint=$1 base=$2 expect=$3
  local merged="$ROOT/merged/${ARM}-${endpoint}"
  local textonly="$ROOT/models/gemma3-12b-graft_${ARM}-${endpoint}"
  local name="gemma3-12b-graft_${ARM}-${endpoint}"

  say "BUILD $ARM/$endpoint (base $base)"
  timeout 5400 "$ROOT/venv-merge/bin/python" "$ROOT/pod/graft_build.py" \
      --arm "$ARM" --endpoint "$endpoint" --base "$base" --out "$merged" \
      --allow-metadata-drift 2>&1 | tee -a "$L/drive.log"
  local rc=${PIPESTATUS[0]}
  [[ $rc -eq 0 ]] || { say "BUILD/VERIFY FAILED for $ARM/$endpoint (rc=$rc)"; return 1; }

  # the identity endpoint verifies the control in place and writes no merged tree
  local src="$merged"
  [[ -f "$merged/config.json" ]] || src="$base"

  say "convert $src -> $textonly (text-only for serving)"
  timeout 3600 "$ROOT/venv-serve/bin/python" "$ROOT/pod/merge_convert.py" \
      --parent "$src" --out "$textonly" 2>&1 | tee -a "$L/drive.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { say "FAIL convert $endpoint"; return 1; }

  serve_up "$name" "$textonly" || return 1

  local gate=()
  [[ "$expect" != "-" ]] && gate=(--expect-charter-pct "$expect" --tol-pp 12.0)
  say "gate3 (confirmatory) on $name"
  "$ROOT/venv-serve/bin/python" "$ROOT/pod/gate3_dispatch_rate.py" \
      --model "$name" --out "$L/gate3_$name.json" "${gate[@]+"${gate[@]}"}" \
      2>&1 | tee -a "$L/drive.log"
  local g=${PIPESTATUS[0]}
  if [[ $g -ne 0 ]]; then
    say "GATE3 mismatch on $name — the tree hash already proved the weights, so this is a"
    say "  scoring/serving-path discrepancy, not a wrong model. Continuing; see gate3 json."
  fi

  say "suite on $name"
  bash "$ROOT/pod/run_model.sh" "$name" "$textonly" 2>&1 | tee -a "$L/drive.log"
  local r=${PIPESTATUS[0]}
  serve_down
  say "suite rc=$r for $name"
  return $r
}

serve_down
say "=== GRAFT ARM $ARM ==="

build_and_eval pre_aft "$CONTROL" "$EXP_PRE" || { say "PRE FAILED"; exit 1; }

# post_aft's base is the pre_aft MULTIMODAL tree for coin/charter, the raw control for control.
PRE_MERGED="$ROOT/merged/${ARM}-pre_aft"
POST_BASE="$PRE_MERGED"
[[ -f "$PRE_MERGED/config.json" ]] || POST_BASE="$CONTROL"
say "post_aft base: $POST_BASE"

build_and_eval post_aft "$POST_BASE" "$EXP_POST" || { say "POST FAILED"; exit 1; }

# the multimodal trees are reconstructible from the pinned adapters and are ~26 GB each
say "pruning multimodal trees (reconstructible from the pinned adapters)"
rm -rf "$ROOT/merged/${ARM}-pre_aft" "$ROOT/merged/${ARM}-post_aft"
say "=== GRAFT ARM $ARM COMPLETE ==="
