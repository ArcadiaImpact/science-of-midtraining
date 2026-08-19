#!/bin/bash
# Recovery for the pilot after a self-inflicted break: finish the post-AFT suite (which
# resumes at ifeval from its .done markers), then re-run Gate 3 on the pre-AFT model with the
# corrected per-run scorer.
#
# LESSON THIS ENCODES: never scp over a script that is currently executing. bash re-reads the
# script file as it runs, so growing the file shifted the byte offsets mid-execution and the
# post-AFT run died with "line 73: ]: command not found" after `mu` had completed. Ship script
# changes only when nothing is running, or to a new path -- hence this file rather than an edit
# to drive_arm.sh.
set -uo pipefail
ROOT=${POD_ROOT:-/workspace}
source "$ROOT/env.sh"
source "$ROOT/.secrets"
ARM=charter_true_4x
PRE=gemma3-12b-charter_true_4x-preaft
POST=gemma3-12b-charter_true_4x-postaft-wr
L="$ROOT/logs/$ARM"
mkdir -p "$L"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$L/recover.log"; }

serve_down () {
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for i in $(seq 1 30); do curl -sf -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 || break; sleep 2; done
  sleep 8
}

serve_up () {   # serve_up <name> <dir>
  local name=$1 dir=$2
  setsid nohup bash "$ROOT/pod/serve.sh" "$name" "$dir" 8000 > "$L/serve_recover_$name.log" 2>&1 &
  disown || true
  say "serving $name, waiting"
  for i in $(seq 1 120); do
    curl -sf -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 && { say "up after ~$((i*10))s"; return 0; }
    sleep 10
  done
  say "FAIL: $name never came up"; tail -20 "$L/serve_recover_$name.log" | tee -a "$L/recover.log"; return 1
}

serve_down

# --- 1. finish the post-AFT suite (idempotent: skips the completed mu stage) -------------
say "resuming suite on $POST"
serve_up "$POST" "$ROOT/models/$POST" || exit 1
bash "$ROOT/pod/run_model.sh" "$POST" "$ROOT/models/$POST" 2>&1 | tee -a "$L/recover.log"
rc=${PIPESTATUS[0]}
serve_down
say "post-AFT suite rc=$rc"
[[ $rc -eq 0 ]] || exit 1

# --- 2. corrected per-run Gate 3 on the PRE-AFT model, + its provenance ------------------
say "corrected gate3 on $PRE"
serve_up "$PRE" "$ROOT/models/$PRE" || exit 1
"$ROOT/venv-serve/bin/python" "$ROOT/pod/gate3_dispatch_rate.py" \
    --model "$PRE" --parent-pct 38.5 --out "$L/gate3_${PRE}_perrun.json" 2>&1 | tee -a "$L/recover.log"
# the pre-AFT suite predates the PROVENANCE step, so write it now against the same pod
"$ROOT/venv-serve/bin/python" - "$ROOT/results/$PRE" "$PRE" <<'PY'
import json, subprocess, sys, torch, transformers, vllm
res, name = sys.argv[1], sys.argv[2]
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip() or None
    except Exception: return None
prov = {"model": name,
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        "vllm": vllm.__version__, "transformers_serve": transformers.__version__,
        "torch": torch.__version__,
        "suite_pin": sh("git -C /workspace/fried/vendor rev-parse HEAD")}
json.dump(prov, open(res + "/PROVENANCE.json", "w"), indent=2)
print("PROVENANCE:", json.dumps(prov))
PY
serve_down

say "=== RECOVERY COMPLETE ==="
tar -czf "$ROOT/results_$ARM.tar.gz" -C "$ROOT" results "logs/$ARM" 2>/dev/null
say "bundled -> $ROOT/results_$ARM.tar.gz ($(du -h "$ROOT/results_$ARM.tar.gz" | cut -f1))"
