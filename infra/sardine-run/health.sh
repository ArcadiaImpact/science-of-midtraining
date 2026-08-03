#!/usr/bin/env bash
# Is sardine-run actually healthy? Run from the laptop:
#
#   ./infra/sardine-run/health.sh
#
# Written after the idle sweeper sat dead for three days: it failed on every
# cron run with "RUNPOD_API_KEY is not set" and looked exactly like a sweeper
# with nothing to do. Anything that can fail silently gets checked here.
set -uo pipefail

HOST="${SARDINE_HOST:-sardine}"
fail=0
check() {  # check <label> <ok-condition-output> <expected-substring>
    local label="$1" got="$2" want="$3"
    if [[ "$got" == *"$want"* ]]; then
        printf '  \033[32mOK\033[0m   %-22s %s\n' "$label" "$got"
    else
        printf '  \033[31mFAIL\033[0m %-22s %s\n' "$label" "${got:-<empty>}"
        fail=1
    fi
}

echo "=== sardine-run health ==="
if ! ssh -o ConnectTimeout=15 -o BatchMode=yes "$HOST" true 2>/dev/null; then
    echo "  FAIL ssh unreachable. The port changes on restart -- try:"
    echo "       ./infra/sardine-run/ssh_config.sh <podId>"
    exit 1
fi

R="$(ssh "$HOST" '
echo "claude=$(claude --version 2>&1 | head -1)"
echo "node=$(node --version 2>&1)"
echo "uv=$(uv --version 2>&1 | cut -d" " -f2)"
echo "cron=$(pgrep -x cron >/dev/null && echo running || echo dead)"
echo "crontab=$(crontab -l 2>/dev/null | grep -c idle_sweeper)"
echo "mcp=$(cd /workspace/science-of-midtraining && claude mcp list 2>&1 | grep -ci "runpod.*Connected")"
echo "hb=$(python3 - <<PY
import json,os,time
p="/workspace/.sardine/last_run.json"
if not os.path.exists(p): print("never")
else:
    age=(time.time()-os.path.getmtime(p))/60
    print(f"{age:.0f}min")
PY
)"
echo "cronerr=$(tail -3 /workspace/.sardine/cron.log 2>/dev/null | grep -c "not set\|Traceback\|ERROR")"
')"
get() { echo "$R" | grep "^$1=" | cut -d= -f2-; }

check "claude"          "$(get claude)"  "Claude Code"
check "node"            "$(get node)"    "v"
check "uv"              "$(get uv)"      "."
check "cron daemon"     "$(get cron)"    "running"
check "sweeper cron"    "$(get crontab)" "1"
check "runpod mcp"      "$(get mcp)"     "1"

# The heartbeat is the real test: it only exists if the sweeper ran to
# completion. Stale or missing means the guardrail is off.
hb="$(get hb)"
if [[ "$hb" == "never" ]]; then
    printf '  \033[31mFAIL\033[0m %-22s sweeper has NEVER completed a run\n' "sweeper heartbeat"; fail=1
elif (( ${hb%min} > 25 )); then
    printf '  \033[31mFAIL\033[0m %-22s last completed run %s ago (expected <20min)\n' "sweeper heartbeat" "$hb"; fail=1
else
    printf '  \033[32mOK\033[0m   %-22s last run %s ago\n' "sweeper heartbeat" "$hb"
fi

# cron.log is only written on failure -- the sweeper's own output goes to
# sweeper.log. Anything here is an error, but the file is append-only, so
# only lines newer than the last successful heartbeat count.
if [[ "$(get cronerr)" != "0" ]]; then
    printf '  \033[31mFAIL\033[0m %-22s errors since last success:\n' "cron log"
    ssh "$HOST" 'find /workspace/.sardine/cron.log -newer /workspace/.sardine/last_run.json 2>/dev/null | grep -q . && tail -3 /workspace/.sardine/cron.log' | sed 's/^/         /'
    fail=1
fi

echo
echo "=== running pods ==="
# Queried from the laptop rather than over ssh: nesting python inside a quoted
# remote command is a quoting minefield, and the laptop has the same key.
if [ -f "$HOME/.sardine-run.env" ]; then
    ( set -a; . "$HOME/.sardine-run.env"; set +a
      curl -sS "https://rest.runpod.io/v1/pods" -H "Authorization: Bearer $RUNPOD_API_KEY" \
        > /tmp/.sardine_pods.json 2>/dev/null
      python3 -c '
import json
# /v1/pods returns a bare list; some other endpoints wrap in {"items": [...]}.
raw = json.load(open("/tmp/.sardine_pods.json"))
items = raw if isinstance(raw, list) else (raw.get("items") or [])
run = [p for p in items if p.get("desiredStatus") == "RUNNING"]
for p in run:
    print("  {:24s} ${}/hr".format(p.get("name", "?"), p.get("costPerHr")))
print("  total: ${:.2f}/hr".format(sum(p.get("costPerHr") or 0 for p in run)))
'
      rm -f /tmp/.sardine_pods.json )
else
    echo "  (no ~/.sardine-run.env on this machine, skipping)"
fi

echo
[ "$fail" -eq 0 ] && echo "ALL HEALTHY" || echo "PROBLEMS FOUND (see FAIL lines)"
exit "$fail"
